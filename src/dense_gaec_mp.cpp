#include "dense_gaec_mp.h"
#include "feature_index.h"
#include "feature_index_faiss.h"
#include "feature_index_brute_force.h"
#include "dense_multicut_utils.h"
#include "union_find.hxx"
#include "time_measure_util.h"
#include "incremental_nns.h"
#include "message_passing.h"
#include "hash_helper.h"

#include <vector>
#include <queue>
#include <numeric>
#include <random>
#include <iostream>

#include <faiss/index_factory.h>
#include <faiss/IndexFlat.h>
#include <faiss/MetaIndexes.h>

#include <faiss/IndexHNSW.h>

#include <faiss/IndexIVFPQ.h>
#include <faiss/index_io.h>
#include <faiss/impl/AuxIndexStructures.h>

namespace DENSE_MULTICUT {

    using pq_type = std::tuple<float, std::array<faiss::Index::idx_t,2>>;

    template<typename REAL>
    std::vector<size_t> dense_gaec_mp_impl(const size_t n, const size_t d, feature_index<REAL>& index, const std::vector<REAL>& features, const bool track_dist_offset, const size_t k_in, const size_t k_cap)
    {
        MEASURE_FUNCTION_EXECUTION_TIME;
        assert(features.size() == n*d);
        const size_t k = std::min(n - 1, k_in);
        const size_t k_cap_eff = std::min(n - 1, k_cap);

        std::cout << "[dense gaec MP] Find multicut for " << n << " nodes with features of dimension " << d << " with k " <<k<<" and K "<<k_cap_eff<<"\n";

        double multicut_cost = cost_disconnected(n, d, features, track_dist_offset);

        const size_t max_nr_ids = 2*n;
        union_find uf(max_nr_ids);

        incremental_nns<REAL> nn_graph;
        auto pq_comp = [](const pq_type& a, const pq_type& b) { return std::get<0>(a) < std::get<0>(b); };
        std::priority_queue<pq_type, std::vector<pq_type>, decltype(pq_comp)> pq(pq_comp);
        {
            MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME2("Initial KNN construction");
            std::vector<faiss::Index::idx_t> all_indices(n);
            std::iota(all_indices.begin(), all_indices.end(), 0);
            const auto [nns, distances] = index.get_nearest_nodes(all_indices, k);
            std::cout<<"[dense gaec MP] Initial NN search complete\n";
            nn_graph = incremental_nns(all_indices, nns, distances, n, k, k_cap_eff);
            size_t index_1d = 0;
            for(size_t i=0; i<n; ++i)
                for(size_t i_k=0; i_k < k; ++i_k, ++index_1d)
                    if(distances[index_1d] > 0.0)
                        pq.push({distances[index_1d], {i, nns[index_1d]}});
        }
        auto insert_into_pq = [&](const std::vector<std::tuple<size_t, size_t, REAL>>& edges) {
            for (const auto [i, j, cost]: edges)
            {
                assert(index.node_active(i));
                assert(index.node_active(j));
                pq.push({cost, {i, j}});
            }
        };

        //initialize MP
        message_passing<REAL> mp = message_passing(&index);
        mp.initialize_triangles();
        mp.message_passing_impl();
        std::unordered_map<std::array<size_t, 2>, std::tuple<REAL,size_t>> edges = mp.get_edges();
        auto pq_comp_2 = [](const std::tuple<REAL, std::array<size_t,2>>& a, const std::tuple<REAL, std::array<size_t,2>>& b) {
            return std::get<0>(a) < std::get<0>(b);
        };
        std::priority_queue<std::tuple<REAL, std::array<size_t,2>>, std::vector<std::tuple<REAL, std::array<size_t,2>>>, decltype(pq_comp_2)> reparam_pq(pq_comp_2);
        for (auto [edge, e_cost]: edges){
            reparam_pq.push(std::make_tuple(std::get<0>(e_cost), edge));
        }
        auto insert_into_reparam = [&](const std::vector<std::tuple<REAL, size_t, size_t>>& edges_) {
            for (auto [cost, a, b]: edges_)
            {
                assert(index.node_active(a));
                assert(index.node_active(b));
                reparam_pq.push({cost, {a, b}});
            }
        };

        // Contraction
        if (reparam_pq.empty()){
            while(!pq.empty())
            {
                const auto [distance, ij] = pq.top();
                pq.pop();
                assert(distance >= 0.0);
                const auto [i,j] = ij;
                assert(i != j);
                if(index.node_active(i) && index.node_active(j))
                {
                    const size_t new_id = index.merge(i, j, true);
                    uf.merge(i, new_id);
                    uf.merge(j, new_id);
                    insert_into_pq(nn_graph.merge_nodes(i, j, new_id, index, true));
                    // std::cout << "[dense gaec inc NN] contracting edge " << i << " and " << j << " with edge cost " << distance <<"\n";
                    multicut_cost -= distance;
                }
            }
        } else {
            while (!pq.empty()) {
                const auto [distance, ij] = pq.top();
                assert(distance >= 0.0);
                auto [i, j] = ij;
                assert(i != j);
                const auto [e_cost, edge] = reparam_pq.top();
                const auto [a, b] = edge;
                assert(a != b);


                if (distance > e_cost) {
                    // i < j for MP necessary
                    if (i > j) {
                        auto temp = i;
                        i = j;
                        j = temp;
                    }
                    if (mp.triangle_edge(i, j)) {
                        pq.pop();
                        continue;
                    }
                    if (index.node_active(i) && index.node_active(j)) {
                        std::tuple<size_t, std::vector<std::tuple<REAL, size_t, size_t>>> temp = mp.contract_edge({i, j});
                        auto [new_id, edges_] = temp;
                        uf.merge(i, new_id);
                        uf.merge(j, new_id);
                        pq.pop();
                        insert_into_pq(nn_graph.merge_nodes(i, j, new_id, index, true));
                        insert_into_reparam(edges_);
                        // std::cout << "[dense gaec MP] contracting edge " << i << " and " << j << " with edge cost " << distance <<"\n";
                        multicut_cost -= distance;
                    } else {
                        pq.pop();
                    }
                    continue;
                }

                if (index.node_active(a) && index.node_active(b)) {
                    std::tuple<size_t, std::vector<std::tuple<REAL, size_t, size_t>>> temp = mp.contract_edge({a, b});
                    auto [new_id, edges_] = temp;
                    uf.merge(a, new_id);
                    uf.merge(b, new_id);
                    reparam_pq.pop();
                    insert_into_pq(nn_graph.merge_nodes(a, b, new_id, index, true));
                    insert_into_reparam(edges_);
                    // std::cout << "[dense gaec MP] contracting edge " << i << " and " << j << " with edge cost " << distance <<"\n";
                    multicut_cost -= index.inner_product(a, b);
                } else {
                    reparam_pq.pop();
                }
            }

            while (!reparam_pq.empty()) {
                const auto [e_cost, edge] = reparam_pq.top();
                const auto [a, b] = edge;
                assert(a != b);
                if (e_cost > 0.0) {
                    if (index.node_active(a) && index.node_active(b)) {
                        std::tuple<size_t, std::vector<std::tuple<REAL, size_t, size_t>>> temp = mp.contract_edge(
                                {a, b});
                        auto [new_id, edges_] = temp;
                        uf.merge(a, new_id);
                        uf.merge(b, new_id);
                        reparam_pq.pop();
                        insert_into_pq(nn_graph.merge_nodes(a, b, new_id, index, true));
                        insert_into_reparam(edges_);
                        // std::cout << "[dense gaec MP] contracting edge " << i << " and " << j << " with edge cost " << distance <<"\n";
                        multicut_cost -= index.inner_product(a, b);
                    } else {
                        reparam_pq.pop();
                    }
                } else {
                    break;
                }
            }
        }

        std::cout << "[dense gaec MP] final nr clusters = " << index.nr_nodes() << "\n";
        std::cout << "[dense gaec MP] final multicut cost = " << multicut_cost << "\n";

        std::vector<size_t> component_labeling(n);
        for(size_t i=0; i<n; ++i)
            component_labeling[i] = uf.find(i);

        // std::cout << "[dense gaec MP] final multicut computed cost = " << labeling_cost(component_labeling, n, d, features, track_dist_offset) << "\n";
        return component_labeling;

    }







    std::vector<size_t> dense_gaec_mp_faiss(const size_t n, const size_t d, const std::vector<float>& features, const std::string index_str, const bool track_dist_offset, const size_t k_in, const size_t k_cap)
    {
        MEASURE_FUNCTION_EXECUTION_TIME;
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;
        std::cout << "Dense GAEC with faiss index: "<<index_str<<"\n";
        std::unique_ptr<feature_index_faiss> index = std::make_unique<feature_index_faiss>(d, n, features, index_str, track_dist_offset);
        return dense_gaec_mp_impl<float>(n, d, *index, features, track_dist_offset, k_in, k_cap);
    }

    template<typename REAL>
    std::vector<size_t> dense_gaec_mp_brute_force(const size_t n, const size_t d, const std::vector<REAL>& features, const bool track_dist_offset, const size_t k_in, const size_t k_cap)
    {
        MEASURE_FUNCTION_EXECUTION_TIME;
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;
        std::cout << "Dense GAEC with brute force\n";
        std::unique_ptr<feature_index_brute_force<REAL>> index = std::make_unique<feature_index_brute_force<REAL>>(
                d, n, features, track_dist_offset);
        return dense_gaec_mp_impl<REAL>(n, d, *index, features, track_dist_offset, k_in, k_cap);
    }

    template std::vector<size_t> dense_gaec_mp_brute_force(const size_t n, const size_t d, const std::vector<float>& features, const bool track_dist_offset, const size_t k_in, const size_t k_cap);
    template std::vector<size_t> dense_gaec_mp_brute_force(const size_t n, const size_t d, const std::vector<double>& features, const bool track_dist_offset, const size_t k_in, const size_t k_cap);
}
