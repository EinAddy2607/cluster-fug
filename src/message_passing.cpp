#include "message_passing.h"
#include "hash_helper.h"
#include "feature_index.h"
#include "time_measure_util.h"

#include <unordered_map>
#include <vector>
#include <iostream>
#include <queue>


#include <faiss/IndexIVFPQ.h>
#include <faiss/impl/AuxIndexStructures.h>

namespace DENSE_MULTICUT{

    template<typename REAL>
    message_passing<REAL>::message_passing(feature_index<REAL> *f_index)
        :f_index_(f_index)
    {
        std::unordered_map<std::array<size_t,3>, std::array<REAL,3>> triangles_;
        std::unordered_map<std::array<size_t, 2>, std::tuple<REAL,size_t>> edges_;
    }

    template<typename REAL>
    void message_passing<REAL>::add_triangle(std::array<size_t,3> t){
        std::array<REAL,3> costs = {0.0,0.0,0.0};

        assert (triangles_.count(t) == 0);
        assert (t[0] < t[1] && t[1] < t[2]);

        triangles_.insert(std::make_pair(t, costs));

        assert (triangles_.count(t) == 1);

        auto const [i,j,k] = t;
        const std::array<size_t,2> ij = {i,j};
        const std::array<size_t,2> ik = {i,k};
        const std::array<size_t,2> jk = {j,k};

        if (edges_.count(ij) == 0){
            std::tuple<REAL,size_t> temp = {f_index_->inner_product(i,j), 1};
            edges_.insert(std::make_pair(ij,temp));
        } else {
            std::get<1>(edges_[ij]) += 1;
        }
        if (edges_.count(ik) == 0){
            std::tuple<REAL,size_t> temp = {f_index_->inner_product(i,k), 1};
            edges_.insert(std::make_pair(ik,temp));
        } else {
            std::get<1>(edges_[ik]) += 1;
        }
        if (edges_.count(jk) == 0){
            std::tuple<REAL,size_t> temp = {f_index_->inner_product(j,k), 1};
            edges_.insert(std::make_pair(jk,temp));
        } else {
            std::get<1>(edges_[jk]) += 1;
        }
    }

    template<typename REAL>
    void message_passing<REAL>::initialize_triangles(){
        assert(triangles_.empty());
        for (size_t i = 0; i < f_index_->nr_nodes(); i++){ //primitive solution, add all possible triangles
            for (size_t j = i + 1; j < f_index_->nr_nodes(); j++){
                for (size_t k = j + 1; k < f_index_->nr_nodes(); k++){
                    REAL c_ij = f_index_->inner_product(i,j);
                    // std::cout<<"[dense gaec MP]" << i << " & " << j << " = " << c_ij << "\n";
                    REAL c_ik = f_index_->inner_product(i,k);
                    // std::cout<<"[dense gaec MP]" << i << " & " << k << " = " << c_ik << "\n";
                    REAL c_jk = f_index_->inner_product(j,k);
                    // std::cout<<"[dense gaec MP]" << j << " & " << k << " = " << c_jk << "\n";
                    size_t count = 0;
                    if (c_ij < 0) {
                        ++count;
                    }
                    if (c_ik < 0) {
                        ++count;
                    }
                    if (c_jk < 0) {
                        ++count;
                    }
                    if (count == 1) {
                        assert(i < j && j < k);
                        std::array<size_t,3> t = {i,j,k};
                        assert(triangles_.count(t) == 0);
                        add_triangle(t);
                    }
                }
            }
        }
        std::cout<<"[dense gaec MP] Initial triangle search complete with "<< triangles_.size() << " triangles\n";
    }

    template<typename REAL>
    std::unordered_map<std::array<size_t, 2>, std::tuple<REAL,size_t>> message_passing<REAL>::get_edges(){
        return edges_;
    }


    template<typename REAL>
    std::tuple<size_t,std::vector<std::tuple<REAL,size_t,size_t>>> message_passing<REAL>::contract_edge(const std::array<size_t,2>& e){
        auto const [i , j] = e;
        assert (i < j);
        assert(f_index_->nr_nodes() > 1);
        assert (f_index_->node_active(i) && f_index_->node_active(j));

        // Adjust cost for edges with i or j
        for (auto [edge, e_cost]: edges_){
            auto const [a,b] = edge;
            // Edge: [i,x] where x != j and [x,i]
            if (a == i && b != j){
                if (b < j) {
                    if (triangle_edge(b,j)){
                        std::get<0>(edges_[edge]) += std::get<0>(edges_[{b,j}]);
                    } else {
                        std::get<0>(edges_[edge]) += f_index_->inner_product(b, j);
                    }
                } else {
                    if (triangle_edge(j,b)){
                        std::get<0>(edges_[edge]) += std::get<0>(edges_[{j,b}]);
                    } else {
                        std::get<0>(edges_[edge]) += f_index_->inner_product(j, b);
                    }
                }
                continue;
            } else if (b == i){
                if (triangle_edge(a,j)){
                    std::get<0>(edges_[edge]) += std::get<0>(edges_[{a,j}]);
                } else {
                    std::get<0>(edges_[edge]) += f_index_->inner_product(a, j);
                }
                continue;
            }

            // Edge: [j,x] and [x,j] where x != i
            if (a == j){
                if (triangle_edge(i,b)){
                    std::get<0>(edges_[edge]) += std::get<0>(edges_[{i,b}]);
                } else {
                    std::get<0>(edges_[edge]) += f_index_->inner_product(i,b);
                }
            } else if (a != i && b == j){
                if (a < i) {
                    if (triangle_edge(a,i)){
                        std::get<0>(edges_[edge]) += std::get<0>(edges_[{a,i}]);
                    } else {
                        std::get<0>(edges_[edge]) += f_index_->inner_product(a, i);
                    }
                } else {
                    if (triangle_edge(i,a)){
                        std::get<0>(edges_[edge]) += std::get<0>(edges_[{i,a}]);
                    } else {
                        std::get<0>(edges_[edge]) += f_index_->inner_product(i, a);
                    }
                }
            }
        }

        // Delete original reparam. edge
        if (triangle_edge(i,j)) {
            edges_.erase(e);
        }
        const size_t m = f_index_->merge(i,j);

        std::unordered_map<std::array<size_t, 2>, std::tuple<REAL,size_t>> newMap;
        std::vector<std::tuple<REAL,size_t,size_t>> newEdges;

        // Adjust reference for edges with i or j to m
        for (auto [edge, e_cost]: edges_){

            auto const [a,b] = edge;
            if (a == i){
                if (b < j) {
                    edges_.erase({b,j});
                } else {
                    edges_.erase({j,b});
                }
                const std::array<size_t,2> temp = {b,m};
                newMap.insert(std::make_pair(temp, e_cost));
                newEdges.push_back(std::make_tuple(std::get<0>(e_cost),b,m));
            } else if (b == i){
                edges_.erase({a,j});
                const std::array<size_t,2> temp = {a,m};
                newMap.insert(std::make_pair(temp, e_cost));
                newEdges.push_back(std::make_tuple(std::get<0>(e_cost),a,m));
            } else if (a == j){
                edges_.erase({i,b});
                const std::array<size_t,2> temp = {b,m};
                newMap.insert(std::make_pair(temp, e_cost));
                newEdges.push_back(std::make_tuple(std::get<0>(e_cost),b,m));
            } else if (b == j){
                if (a < i) {
                    edges_.erase({a,i});
                } else {
                    edges_.erase({i,a});
                }
                const std::array<size_t,2> temp = {a,m};
                newMap.insert(std::make_pair(temp, e_cost));
                newEdges.push_back(std::make_tuple(std::get<0>(e_cost),a,m));
            } else {
                newMap.insert(std::make_pair(edge, e_cost));
            }
        }
        edges_ = newMap;
        return std::make_tuple(m,newEdges);
    }


    template<typename REAL>
    bool message_passing<REAL>::edge_in_triangle(std::array<size_t,3> triangle, const std::array<size_t,2>& edge)
    {
        auto const [a,b,c] = triangle;
        auto const [i,j] = edge;

        assert (i < j);
        assert (triangle[0] < triangle[1] && triangle[1] < triangle[2]);

        if (a == i){
            if (b == j || c == j){
                return true;
            }
        } else if (b == i && c == j){
            return true;
        }
        return false;
    }

    template<typename REAL>
    size_t message_passing<REAL>::get_index (std::array<size_t,3> triangle, const std::array<size_t,2>& edge){
        //Requirement: for edge: i<j; for triangles i<j<k & ij,ik,jk
        auto const [i,j] = edge;

        assert (i < j);
        assert (triangle[0] < triangle[1] && triangle[1] < triangle[2]);
        assert (edge_in_triangle(triangle, edge) == true);

        if (triangle[0] == i){
            if(triangle[1] == j){
                return 0;
            } else {
                return 1;
            }
        } else {
            return 2;
        }
    }
    

    template<typename REAL>
    REAL message_passing<REAL>::min_marginal(const REAL a, const REAL b, const REAL c)
    {
        REAL one_marg = std::numeric_limits<REAL>::infinity();
        REAL zero_marg = std::numeric_limits<REAL>::infinity();
        REAL zero = 0.0;

        one_marg = std::min(a+b+c, std::min(a+b, a+c));
        zero_marg = std::min(zero, b+c);

        assert(one_marg < std::numeric_limits<REAL>::infinity());
        assert(zero_marg < std::numeric_limits<REAL>::infinity());
        // std::cout << "[test message passing] test min_marg = " << (one_marg - zero_marg) <<" \n";
        return (one_marg - zero_marg); 
    }


    template<typename REAL>
    void message_passing<REAL>::message_passing_routine()
    {
        std::vector<std::array<size_t,3>> t_set;
        std::vector<std::array<REAL,3>> t_costs;
        for(auto kv:triangles_){
            t_set.push_back(kv.first);
            t_costs.push_back(kv.second);
        }

        // Pass Messages from edges to triangles
        for (auto [edge, e_cost] : edges_){
            REAL a = std::get<0>(e_cost);
            REAL count = std::get<1>(e_cost);
            assert (count > 0);
            REAL temp = a/count;
            for (size_t k=0; k < t_set.size(); ++k){
                if (edge_in_triangle(t_set[k], edge)){
                    auto [l_a, l_b, l_c] = t_costs[k];
                    size_t index = get_index(t_set[k], edge);
                    switch(index){
                        case 0:
                            l_a += temp;
                            break;
                        case 1:
                            l_b += temp;
                            break;
                        case 2:
                            l_c += temp;
                            break;
                    }
                    std::array<REAL,3> costs = {l_a,l_b,l_c};
                    t_costs[k] = costs;
                }
            }
            std::get<0>(edges_[edge]) = 0.0;
       }

        // Pass Messages from triangles to edges
        for (size_t r=0; r < t_set.size(); ++r){
            const auto [i, j, k] = t_set[r];
            assert(i < j && j < k);
            std::array<REAL,3> current_costs = {0.0,0.0,0.0};
            current_costs = t_costs[r];
            const std::array<size_t,2> ij = {i,j};
            const std::array<size_t,2> ik = {i,k};
            const std::array<size_t,2> jk = {j,k};
            REAL c_ij = current_costs[0];
            REAL c_ik = current_costs[1];
            REAL c_jk = current_costs[2];

            REAL l_ij = 0.0;
            REAL l_ik = 0.0;
            REAL l_jk = 0.0;

            l_ij = min_marginal(c_ij, c_ik, c_jk)/3.0;
            c_ij -= l_ij;
            std::get<0>(edges_[ij]) += l_ij;
            l_ik = min_marginal(c_ik,c_ij, c_jk)/2.0;
            c_ik -= l_ik;
            std::get<0>(edges_[ik]) += l_ik;
            l_jk = min_marginal(c_jk, c_ij, c_ik);
            c_jk -= l_jk;
            std::get<0>(edges_[jk]) += l_jk;
            l_ij = min_marginal(c_ij, c_ik, c_jk)/2.0;
            c_ij -= l_ij;
            std::get<0>(edges_[ij]) += l_ij;
            l_ik = min_marginal(c_ik,c_ij, c_jk);
            c_ik -= l_ik;
            std::get<0>(edges_[ik]) += l_ik;
            l_jk = min_marginal(c_jk, c_ij, c_ik);
            c_jk -= l_jk;
            std::get<0>(edges_[jk]) += l_jk;


            // std::cout << " min-marginals: " << min_marginal(current_costs, ij_index) << " , " << min_marginal(current_costs, ik_index) << " , " << min_marginal(current_costs, jk_index) << "\n";

            triangles_[t_set[r]] = {c_ij, c_ik, c_jk};
        }
    }

    template<typename REAL>
    bool message_passing<REAL>::triangle_edge(size_t i, size_t j) const{
        assert(i < j);
        std::array<size_t,2> edge = {i,j};
        return (edges_.count(edge) != 0);
    }


    template<typename REAL>
    REAL message_passing<REAL>::lower_bound() const
    {
        // lower bound for edge
        REAL lb = 0.0;
        for(size_t i=0; i < f_index_->nr_nodes(); ++i) {
            for(size_t j=i+1; j < f_index_->nr_nodes(); ++j){
                assert(i < j);
                REAL edge_cost = f_index_->inner_product(i,j);
                if(triangle_edge(i,j)){
                    std::array<size_t, 2> edge = {i,j};
                    edge_cost = std::get<0>(edges_.find(edge)->second);
                }
                REAL zero = 0.0;
                lb += std::min(zero, edge_cost);
            }
        }

        // lower bound for triangles
        for (auto const [t,c]:triangles_)
        {
            const auto [lambda_ij, lambda_ik, lambda_jk] = c;
            REAL zero = 0.0;
            std::array<REAL, 5> list =  {
                zero,
                lambda_ij + lambda_ik,
                lambda_ij + lambda_jk,
                lambda_ik + lambda_jk,
                lambda_ik + lambda_ij + lambda_jk
                };
            REAL min = std::numeric_limits<REAL>::infinity();
            for (REAL x:list){
                if(x < min){
                    min = x;
                }
            }
            assert (min <= 0.0);
            lb += min;
        }
        return lb;
    }

    template<typename REAL>
    void message_passing<REAL>::message_passing_impl()
    {
        REAL lb_prior = -std::numeric_limits<REAL>::infinity();
        for(size_t i = 0; i<500; ++i){
            message_passing_routine();
            // std::cout<<"[dense gaec MP] " << i+1 <<" Lower bound = " << lower_bound() << "\n";
            if (lb_prior == lower_bound()){
                break;
            }
            lb_prior = lower_bound();
        }
        std::cout<<"[dense gaec MP] Message passing complete\n";
    }

    template class message_passing<float>;
    template class message_passing<double>;
}
