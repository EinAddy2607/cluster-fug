#include "test.h"
#include "message_passing.h"
#include "hash_helper.h"
#include "feature_index.h"
#include "dense_gaec_mp.h"
#include "dense_gaec_inc_nn.h"
#include "dense_multicut_utils.h"

#include <unordered_map>
#include <vector>
#include <iostream>
#include <random>

using namespace DENSE_MULTICUT;

template<typename REAL>
void test_edge_in_triangle(message_passing<REAL> mp, std::array<size_t,3> triangle, const std::array<size_t,2>& edge)
{
    std::cout << "[test message passing] test static problem 1 for edge in triangle check\n";
    test(mp.edge_in_triangle(triangle, edge));
    std::cout << "\n[test message passing] test over\n";
}

template<typename REAL>
void test_edge_in_triangle_2(message_passing<REAL> mp, std::array<size_t,3> triangle, const std::array<size_t,2>& edge)
{
    std::cout << "[test message passing] test static problem 2 for edge in triangle check\n";
    test(!mp.edge_in_triangle(triangle, edge));
    std::cout << "\n[test message passing] test over\n";
}

template<typename REAL>
void test_get_index(message_passing<REAL> mp, std::array<size_t,3> triangle, const std::array<size_t,2>& edge)
{
    std::cout << "[test message passing] test static problem 1 for fetching index\n";
    test(mp.get_index(triangle, edge) == 1);
    std::cout << "\n[test message passing] test over\n";
}

template<typename REAL>
void test_get_index_2(message_passing<REAL> mp, std::array<size_t,3> triangle, const std::array<size_t,2>& edge)
{
    std::cout << "[test message passing] test static problem 2 for fetching index\n";
    test(mp.get_index(triangle, edge) == 0);
    std::cout << "\n[test message passing] test over\n";
}

template<typename REAL>
void test_min_marginal(message_passing<REAL> mp, std::array<REAL,3> t_cost, const size_t index)
{
    std::cout << "[test message passing] test static problem 1 for calculating the min marginal\n";
    double min_marg = 0.0;

    const REAL c_ij = t_cost[0];
    const REAL c_ik = t_cost[1];
    const REAL c_jk = t_cost[2];

    if (index == 0) {
        min_marg = mp.min_marginal(c_ij, c_ik, c_jk);
    } else if (index == 1) {
        min_marg = mp.min_marginal(c_ik, c_ij, c_jk);
    } else if (index == 2) {
        min_marg = mp.min_marginal(c_jk, c_ij, c_ik);
    }

    //std::cout << "[test message passing] test result for static problem 1 is" << min_marg <<"\n";
    test(min_marg == 2.5);
    std::cout << "\n[test message passing] test over\n";
}

template<typename REAL>
void test_min_marginal_2(message_passing<REAL> mp, std::array<REAL,3> t_cost, const size_t index)
{
    std::cout << "[test message passing] test static problem 2 for calculating the min marginal\n";
    double min_marg = 0.0;

    const REAL c_ij = t_cost[0];
    const REAL c_ik = t_cost[1];
    const REAL c_jk = t_cost[2];


    if (index == 0) {
        min_marg = mp.min_marginal(c_ij, c_ik, c_jk);
    } else if (index == 1) {
        min_marg = mp.min_marginal(c_ik, c_ij, c_jk);
    } else if (index == 2) {
        min_marg = mp.min_marginal(c_jk, c_ij, c_ik);
    }


    //std::cout << "[test message passing] test result for static problem 2 is" << min_marg <<"\n";
    test(min_marg == 0.0);
    std::cout << "\n[test message passing] test over\n";
}

/*template<typename REAL>
void test_t_edge_cost (const feature_index<REAL>& f_index, std::vector<std::array<size_t,3>> t_set, std::vector<std::array<REAL,3>> t_costs, const std::array<size_t,2>& edge)
{
    std::cout << "[test message passing] test static problem for calculating the triangle edge cost\n";
    const auto [i,j] = edge;
    REAL inner_product = f_index.inner_product(i,j);
    // std::cout << "[test message passing] test inner product for static problem 1 is " << inner_product <<"\n";

    const feature_index<REAL> *f_index_ = & f_index;
    message_passing<REAL> mp = message_passing(f_index_);
    for (size_t i = 0; i < t_set.size(); ++i){
        mp.add_triangle(t_set[i]);
    }
    
    REAL costs = mp.t_edge_cost(t_costs, t_set, edge);
    // std::cout << "[test message passing] test t_edge_cost for static problem 1 is " << costs <<"\n";
    REAL comparison = (inner_product + 4.5);
    // std::cout << "[test message passing] test comparison value for static problem 1 is " << comparison <<"\n";
    test(costs == comparison);
    std::cout << "\n[test message passing] test over\n";
}*/

// andere LB test

template<typename REAL>
void test_msg_passing_lb(const feature_index<REAL>& f_index, std::vector<std::array<size_t,3>> t_set)
{
    std::cout << "[test message passing] test static problem for calculating the lower bound \n";
    const feature_index<REAL> *f_index_ = & f_index;
    message_passing<REAL> mp = message_passing(f_index_);
    for (size_t i = 0; i < t_set.size(); ++i){
        mp.add_triangle(t_set[i]);
    }
    const REAL lb_before = mp.lower_bound();
    mp.message_passing_routine();
    const REAL lb_after = mp.lower_bound();
    // std::cout << "[test message passing] test static problem lb before = " << lb_before << " , lb after = " << lb_after <<" \n";
    test(lb_before <= lb_after + 1e-5);
    std::cout << "\n[test message passing] test over\n";
}

template<typename REAL>
void test_msg_passing_impl(const feature_index<REAL>& f_index, std::vector<std::array<size_t,3>> t_set)
{
    std::cout << "[test message passing] test static problem for calculating the lower bound on impl \n";
    const feature_index<REAL> *f_index_ = & f_index;
    message_passing<REAL> mp = message_passing(f_index_);
    for (size_t i = 0; i < t_set.size(); ++i){
        mp.add_triangle(t_set[i]);
    }
    const REAL lb_before = mp.lower_bound();
    mp.message_passing_impl();
    const REAL lb_after = mp.lower_bound();
    // std::cout << "[test message passing] test static problem lb before = " << lb_before << " , lb after = " << lb_after <<" \n";
    test(lb_before <= lb_after + 1e-5);
    std::cout << "\n[test message passing] test over\n";
}

template<typename REAL>
void test_triangle_lb(const feature_index<REAL>& f_index)
{
    std::cout << "[test message passing] test static triangle problem for calculating the lower bound\n";


    const size_t n = 3;
    const size_t d = 2;
    std::vector<float> features(n*d);
    std::mt19937 generator(1); // for deterministic behaviour
    for(int experiment=0; experiment<10; ++experiment) {
        std::uniform_real_distribution<float> distr(-1.0, 1.0);
        for (size_t i = 0; i < n * d; ++i) { features[i] = distr(generator); }
        feature_index_faiss index = feature_index_faiss(d, n, features, "Flat");

        message_passing<REAL> mp = message_passing(&index);

        test(index.nr_nodes() == 3);

        std::array<size_t, 3> triangle = {0, 1, 2};
        mp.add_triangle(triangle);


        /* std::cout << "[test message passing] test edge costs = " << index.inner_product(0, 1) << " "
                  << index.inner_product(0, 2) << " " << index.inner_product(1, 2) << " \n";*/
        const REAL lb_before = mp.lower_bound();
        // std::cout << "[test message passing] test static problem lb before = " << lb_before << " \n";

        const REAL lb_before_by_hand = std::min(0.0, index.inner_product(0,1)) + std::min(0.0, index.inner_product(0,2)) + std::min(0.0, index.inner_product(1,2));
        // std::cout << "[test message passing] test static problem lb by hand = " << lb_before_by_hand << " \n";
        test(floor(lb_before_by_hand * 100000) == floor(lb_before * 100000));

        mp.message_passing_routine();
        const REAL lb_after_one = mp.lower_bound();
        // std::cout << "[test message passing] test static problem lb after one loop = " << lb_after_one << " \n";
        //mp.message_passing_routine();
        // std::cout << "[test message passing] test static problem lb before = " << lb_before << " , lb end = " << lb_end <<" \n";
        test(lb_before <= lb_after_one + 1e-5);
        //test(lb_end == lb_after_one);
    }
    std::cout << "\n[test message passing] test over\n";
}

template<typename REAL>
void test_square_lb(const feature_index<REAL>& f_index)
{
    std::cout << "[test message passing] test static square problem for calculating the lower bound\n";
    // message_passing<REAL> mp = message_passing(f_index_);

    size_t n = 4;
    size_t d = 3;
    std::vector<float> features(n*d);
    std::mt19937 generator(0); // for deterministic behaviour
    for (int experiment=0; experiment<50; ++experiment) {
        std::uniform_real_distribution<float> distr(-5.0, 5.0);
        for (size_t i = 0; i < n * d; ++i) { features[i] = distr(generator); }
        feature_index_faiss index = feature_index_faiss(d, n, features, "Flat");

        message_passing<REAL> mp = message_passing(&index);

        std::array<size_t, 3> triangle_a = {0, 1, 2};
        std::array<size_t, 3> triangle_c = {0, 1, 3};
        std::array<size_t, 3> triangle_d = {0, 2, 3};
        std::array<size_t, 3> triangle_b = {1, 2, 3};
        mp.add_triangle(triangle_a);
        mp.add_triangle(triangle_b);
        mp.add_triangle(triangle_c);
        mp.add_triangle(triangle_d);

        const REAL lb_before = mp.lower_bound();
        mp.message_passing_impl();
        const REAL lb_after = mp.lower_bound();
        /* std::cout << "[test message passing] test static problem lb before = " << lb_before << " , lb after = "
                  << lb_after << " \n"; */
        test(lb_before <= lb_after + 1e-5);
    }
    std::cout << "\n[test message passing] test over\n";
}

template<typename REAL>
void test_perfect_square_lb(const feature_index<REAL>& f_index)
{
    std::cout << "[test message passing] test static square problem for calculating the lower bound\n";
    // message_passing<REAL> mp = message_passing(f_index_);
    size_t n = 4;
    size_t d = 3;
    std::vector<float> features(n*d);
    std::mt19937 generator(0); // for deterministic behaviour
    for (int experiment=0; experiment<100; ++experiment) {
        std::uniform_real_distribution<float> distr(-1.0, 1.0);
        for (size_t i = 0; i < n * d; ++i) { features[i] = distr(generator); }

        feature_index_faiss index = feature_index_faiss(d, n, features, "Flat");

        message_passing<REAL> mp = message_passing(&index);

        std::array<size_t,3> triangle_a = {0,1,2};
        std::array<size_t,3> triangle_b = {0,2,3};
        mp.add_triangle(triangle_a);
        mp.add_triangle(triangle_b);

        const REAL lb_before = mp.lower_bound();
        mp.message_passing_impl();
        const REAL lb_after = mp.lower_bound();
        // std::cout << "[test message passing] test static problem lb before = " << lb_before << " , lb after = " << lb_after <<" \n";
        // std::cout << "[test message passing] test static problem difference = " << lb_before - lb_after <<" \n";
        test(lb_before <= lb_after + 1e-5);
    }
    std::cout << "\n[test message passing] test over\n";
}

template<typename REAL>
void test_contract_edge(const feature_index<REAL>& f_index)
{
    std::cout << "[test message passing] test singular edge contraction\n";
    // message_passing<REAL> mp = message_passing(f_index_);
    size_t n = 4;
    size_t d = 2;
    std::vector<float> features(n*d);
    std::mt19937 generator(0); // for deterministic behaviour
    for (int experiment=0; experiment<100; ++experiment) {
        std::uniform_real_distribution<float> distr(-1.0, 1.0);
        for (size_t i = 0; i < n * d; ++i) { features[i] = distr(generator); }

        feature_index_faiss index = feature_index_faiss(d, n, features, "Flat");

        message_passing<REAL> mp = message_passing(&index);

        std::array<size_t, 3> triangle_a = {0, 1, 2};
        std::array<size_t, 3> triangle_b = {0, 2, 3};
        std::array<size_t, 3> triangle_c = {0, 1, 3};
        std::array<size_t, 3> triangle_d = {1, 2, 3};
        mp.add_triangle(triangle_a);
        mp.add_triangle(triangle_b);
        mp.add_triangle(triangle_c);
        mp.add_triangle(triangle_d);

        std::array<size_t, 2> edge = {0,2};
        std::unordered_map<std::array<size_t, 2>, std::tuple<REAL,size_t>> edges_before = mp.get_edges();
        mp.contract_edge(edge);
        std::unordered_map<std::array<size_t, 2>, std::tuple<REAL,size_t>> edges_after = mp.get_edges();
        // std::cout << "\n[test message passing] size = " << edges.size() <<"\n";
        test(edges_before.size() == 6);
        test(edges_after.size() == 3);
        test(std::get<0>(edges_after[{3,4}]) == std::get<0>(edges_before[{0,3}]) + std::get<0>(edges_before[{2,3}]));
        test(std::get<0>(edges_after[{1,4}]) == std::get<0>(edges_before[{0,1}]) + std::get<0>(edges_before[{1,2}]));
    }
    std::cout << "\n[test message passing] test over\n";
}

void test_full_contraction(size_t n, size_t d)
{
    std::cout << "[test message passing] test full contraction\n";
    size_t mp_count = 0;
    size_t std_count = 0;
    size_t eq_count = 0;
    // message_passing<REAL> mp = message_passing(f_index_);
    std::vector<double> features(n*d);
    std::mt19937 generator(0); // for deterministic behaviour
    for (int experiment=0; experiment<50; ++experiment) {
        std::uniform_real_distribution<double> distr(-1.0, 1.0);
        for (size_t i = 0; i < n * d; ++i) { features[i] = distr(generator); }

        const auto gaec_inc_nn_bf_cost = labeling_cost<double>(dense_gaec_inc_nn_brute_force<double>(n, d, features, false, 1, 1), n, d, features);
        const auto mp_inc_nn_bf_cost = labeling_cost<double>(dense_gaec_mp_brute_force<double>(n, d, features, false, 1, 1), n, d, features);

        std::cout << "[test message passing] gaec cost = " << gaec_inc_nn_bf_cost << " & mp cost = " << mp_inc_nn_bf_cost << "\n \n";

        if (mp_inc_nn_bf_cost > gaec_inc_nn_bf_cost) {
            ++std_count;
        } else if (mp_inc_nn_bf_cost < gaec_inc_nn_bf_cost) {
            ++mp_count;
        } else {
            ++eq_count;
        }
        // test(gaec_inc_nn_bf_cost >= mp_inc_nn_bf_cost);
    }
    std::cout << "\n[test message passing] MP " << mp_count << " - " << eq_count << " - " << std_count << " STD \n";
    std::cout << "\n[test message passing] test over\n";
}

int main(int argc, char** argv)
{
    message_passing<float> mp;

    std::array<size_t,3> triangle_a = {1,3,4};
    std::array<size_t,3> triangle_b = {1,2,3};
    std::array<size_t,2> edge_a = {1,4};
    std::array<size_t,2> edge_b = {1,3};
    test_edge_in_triangle(mp, triangle_a, edge_a);
    test_edge_in_triangle_2(mp, triangle_b, edge_a);
    test_get_index(mp, triangle_a, edge_a);
    test_get_index_2(mp, triangle_a, edge_b);

    std::array<size_t,3> triangle_c = {1,2,4};
    std::array<size_t,3> triangle_d = {1,2,5};
    std::array<size_t,3> triangle_e = {1,4,5};
    std::array<size_t,3> triangle_f = {2,3,4};
    // std::array<size_t,3> triangle_g = {3,4,5};
    // std::array<size_t,3> triangle_h = {1,5,6};

    std::vector<std::array<size_t,3>> t_set;
    t_set.push_back(triangle_a);
    t_set.push_back(triangle_b);
    t_set.push_back(triangle_c);
    t_set.push_back(triangle_d);
    t_set.push_back(triangle_e);
    t_set.push_back(triangle_f);
    // t_set.push_back(triangle_g);
    // t_set.push_back(triangle_h);

    //test_count(mp, t_set, edge_a);
    // test_count_2(mp, t_set, edge_b);

    std::array<float,3> t_cost_a = {1.0,1.5,2.0};
    std::array<float,3> t_cost_b = {-1.0,1.0,2.0};
    /*std::array<float,3> t_cost_c = {1.0,2.0,0.0};
    std::array<float,3> t_cost_d = {0.0,-1.0,-2.0};
    std::array<float,3> t_cost_e = {1.0,1.0,0.0};
    std::array<float,3> t_cost_f = {2.0,-1.0,-2.0};
    
    std::vector<std::array<float,3>> t_costs;
    t_costs.push_back(t_cost_a);
    t_costs.push_back(t_cost_b);
    t_costs.push_back(t_cost_c);
    t_costs.push_back(t_cost_d);
    t_costs.push_back(t_cost_e);
    t_costs.push_back(t_cost_f);*/


    test_min_marginal(mp, t_cost_a, 1);
    test_min_marginal_2(mp, t_cost_b, 0);


    size_t n = 6;
    size_t d = 2;

    std::vector<float> features(n*d);
    std::mt19937 generator(0); // for deterministic behaviour
    std::uniform_real_distribution<float>  distr(-1.0, 1.0);
    for(size_t i=0; i<n*d; ++i){features[i] = distr(generator);}

    feature_index_faiss index = feature_index_faiss(d, n, features, "Flat");
    // std::unique_ptr<feature_index_faiss> index = std::make_unique<feature_index_faiss>(d, n, features, "Flat", 0);

    std::unordered_map<std::array<size_t,3>, std::array<float,3>, std::hash<std::array<std::size_t,3>>> triangles;
    
    /*for(size_t p = 0; p < t_set.size(); ++p){
        triangles.insert({t_set[p],t_costs[p]});
    }*/



    //test_t_edge_cost(index, t_set, t_costs, edge_a);
    // double costs = t_edge_cost(index, features, triangles, edge_a);

    // test_msg_pass_rout(index, features, triangles);
    test_triangle_lb(index);
    // test_square_lb(index);
    test_perfect_square_lb(index);
    // test_msg_passing_impl(index, t_set);
    // test_msg_passing_lb(index, t_set);
    test_contract_edge(index);

    const std::vector<size_t> nr_nodes = {4,10,20,50};
    const std::vector<size_t> nr_dims = {2,4,6,16,32};
    for(const size_t n : nr_nodes)
        for(const size_t d : nr_dims)
            test_full_contraction(n, d);

}
