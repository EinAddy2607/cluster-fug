#include "feature_index.h"
#include "feature_index_faiss.h"
#include "feature_index_brute_force.h"

#include <unordered_map>
#include <vector>
#include <queue>

namespace DENSE_MULTICUT{

    template<typename REAL>
    class message_passing{
        public:

            message_passing() {}
            message_passing(
                feature_index<REAL> *f_index
            );

            void add_triangle(std::array<size_t,3> t); // add triangle to set, check edge in edge, if not add
            void initialize_triangles();
            std::unordered_map<std::array<size_t, 2>, std::tuple<REAL,size_t>> get_edges();
            std::tuple<size_t,std::vector<std::tuple<REAL,size_t,size_t>>> contract_edge(const std::array<size_t,2>& e);
            static size_t get_index (std::array<size_t,3> triangle, const std::array<size_t,2>& edge);

            // REAL t_edge_cost(std::vector<std::array<REAL,3>> t_costs, std::vector<std::array<size_t,3>> t_set, const std::array<size_t,2>& edge);  //const
            REAL min_marginal(const REAL a, const REAL b, const REAL c); //const wenn keine Änderung der Daten

            void message_passing_routine ();
            void message_passing_impl ();

            bool triangle_edge(size_t i, size_t j) const;
            static bool edge_in_triangle(std::array<size_t,3> triangle, const std::array<size_t,2>& edge); // check edge in edges_
            REAL lower_bound() const; // check RAMA def

        private:
            feature_index<REAL> *f_index_;
            std::unordered_map<std::array<size_t,3>, std::array<REAL,3>> triangles_;
            std::unordered_map<std::array<size_t, 2>, std::tuple<REAL,size_t>> edges_; // (i,j) -> reparametrized cost, #triangles, in which ij is in


    };

}
