#ifndef OPTIMIZE_HPP
#define OPTIMIZE_HPP

#include <utility>

/**
 * @brief optimize_goldensectionsearch is a line optimization strategy
 * @param init start value
 * @param rng range to look in
 * @param function cost function
 * @return instance of T for which function is minimal
 */
template <typename T, typename F>
T optimize_goldensectionsearch(T init, T rng, F function)
{
   T sta = init - 0.382*rng;
   T end = init + 0.618*rng;
   T c = (end - (end-sta)/1.618);
   T d = (sta + (end-sta)/1.618);

   while (fabs(c-d) > 0.005) {
      if (function(c) < function(d)) {
         end = d;
      } else {
         sta = c;
      }

      c = (end - (end-sta)/1.618);
      d = (sta + (end-sta)/1.618);
   }

   return (end+sta)/2;
}

/**
 * @brief optimize_powell is a strategy to optimize a parameter space for a
 *        given cost function
 * @param init range with the initial values, optimized values are stored in
 *        there when the function returns
 * @param rng range containing the ranges in which each parameter is optimized
 * @param cost_function cost function for which the parameters are optimized
 */
template <typename Iter, typename Cf>
void optimize_powell(std::pair<Iter, Iter> init,
                     std::pair<Iter, Iter> rng,
                     Cf cost_function)
{
   using TPS = typename std::remove_reference<decltype(*init.first)>::type;

   bool converged = false;

   const double eps = 0.00005;

   double last_mutualinf = 100000.0;
   while (!converged) {
      converged = true;

      for (auto it = init.first; it != init.second; ++it) {
         std::size_t pos = it - init.first;
         auto curr_param = init.first[pos];
         auto curr_rng = rng.first[pos];

         auto fn = [pos, init, &cost_function](TPS p)
         {
            init.first[pos] = p;
            return cost_function(init.first);
         };

         auto param_optimized = optimize_goldensectionsearch(curr_param, curr_rng, fn);

         auto curr_mutualinf = cost_function(init.first);
         init.first[pos] = curr_param;
         if (last_mutualinf - curr_mutualinf > eps) {
            *it = param_optimized;
            last_mutualinf = curr_mutualinf;
            converged = false;
         } else {
            *it = curr_param;
         }
      }
   }
}

#endif // OPTIMIZE_HPP
