#ifndef OPTIMIZE_HPP
#define OPTIMIZE_HPP

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

   while (abs(c-d) > 0.005) {
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

#endif // OPTIMIZE_HPP
