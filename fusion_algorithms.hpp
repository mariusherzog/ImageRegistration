#ifndef FUSION_ALGORITHMS_HPP
#define FUSION_ALGORITHMS_HPP

#include <map>
#include <string>
#include <algorithm>

#include "fusion.hpp"


static alphablend alpblnd {0.5};

/**
 * @brief The fusion_algorithms class is a facade which facilitates easy access
 *        to all available fusion strategies. Each strategies is uniquely
 *        identified by its name and can be accessed with it.
 */
class fusion_algorithms
{
   private:
      static const std::map<std::string, fusion*> algorithms;

   public:
      /**
       * @brief pick returns the fusion algorithm for the given name
       * @param name identifying name / key of the fusion strategy
       * @return instance of the fusion strategy
       */
      static fusion& pick(std::string name)
      {
         if (algorithms.find(name) == algorithms.end())
         {
            name = "alphablend";
         }
         return *algorithms.at(name);
      }

      /**
       * @brief available returns a list of the names of all available fusion
       *        algorithms.
       * @return list of names of fusion algorithms
       */
      static std::vector<std::string> available()
      {
         std::vector<std::string> names;
         std::transform(algorithms.begin(), algorithms.end(),
                        std::back_inserter(names),
                        [](std::pair<std::string, fusion*> pair) { return pair.first; });
         return names;
      }
};

const std::map<std::string, fusion*> fusion_algorithms::algorithms
{
   { "alphablend", &alpblnd }
};

#endif // FUSION_ALGORITHMS_HPP
