#ifndef FUSION_ALGORITHMS_HPP
#define FUSION_ALGORITHMS_HPP

#include <string>
#include <algorithm>
#include <memory>
#include <cassert>

#include "fusion.hpp"

/**
 * @brief The fusion_algorithms class is a facade which facilitates easy access
 *        to all available fusion strategies. Each strategies is uniquely
 *        identified by its name and can be accessed with it.
 */
class fusion_algorithms
{
   private:
      static const std::vector<std::string> algorithms;

   public:
      /**
       * @brief pick returns the fusion algorithm for the given name
       * @param name identifying name / key of the fusion strategy
       * @return instance of the fusion strategy
       */
      static std::unique_ptr<fusion> pick(std::string name)
      {
         if (std::find(algorithms.begin(), algorithms.end(), name) == algorithms.end())
         {
            name = "alphablend";
         }

         if (name == "alphablend")
         {
            return std::make_unique<alphablend>(0.5);
         }
         return nullptr;
      }

      /**
       * @brief available returns a list of the names of all available fusion
       *        algorithms.
       * @return list of names of fusion algorithms
       */
      static std::vector<std::string> available()
      {
         return algorithms;
      }
};

/**
 * @todo maybe replace by enum
 */
const std::vector<std::string> fusion_algorithms::algorithms
{
   "alphablend"
};

#endif // FUSION_ALGORITHMS_HPP
