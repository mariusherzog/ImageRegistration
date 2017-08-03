#ifndef REGISTER_ALGORITHMS_HPP
#define REGISTER_ALGORITHMS_HPP

#include <string>
#include <algorithm>
#include <memory>
#include <cassert>

#include "register.hpp"

/**
 * @brief The fusion_algorithms class is a facade which facilitates easy access
 *        to all available fusion strategies. Each strategies is uniquely
 *        identified by its name and can be accessed with it.
 */
class register_algorithms
{
   private:
      static const std::vector<std::string> algorithms;

   public:
      /**
       * @brief pick returns the fusion algorithm for the given name
       * @param name identifying name / key of the fusion strategy
       * @return instance of the fusion strategy
       */
      static std::unique_ptr<registration> pick(std::string name)
      {
         if (std::find(algorithms.begin(), algorithms.end(), name) == algorithms.end())
         {
            name = "mutualinformation";
         }

         if (name == "mutualinformation")
         {
            return std::make_unique<mutualinformation>();
         }
         return nullptr;
      }

      /**
       * @brief available returns a list of the names of all available
       *        registration algorithms.
       * @return list of names of registration algorithms
       */
      static std::vector<std::string> available()
      {
         return algorithms;
      }
};

/**
 * @todo maybe replace by enum
 */
const std::vector<std::string> register_algorithms::algorithms
{
   "mutualinformation"
};

#endif // REGISTER_ALGORITHMS_HPP
