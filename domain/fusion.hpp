#ifndef DOMAIN_FUSION_HPP
#define DOMAIN_FUSION_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include "../core/fusion_algorithms.hpp"
#include "../core/register_algorithms.hpp"

/**
 * @brief available_fusion_algorithms
 * @return available strategies for fusion
 */
std::vector<std::string> available_fusion_algorithms()
{
   return fusion_algorithms::available();
}

/**
 * @brief available_registration_algorithms
 * @return availabe strategies for registration
 */
std::vector<std::string> available_registration_algorithms()
{
   return register_algorithms::available();
}


/**
 * @brief fusion performs a fusion by registering the floating to the reference
 *        image and then perform the fusion
 * @param ref reference image
 * @param flt floating image
 * @param register_strategy method for registration
 * @param fusion_strategy method for fusion
 * @return fused image
 */
cv::Mat fuse_images(cv::Mat ref, cv::Mat flt,
           std::string register_strategy, std::string fusion_strategy)
{
   using namespace cv;
   Size origsize(512, 512);
   resize(ref, ref, origsize);
   resize(flt, flt, origsize);

   std::unique_ptr<fusion> fusion_algorithm =
         fusion_algorithms::pick(fusion_strategy);
   std::unique_ptr<registration> registration_algorithm =
         register_algorithms::pick(register_strategy);

   // register to align images
   Mat fin = registration_algorithm->register_images(ref, flt);

   // now do the fusion
   Mat fused = fusion_algorithm->fuse(ref, fin);

   return fused;
}



#endif // FUSION_HPP
