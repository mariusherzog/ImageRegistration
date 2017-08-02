#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <iostream>
#include <algorithm>
#include <functional>
#include <memory>
#include <iterator>

#include "register.hpp"
#include "fusion.hpp"
#include "fusion_algorithms.hpp"

using namespace cv;
using namespace std;
using namespace std::placeholders;

int main()
{
   Mat image = imread("mrit1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
   Mat pet = imread("mrit2.jpg", CV_LOAD_IMAGE_GRAYSCALE);

   cv::Size origsize(512, 512);
   cv::resize(image, image, origsize);
   cv::resize(pet, pet, origsize);

   //pet = transform(pet, 9, -13, 0.97, -0.08, 0.08, 1.06);
   //pet = transform(pet, 0, 0, cos(M_PI/4), -sin(M_PI/4), sin(M_PI/4), cos(M_PI/4));

   auto available_fusion_names = fusion_algorithms::available();
   std::copy(available_fusion_names.begin(),
             available_fusion_names.end(),
             std::ostream_iterator<std::string>(std::cout, "\n"));

   fusion& fus_alg = fusion_algorithms::pick("alphablend");
   std::unique_ptr<fusion> fusion_algorithm = std::make_unique<alphablend>(0.5);
   std::unique_ptr<registration> registration_algorithm =
         std::make_unique<mutualinformation>();

   Mat fin = registration_algorithm->perform(image, pet);


   // now do the fusion
   //Mat fused = fusion_algorithm->perform(image, fin);
   Mat fused = fus_alg.perform(image, fin);
   Mat fused_unregistered = fusion_algorithm->perform(image, pet);

   imshow("floating image", pet);
   imshow("original image", image);
   imshow("fused transformed", fused);
   imshow("fused unregistered", fused_unregistered);

   waitKey(0);
}


