#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <iostream>
#include <algorithm>
#include <functional>
#include <memory>
#include <iterator>

#include "fusion_algorithms.hpp"
#include "register_algorithms.hpp"

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

   auto available_register_names = register_algorithms::available();
   std::copy(available_register_names.begin(),
             available_register_names.end(),
             std::ostream_iterator<std::string>(std::cout, "\n"));


   std::unique_ptr<fusion> fusion_algorithm = fusion_algorithms::pick("alphablend");
   std::unique_ptr<registration> registration_algorithm =
         register_algorithms::pick("mutualinformation");

   // register to align images
   Mat fin = registration_algorithm->register_images(image, pet);

   // now do the fusion
   Mat fused = fusion_algorithm->fuse(image, fin);
   Mat fused_unregistered = fusion_algorithm->fuse(image, pet);

   imshow("floating image", pet);
   imshow("original image", image);
   imshow("fused transformed", fused);
   imshow("fused unregistered", fused_unregistered);

   waitKey(0);
}


