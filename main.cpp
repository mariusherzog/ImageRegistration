#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <iostream>
#include <algorithm>
#include <functional>
#include <memory>
#include <iterator>

#include "core/fusion_algorithms.hpp"
#include "core/register_algorithms.hpp"
#include "app/imagefusion.hpp"

using namespace cv;
using namespace std;
using namespace std::placeholders;

int main()
{
   Mat image = imread("mrit1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
   Mat pet = imread("mrit2.jpg", CV_LOAD_IMAGE_GRAYSCALE);

   //pet = transform(pet, 9, -13, 0.97, -0.08, 0.08, 1.06);
   //pet = transform(pet, 0, 0, cos(M_PI/4), -sin(M_PI/4), sin(M_PI/4), cos(M_PI/4));

   auto available_fusion_names = imagefusion::fusion_strategies();
   std::copy(available_fusion_names.begin(),
             available_fusion_names.end(),
             std::ostream_iterator<std::string>(std::cout, "\n"));

   auto available_register_names = imagefusion::register_strategies();
   std::copy(available_register_names.begin(),
             available_register_names.end(),
             std::ostream_iterator<std::string>(std::cout, "\n"));

   Mat fused = imagefusion::perform_fusion("mutualinformation", "alphablend");

   //std::unique_ptr<fusion> fusion_algorithm = fusion_algorithms::pick("alphablend");

   //Mat fused = fuse_images(image, pet, "mutualinformation", "alphablend");

   Mat fused_unregistered = imagefusion::perform_fusion("identity", "alphablend");

   imshow("floating image", pet);
   imshow("original image", image);
   imshow("fused transformed", fused);
   imshow("fused unregistered", fused_unregistered);

   waitKey(0);
}


