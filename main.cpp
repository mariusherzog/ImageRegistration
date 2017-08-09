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

   Mat fused = imagefusion::perform_fusion_from_files("mrit1.jpg", "mrit2.jpg", "mutualinformation", "alphablend");

   Mat fused_unregistered = imagefusion::perform_fusion_from_files("mrit1.jpg", "mrit2.jpg", "identity", "alphablend");

   imshow("floating image", pet);
   imshow("original image", image);
   imshow("fused transformed", fused);
   imshow("fused unregistered", fused_unregistered);

   waitKey(0);
}


