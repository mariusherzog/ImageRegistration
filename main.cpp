#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <iostream>
#include <algorithm>
#include <functional>

#include "register.hpp"

using namespace cv;
using namespace std;
using namespace std::placeholders;


Mat fusion_alphablend(Mat ref, Mat flt, double alpha)
{
   assert(abs(alpha) < 1.0);

   Mat color(flt.cols, flt.rows, CV_8UC3);
   cv::cvtColor(flt, color, cv::COLOR_GRAY2BGR);
   Mat channel[3];
   split(color, channel);
   channel[1] = Mat::zeros(flt.rows, flt.cols, CV_8UC1);
   merge(channel, 3, color);

   cv::cvtColor(ref, ref, cv::COLOR_GRAY2BGR);

   double beta = 1-alpha;
   Mat dst = ref.clone();
   addWeighted(ref, alpha, color, beta, 0.0, dst);
   return dst;
}

int main()
{
   Mat image = imread("mrit1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
   Mat pet = imread("mrit2.jpg", CV_LOAD_IMAGE_GRAYSCALE);

   cv::Size origsize(512, 512);
   cv::resize(image, image, origsize);
   cv::resize(pet, pet, origsize);

   //pet = transform(pet, 9, -13, 0.97, -0.08, 0.08, 1.06);
   //pet = transform(pet, 0, 0, cos(M_PI/4), -sin(M_PI/4), sin(M_PI/4), cos(M_PI/4));

   Mat fin = register_images(image, pet);


   // now do the fusion
   Mat fused = fusion_alphablend(image, fin, 0.5);
   Mat fused_unregistered = fusion_alphablend(image, pet, 0.5);

   imshow("floating image", pet);
   imshow("original image", image);
   imshow("fused transformed", fused);
   imshow("fused unregistered", fused_unregistered);

   waitKey(0);
}


