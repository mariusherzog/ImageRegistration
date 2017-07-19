#ifndef FUSION_HPP
#define FUSION_HPP

#include <cassert>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

/**
 * @brief fusion_alphablend performs a fusion by alpha blending both images
 * @param ref
 * @param flt
 * @param alpha
 * @return fused image
 */
cv::Mat fusion_alphablend(cv::Mat ref, cv::Mat flt, double alpha)
{
   assert(abs(alpha) < 1.0);

   cv::Mat color(flt.cols, flt.rows, CV_8UC3);
   cv::cvtColor(flt, color, cv::COLOR_GRAY2BGR);
   cv::Mat channel[3];
   split(color, channel);
   channel[1] = cv::Mat::zeros(flt.rows, flt.cols, CV_8UC1);
   merge(channel, 3, color);

   cv::cvtColor(ref, ref, cv::COLOR_GRAY2BGR);

   double beta = 1-alpha;
   cv::Mat dst = ref.clone();
   cv::addWeighted(ref, alpha, color, beta, 0.0, dst);
   return dst;
}

#endif // FUSION_HPP
