#ifndef FUSION_HPP
#define FUSION_HPP

#include <cassert>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>


/**
 * @brief The fusion interface defines the signature of a fusion operation on
 *        two images (Mats).
 */
class fusion
{
   public:
      virtual cv::Mat fuse(cv::Mat a, cv::Mat b) = 0;
      virtual ~fusion() = 0;
};

fusion::~fusion()
{
}

/**
 * @brief The alphablend strategy performs a fusion by alpha blending both
 *        images
 */
class alphablend : public fusion
{
   public:
      alphablend(double alpha) :
         alpha {alpha}
      {
      }

      cv::Mat fuse(cv::Mat ref, cv::Mat flt) override
      {
         assert(abs(alpha) < 1.0);

         cv::Mat color(flt.cols, flt.rows, CV_8UC3);
         cv::cvtColor(flt, color, cv::COLOR_GRAY2BGR);
         cv::Mat channel[3];
         split(color, channel);
         channel[1] = cv::Mat::zeros(flt.rows, flt.cols, CV_8UC1);
         merge(channel, 3, color);

         cv::cvtColor(ref, ref, cv::COLOR_GRAY2BGR);

         const double beta = 1-alpha;
         cv::Mat dst = ref.clone();
         cv::addWeighted(ref, alpha, color, beta, 0.0, dst);
         return dst;
      }

   private:
      const double alpha;
};

#endif // FUSION_HPP
