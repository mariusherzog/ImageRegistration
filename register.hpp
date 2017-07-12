#ifndef REGISTER_HPP
#define REGISTER_HPP

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include "optimize.hpp"

static double mutual_information(cv::Mat ref, cv::Mat flt)
{

   cv::Mat joint_histogram(256, 256, CV_64FC1, cv::Scalar(0));

   for (int i=0; i<ref.cols; ++i) {
      for (int j=0; j<ref.rows; ++j) {
         int ref_intensity = ref.at<uchar>(j,i);
         int flt_intensity = flt.at<uchar>(j,i);
         joint_histogram.at<double>(ref_intensity, flt_intensity) = joint_histogram.at<double>(ref_intensity, flt_intensity)+1;
         double v = joint_histogram.at<double>(ref_intensity, flt_intensity);
      }
   }



   for (int i=0; i<256; ++i) {
      for (int j=0; j<256; ++j) {
         joint_histogram.at<double>(j, i) = joint_histogram.at<double>(j, i)/(1.0*ref.rows*ref.cols);
         double v = joint_histogram.at<double>(j, i);
      }
   }

   cv::Size ksize(7, 7);
   cv::GaussianBlur(joint_histogram, joint_histogram, ksize, 7, 7);


   double entropy = 0.0;
   for (int i=0; i<256; ++i) {
      for (int j=0; j<256; ++j) {
         double v = joint_histogram.at<double>(j, i);
         if (v > 0.000000000000001) {
            entropy += v*log(v)/log(2);
         }
      }
   }
   entropy *= -1;

   //    std::cout << entropy << "###";



   std::vector<double> hist_ref(256, 0.0);
   for (int i=0; i<joint_histogram.rows; ++i) {
      for (int j=0; j<joint_histogram.cols; ++j) {
         hist_ref[i] += joint_histogram.at<double>(i, j);
      }
   }

   cv::Size ksize2(5,0);
   //  cv::GaussianBlur(hist_ref, hist_ref, ksize2, 5);


   std::vector<double> hist_flt(256, 0.0);
   for (int i=0; i<joint_histogram.cols; ++i) {
      for (int j=0; j<joint_histogram.rows; ++j) {
         hist_flt[i] += joint_histogram.at<double>(j, i);
      }
   }

   //   cv::GaussianBlur(hist_flt, hist_flt, ksize2, 5);



   double entropy_ref = 0.0;
   for (int i=0; i<256; ++i) {
      if (hist_ref[i] > 0.000000000001) {
         entropy_ref += hist_ref[i] * log(hist_ref[i])/log(2);
      }
   }
   entropy_ref *= -1;
   //std::cout << entropy_ref << "~~ ";

   double entropy_flt = 0.0;
   for (int i=0; i<256; ++i) {
      if (hist_flt[i] > 0.000000000001) {
         entropy_flt += hist_flt[i] * log(hist_flt[i])/log(2);
      }
   }
   entropy_flt *= -1;
   // std::cout << entropy_flt << "++ ";

   double mutual_information = entropy_ref + entropy_flt - entropy;
   return mutual_information;
}

static cv::Mat transform(cv::Mat image, double tx, double ty, double a11, double a12, double a21, double a22)
{
   cv::Mat trans_mat = (cv::Mat_<double>(2,3) << a11, a12, tx, a21, a22, ty);

   cv::Mat out = image.clone();
   warpAffine(image, out, trans_mat, image.size());
   return out;
}


static double cost_function(cv::Mat ref, cv::Mat flt, std::vector<double>::iterator affine_params)
{
   const double tx = affine_params[0];
   const double ty = affine_params[1];
   const double a11 = affine_params[2];
   const double a12 = affine_params[3];
   const double a21 = affine_params[4];
   const double a22 = affine_params[5];
   return exp(-mutual_information(ref, transform(flt, tx, ty, a11, a12, a21, a22)));
}


static void estimate_initial(cv::Mat ref, cv::Mat flt, double& tx, double& ty, double& a11, double& a12, double& a21, double& a22)
{
   cv::Moments im_mom = moments(ref);
   cv::Moments pt_mom = moments(flt);

   cv::Mat ref_bin = ref.clone();
   cv::Mat flt_bin = flt.clone();
   cv::threshold(ref, ref_bin, 40, 256, 0);
   cv::threshold(flt, flt_bin, 40, 256, 0);

   cv::Moments ref_mom_bin = moments(ref_bin);
   cv::Moments flt_mom_bin = moments(flt_bin);

   double pt_avg_10 = pt_mom.m10/pt_mom.m00;
   double pt_avg_01 = pt_mom.m01/pt_mom.m00;
   double pt_mu_20 = (pt_mom.m20/pt_mom.m00*1.0)-(pt_avg_10*pt_avg_10);
   double pt_mu_02 = (pt_mom.m02/pt_mom.m00*1.0)-(pt_avg_01*pt_avg_01);
   double pt_mu_11 = (pt_mom.m11/pt_mom.m00*1.0)-(pt_avg_01*pt_avg_10);

   double im_avg_10 = im_mom.m10/im_mom.m00;
   double im_avg_01 = im_mom.m01/im_mom.m00;
   double im_mu_20 = (im_mom.m20/im_mom.m00*1.0)-(im_avg_10*im_avg_10);
   double im_mu_02 = (im_mom.m02/im_mom.m00*1.0)-(im_avg_01*im_avg_01);
   double im_mu_11 = (im_mom.m11/im_mom.m00*1.0)-(im_avg_01*im_avg_10);

   tx = im_mom.m10/im_mom.m00 - pt_mom.m10/pt_mom.m00;
   ty = im_mom.m01/im_mom.m00 - pt_mom.m01/pt_mom.m00;

   double scale = ref_mom_bin.m00/flt_mom_bin.m00;

   double rho = 0.5f * atan((2.0*pt_mu_11)/(pt_mu_20 - pt_mu_02));
   double rho_im = 0.5f * atan((2.0*im_mu_11)/(im_mu_20 - im_mu_02));

   const double rho_diff = rho_im - rho;

   const double roundness = (pt_mom.m20/pt_mom.m00) / (pt_mom.m02/pt_mom.m00);
   if (abs(roundness-1.0) >= 0.3) {
      a11 = cos(rho_diff);
      a12 = -sin(rho_diff);
      a21 = sin(rho_diff);
      a22 = cos(rho_diff);
   } else {
      a11 = 1.0;
      a12 = 0.0;
      a21 = 0.0;
      a22 = 1.0;
   }
}

static bool is_inverted(cv::Mat ref, cv::Mat flt)
{
   std::vector<double> hist_ref(256, 0);
   std::vector<double> hist_flt(256, 0);

   for (int i=0; i<ref.rows; ++i) {
      for (int j=0; j<ref.cols; ++j) {
         int val = ref.at<uchar>(i, j);
         hist_ref[val]++;
      }
   }

   for (int i=0; i<flt.rows; ++i) {
      for (int j=0; j<flt.cols; ++j) {
         int val = flt.at<uchar>(i, j);
         hist_flt[val]++;
      }
   }

   std::transform(hist_ref.begin(), hist_ref.end(), hist_ref.begin(), [&ref](int val) { return 1.0*val / (1.0*ref.cols*ref.rows); });
   std::transform(hist_flt.begin(), hist_flt.end(), hist_flt.begin(), [&flt](int val) { return 1.0*val / (1.0*flt.cols*flt.rows); });

   std::vector<double> distances(256, 0);
   std::transform(hist_ref.begin(), hist_ref.end(), hist_flt.begin(), distances.begin(),
                  [](double ref_val, double flt_val) { return abs(ref_val - flt_val); });

   double distance_flt = std::accumulate(distances.begin(), distances.end(), 0.0);

   // invert
   std::reverse(hist_flt.begin(), hist_flt.end());

   std::transform(hist_ref.begin(), hist_ref.end(), hist_flt.begin(), distances.begin(),
                  [](double ref_val, double inv_val) { return abs(ref_val - inv_val); });

   double distance_inv = std::accumulate(distances.begin(), distances.end(), 0.0);

   return distance_flt < distance_inv;
}

/**
 * @brief register_images uses mutual inforcv::Mation to perform registration of the
 *        images given by the parameter
 * @param ref reference image
 * @param flt floating image which shall be registered to image
 * @return transformed floating image which cv::Matches the reference
 */
cv::Mat register_images(cv::Mat ref, cv::Mat flt)
{
   cv::Size origsize(512, 512);
   cv::resize(ref, ref, origsize);

   bool inverted = false;
   if (is_inverted(ref, flt)) {
      cv::bitwise_not(flt, flt);
      inverted = true;
   }

   cv::Size ksize(5,5);
   cv::GaussianBlur(ref, ref, ksize, 10, 10);

   auto max_size = ref.size();
   cv::resize(flt, flt, max_size);

   double tx, ty, a11, a12, a21, a22;
   estimate_initial(ref, flt, tx, ty, a11, a12, a21, a22);


   std::vector<double> init {tx, ty, a11, a12, a21, a22};
   std::vector<double> rng {80.0, 80.0, 1.0, 1.0, 1.0, 1.0};

   std::pair<std::vector<double>::iterator, std::vector<double>::iterator> o {init.begin(), init.end()};
   optimize_powell(o, {rng.begin(), rng.end()}, std::bind(cost_function, ref, flt, std::placeholders::_1));


   tx = init[0];
   ty = init[1];
   a11 = init[2];
   a12 = init[3];
   a21 = init[4];
   a22 = init[5];

   cv::Mat fin = transform(flt, tx, ty, a11, a12, a21, a22);

//   double mutual_inf = mutual_inforcv::Mation(image, fin);
//   std::cout << exp(-mutual_inf) << "*** \n";

   if (inverted) {
      cv::bitwise_not(fin, fin);
      cv::bitwise_not(flt, flt);
   }

   return fin.clone();
}


#endif // REGISTER_HPP
