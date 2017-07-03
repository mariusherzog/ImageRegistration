#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <iostream>
#include <algorithm>
#include <functional>

using namespace cv;
using namespace std;
using namespace std::placeholders;


double mutual_information(Mat ref, Mat flt)
{

   Mat joint_histogram(256, 256, CV_64FC1, Scalar(0));

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

Mat transform(Mat image, double tx, double ty, double a11, double a12, double a21, double a22)
{
   Mat trans_mat = (Mat_<double>(2,3) << a11, a12, tx, a21, a22, ty);

   Mat out = image.clone();
   warpAffine(image, out, trans_mat, image.size());
   return out;
}

template <typename F>
double optimize_goldensectionsearch(double init, double rng, F function)
{
   double sta = init - 0.382*rng;
   double end = init + 0.618*rng;
   double c = (end - (end-sta)/1.618);
   double d = (sta + (end-sta)/1.618);

   while (abs(c-d) > 0.005) {
      if (function(c) < function(d)) {
         end = d;
      } else {
         sta = c;
      }

      c = (end - (end-sta)/1.618);
      d = (sta + (end-sta)/1.618);
   }

   return (end+sta)/2;
}

double cost_function(Mat ref, Mat flt, double tx, double ty, double a11, double a12, double a21, double a22)
{
   return exp(-mutual_information(ref, transform(flt, tx, ty, a11, a12, a21, a22)));
}


void estimate_initial(Mat ref, Mat flt, double& tx, double& ty, double& a11, double& a12, double& a21, double& a22)
{
   Moments im_mom = moments(ref);
   Moments pt_mom = moments(flt);

   Mat ref_bin = ref.clone();
   Mat flt_bin = flt.clone();
   threshold(ref, ref_bin, 40, 256, 0);
   threshold(flt, flt_bin, 40, 256, 0);

   Moments ref_mom_bin = moments(ref_bin);
   Moments flt_mom_bin = moments(flt_bin);

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

bool is_inverted(Mat ref, Mat flt)
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

int main()
{
   Mat image = imread("brainct.jpg", CV_LOAD_IMAGE_GRAYSCALE);
   Mat pet = imread("brainpet.jpg", CV_LOAD_IMAGE_GRAYSCALE);

   //pet = transform(pet, 9, -13, 0.9, -0.08, 0.08, 1.06);
   //pet = transform(pet, 0, 0, cos(M_PI/4), -sin(M_PI/4), sin(M_PI/4), cos(M_PI/4));


   Size origsize(512, 512);
   resize(image, image, origsize);

   bool inverted = false;
   if (is_inverted(image, pet)) {
      bitwise_not(pet, pet);
      inverted = true;
   }


   //Mat trans_mat = (Mat_<double>(2,3) << 1.04*cos(-0.05), sin(-0.05), 5, -sin(-0.05), 1.01*cos(-0.05), 3);
   //warpAffine(pet,pet,trans_mat,pet.size());
   //    transform(pet, 5, 0, -0.05, 1.035);

   cv::Size ksize(5,5);
   cv::GaussianBlur(image, image, ksize, 10, 10);

   Size max_size = image.size();
   resize(pet, pet, max_size);

   double tx, ty, a11, a12, a21, a22;
   estimate_initial(image, pet, tx, ty, a11, a12, a21, a22);

   bool converged = false;

   double last_mutualinf = 100.0;
   double curr_mutualinf = 0.0;
   double tx_opt;
   double ty_opt;
   double a11_opt;
   double a12_opt;
   double a21_opt;
   double a22_opt;

   //ty /= 2;
   while (!converged) {
      converged  = true;
      auto optimize_tx = std::bind(cost_function, image, pet, _1, ty, a11, a12, a21, a22);
      //tx_opt = optimize_tx(image, pet, tx, 80, ty, a11, a12, a21, a22);
      tx_opt = optimize_goldensectionsearch(tx, 80, optimize_tx);
      curr_mutualinf = exp(-mutual_information(image, transform(pet, tx_opt, ty, a11, a12, a21, a22)));
      if (last_mutualinf - curr_mutualinf > 0.00005) {
         tx = tx_opt;
         last_mutualinf = curr_mutualinf;
         converged = false;
      }

      std::cout << last_mutualinf - curr_mutualinf << "++\n";

      auto optimize_ty = std::bind(cost_function, image, pet, tx, _1, a11, a12, a21, a22);
      ty_opt = optimize_goldensectionsearch(ty, 80, optimize_ty);
      curr_mutualinf = exp(-mutual_information(image, transform(pet, tx, ty_opt, a11, a12, a21, a22)));
      if (last_mutualinf - curr_mutualinf > 0.00005) {
         ty = ty_opt;
         last_mutualinf = curr_mutualinf;
         converged = false;
      }


      auto optimize_a11 = std::bind(cost_function, image, pet, tx, ty, _1, a12, a21, a22);
      a11_opt = optimize_goldensectionsearch(a11, 1.0, optimize_a11);
      curr_mutualinf = exp(-mutual_information(image, transform(pet, tx, ty, a11_opt, a12, a21, a22)));
      if (last_mutualinf - curr_mutualinf > 0.00005) {
         a11 = a11_opt;
         last_mutualinf = curr_mutualinf;
         converged = false;
      }

      auto optimize_a12 = std::bind(cost_function, image, pet, tx, ty, a11, _1, a21, a22);
      a12_opt = optimize_goldensectionsearch(a12, 1.0, optimize_a12);
      curr_mutualinf = exp(-mutual_information(image, transform(pet, tx, ty, a11, a12_opt, a21, a22)));
      std::cout << last_mutualinf - curr_mutualinf << "##";
      if (last_mutualinf - curr_mutualinf > 0.00005) {
         a12 = a12_opt;
         last_mutualinf = curr_mutualinf;
         converged = false;
      }

      auto optimize_a21 = std::bind(cost_function, image, pet, tx, ty, a11, a12, _1, a22);
      a21_opt = optimize_goldensectionsearch(a21, 1.0, optimize_a21);
      curr_mutualinf = exp(-mutual_information(image, transform(pet, tx, ty, a11, a12, a21_opt, a22)));
      std::cout << last_mutualinf - curr_mutualinf << "##";
      if (last_mutualinf - curr_mutualinf > 0.00005) {
         a21 = a21_opt;
         last_mutualinf = curr_mutualinf;
         converged = false;
      }

      auto optimize_a22 = std::bind(cost_function, image, pet, tx, ty, a11, a12, a21, _1);
      a22_opt = optimize_goldensectionsearch(a22, 1.0, optimize_a22);
      curr_mutualinf = exp(-mutual_information(image, transform(pet, tx, ty, a11, a12, a21, a22_opt)));
      std::cout << last_mutualinf - curr_mutualinf << "##";
      if (last_mutualinf - curr_mutualinf > 0.00005) {
         a22 = a22_opt;
         last_mutualinf = curr_mutualinf;
         converged = false;
      }
   }


   std::cout << "!" << tx << " " << ty << "#";
   Mat fin = transform(pet, tx, ty, a11, a12, a21, a22);

   double mutual_inf = mutual_information(image, fin);
   std::cout << exp(-mutual_inf) << "*** \n";

   if (inverted) {
      bitwise_not(fin, fin);
      bitwise_not(pet, pet);
   }

   // now do the fusion
   Mat fused = fusion_alphablend(image, fin, 0.5);
   Mat fused_unregistered = fusion_alphablend(image, pet, 0.5);

   imshow("floating image", pet);
   imshow("original image", image);
   imshow("fused transformed", fused);
   imshow("fused unregistered", fused_unregistered);

   waitKey(0);
}


