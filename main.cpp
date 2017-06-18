#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <iostream>
#include <algorithm>

using namespace cv;
using namespace std;


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

    cv::Size ksize(5,5);
//    cv::GaussianBlur(joint_histogram, joint_histogram, ksize, 5, 5);


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
    for (int i=0; i<ref.rows; ++i) {
       for (int j=0; j<ref.cols; ++j) {
          int intensity = ref.at<uchar>(j, i);
          hist_ref[intensity] = hist_ref[intensity]+1;
       }
    }

    for (int i=0; i<256; ++i) {
          hist_ref[i] = hist_ref[i]/(1.0*ref.rows*ref.cols);
    }

    cv::Size ksize2(5,0);
  //  cv::GaussianBlur(hist_ref, hist_ref, ksize2, 5);


    std::vector<double> hist_flt(256, 0.0);
    for (int i=0; i<flt.rows; ++i) {
       for (int j=0; j<flt.cols; ++j) {
          int intensity = flt.at<uchar>(j, i);
          hist_flt[intensity] = hist_flt[intensity]+1;
       }
    }


    for (int i=0; i<256; ++i) {
          hist_flt[i] = hist_flt[i]/(1.0*flt.rows*flt.cols);
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


/*
double optimize_ty(Mat ref, Mat flt, double ty, double rng, const double tx, const double a11, const double a12, const double a21, const double a22)
{
    double sta = ty - 0.382*rng;
    double end = ty + 0.618*rng;
    double f3pos = ty;

    std::cout << "sta: " << sta << "\n";
    std::cout << "end: " << end << "\n";
    std::cout << "ty: " << ty << "\n";

    while (true) {
        double f1 = mutual_information(ref, transform(flt, tx, sta, a11, a12, a21, a22));
        double f3 = mutual_information(ref, transform(flt, tx, end, a11, a12, a21, a22));
        double f2 = mutual_information(ref, transform(flt, tx, ty, a11, a12, a21, a22));

        double f4pos = ty + 0.312*(end-ty);
        std::cout << "f4pos " << f4pos << "\n";
        double f4val = mutual_information(ref, transform(flt, tx, f4pos, a11, a12, a21, a22));
        //std::cout << f4pos << "''" << f4val << "##" << f2 <<"\n";
        double dist = f4pos -ty;
        if (f4val > f2) {
            sta = ty;
            ty = f4pos;
        } else {
            end = f4pos;
        }

        //std::cout << dist << "  " << f4pos << ":"  << exp(-f4val) << "ä\n" << std::flush;

        if (dist < 0.5) break;
    }

    return ty;
}*/

double optimize_ty(Mat ref, Mat flt, double ty, double rng, const double tx, const double a11, const double a12, const double a21, const double a22)
{
    double sta = ty - 0.382*rng;
    double end = ty + 0.618*rng;
    double f3pos = ty;

    double c = (end - (end-sta)/1.618);
    double d = (sta + (end-sta)/1.618);

    while (abs(c-d) > 0.5) {
       if (exp(-mutual_information(ref, transform(flt, tx, c, a11, a12, a21, a22)))
           < exp(-mutual_information(ref, transform(flt, tx, d, a11, a12, a21, a22)))) {
          end = d;
       } else {
          sta = c;
       }

       c = (end - (end-sta)/1.618);
       d = (sta + (end-sta)/1.618);
    }

    return (end+sta)/2;
}

double optimize_tx(Mat ref, Mat flt, double tx, double rng, const double ty, const double a11, const double a12, const double a21, const double a22)
{
    double sta = tx - 0.382*rng;
    double end = tx + 0.618*rng;
    double f3pos = tx;

    double c = (end - (end-sta)/1.618);
    double d = (sta + (end-sta)/1.618);

    while (abs(c-d) > 0.5) {
       if (exp(-mutual_information(ref, transform(flt, c, ty, a11, a12, a21, a22)))
           < exp(-mutual_information(ref, transform(flt, d, ty, a11, a12, a21, a22)))) {
          end = d;
       } else {
          sta = c;
       }

       c = (end - (end-sta)/1.618);
       d = (sta + (end-sta)/1.618);
    }

    return (end+sta)/2;
}

double optimize_a11(Mat ref, Mat flt, double a11, double rng, const double tx, const double ty, const double a12, const double a21, const double a22)
{
   double sta = a11 - 0.382*rng;
   double end = a11 + 0.618*rng;
   double f3pos = a11;

   double c = (end - (end-sta)/1.618);
   double d = (sta + (end-sta)/1.618);

   while (abs(c-d) > 0.005) {
      if (exp(-mutual_information(ref, transform(flt, tx, ty, c, a12, a21, a22)))
          < exp(-mutual_information(ref, transform(flt, tx, ty, d, a12, a21, a22)))) {
         end = d;
      } else {
         sta = c;
      }

      c = (end - (end-sta)/1.618);
      d = (sta + (end-sta)/1.618);
   }

   return (end+sta)/2;
}

double optimize_a12(Mat ref, Mat flt, double a12, double rng, const double tx, const double ty, const double a11, const double a21, const double a22)
{
   double sta = a12 - 0.382*rng;
   double end = a12 + 0.618*rng;
   double f3pos = a12;

   double c = (end - (end-sta)/1.618);
   double d = (sta + (end-sta)/1.618);

   while (abs(c-d) > 0.005) {
      if (exp(-mutual_information(ref, transform(flt, tx, ty, a11, c, a21, a22)))
          < exp(-mutual_information(ref, transform(flt, tx, ty, a11, d, a21, a22)))) {
         end = d;
      } else {
         sta = c;
      }

      c = (end - (end-sta)/1.618);
      d = (sta + (end-sta)/1.618);
   }

   return (end+sta)/2;
}


int main()
{
  Mat image = imread("ctchest.png", CV_LOAD_IMAGE_GRAYSCALE);
  Mat pet = imread("petchest.jpg", CV_LOAD_IMAGE_GRAYSCALE);

  Size origsize(512, 512);
  resize(image, image, origsize);
  bitwise_not(pet, pet);

    //Mat trans_mat = (Mat_<double>(2,3) << 1.04*cos(-0.05), sin(-0.05), 5, -sin(-0.05), 1.01*cos(-0.05), 3);
    //warpAffine(pet,pet,trans_mat,pet.size());
//    transform(pet, 5, 0, -0.05, 1.035);

    cv::Size ksize(5,5);
    cv::GaussianBlur(image, image, ksize, 10, 10);

    Size max_size = image.size();
    resize(pet, pet, max_size);


  Moments im_mom = moments(image);
  Moments pt_mom = moments(pet);

  std::cout << im_mom.m10/im_mom.m00 << " " << im_mom.m01/im_mom.m00 << " \n\n";
  std::cout << pt_mom.m10/pt_mom.m00 << " " << pt_mom.m01/pt_mom.m00 << " \n\n";


  double tx = im_mom.m10/im_mom.m00 - pt_mom.m10/pt_mom.m00;
  double ty = im_mom.m01/im_mom.m00 - pt_mom.m01/pt_mom.m00;
  double a11 = 1.0;
  double a12 = 0.0;
  double a21 = 0.0;
  double a22 = 1.0;
  std::cout << "???" << tx << " " << ty << "???\n";
  const double trng = 25;

  double a = ty-25;
  double b = ty+25*1.61;
  double xa = tx-25;
  double xb = ty+25*1.61;
  bool converged = false;

  double last_mutualinf = 100.0;
  double curr_mutualinf = 0.0;
  double tx_opt;
  double ty_opt;
  double a11_opt;
  double a12_opt;

  //ty /= 2;
  while (!converged) {
    converged  = true;
    tx_opt = optimize_tx(image, pet, tx, 80, ty, a11, a12, a21, a22);
    curr_mutualinf = exp(-mutual_information(image, transform(pet, tx_opt, ty, a11, a12, a21, a22)));
    if (last_mutualinf - curr_mutualinf > 0.0005) {
        tx = tx_opt;
        last_mutualinf = curr_mutualinf;
        converged = false;
    }

    std::cout << last_mutualinf - curr_mutualinf << "++\n";

    ty_opt = optimize_ty(image, pet, ty, 80, tx, a11, a12, a21, a22);
    curr_mutualinf = exp(-mutual_information(image, transform(pet, tx, ty_opt, a11, a12, a21, a22)));
    if (last_mutualinf - curr_mutualinf > 0.00005) {
        ty = ty_opt;
        last_mutualinf = curr_mutualinf;
        converged = false;
    }


    a11_opt = optimize_a11(image, pet, a11, 1.0, tx, ty, a12, a21, a22);
    curr_mutualinf = exp(-mutual_information(image, transform(pet, tx, ty, a11_opt, a12, a21, a22)));
    if (last_mutualinf - curr_mutualinf > 0.00005) {
        a11 = a11_opt;
        last_mutualinf = curr_mutualinf;
        converged = false;
    }

    a12_opt = optimize_a12(image, pet, a12, 1.0, tx, ty, a11, a21, a22);
    curr_mutualinf = exp(-mutual_information(image, transform(pet, tx, ty, a11, a12_opt, a21, a22)));
    std::cout << last_mutualinf - curr_mutualinf << "##";
    if (last_mutualinf - curr_mutualinf > 0.00005) {
        a12 = a12_opt;
        last_mutualinf = curr_mutualinf;
        converged = false;
    }
  }


   std::cout << "!" << tx << " " << ty << "#";
   Mat fin = transform(pet, tx, ty, a11, a12, 0, 1);

  double mutual_inf = mutual_information(image, fin);
  std::cout << exp(-mutual_inf) << "*** \n";

  Mat color(fin.cols, fin.rows, CV_8UC3);
  cv::cvtColor(fin, color, cv::COLOR_GRAY2BGR);
  Mat channel[3];
    split(color, channel);
  channel[1] = Mat::zeros(fin.rows, fin.cols, CV_8UC1);
  merge(channel, 3, color);

  imshow("lena", image);
  imshow("lena_original", pet);
  imshow("fin", color);
  waitKey(0);
}


