#ifndef IMAGE_FROM_FILE_HPP
#define IMAGE_FROM_FILE_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <string>

#include "../interfaces/image_repository.hpp"

/**
 * @brief The file_repository class implements the repository by loading the
 *        image from files.
 */
class file_repository : public Iimage_repository
{
   private:
      std::string path_reference;
      std::string path_floating;

   public:
      file_repository(std::string path_reference, std::string path_floating) :
         path_reference {path_reference},
         path_floating {path_floating}
      {
      }

      cv::Mat reference_image() override
      {
         return cv::imread(path_reference, CV_LOAD_IMAGE_GRAYSCALE);
      }

      cv::Mat floating_image() override
      {
         return cv::imread(path_floating, CV_LOAD_IMAGE_GRAYSCALE);
      }
};

#endif // IMAGE_FROM_FILE_HPP
