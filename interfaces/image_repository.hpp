#ifndef IMAGE_REPOSITORY_HPP
#define IMAGE_REPOSITORY_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

/**
 * @brief The Iimage_repository interface is used for application services
 *        which need to access image data to pass it on to the core domain
 * Later we can use this interface to provide an implementation for DICOM
 * access, for example.
 */
class Iimage_repository
{
   public:
      virtual cv::Mat reference_image() = 0;
      virtual cv::Mat floating_image() = 0;
      virtual ~Iimage_repository() = 0;
};

Iimage_repository::~Iimage_repository()
{
}


#endif // IMAGE_REPOSITORY_HPP
