#ifndef IMAGEFUSION_HPP
#define IMAGEFUSION_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "../core/domain/fusion_services.hpp"
#include "../interfaces/image_repository.hpp"
#include "../infastructure/file_repository.hpp"

class imagefusion
{
   public:
      static cv::Mat perform_fusion(std::string register_strategy,
                                    std::string fusion_strategy)
      {
         file_repository files("mrit1.jpg", "mrit2.jpg");
         cv::Mat reference_image = files.reference_image();
         cv::Mat floating_image = files.floating_image();

         return fuse_images(reference_image, floating_image,
                            register_strategy, fusion_strategy);
      }

      static std::vector<std::string> fusion_strategies()
      {
         return available_fusion_algorithms();
      }

      static std::vector<std::string> register_strategies()
      {
         return available_registration_algorithms();
      }
};


#endif // IMAGEFUSION_HPP
