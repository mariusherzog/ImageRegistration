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
      /**
       * @brief perform_fusion_from_files is an application service which
       *        performs fusion of two images and the given register / fusion
       *        algorithms
       * @param path_reference_image path to the reference image
       * @param path_floating_image path to the floating image
       * @param register_strategy name of the register strategy
       * @param fusion_strategy name of the fusion strategy
       * @return fused image
       */
      static cv::Mat perform_fusion_from_files(
            std::string path_reference_image,
            std::string path_floating_image,
            std::string register_strategy,
            std::string fusion_strategy)
      {
         file_repository files(path_reference_image, path_floating_image);
         cv::Mat reference_image = files.reference_image();
         cv::Mat floating_image = files.floating_image();

         return fuse_images(reference_image, floating_image,
                            register_strategy, fusion_strategy);
      }

      /**
       * @brief fusion_strategies queries availabe fusion strategies
       * @return a list of availabe fusion strategies
       */
      static std::vector<std::string> fusion_strategies()
      {
         return available_fusion_algorithms();
      }

      /**
       * @brief register_strategies queries availabe register strategies
       * @return a list of availabe register strategies
       */
      static std::vector<std::string> register_strategies()
      {
         return available_registration_algorithms();
      }
};


#endif // IMAGEFUSION_HPP
