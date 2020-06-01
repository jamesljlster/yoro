#ifndef __YORO_IMPL_HPP__
#define __YORO_IMPL_HPP__

#include <string>
#include <vector>

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include "yoro_api.hpp"

namespace yoro_api
{
class Detector::Impl
{
   public:
    explicit Impl(const char* modelPath, const Detector::DeviceType& devType);
    explicit Impl(const std::string& modelPath,
                  const Detector::DeviceType& devType)
        : Impl(modelPath.c_str(), devType)
    {
    }

    std::vector<RBox> detect(const cv::Mat& image, float confTh, float nmsTh);

   protected:
    torch::jit::Module model;
    torch::DeviceType device = torch::kCPU;
    torch::ScalarType scalarType = torch::kFloat;
    torch::TensorOptions opt;

    int netWidth;
    int netHeight;

    std::string make_error_msg(const char* msg);
};

}  // namespace yoro_api

#endif
