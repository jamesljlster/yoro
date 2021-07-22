#ifndef __YORO_IMPL_HPP__
#define __YORO_IMPL_HPP__

#include <string>
#include <vector>

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include "yoro_api.hpp"

namespace yoro_api
{
class GeneralDetector
{
   public:
    explicit GeneralDetector(const char* modelPath, const DeviceType& devType);
    explicit GeneralDetector(
        const std::string& modelPath, const DeviceType& devType)
        : GeneralDetector(modelPath.c_str(), devType)
    {
    }

    torch::jit::IValue detect(const cv::Mat& image);

   protected:
    torch::jit::Module model;
    torch::DeviceType device = torch::kCPU;
    torch::ScalarType scalarType = torch::kFloat;
    torch::TensorOptions opt;

    int netWidth;
    int netHeight;

    std::string make_error_msg(const char* msg);
};

class YORODetector::Impl : public GeneralDetector
{
   public:
    explicit Impl(const char* modelPath, const DeviceType& devType)
        : GeneralDetector(modelPath, devType)
    {
    }

    explicit Impl(const std::string& modelPath, const DeviceType& devType)
        : GeneralDetector(modelPath, devType)
    {
    }

    std::vector<RBox> detect(const cv::Mat& image, float confTh, float nmsTh);
};

class RotationDetector::Impl : public GeneralDetector
{
   public:
    explicit Impl(const char* modelPath, const DeviceType& devType)
        : GeneralDetector(modelPath, devType)
    {
    }

    explicit Impl(const std::string& modelPath, const DeviceType& devType)
        : GeneralDetector(modelPath, devType)
    {
    }

    float detect(const cv::Mat& image);
};

}  // namespace yoro_api

#endif
