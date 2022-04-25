#ifndef __YORO_IMPL_HPP__
#define __YORO_IMPL_HPP__

#include <string>
#include <vector>

#include <torch/extension.h>

#include "yoro_api.hpp"

#ifdef WITH_OPENCV
#    include <opencv2/opencv.hpp>
#endif

namespace yoro_api
{
torch::Tensor from_image(
    const uint8_t* image, int width, int height, int channels);

#ifdef WITH_OPENCV
torch::Tensor from_image(const cv::Mat& image);
cv::Mat to_image(const torch::Tensor& source);
#endif

class GeneralDetector
{
   public:
    explicit GeneralDetector(const char* modelPath, const DeviceType& devType);
    explicit GeneralDetector(
        const std::string& modelPath, const DeviceType& devType)
        : GeneralDetector(modelPath.c_str(), devType)
    {
    }

    int network_width() const;
    int network_height() const;

    // TODO: Add autoResize parameter
    torch::jit::IValue detect(const torch::Tensor& image);

   protected:
    torch::jit::Module model;
    torch::DeviceType device = torch::kCPU;
    torch::ScalarType scalarType = torch::kFloat;
    torch::TensorOptions opt;

    int netWidth;
    int netHeight;

    std::string make_error_msg(const char* msg) const;
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

    // TODO: Add autoResize, autoPad parameter
    std::vector<RBox> detect(
        const torch::Tensor& image, float confTh, float nmsTh);
    std::vector<RBox> detect(
        const uint8_t* image,
        int width,
        int height,
        int channels,
        float confTh,
        float nmsTh);

#ifdef WITH_OPENCV
    std::vector<RBox> detect(const cv::Mat& image, float confTh, float nmsTh);
#endif
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

    // TODO: Add autoResize, autoPad parameter
    float detect(const torch::Tensor& image);
    float detect(const uint8_t* image, int width, int height, int channels);

#ifdef WITH_OPENCV
    float detect(const cv::Mat& image);
#endif
};

}  // namespace yoro_api

#endif
