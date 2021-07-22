#ifndef __YORO_API_HPP__
#define __YORO_API_HPP__

#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

namespace yoro_api
{
std::tuple<cv::Mat, int, int> pad_to_aspect(
    const cv::Mat& src, float aspectRatio);

struct RBox
{
    /** Constructor */
    RBox() {}
    RBox(
        float conf, int label, float degree, float x, float y, float w, float h)
        : conf(conf), label(label), degree(degree), x(x), y(y), w(w), h(h)
    {
    }

    /** String conversion */
    operator std::string() const;
    std::string to_string() const;

    /** Attributes of rotated bounding box */
    float conf = 0;    // Confidence (objectness * class probability)
    int label = 0;     // Class label
    float degree = 0;  // Degree
    float x = 0;       // Center x
    float y = 0;       // Center y
    float w = 0;       // Width
    float h = 0;       // Height
};

enum class DeviceType
{
    Auto,
    CPU,
    CUDA
};

class YORODetector
{
   public:
    explicit YORODetector(
        const char* modelPath, const DeviceType& devType = DeviceType::Auto);
    explicit YORODetector(
        const std::string& modelPath,
        const DeviceType& devType = DeviceType::Auto)
        : YORODetector(modelPath.c_str(), devType)
    {
    }

    std::vector<RBox> detect(const cv::Mat& image, float confTh, float nmsTh);

   protected:
    class Impl;
    std::shared_ptr<Impl> impl;
};

class RotationDetector
{
   public:
    explicit RotationDetector(
        const char* modelPath, const DeviceType& devType = DeviceType::Auto);
    explicit RotationDetector(
        const std::string& modelPath,
        const DeviceType& devType = DeviceType::Auto)
        : RotationDetector(modelPath.c_str(), devType)
    {
    }

    float detect(const cv::Mat& image);

   protected:
    class Impl;
    std::shared_ptr<Impl> impl;
};

}  // namespace yoro_api

#endif
