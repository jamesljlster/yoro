#ifndef __YORO_API_HPP__
#define __YORO_API_HPP__

#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

namespace yoro_api
{
struct RBox
{
    /** Constructor */
    RBox(float conf, int label, float degree, float x, float y, float w,
         float h)
        : conf(conf), label(label), degree(degree), x(x), y(y), w(w), h(h)
    {
    }

    /** String conversion */
    operator std::string() const;
    std::string to_string() const;

    /** Attributes of rotated bounding box */
    float conf;    // Confidence (objectness * class probability)
    int label;     // Class label
    float degree;  // Degree
    float x;       // Center x
    float y;       // Center y
    float w;       // Width
    float h;       // Height
};

class Detector
{
   public:
    explicit Detector(const char* modelPath);
    explicit Detector(const std::string& modelPath)
        : Detector(modelPath.c_str())
    {
    }

    std::vector<RBox> detect(const cv::Mat& image, float confTh, float nmsTh);

   protected:
    class Impl;
    std::shared_ptr<Impl> impl;
};

}  // namespace yoro_api

#endif
