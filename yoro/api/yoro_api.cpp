#include "yoro_api.hpp"
#include "yoro_impl.hpp"

using namespace std;

namespace yoro_api
{
YORODetector::YORODetector(const char* modelPath, const DeviceType& devType)
{
    this->impl = shared_ptr<Impl>(new Impl(modelPath, devType));
}

vector<RBox> YORODetector::detect(
    const uint8_t* image,
    int width,
    int height,
    int channels,
    float confTh,
    float nmsTh)
{
    return this->impl->detect(image, width, height, channels, confTh, nmsTh);
}

#ifdef WITH_OPENCV
vector<RBox> YORODetector::detect(
    const cv::Mat& image, float confTh, float nmsTh)
{
    return this->impl->detect(image, confTh, nmsTh);
}
#endif

RotationDetector::RotationDetector(
    const char* modelPath, const DeviceType& devType)
{
    this->impl = shared_ptr<Impl>(new Impl(modelPath, devType));
}

float RotationDetector::detect(
    const uint8_t* image, int width, int height, int channels)
{
    return this->impl->detect(image, width, height, channels);
}

#ifdef WITH_OPENCV
float RotationDetector::detect(const cv::Mat& image)
{
    return this->impl->detect(image);
}
#endif

}  // namespace yoro_api
