#include "yoro_api.hpp"
#include "yoro_impl.hpp"

using namespace cv;
using namespace std;

namespace yoro_api
{
Detector::Detector(const char* modelPath, const Detector::DeviceType& devType)
{
    this->impl = shared_ptr<Impl>(new Impl(modelPath, devType));
}

vector<RBox> Detector::detect(const Mat& image, float confTh, float nmsTh)
{
    return this->impl->detect(image, confTh, nmsTh);
}

}  // namespace yoro_api
