#include "yoro_api.hpp"
#include "yoro_impl.hpp"

using namespace cv;
using namespace std;

namespace yoro_api
{
YORODetector::YORODetector(const char* modelPath, const DeviceType& devType)
{
    this->impl = shared_ptr<Impl>(new Impl(modelPath, devType));
}

vector<RBox> YORODetector::detect(const Mat& image, float confTh, float nmsTh)
{
    return this->impl->detect(image, confTh, nmsTh);
}

}  // namespace yoro_api
