#include "yoro_api.hpp"
#include "yoro_impl.hpp"

using namespace cv;
using namespace std;

namespace yoro_api
{
Detector::Detector(const char* modelPath)
{
    this->impl = shared_ptr<Impl>(new Impl(modelPath));
}

vector<RBox> Detector::detect(const Mat& image, float confTh, float nmsTh)
{
    return this->impl->detect(image, confTh, nmsTh);
}

}  // namespace yoro_api
