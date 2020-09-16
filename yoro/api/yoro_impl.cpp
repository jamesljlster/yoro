#include <exception>
#include <stdexcept>

#include <torch/script.h>

#include "calc_ops.hpp"
#include "yoro_impl.hpp"

using torch::from_blob;
using torch::ScalarType;
using torch::Tensor;
using torch::tensor;
using torch::indexing::Slice;
using torch::jit::Object;

namespace yoro_api
{
Detector::Impl::Impl(const char* modelPath, const Detector::DeviceType& devType)
{
    // Detect devices and set tensor options
    bool cudaAvail = torch::cuda::is_available();
    switch (devType)
    {
        case DeviceType::Auto:
            this->device = cudaAvail ? torch::kCUDA : torch::kCPU;
            break;

        case DeviceType::CPU:
            this->device = torch::kCPU;
            break;

        case DeviceType::CUDA:
            this->device = torch::kCUDA;
            break;
    }

    if ((this->device == torch::kCUDA) && (!cudaAvail))
    {
        throw std::runtime_error(
            this->make_error_msg("CUDA device is unavailable."));
    }

    this->opt = this->opt.device(this->device).dtype(this->scalarType);

    // Import model and settings
    this->model = torch::jit::load(modelPath);
    Object yoroLayer = model.attr("suffix").toObject();

    this->netWidth = yoroLayer.attr("width").toInt();
    this->netHeight = yoroLayer.attr("height").toInt();

    this->model.to(this->device, this->scalarType);
    this->model.eval();
}

std::vector<RBox> Detector::Impl::detect(
    const cv::Mat& image, float confTh, float nmsTh)
{
    if (image.empty())
    {
        throw std::invalid_argument(this->make_error_msg("Empty image."));
    }

    // Conver BGR to RGB
    cv::Mat src;
    cvtColor(image, src, cv::COLOR_BGR2RGB);

    // Pad to square
    int tarSize = std::max(src.rows, src.cols);
    cv::Mat mat = cv::Mat(tarSize, tarSize, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Rect roi = cv::Rect(
        (tarSize - src.cols) / 2, (tarSize - src.rows) / 2, src.cols, src.rows);
    src.copyTo(mat(roi));

    float startX = (tarSize - src.cols) / 2;
    float startY = (tarSize - src.rows) / 2;
    float scale = float(tarSize) / float(netWidth);

    // Resizing
    cv::resize(mat, mat, cv::Size(this->netWidth, this->netHeight));

    // Normalize image to tensor
    Tensor inputs =
        from_blob(mat.ptr<char>(), {1, mat.rows, mat.cols, 3}, ScalarType::Byte)
            .to(this->opt)
            .permute({0, 3, 1, 2})
            .contiguous() /
        255.0;

    // Forward
    auto outputs = model.forward({inputs}).toTuple();
    auto listRef = outputs->elements();
    Tensor predConf = listRef[0].toTensor();
    Tensor predClass = listRef[1].toTensor();
    Tensor predClassConf = listRef[2].toTensor();
    Tensor predBox = listRef[3].toTensor();
    Tensor predDeg = listRef[4].toTensor();

    // Denormalize
    predBox.mul_(scale);
    predBox.index({"...", Slice(0, 2)})
        .sub_(tensor({{{startX, startY}}}, this->device));

    // Processing non-maximum suppression
    std::vector<std::vector<RBox>> nmsOut = yoro_api::non_maximum_suppression(
        {predConf, predClass, predClassConf, predBox, predDeg}, confTh, nmsTh);

    return nmsOut[0];
}

std::string Detector::Impl::make_error_msg(const char* msg)
{
    return std::string("[YORO API (Error)] ") + std::string(msg);
}

}  // namespace yoro_api
