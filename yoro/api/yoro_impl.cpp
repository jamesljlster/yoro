#include <cmath>
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
using torch::jit::IValue;
using torch::jit::Object;

namespace yoro_api
{
cv::Mat pad_to_aspect(
    const cv::Mat& src, float aspectRatio, int* startXPtr, int* startYPtr)
{
    int width = src.cols;
    int height = src.rows;

    // Find target width, height
    cv::Mat imSize = cv::Mat({width, height});
    cv::Mat cand1 = cv::Mat({width, (int)std::round(width / aspectRatio)});
    cv::Mat cand2 = cv::Mat({(int)std::round(height * aspectRatio), height});
    cv::Mat tarSize = (cv::sum((cand1 - imSize) < 0)[0] == 0) ? cand1 : cand2;

    int tarWidth = tarSize.at<int>(0);   // Target width
    int tarHeight = tarSize.at<int>(1);  // Target height

    // Find padding parameters
    float wPad = (float)(tarWidth - width) / 2.0;
    float hPad = (float)(tarHeight - height) / 2.0;

    int lPad = (int)std::floor(wPad);
    int tPad = (int)std::floor(hPad);

    // Padding image
    cv::Mat mat = cv::Mat(tarHeight, tarWidth, src.type(), cv::Scalar(0, 0, 0));
    cv::Rect roi = cv::Rect(lPad, tPad, src.cols, src.rows);
    src.copyTo(mat(roi));

    // Assign shifted image origin point
    if (startXPtr) *startXPtr = lPad;
    if (startYPtr) *startYPtr = tPad;

    return mat;
}

GeneralDetector::GeneralDetector(
    const char* modelPath, const DeviceType& devType)
{
    // Detect devices and set tensor options
    bool cudaAvail = torch::cuda::is_available();
    switch (devType)
    {
        case DeviceType::Auto:
            if (cudaAvail)
                this->device = torch::kCUDA;
            else
                this->device = torch::kCPU;
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
    Object suffixLayer = model.attr("suffix").toObject();

    this->netWidth = suffixLayer.attr("width").toInt();
    this->netHeight = suffixLayer.attr("height").toInt();

    this->model.to(this->device, this->scalarType);
    this->model.eval();
}

torch::jit::IValue GeneralDetector::detect(const cv::Mat& image)
{
    if (image.empty())
    {
        throw std::invalid_argument(this->make_error_msg("Empty image."));
    }

    // Conver BGR to RGB
    cv::Mat mat;
    cvtColor(image, mat, cv::COLOR_BGR2RGB);

    // Resizing
    cv::resize(mat, mat, cv::Size(this->netWidth, this->netHeight));

    // Convert image to tensor
    Tensor inputs =
        from_blob(mat.ptr<char>(), {1, mat.rows, mat.cols, 3}, ScalarType::Byte)
            .to(this->opt)
            .permute({0, 3, 1, 2})
            .contiguous() /
        255.0;

    // Forward
    IValue outputs = model.forward({inputs});

    return outputs;
}

std::string GeneralDetector::make_error_msg(const char* msg)
{
    return std::string("[YORO API (Error)] ") + std::string(msg);
}

std::vector<RBox> YORODetector::Impl::detect(
    const cv::Mat& image, float confTh, float nmsTh)
{
    int width = this->netWidth;
    int height = this->netHeight;

    // Pad to aspect ratio
    int startX = 0, startY = 0;
    cv::Mat mat = pad_to_aspect(image, (float)width / height, &startX, &startY);
    float scale = float(mat.cols) / float(width);

    // Forward
    auto outputs = GeneralDetector::detect(mat).toTuple();
    auto listRef = outputs->elements();
    Tensor predConf = listRef[0].toTensor();
    Tensor predClass = listRef[1].toTensor();
    Tensor predBox = listRef[2].toTensor();
    Tensor predDeg = listRef[3].toTensor();

    // Denormalize
    predBox.mul_(scale);
    predBox.index({"...", Slice(0, 2)})
        .sub_(tensor({{{startX, startY}}}, this->device));

    // Processing non-maximum suppression
    std::vector<std::vector<RBox>> nmsOut = yoro_api::non_maximum_suppression(
        {predConf, predClass, predBox, predDeg}, confTh, nmsTh);

    return nmsOut[0];
}

float RotationDetector::Impl::detect(const cv::Mat& image)
{
    int width = this->netWidth;
    int height = this->netHeight;

    // Pad to aspect ratio
    cv::Mat mat = pad_to_aspect(image, (float)width / height);

    // Forward
    Tensor outputs = GeneralDetector::detect(mat).toTensor();

    return outputs.to(torch::kFloat32).item<float>();
}

}  // namespace yoro_api
