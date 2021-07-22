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
using torch::indexing::Ellipsis;
using torch::indexing::Slice;
using torch::jit::IValue;
using torch::jit::Object;

namespace yoro_api
{
std::tuple<cv::Mat, int, int> pad_to_aspect(
    const cv::Mat& src, float aspectRatio)
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

    // Return padded image and shifted image origin point
    return {mat, lPad, tPad};
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
    std::tuple<cv::Mat, int, int> padRet =
        pad_to_aspect(image, (float)width / height);
    cv::Mat mat = std::get<0>(padRet);
    int startX = std::get<1>(padRet);
    int startY = std::get<2>(padRet);
    float scale = float(mat.cols) / float(width);

    // Forward, denormalize and flatten predictions
    std::vector<std::tuple<Tensor, Tensor, Tensor>> fwOutputs;
    auto outList = GeneralDetector::detect(mat).toList();
    for (const IValue& tup : outList)
    {
        // Extract tensors
        auto tupRef = tup.toTuple()->elements();
        Tensor conf = tupRef[0].toTensor();
        Tensor label = tupRef[1].toTensor();
        Tensor rboxes = tupRef[2].toTensor();

        // Denormalize
        rboxes.index({Ellipsis, Slice(1, 5)}).mul_(scale);
        rboxes.index({Ellipsis, Slice(1, 3)})
            .sub_(tensor({{{{{startX, startY}}}}}, this->device));

        fwOutputs.push_back({conf, label, rboxes});
    }

    auto flatOut = flatten_prediction(fwOutputs);

    // Processing non-maximum suppression
    std::vector<std::vector<RBox>> nmsOut =
        yoro_api::non_maximum_suppression(flatOut, confTh, nmsTh);

    return nmsOut[0];
}

float RotationDetector::Impl::detect(const cv::Mat& image)
{
    int width = this->netWidth;
    int height = this->netHeight;

    // Pad to aspect ratio
    cv::Mat mat = std::get<0>(pad_to_aspect(image, (float)width / height));

    // Forward
    Tensor outputs = GeneralDetector::detect(mat).toTensor();

    return outputs.to(torch::kFloat32).item<float>();
}

}  // namespace yoro_api
