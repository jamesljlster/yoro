#ifndef __CALC_OPS_HPP__
#define __CALC_OPS_HPP__

#include <torch/extension.h>
#include <tuple>
#include <vector>

#include "yoro_api.hpp"

namespace yoro_api
{
torch::Tensor bbox_to_corners(const torch::Tensor& bbox);
torch::Tensor rbox_similarity(
    const torch::Tensor& rbox1, const torch::Tensor& rbox2);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> flatten_prediction(
    const std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>>&
        predList);

std::vector<std::vector<RBox>> non_maximum_suppression(
    const std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>& pred,
    float confTh,
    float nmsTh);

}  // namespace yoro_api

#endif
