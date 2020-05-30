#ifndef __CALC_OPS_HPP__
#define __CALC_OPS_HPP__

#include <torch/extension.h>
#include <vector>

#include "yoro_api.hpp"

namespace yoro_api
{
template <typename T>
T deg2rad(T deg)
{
    return deg * 3.1415927410125732 / 180.0;
}

torch::Tensor bbox_to_corners(const torch::Tensor& bbox);
torch::Tensor rbox_similarity(const torch::Tensor& pred1,
                              const torch::Tensor& pred2);
std::vector<std::vector<RBox>> non_maximum_suppression(
    const torch::Tensor& predList, float confTh, float nmsTh);

}  // namespace yoro_api

#endif
