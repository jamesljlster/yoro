#ifndef __CALC_OPS_HPP__
#define __CALC_OPS_HPP__

#include <torch/extension.h>

namespace yoro_api
{
template <typename T>
T deg2rad(T deg)
{
    return deg * 3.1415927410125732 / 180.0;
}

torch::Tensor bbox_to_corners(const torch::Tensor& bbox);

}  // namespace yoro_api

#endif
