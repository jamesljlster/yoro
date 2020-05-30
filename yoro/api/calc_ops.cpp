#include "calc_ops.hpp"

using namespace torch;
using namespace torch::indexing;

namespace yoro_api
{
Tensor bbox_to_corners(const Tensor& bbox)
{
    Tensor corners = torch::zeros_like(bbox);

    corners.index({Ellipsis, 0}) =
        bbox.index({Ellipsis, 0}) - bbox.index({Ellipsis, 2}) / 2.0;
    corners.index({Ellipsis, 1}) =
        bbox.index({Ellipsis, 1}) - bbox.index({Ellipsis, 3}) / 2.0;
    corners.index({Ellipsis, 2}) =
        bbox.index({Ellipsis, 0}) + bbox.index({Ellipsis, 2}) / 2.0;
    corners.index({Ellipsis, 3}) =
        bbox.index({Ellipsis, 1}) + bbox.index({Ellipsis, 3}) / 2.0;

    return corners;
}

}  // namespace yoro_api
