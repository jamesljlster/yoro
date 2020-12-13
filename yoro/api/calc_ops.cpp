#include "calc_ops.hpp"

#define PROC_DTYPE torch::kFloat

using namespace torch;
using namespace torch::indexing;

using std::tuple;
using std::vector;

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

Tensor rbox_similarity(const Tensor& pred1, const Tensor& pred2)
{
    // BBox to corners
    Tensor corners1 = bbox_to_corners(pred1.index({Ellipsis, Slice(4, 8)}));
    Tensor corners2 = bbox_to_corners(pred2.index({Ellipsis, Slice(4, 8)}));

    // Find IoU scores
    Tensor interX1 =
        max(corners1.index({Ellipsis, 0}), corners2.index({Ellipsis, 0}));
    Tensor interY1 =
        max(corners1.index({Ellipsis, 1}), corners2.index({Ellipsis, 1}));
    Tensor interX2 =
        min(corners1.index({Ellipsis, 2}), corners2.index({Ellipsis, 2}));
    Tensor interY2 =
        min(corners1.index({Ellipsis, 3}), corners2.index({Ellipsis, 3}));

    Tensor interArea =
        clamp(interX2 - interX1, 0) * clamp(interY2 - interY1, 0);
    Tensor unionArea = pred1.index({Ellipsis, 6}) * pred2.index({Ellipsis, 6}) +
                       pred1.index({Ellipsis, 7}) * pred2.index({Ellipsis, 7}) -
                       interArea;
    Tensor ious = interArea / (unionArea + 1e-4);

    // Find degree similarity
    Tensor rad1 = deg2rad(pred1.index({Ellipsis, 3}));
    Tensor rad2 = deg2rad(pred2.index({Ellipsis, 3}));
    Tensor ang1 = stack({sin(rad1), cos(rad1)}, 1);
    Tensor ang2 = stack({sin(rad2), cos(rad2)}, 1);
    Tensor angSim = (matmul(ang1, ang2.t()) + 1.0) / 2.0;

    return ious * angSim;
}

vector<vector<RBox>> non_maximum_suppression(
    const Tensor& predIn, float confTh, float nmsTh)
{
    int batch = predIn.size(0);
    vector<vector<RBox>> nmsOut(batch, vector<RBox>());

    // Detach tensor
    predIn.detach_();

    // Processing NMS on mini-batch
    for (int n = 0; n < batch; n++)
    {
        Tensor pred = predIn.index({n, "..."});

        // Objectness filtering
        pred = pred.index(pred.index({Ellipsis, 0}) >= confTh);

        // RBox similarity filtering
        vector<RBox> boxes;
        while (pred.size(0))
        {
            // Sort rbox with confidence
            Tensor confScore =
                pred.index({Ellipsis, 0}) * pred.index({Ellipsis, 2});
            pred = pred.index({confScore.argsort(0, true)});

            // Get indices of rbox with same label
            Tensor labelMatch =
                (pred.index({0, 1}) == pred.index({Ellipsis, 1}));

            // Get indices of rbox with high similarity
            Tensor highSim =
                (rbox_similarity(
                     pred.index({0, Ellipsis}).unsqueeze(0), pred) >= nmsTh);

            // Find indices and weights of rbox that ready to be merged.
            Tensor rmIdx = logical_and(labelMatch, highSim).squeeze(0);
            Tensor weight = confScore.index(rmIdx).unsqueeze(0);

            // Find merged rbox
            Tensor rbox =
                (matmul(weight, pred.index({rmIdx, Ellipsis})) / weight.sum())
                    .squeeze(0);

            auto data = rbox.accessor<float, 1>();
            boxes.push_back(RBox(
                data[0] * data[2],  // Confidence
                (int)data[1],       // Label
                data[3],            // Degree
                data[4],            // Center x
                data[5],            // Center y
                data[6],            // Width
                data[7]             // Height
                ));

            // Remove processed rbox
            pred = pred.index({logical_not(rmIdx)});
        }

        nmsOut[n] = boxes;
    }

    return nmsOut;
}

vector<vector<RBox>> non_maximum_suppression(
    const Tensor& predConf,
    const Tensor& predClass,
    const Tensor& predClassConf,
    const Tensor& predBox,
    const Tensor& predDeg,
    float confTh,
    float nmsTh)
{
    // Concatenate tensor
    Tensor pred = torch::cat(
                      {predConf.unsqueeze(-1).to(PROC_DTYPE),
                       predClass.unsqueeze(-1).to(PROC_DTYPE),
                       predClassConf.unsqueeze(-1).to(PROC_DTYPE),
                       predDeg.unsqueeze(-1).to(PROC_DTYPE),
                       predBox.to(PROC_DTYPE)},
                      2)
                      .to(torch::kCPU);

    // Processing non-maximum suppression
    return non_maximum_suppression(pred, confTh, nmsTh);
}

vector<vector<RBox>> non_maximum_suppression(
    const tuple<Tensor, Tensor, Tensor, Tensor, Tensor>& outputs,
    float confTh,
    float nmsTh)
{
    // Unpack tuple
    Tensor predConf = std::get<0>(outputs);
    Tensor predClass = std::get<1>(outputs);
    Tensor predClassConf = std::get<2>(outputs);
    Tensor predBox = std::get<3>(outputs);
    Tensor predDeg = std::get<4>(outputs);

    // Processing non-maximum suppression
    return non_maximum_suppression(
        predConf, predClass, predClassConf, predBox, predDeg, confTh, nmsTh);
}

}  // namespace yoro_api
