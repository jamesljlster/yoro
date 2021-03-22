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

// rbox: deg, x, y, w, h
Tensor rbox_similarity(const Tensor& rbox1, const Tensor& rbox2)
{
    // BBox to corners
    Tensor corners1 = bbox_to_corners(rbox1.index({Ellipsis, Slice(1, 5)}));
    Tensor corners2 = bbox_to_corners(rbox2.index({Ellipsis, Slice(1, 5)}));

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
    Tensor unionArea = rbox1.index({Ellipsis, 3}) * rbox1.index({Ellipsis, 4}) +
                       rbox2.index({Ellipsis, 3}) * rbox2.index({Ellipsis, 4}) -
                       interArea;
    Tensor ious = interArea / (unionArea + 1e-4);

    // Find degree similarity
    Tensor rad1 = deg2rad(rbox1.index({Ellipsis, 0}));
    Tensor rad2 = deg2rad(rbox2.index({Ellipsis, 0}));
    Tensor ang1 = stack({sin(rad1), cos(rad1)}, 1);
    Tensor ang2 = stack({sin(rad2), cos(rad2)}, 1);
    Tensor angSim = (matmul(ang1, ang2.t()) + 1.0) / 2.0;

    return ious * angSim;
}

// pred: conf, label, deg, x, y, w, h
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

        // Confidence filtering
        pred = pred.index({pred.index({Ellipsis, 0}) >= confTh});

        // RBox similarity filtering
        vector<RBox> boxes;
        while (pred.size(0))
        {
            // Sort rbox with confidence
            Tensor confScore = pred.index({Ellipsis, 0});
            pred = pred.index({confScore.argsort(0, true)});

            // Get indices of rbox with same label
            Tensor labelMatch =
                (pred.index({0, 1}) == pred.index({Ellipsis, 1}));

            // Get indices of rbox with high similarity
            Tensor highSim =
                (rbox_similarity(
                     pred.index({0, Slice(2, 7)}).unsqueeze(0),
                     pred.index({Ellipsis, Slice(2, 7)})) >= nmsTh);

            // Find indices and weights of rbox that ready to be merged.
            Tensor rmIdx = logical_and(labelMatch, highSim).squeeze(0);
            if (!rmIdx.sum().item<bool>())
            {
                rmIdx.accessor<bool, 1>()[0] = true;
            }

            Tensor weight = confScore.index({rmIdx}).unsqueeze(0);

            // Find merged rbox
            Tensor rbox =
                (matmul(weight, pred.index({rmIdx, Ellipsis})) / weight.sum())
                    .squeeze(0);

            auto data = rbox.accessor<float, 1>();
            boxes.push_back(RBox(
                data[0],       // Confidence
                (int)data[1],  // Label
                data[2],       // Degree
                data[3],       // Center x
                data[4],       // Center y
                data[5],       // Width
                data[6]        // Height
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
    const Tensor& predBox,
    const Tensor& predDeg,
    float confTh,
    float nmsTh)
{
    // Concatenate tensor
    Tensor pred = torch::cat(
                      {predConf.unsqueeze(-1).to(PROC_DTYPE),
                       predClass.unsqueeze(-1).to(PROC_DTYPE),
                       predDeg.unsqueeze(-1).to(PROC_DTYPE),
                       predBox.to(PROC_DTYPE)},
                      2)
                      .to(torch::kCPU);

    // Processing non-maximum suppression
    return non_maximum_suppression(pred, confTh, nmsTh);
}

vector<vector<RBox>> non_maximum_suppression(
    const tuple<Tensor, Tensor, Tensor, Tensor>& outputs,
    float confTh,
    float nmsTh)
{
    // Unpack tuple
    Tensor predConf = std::get<0>(outputs);
    Tensor predClass = std::get<1>(outputs);
    Tensor predBox = std::get<2>(outputs);
    Tensor predDeg = std::get<3>(outputs);

    // Processing non-maximum suppression
    return non_maximum_suppression(
        predConf, predClass, predBox, predDeg, confTh, nmsTh);
}

}  // namespace yoro_api
