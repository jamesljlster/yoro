#include <algorithm>
#include <exception>
#include <initializer_list>
#include <stdexcept>

#include "calc_ops.hpp"

#define PROC_DTYPE torch::kFloat

using namespace torch;
using namespace torch::indexing;

using std::invalid_argument;
using std::tuple;
using std::vector;

namespace yoro_api
{
template <typename T>
bool is_in(const T& value, std::initializer_list<T> list)
{
    return (std::find(std::begin(list), std::end(list), value)) !=
           std::end(list);
}

Tensor resize(const Tensor& source, const vector<long>& outputSize)
{
    Tensor data = source;
    ScalarType dtype = source.scalar_type();

    // Unsqueeze and cast
    bool needSqueeze = false;
    bool needCast = !is_in<ScalarType>(dtype, {kFloat, kDouble});

    if (data.dim() < 4)
    {
        needSqueeze = true;
        data = data.unsqueeze(0);
    }

    if (needCast)
    {
        data = data.to(PROC_DTYPE);
    }

    // Resize with bilinear interpolation
    data = upsample_bilinear2d(data, outputSize, true);

    // Squeeze an cast back
    if (needSqueeze)
    {
        data = data.squeeze(0);
    }

    if (needCast)
    {
        if (is_in(dtype, {kUInt8, kInt8, kInt16, kInt32, kInt64}))
        {
            data = round(data);
        }
        data = data.to(dtype);
    }

    return data;
}

tuple<Tensor, vector<long>> pad_to_aspect(
    const Tensor& source, float aspectRatio)
{
    // Get source resolution
    if (source.dim() < 2)
    {
        throw invalid_argument(
            "Input source should contain at least 2 dimensions.");
    }

    int width = source.size(-1);
    int height = source.size(-2);

    // Find target size
    Tensor imSize = tensor({width, height}, kLong);
    Tensor cand1 = tensor({width, (int)std::round(width / aspectRatio)});
    Tensor cand2 = tensor({(int)std::round(height * aspectRatio), height});
    Tensor tarSize = any((cand1 - imSize) < 0).item<bool>() ? cand2 : cand1;

    // Find padding parameters
    Tensor sizeDiff = tarSize - imSize;
    Tensor ltPad = bitwise_right_shift(sizeDiff, 1);
    Tensor rbPad = sizeDiff - ltPad;

    Tensor pad = stack({ltPad, rbPad}).t().flatten();
    std::vector<long> padArray(
        pad.data_ptr<long>(), pad.data_ptr<long>() + pad.numel());

    // Padding tensor
    Tensor output = constant_pad_nd(source, padArray, 0);

    return {output, padArray};
}

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
    Tensor lt =
        max(corners1.index({Ellipsis, None, Slice(None, 2)}),
            corners2.index({Ellipsis, Slice(None, 2)}));
    Tensor rb =
        min(corners1.index({Ellipsis, None, Slice(2, None)}),
            corners2.index({Ellipsis, Slice(2, None)}));
    Tensor wh = clamp(rb - lt, 0);

    Tensor interArea = wh.index({Ellipsis, 0}) * wh.index({Ellipsis, 1});
    Tensor unionArea = (rbox1.index({Ellipsis, 3}) * rbox1.index({Ellipsis, 4}))
                           .index({Ellipsis, None}) +
                       rbox2.index({Ellipsis, 3}) * rbox2.index({Ellipsis, 4}) -
                       interArea;
    Tensor ious = interArea / unionArea;

    // Find degree similarity
    Tensor rad1 = deg2rad(rbox1.index({Ellipsis, 0}));
    Tensor rad2 = deg2rad(rbox2.index({Ellipsis, 0}));
    Tensor ang1 = stack({sin(rad1), cos(rad1)}, 1);
    Tensor ang2 = stack({sin(rad2), cos(rad2)}, 1);
    Tensor angSim = (matmul(ang1, ang2.t()) + 1.0) / 2.0;

    return ious * angSim;
}

tuple<Tensor, Tensor, Tensor> flatten_prediction(
    const vector<tuple<Tensor, Tensor, Tensor>>& predList)
{
    vector<Tensor> totalConf, totalLabel, totalRBox;
    for (tuple<Tensor, Tensor, Tensor> inst : predList)
    {
        Tensor conf = std::get<0>(inst);
        Tensor label = std::get<1>(inst);
        Tensor rbox = std::get<2>(inst);

        auto batch = conf.size(0);
        totalConf.push_back(conf.view({batch, -1}));
        totalLabel.push_back(label.view({batch, -1}));
        totalRBox.push_back(rbox.view({batch, -1, 5}));
    }

    return {cat(totalConf, 1), cat(totalLabel, 1), cat(totalRBox, 1)};
}

vector<vector<RBox>> non_maximum_suppression(
    const tuple<Tensor, Tensor, Tensor>& pred, float confTh, float nmsTh)
{
    Tensor predConf = std::get<0>(pred).to(torch::kCPU);
    Tensor predClass = std::get<1>(pred).to(torch::kCPU);
    Tensor predRBox = std::get<2>(pred).to(torch::kCPU);

    auto batch = predConf.size(0);
    vector<vector<RBox>> nmsOut(batch, vector<RBox>());

    // Processing NMS on mini-batch
    for (auto n = 0; n < batch; n++)
    {
        // Confidence filtering
        Tensor mask = (predConf.index({n}) >= confTh);
        Tensor conf = predConf.index({n, mask});
        Tensor cls = predClass.index({n, mask});
        Tensor rbox = predRBox.index({n, mask});
        Tensor sim = rbox_similarity(rbox, rbox);

        if (conf.size(-1) == 0)
        {
            continue;
        }

        while (true)
        {
            // Start with the maximum confident instance
            tuple<Tensor, Tensor> ret = max(conf, -1);
            float maxConf = std::get<0>(ret).item<float>();
            long ind = std::get<1>(ret).item<long>();
            if (maxConf < confTh)
            {
                break;
            }

            // Merge instances with high similarity
            long curClass = cls.accessor<long, 1>()[ind];
            Tensor candMask = (conf >= confTh) & (cls == curClass) &
                              (sim.index({ind}) >= nmsTh);
            if (!candMask.sum().item<long>())
            {
                candMask.accessor<bool, 1>()[ind] = true;
            }

            Tensor weight = conf.index({candMask});
            Tensor resultRBox =
                matmul(weight, rbox.index({candMask})) / weight.sum();
            Tensor resultConf = matmul(weight, weight) / weight.sum();

            // Clear merged RBox
            conf.index_put_({candMask}, -1.0);

            // Append result
            auto data = resultRBox.accessor<float, 1>();
            nmsOut[n].push_back(RBox(
                resultConf.item<float>(),  // Confidence
                curClass,                  // Label
                data[0],                   // Degree
                data[1],                   // Center x
                data[2],                   // Center y
                data[3],                   // Width
                data[4]                    // Height
                ));
        }
    }

    return nmsOut;
}

}  // namespace yoro_api
