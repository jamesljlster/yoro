#include <stdexcept>
#include <tuple>
#include <vector>

#include <pybind11/numpy.h>

#include "calc_ops.hpp"
#include "yoro_api.hpp"
#include "yoro_api_pym.hpp"

using namespace yoro_api;
using namespace pybind11::detail;

std::tuple<const uint8_t*, int, int, int> array_to_image(
    const py::array& buffer)
{
    // Check if input array is contiguous
    if (!(buffer.flags() & npy_api::constants::NPY_ARRAY_C_CONTIGUOUS_))
    {
        throw std::invalid_argument(
            "Input array should be C-style contiguous!");
    }

    // Check array dtype
    py::dtype bufType = buffer.dtype();
    if (!(bufType.is(py::dtype::of<unsigned char>()) ||
          bufType.is(py::dtype::of<signed char>())))
    {
        throw std::invalid_argument(
            "Dtype of input array should be either np.uint8 or np.int8");
    }

    // Check array dimensions
    int ndim = buffer.ndim();
    std::vector<int> sizes({0, 0, 1});
    if (ndim < 2 || ndim > 3)
    {
        throw std::invalid_argument(
            "Dimensions of input array should be 2 or 3!");
    }

    for (int i = 0; i < ndim; i++)
    {
        sizes[i] = buffer.shape(i);
    }

    return {(const uint8_t*)buffer.data(), sizes[0], sizes[1], sizes[2]};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::class_<RBox>(m, "RBox")
        .def(py::init())
        .def(py::init<float, int, float, float, float, float, float>())
        .def("__repr__", &RBox::to_string)
        .def(
            "to_dict",
            [](const RBox& rbox) -> py::dict
            {
                py::dict ret;
                ret["conf"] = rbox.conf;
                ret["label"] = rbox.label;
                ret["degree"] = rbox.degree;
                ret["x"] = rbox.x;
                ret["y"] = rbox.y;
                ret["w"] = rbox.w;
                ret["h"] = rbox.h;
                return ret;
            })
        .def_readwrite("conf", &RBox::conf)
        .def_readwrite("label", &RBox::label)
        .def_readwrite("degree", &RBox::degree)
        .def_readwrite("x", &RBox::x)
        .def_readwrite("y", &RBox::y)
        .def_readwrite("w", &RBox::w)
        .def_readwrite("h", &RBox::h);

    py::enum_<DeviceType>(m, "DeviceType")
        .value("Auto", DeviceType::Auto)
        .value("CPU", DeviceType::CPU)
        .value("CUDA", DeviceType::CUDA)
        .export_values();

    py::class_<YORODetector>(m, "YORODetector")
        .def(py::init<const std::string&, const DeviceType&>())
        .def(py::init<const std::string&>())
        .def(
            "detect",
            [](YORODetector& detector,
               const py::array& image,
               float confTh,
               float nmsTh)
            {
                std::tuple<const uint8_t*, int, int, int> data =
                    array_to_image(image);
                return detector.detect(
                    std::get<0>(data),
                    std::get<1>(data),
                    std::get<2>(data),
                    std::get<3>(data),
                    confTh,
                    nmsTh);
            });

    py::class_<RotationDetector>(m, "RotationDetector")
        .def(py::init<const std::string&, const DeviceType&>())
        .def(py::init<const std::string&>())
        .def(
            "detect",
            [](RotationDetector& detector, const py::array& image)
            {
                std::tuple<const uint8_t*, int, int, int> data =
                    array_to_image(image);
                return detector.detect(
                    std::get<0>(data),
                    std::get<1>(data),
                    std::get<2>(data),
                    std::get<3>(data));
            });

    m.def("bbox_to_corners", bbox_to_corners);
    m.def("rbox_similarity", rbox_similarity);
    m.def("flatten_prediction", flatten_prediction);
    m.def("non_maximum_suppression", non_maximum_suppression);
    m.def("resize", resize);
    m.def("pad_to_aspect", pad_to_aspect);
}
