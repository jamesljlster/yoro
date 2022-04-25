
#include "yoro_api_pym.hpp"
#include "calc_ops.hpp"
#include "yoro_api.hpp"

using namespace yoro_api;

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
#ifdef WITH_OPENCV
        .def(
            "detect",
            py::overload_cast<const cv::Mat&, float, float>(
                &YORODetector::detect))
#endif
        ;

    py::class_<RotationDetector>(m, "RotationDetector")
        .def(py::init<const std::string&, const DeviceType&>())
        .def(py::init<const std::string&>())
#ifdef WITH_OPENCV
        .def(
            "detect",
            py::overload_cast<const cv::Mat&>(&RotationDetector::detect))
#endif
        ;

    m.def("bbox_to_corners", bbox_to_corners);
    m.def("rbox_similarity", rbox_similarity);
    m.def("flatten_prediction", flatten_prediction);
    m.def("non_maximum_suppression", non_maximum_suppression);
    m.def("resize", resize);
    m.def("pad_to_aspect", pad_to_aspect);
}
