
#include "yoro_api_pym.hpp"
#include "calc_ops.hpp"
#include "yoro_api.hpp"

using namespace yoro_api;

int cv_imshow(const cv::Mat& src, int ms)
{
    cv::imshow("Window", src);
    int ret = cv::waitKey(ms);
    return ret;
}

cv::Mat cv_imread(const std::string& path)
{
    return cv::imread(path, cv::IMREAD_COLOR);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::class_<RBox>(m, "RBox")
        .def(py::init())
        .def(py::init<float, float, float, float, float, float, float>())
        .def("__repr__", &RBox::to_string)
        .def(
            "to_dict",
            [](const RBox& rbox) -> py::dict {
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

    m.def(
        "non_maximum_suppression",
        py::overload_cast<
            const std::tuple<
                torch::Tensor,
                torch::Tensor,
                torch::Tensor,
                torch::Tensor,
                torch::Tensor>&,
            float,
            float>(non_maximum_suppression));

    m.def("cv_imshow", &cv_imshow);
    // m.def("cv_imread", &cv_imread);
}
