#include <torch/extension.h>

#include "yoro_api.hpp"

using namespace yoro_api;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::class_<RBox>(m, "RBox")
        .def(py::init())
        .def(py::init<float, float, float, float, float, float, float>())
        .def("__repr__", &RBox::to_string)
        .def_readwrite("conf", &RBox::conf)
        .def_readwrite("label", &RBox::label)
        .def_readwrite("degree", &RBox::degree)
        .def_readwrite("x", &RBox::x)
        .def_readwrite("y", &RBox::y)
        .def_readwrite("w", &RBox::w)
        .def_readwrite("h", &RBox::h);
}
