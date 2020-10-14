#ifndef __YORO_API_PYM_HPP__
#define __YORO_API_PYM_HPP__

#include <pybind11/numpy.h>
#include <torch/extension.h>
#include <opencv2/opencv.hpp>

namespace pybind11
{
namespace detail
{
template <>
struct type_caster<cv::Mat>
{
   public:
    PYBIND11_TYPE_CASTER(cv::Mat, _("cv::Mat"));

    bool load(py::handle src, bool convert)
    {
        // Convert handle to python array
        py::array buffer = py::array::ensure(src);
        if (!buffer)
        {
            return false;
        }

        // Find Mat type
        py::dtype bufType = buffer.dtype();

        int type;
        if (bufType.is(py::dtype::of<unsigned char>()))
            type = CV_8U;
        else if (bufType.is(py::dtype::of<signed char>()))
            type = CV_8S;
        else if (bufType.is(py::dtype::of<unsigned short>()))
            type = CV_16U;
        else if (bufType.is(py::dtype::of<signed short>()))
            type = CV_16S;
        else if (bufType.is(py::dtype::of<int>()))
            type = CV_32S;
        else if (bufType.is(py::dtype::of<float>()))
            type = CV_32F;
        else if (bufType.is(py::dtype::of<double>()))
            type = CV_64F;
        else
            return false;

        // Get buffer shape and dimension
        int dim = buffer.ndim();
        std::vector<int> sizes;
        for (int i = 0; i < dim - 1; i++)
        {
            sizes.push_back(buffer.shape(i));
        }

        // Convert buffer to Mat
        int depth = buffer.shape(dim - 1);
        this->value =
            cv::Mat(sizes, CV_MAKETYPE(type, depth), (void*)buffer.data())
                .clone();

        return true;
    }

    static py::handle cast(const cv::Mat& src, return_value_policy, handle)
    {
        // Find numpy array type corresponding to given opencv mat
        int depth = src.depth();

        py::dtype dtype;
        if (depth == CV_8U)
            dtype = py::dtype::of<unsigned char>();
        else if (depth == CV_8S)
            dtype = py::dtype::of<signed char>();
        else if (depth == CV_16U)
            dtype = py::dtype::of<unsigned short>();
        else if (depth == CV_16S)
            dtype = py::dtype::of<signed short>();
        else if (depth == CV_32S)
            dtype = py::dtype::of<int>();
        else if (depth == CV_32F)
            dtype = py::dtype::of<float>();
        else if (depth == CV_64F)
            dtype = py::dtype::of<double>();
        else
            return py::none();

        // Construct python buffer
        std::vector<ssize_t> shape;
        shape.push_back(src.rows);
        shape.push_back(src.cols);
        shape.push_back(src.channels());

        return py::array(dtype, shape, src.ptr()).release();
    }
};

}  // namespace detail
}  // namespace pybind11

#endif
