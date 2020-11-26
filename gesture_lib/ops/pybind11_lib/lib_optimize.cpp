#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

using namespace std;
namespace py = pybind11;


float dist(const py::array_t<float>& kernel,
		   const py::array_t<float>& img,
		   const int& x,
		   const int& y,
		   const int& k_w,
		   const int& k_h)
{
	float d;
	py::array_t<float> region;


}
