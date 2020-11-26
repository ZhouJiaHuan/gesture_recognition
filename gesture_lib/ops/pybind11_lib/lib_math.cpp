#include <iostream>
#include <cmath>
#include <algorithm>
#include <pybind11/pybind11.h>

using namespace std;
namespace py = pybind11;


py::list cross_point(const py::list& point11,
					 const py::list& point12,
					 const py::list& point21,
					 const py::list& point22,
					 const py::tuple& img_size)
{
	py::list point(2);
	point[0] = -1;
	point[1] = -1;
	int w = img_size[0].cast<float>();
	int h = img_size[1].cast<float>();

	float x11 = point11[0].cast<float>();
	float y11 = point11[1].cast<float>();
	float x12 = point12[0].cast<float>();
	float y12 = point12[1].cast<float>();
	float x21 = point21[0].cast<float>();
	float y21 = point21[1].cast<float>();
	float x22 = point22[0].cast<float>();
	float y22 = point22[1].cast<float>();

	float x, y, k1, b1, k2, b2;

	// case 1: 2 vertical lines
	if (x11 == x12 && x21 == x22)
	{
		return point;
	}

	// case 2: 1 vertical line
	if (x11 == x12 && x21 != x22)
	{
		k2 = (y22-y21) / (x22-x21);
		b2 = y21 - k2 * x21;
		x = x11;
		y = k2 * x + b2;
		point[0] = x;
		point[1] = y;
		return point;
	}

	if (x11 != x12 && x21 == x22)
	{
		k1 = (y12-y11) / (x12-x11);
		b1 = y11 - k1 * x11;
		x = x21;
		y = k1 * x + b1;
		point[0] = x;
		point[1] = y;
		return point;
	}

	k1 = (y12-y11) / (x12-x11);
	k2 = (y22-y21) / (x22-x21);

	// case 3: 2 parallel lines
	if (k1 == k2)
	{
		return point;
	}

	// case 4:
	b1 = y11 - k1 * x11;
	b2 = y21 - k2 * x21;
	x = (b2-b1) / (k1-k2);
	y = x * k1 + b1;
	if (x<0 || x>w ||y<0 || y>h)
	{
		return point;
	}

	point[0] = x;
	point[1] = y;
	return point;
}


py::list cross_point_in(const py::list& point11,
						const py::list& point12,
						const py::list& point21,
						const py::list& point22,
						const py::tuple& img_size)
{
	py::list default_point(2);
	default_point[0] = -1;
	default_point[1] = -1;

	py::list point = cross_point(point11, point12, point21, point22, img_size);

	float x = point[0].cast<float>();
	float y = point[1].cast<float>();
	float p1, p2;

	p1 = point11[0].cast<float>();
	p2 = point12[0].cast<float>();
	if (x < min(p1, p2) || x > max(p1, p2))
	{
		return default_point;
	}

	p1 = point11[1].cast<float>();
	p2 = point12[1].cast<float>();
	if (y < min(p1, p2) || y > max(p1, p2))
	{
		return default_point;
	}

	p1 = point21[0].cast<float>();
	p2 = point22[0].cast<float>();
	if (x < min(p1, p2) || x > max(p1, p2))
	{
		return default_point;
	}

	p1 = point21[1].cast<float>();
	p2 = point22[1].cast<float>();
	if (y < min(p1, p2) || y > max(p1, p2))
	{
		return default_point;
	}

	return point;
}


py::list computeP3withD(const py::tuple& point1,
						const py::tuple& point2,
						const float& d)
{
	int dim = point1.size();
	float x1, y1, x2, y2, k, x, y;

	x1 = point1[0].cast<float>();
	y1 = point1[1].cast<float>();
	x2 = point2[0].cast<float>();
	y2 = point2[1].cast<float>();

	if (dim == 2)
	{
		py::list point(2);
		k = pow(x2-x1, 2) + pow(y2-y1, 2);
		k = d / sqrt(k);
		point[0] = k * (x2-x1) + x1;
		point[1] = k * (y2-y1) + y1;
		return point;
	}

	py::list point(3);
	float z1, z2;
	z1 = point1[2].cast<float>();
	z2 = point2[2].cast<float>();
	k = pow(x2-x1, 2) + pow(y2-y1, 2) + pow(z2-z1, 2);
	k = d / sqrt(k);
	point[0] = k * (x2-x1) + x1;
	point[1] = k * (y2-y1) + y1;
	point[2] = k * (z2-z1) + z1;
	return point;
}

PYBIND11_MODULE(lib_math, m)
{
	m.doc() = "general math lib";
	m.def("cross_point", &cross_point, "compute cross point",
		  py::arg("point11"), py::arg("point12"),
		  py::arg("point21"), py::arg("point22"),
		  py::arg("img_size"));
	m.def("cross_point_in", &cross_point_in, "compute inline cross point",
		  py::arg("point11"), py::arg("point12"),
		  py::arg("point21"), py::arg("point22"),
		  py::arg("img_size"));
	m.def("computeP3withD", &computeP3withD, "compute point with 2 points and distance",
		  py::arg("point1"), py::arg("point2"), py::arg("d"));
}
