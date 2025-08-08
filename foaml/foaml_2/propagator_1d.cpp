#include <vector>
#include <complex>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

py::array_t<std::complex<double>> propagate_1d_cpp(
    py::array_t<std::complex<double>> field,
    double wavelength, double distance) {
    // ... (FFT using FFTW library)
}

PYBIND11_MODULE(propagator_1d_cpp, m) {
    m.def("propagate", &propagate_1d_cpp);
}