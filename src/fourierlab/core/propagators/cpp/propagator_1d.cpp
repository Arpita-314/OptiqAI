#include <pybind11/pybind11.h>
#include <vector>
#include <complex>

namespace py = pybind11;

std::vector<std::complex<double>> propagate_1d(
    const std::vector<std::complex<double>>& field,
    double wavelength,
    double distance) {
    // FFT-based propagation kernel
    // ... implement your physics here ...
}

PYBIND11_MODULE(propagator_core, m) {
    m.def("propagate_1d", &propagate_1d);
}