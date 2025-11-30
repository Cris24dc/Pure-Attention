#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <core/Tensor.h>
#include <layers/Linear.h>
#include <layers/ReLU.h>

namespace py = pybind11;
using namespace core;
using namespace layers;

class PyModule : public Module {
public:
    using Module::Module;

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor> &input) override {
        // PURE inseamna: Daca Python nu are metoda 'forward', CRAPA cu eroare.
        PYBIND11_OVERRIDE_PURE(
            std::shared_ptr<Tensor>, // Return type
            Module,                  // Base class
            forward,                 // Nume functie
            input                    // Argumente
        );
    }

    // Implementam parameters apeland Python-ul
    std::vector<std::shared_ptr<Tensor>> parameters() override {
        PYBIND11_OVERRIDE_PURE(
            std::vector<std::shared_ptr<Tensor>>,
            Module,
            parameters,
            // fara argumente
        );
    }
};

PYBIND11_MODULE(PureAttention, m) {
    m.doc() = "Framework-ul meu de Deep Learning in C++";

    // 1. Expunem clasa TENSOR
    // <Tensor, std::shared_ptr<Tensor>> ii spune lui Python ca obiectul e managed de shared_ptr
    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
        .def(py::init<std::vector<uint32_t>, bool>(), py::arg("shape"), py::arg("requires_grad")=false)
        .def("backward", &Tensor::backward)
        .def("to_host", &Tensor::to_host)
        .def("to_device", &Tensor::to_device)
        .def("shape", &Tensor::get_shape);

    // 2. Expunem clasa LINEAR
    py::class_<Linear>(m, "Linear")
        .def(py::init<uint32_t, uint32_t>())
        .def("forward", &Linear::forward);

    // 3. Expunem clasa RELU
    py::class_<ReLU>(m, "ReLU")
        .def(py::init<>())
        .def("forward", &ReLU::forward);

    py::class_<Module, PyModule, std::shared_ptr<Module>>(m, "Module")
            .def(py::init<>()) // Constructorul de baza (necesar pt super().__init__)
            .def("forward", &Module::forward)
            .def("parameters", &Module::parameters);
}