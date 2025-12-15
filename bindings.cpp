// headers
#include <core/Tensor.h>
#include <layers/Module.h>
#include <layers/Linear.h>
#include <layers/ReLU.h>
#include <loss/MSE.h>
#include <core/Functional.h>
#include <core/Adam.h>

// libs
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace py = pybind11;
using namespace core;
using namespace layers;

class PyModule : public Module {
public:
    using Module::Module;
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor> &input) override {
        PYBIND11_OVERRIDE_PURE(std::shared_ptr<Tensor>, Module, forward, input);
    }
    std::vector<std::shared_ptr<Tensor>> parameters() override {
        PYBIND11_OVERRIDE_PURE(std::vector<std::shared_ptr<Tensor>>, Module, parameters,);
    }
};

PYBIND11_MODULE(PureAttention, m) {
    m.doc() = "Deep Learning Framework with Debugging Tools";

    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
        .def(py::init<std::vector<uint32_t>, bool>(), py::arg("shape"), py::arg("requires_grad")=false)
        .def("backward", &Tensor::backward)
        .def("to_host", &Tensor::to_host)
        .def("to_device", &Tensor::to_device)
        .def("grad_to_host", &Tensor::grad_to_host)
        .def("shape", &Tensor::get_shape)

        .def("get_grad_stats", [](const Tensor &t) {
            std::vector<float> grads = t.grad_to_host();
            
            if (grads.empty()) {
                return py::dict();
            }

            double sum = std::accumulate(grads.begin(), grads.end(), 0.0);
            double mean = sum / grads.size();

            double sq_sum = std::inner_product(grads.begin(), grads.end(), grads.begin(), 0.0);
            double std_dev = std::sqrt(sq_sum / grads.size() - mean * mean);

            float min_val = *std::min_element(grads.begin(), grads.end());
            float max_val = *std::max_element(grads.begin(), grads.end());

            py::dict stats;
            stats["mean"] = mean;
            stats["std"] = std_dev;
            stats["min"] = min_val;
            stats["max"] = max_val;
            return stats;
        }, "Gradient stats");

    py::class_<Module, PyModule, std::shared_ptr<Module>>(m, "Module")
        .def(py::init<>())
        .def("forward", &Module::forward)
        .def("parameters", &Module::parameters);

    py::class_<Linear, Module, std::shared_ptr<Linear>>(m, "Linear")
        .def(py::init<uint32_t, uint32_t>())
        .def("forward", &Linear::forward)
        .def("parameters", &Linear::parameters)
        .def_readwrite("weight", &Linear::weight)
        .def_readwrite("bias", &Linear::bias);

    py::class_<ReLU, Module, std::shared_ptr<ReLU>>(m, "ReLU")
        .def(py::init<>())
        .def("forward", &ReLU::forward)
        .def("parameters", &ReLU::parameters);

    py::class_<loss::MSE, std::shared_ptr<loss::MSE>>(m, "MSE")
        .def(py::init<>())
        .def("forward", py::overload_cast<const std::shared_ptr<Tensor>&, const std::shared_ptr<Tensor>&>(&loss::MSE::forward))
        .def("__call__", py::overload_cast<const std::shared_ptr<Tensor>&, const std::shared_ptr<Tensor>&>(&loss::MSE::forward));

    py::class_<optim::Adam, std::shared_ptr<optim::Adam>>(m, "Adam")
        .def(py::init<std::vector<std::shared_ptr<Tensor>>, float, float, float, float>(), 
             py::arg("params"), py::arg("lr")=0.001f, py::arg("beta1")=0.9f, py::arg("beta2")=0.999f, py::arg("eps")=1e-8f)
        .def("step", &optim::Adam::step)
        .def("zero_grad", &optim::Adam::zero_grad);
}
