// headers
#include <core/Adam.h>
#include <backend/Launchers.cuh>
#include <core/Context.h>
#include <core/Functional.h>

// libs
#include <cmath>

namespace optim {
    Adam::Adam(std::vector<std::shared_ptr<core::Tensor>> params, float lr, float beta1, float beta2, float eps)
        : parameters(params), lr(lr), beta1(beta1), beta2(beta2), epsilon(eps), step_count(0) 
    {
        for (const auto& p : parameters) {
            auto m = std::make_shared<core::Tensor>(p->get_shape(), false, false);
            auto v = std::make_shared<core::Tensor>(p->get_shape(), false, false);

            const cudaStream_t& stream = CudaContext::getStream();

            core::pop_data_zeros(m,stream);
            core::pop_data_zeros(v,stream);

            m_states.push_back(m);
            v_states.push_back(v);
        }
    }

    void Adam::step() {
        step_count++;

        float bias_correction1 = 1.0f - std::pow(beta1, step_count);
        float bias_correction2 = 1.0f - std::pow(beta2, step_count);

        for (size_t i = 0; i < parameters.size(); ++i) {
            auto& param = parameters[i];
            
            if (!param->requires_grad() || param->get_gradient_ptr() == nullptr) {
                continue;
            }

            uint32_t size = 1;
            for(auto s : param->get_shape()) size *= s;

            launch_adam_step(
                param->get_data_ptr(),
                param->get_gradient_ptr(),
                m_states[i]->get_data_ptr(),
                v_states[i]->get_data_ptr(),
                size,
                beta1,
                beta2,
                epsilon,
                lr,
                bias_correction1,
                bias_correction2,
                CudaContext::getStream()
            );
        }
    }

    void Adam::zero_grad() {
        for (auto& param : parameters) {
            if (param->requires_grad()) {
                const cudaStream_t& stream = CudaContext::getStream();
                core::pop_grad_zeros(param,stream);
            }
        }
    }
}
