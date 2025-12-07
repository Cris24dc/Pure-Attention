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
        // Initializam m si v cu 0 pentru fiecare parametru
        for (const auto& p : parameters) {
            // Creăm tensori de aceeași formă, fără gradienți
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

        for (size_t i = 0; i < parameters.size(); ++i) {
            auto& param = parameters[i];
            
            // Dacă parametrul nu are gradient calculat, sărim peste el
            if (!param->requires_grad() || param->get_gradient_ptr() == nullptr) {
                continue;
            }

            uint32_t size = 1;
            for(auto s : param->get_shape()) size *= s;

            // Lansăm kernel-ul CUDA definit anterior
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
                step_count,
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