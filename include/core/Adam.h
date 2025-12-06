#pragma once

// libs
#include <vector>
#include <memory>
#include <core/Tensor.h>

namespace optim {

    class Adam {
    private:
        std::vector<std::shared_ptr<core::Tensor>> parameters;
        
        // Stocăm m și v pentru fiecare parametru. 
        std::vector<std::shared_ptr<core::Tensor>> m_states;
        std::vector<std::shared_ptr<core::Tensor>> v_states;

        float lr;
        float beta1;
        float beta2;
        float epsilon;
        int step_count;

    public:
        Adam(std::vector<std::shared_ptr<core::Tensor>> params, float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f);

        void step();
        void zero_grad();
    };
}