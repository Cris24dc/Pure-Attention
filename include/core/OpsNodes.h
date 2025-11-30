#pragma once

#include <core/Tensor.h>

namespace core {
    struct MatMulFunction : public Function {
        std::shared_ptr<Tensor> X_input;
        std::shared_ptr<Tensor> W_input;
        std::weak_ptr<Tensor> Y_output;

        MatMulFunction(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b, std::shared_ptr<Tensor> c);

        void apply_backward() override;
    };

    struct AddFunction : public Function {
        std::shared_ptr<Tensor> Input_A;
        std::shared_ptr<Tensor> Input_B;
        std::weak_ptr<Tensor> Output_C;

        AddFunction(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b, std::shared_ptr<Tensor> &c);

        void apply_backward() override;
    };

    struct ReLUFunction : public Function {
        std::shared_ptr<Tensor> Input;
        std::weak_ptr<Tensor> Output;

        ReLUFunction(std::shared_ptr<Tensor> in, std::shared_ptr<Tensor> out);

        void apply_backward() override ;
    };
};