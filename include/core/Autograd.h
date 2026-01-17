#pragma once

#include <core/Tensor.h>

namespace core {
    struct Function {
        virtual void apply_backward() = 0;
        virtual ~Function() = default;
    };

    struct MatMulFunction : public Function {
        std::shared_ptr<Tensor> X_input;
        std::shared_ptr<Tensor> W_input;
        std::weak_ptr<Tensor> Y_output;

        MatMulFunction(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> w, std::shared_ptr<Tensor> y);

        void apply_backward() override;
    };

    struct AddFunction : public Function {
        std::shared_ptr<Tensor> X_input;
        std::shared_ptr<Tensor> bias_input;
        std::weak_ptr<Tensor> Y_output;

        AddFunction(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> bias, std::shared_ptr<Tensor> y);

        void apply_backward() override;
    };

    struct ReLUFunction : public Function {
        std::shared_ptr<Tensor> Input;
        std::weak_ptr<Tensor> Output;

        ReLUFunction(std::shared_ptr<Tensor> in, std::shared_ptr<Tensor> out);

        void apply_backward() override ;
    };

    struct MSEFunction : public Function {
        std::shared_ptr<Tensor> predictions;
        std::shared_ptr<Tensor> targets;
        std::weak_ptr<Tensor> output_loss;

        MSEFunction(const std::shared_ptr<Tensor>& preds,
                    const std::shared_ptr<Tensor>& targs,
                    const std::shared_ptr<Tensor>& out);

        void apply_backward() override;
    };

    struct FlashAttentionFunction : public Function {
        std::shared_ptr<Tensor> Q_input;
        std::shared_ptr<Tensor> K_input;
        std::shared_ptr<Tensor> V_input;
        std::weak_ptr<Tensor> O_output;
        std::shared_ptr<Tensor> L_cache;

        FlashAttentionFunction(const std::shared_ptr<Tensor>& q,
                               const std::shared_ptr<Tensor>& k,
                               const std::shared_ptr<Tensor>& v,
                               const std::shared_ptr<Tensor>& o,
                               const std::shared_ptr<Tensor>& lcache);

        void apply_backward() override;
    };
};