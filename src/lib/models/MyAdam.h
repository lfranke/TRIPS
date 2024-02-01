/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#pragma once

#include <torch/nn/module.h>
#include <torch/optim/adam.h>
#include <torch/optim/optimizer.h>
#include <torch/optim/serialize.h>
#include <utility>
#include <vector>

namespace torch
{
namespace serialize
{
class OutputArchive;
class InputArchive;
}  // namespace serialize
}  // namespace torch

namespace torch
{
namespace optim
{

struct MyAdamOptions : public OptimizerCloneableOptions<MyAdamOptions>
{
    MyAdamOptions(double lr = 1e-3) { set_lr(lr); };
    TORCH_ARG(double, lr) = 1e-3;
    typedef std::tuple<double, double> betas_t;
    TORCH_ARG(betas_t, betas)                    = std::make_tuple(0.9, 0.999);
    TORCH_ARG(double, eps)                       = 1e-8;
    TORCH_ARG(double, relative_weight_decay)     = 0;
    TORCH_ARG(double, absolute_weight_decay)     = 0;
    TORCH_ARG(double, weight_clipping_magnitude) = 0;
    TORCH_ARG(bool, adabound)                    = false;

   public:
    // void serialize(torch::serialize::InputArchive& archive) override;
    // void serialize(torch::serialize::OutputArchive& archive) const override;
    // friend bool operator==(const MyAdamOptions& lhs, const MyAdamOptions& rhs);
    ~MyAdamOptions() override = default;
    double get_lr() const { return lr(); }
    void set_lr(const double lr) { this->lr(lr); }
};

class MyAdam : public Optimizer
{
   public:
    explicit MyAdam(std::vector<OptimizerParamGroup> param_groups, MyAdamOptions defaults = {})
        : Optimizer(std::move(param_groups), std::make_unique<MyAdamOptions>(defaults))
    {
        TORCH_CHECK(defaults.lr() >= 0, "Invalid learning rate: ", defaults.lr());
        TORCH_CHECK(defaults.eps() >= 0, "Invalid epsilon value: ", defaults.eps());
        auto betas = defaults.betas();
        TORCH_CHECK(0 <= std::get<0>(betas) && std::get<0>(betas) < 1.0,
                    "Invalid beta parameter at index 0: ", std::get<0>(betas));
        TORCH_CHECK(0 <= std::get<1>(betas) && std::get<1>(betas) < 1.0,
                    "Invalid beta parameter at index 1: ", std::get<1>(betas));
        TORCH_CHECK(defaults.relative_weight_decay() >= 0,
                    "Invalid relative_weight_decay value: ", defaults.relative_weight_decay());
        TORCH_CHECK(defaults.absolute_weight_decay() >= 0,
                    "Invalid absolute_weight_decay value: ", defaults.absolute_weight_decay());
    }
    explicit MyAdam(std::vector<Tensor> params, MyAdamOptions defaults = {})
        : MyAdam({std::move(OptimizerParamGroup(std::move(params)))}, defaults)
    {
    }

    torch::Tensor step(LossClosure closure = nullptr) override;
    void save(serialize::OutputArchive& archive) const override;
    void load(serialize::InputArchive& archive) override;
    void shrinkInternalState(int param_group_index, torch::Tensor indices_to_keep);
    void appendToInternalState(int param_group_index, int new_size);

   private:
    // template <typename Self, typename Archive>
    // static void serialize(Self& self, Archive& archive)
    //{
    //     _TORCH_OPTIM_SERIALIZE_WITH_TEMPLATE_ARG(Adam);
    // }
};
}  // namespace optim
}  // namespace torch

namespace Saiga
{
inline void UpdateOptimLR(torch::optim::Optimizer* optimizer, double factor)
{
    for (auto& pg : optimizer->param_groups())
    {
        auto opt_my_adam = dynamic_cast<torch::optim::MyAdamOptions*>(&pg.options());
        if (opt_my_adam)
        {
            opt_my_adam->lr() = opt_my_adam->lr() * factor;
        }
        auto opt_adam = dynamic_cast<torch::optim::AdamOptions*>(&pg.options());
        if (opt_adam)
        {
            opt_adam->lr() = opt_adam->lr() * factor;
        }

        auto opt_sgd = dynamic_cast<torch::optim::SGDOptions*>(&pg.options());
        if (opt_sgd)
        {
            opt_sgd->lr() = opt_sgd->lr() * factor;
        }

        auto opt_rms = dynamic_cast<torch::optim::RMSpropOptions*>(&pg.options());
        if (opt_rms)
        {
            opt_rms->lr() = opt_rms->lr() * factor;
        }
    }
}
}  // namespace Saiga