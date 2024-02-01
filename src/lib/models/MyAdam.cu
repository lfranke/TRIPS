/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "saiga/core/util/assert.h"
#include "saiga/cuda/cudaHelper.h"
#include "saiga/vision/torch/CudaHelper.h"
#include "saiga/vision/torch/TorchHelper.h"

#include "MyAdam.h"

#include <ATen/ATen.h>
#include <c10/util/irange.h>
#include <cmath>
#include <functional>
#include <torch/csrc/autograd/variable.h>
#include <torch/nn/module.h>
#include <torch/serialize/archive.h>
#include <torch/utils.h>

using namespace Saiga;

/*
 * adapted from https://github.com/NVlabs/tiny-cuda-nn/blob/master/include/tiny-cuda-nn/optimizers/adam.h
 */

__device__ inline float weight_decay(float relative_weight_decay, float absolute_weight_decay, float weight)
{
    // Relative weight decay is closely related to l2 regularization, whereas absolute weight decay corresponds to l1
    // regularization
    // copysignf: If no errors occur, the floating point value with the magnitude of x and the sign of y is returned.
    return (1 - relative_weight_decay) * weight - copysignf(absolute_weight_decay, weight);
}
template <typename T>
__device__ T clamp(T val, T lower, T upper)
{
    return val < lower ? lower : (upper < val ? upper : val);
}

template <typename T, typename Tfull>
__global__ void adam_step(const uint32_t n_elements, const float relative_weight_decay,
                          const float absolute_weight_decay, const float weight_clipping_magnitude,
                          const float loss_scale, float learning_rate, const float beta1, const float beta2,
                          const float epsilon, const float lower_lr_bound, const float upper_lr_bound,
                          StaticDeviceTensor<Tfull, 1> weights_full_precision,  // StaticDeviceTensor<T, 1> weights,
                          StaticDeviceTensor<T, 1> gradients, StaticDeviceTensor<float, 1> first_moments,
                          StaticDeviceTensor<float, 1> second_moments, StaticDeviceTensor<int64_t, 1> param_steps)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements) return;

    float gradient        = (float)gradients(i) / loss_scale;
    const float weight_fp = weights_full_precision(i);

    // gradient zero = parameter not seen in this optimization process
    if (gradient == 0.f)
    {
        // weights(i) = (T)weight_fp;
        return;
    }

    const float gradient_sq = gradient * gradient;

    float first_moment = first_moments(i) = beta1 * first_moments(i) + (1 - beta1) * gradient;
    const float second_moment = second_moments(i) = beta2 * second_moments(i) + (1 - beta2) * gradient_sq;

    // Debiasing. Since some parameters might see fewer steps than others, they each need their own step counter.
    param_steps(i)              = param_steps(i) + 1;
    const uint32_t current_step = param_steps(i);
    learning_rate *= sqrtf(1 - powf(beta2, (float)current_step)) / (1 - powf(beta1, (float)current_step));

    // Follow AdaBound paradigm
    const float effective_learning_rate =
        fminf(fmaxf(learning_rate / (sqrtf(second_moment) + epsilon), lower_lr_bound), upper_lr_bound);

    const float decayed_weight =
        weight_decay(relative_weight_decay * learning_rate, absolute_weight_decay * learning_rate, weight_fp);
    float new_weight = decayed_weight - effective_learning_rate * first_moment;

    if (weight_clipping_magnitude != 0.0f)
    {
        new_weight = clamp(new_weight, -weight_clipping_magnitude, weight_clipping_magnitude);
    }

    weights_full_precision(i) = new_weight;
    //   weights(i)                = (T)new_weight;
}

template <typename T, typename Tfull>
__global__ void adam_step_old(const uint32_t n_elements, const float relative_weight_decay,
                              const float absolute_weight_decay, const float weight_clipping_magnitude,
                              const float loss_scale, float learning_rate, const float beta1, const float beta2,
                              const float epsilon, const float lower_lr_bound, const float upper_lr_bound,
                              StaticDeviceTensor<Tfull, 1> weights_full_precision, StaticDeviceTensor<T, 1> weights,
                              StaticDeviceTensor<T, 1> gradients, StaticDeviceTensor<float, 1> first_moments,
                              StaticDeviceTensor<float, 1> second_moments, StaticDeviceTensor<int64_t, 1> param_steps,
                              int64_t step)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements) return;

    float gradient          = (float)gradients(i) / loss_scale;
    const float gradient_sq = gradient * gradient;

    // exp_avg
    float first_moment = first_moments(i) = beta1 * first_moments(i) + (1 - beta1) * gradient;
    // exp_avg_sq
    const float second_moment = second_moments(i) = beta2 * second_moments(i) + (1 - beta2) * gradient_sq;

    auto bias_correction1 = 1.f - pow(beta1, float(step));
    auto bias_correction2 = 1.f - pow(beta2, float(step));


    float denom     = sqrt(second_moment) / sqrt(bias_correction2) + epsilon;
    float step_size = learning_rate / bias_correction1;
    float w         = (first_moment / denom) * -step_size;

    const float weight_fp = weights_full_precision(i);

    float new_weight = weight_fp + w;

    weights_full_precision(i) = new_weight;
    weights(i)                = (T)new_weight;
}

template <typename T>
T div_round_up(T val, T divisor)
{
    return (val + divisor - 1) / divisor;
}
constexpr uint32_t n_threads_linear = 128;

template <typename T>
constexpr uint32_t n_blocks_linear(T n_elements)
{
    return (uint32_t)div_round_up(n_elements, (T)n_threads_linear);
}

namespace torch
{
namespace optim
{


struct MyAdamParamState : public OptimizerCloneableParamState<MyAdamParamState>
{
    // TORCH_ARG(int64_t, step) = 0;
    TORCH_ARG(torch::Tensor, param_steps);
    TORCH_ARG(torch::Tensor, first_moments);
    TORCH_ARG(torch::Tensor, second_moments);
    // TORCH_ARG(torch::Tensor, max_exp_avg_sq) = {};
    TORCH_ARG(int64_t, current_step) = 0;



   public:
    // void serialize(torch::serialize::InputArchive& archive) override;
    // void serialize(torch::serialize::OutputArchive& archive) const override;
    // friend bool operator==(const AdamParamState& lhs, const AdamParamState& rhs);
    ~MyAdamParamState() override = default;
};

torch::Tensor MyAdam::step(LossClosure closure)
{
    NoGradGuard no_grad;
    Tensor loss = {};
    if (closure != nullptr)
    {
        at::AutoGradMode enable_grad(true);
        loss = closure();
    }
    for (auto& group : param_groups_)
    {
        for (auto& p : group.params())
        {
            if (!p.grad().defined())
            {
                continue;
            }
            auto grad = p.grad();
            TORCH_CHECK(!grad.is_sparse(),
                        "Adam does not support sparse gradients" /*, please consider SparseAdam instead*/);
            auto param_state = state_.find(c10::guts::to_string(p.unsafeGetTensorImpl()));
            auto& options    = static_cast<MyAdamOptions&>(group.options());

            // State initialization
            if (param_state == state_.end())
            {
                auto state = std::make_unique<MyAdamParamState>();
                state->param_steps(torch::zeros_like(p, MemoryFormat::Preserve).to(torch::kInt64));
                // Exponential moving average of gradient values
                state->first_moments(torch::zeros_like(p, MemoryFormat::Preserve).to(torch::kFloat32));
                // Exponential moving average of squared gradient values
                state->second_moments(torch::zeros_like(p, MemoryFormat::Preserve).to(torch::kFloat32));
                state->current_step(0);
                // if (options.amsgrad())
                //{
                //     // Maintains max of all exp. moving avg. of sq. grad. values
                //     state->max_exp_avg_sq(torch::zeros_like(p, MemoryFormat::Preserve));
                // }
                state_[c10::guts::to_string(p.unsafeGetTensorImpl())] = std::move(state);
            }

            auto& state = static_cast<MyAdamParamState&>(*state_[c10::guts::to_string(p.unsafeGetTensorImpl())]);
            state.current_step(state.current_step() + 1);

            auto& first_moments  = state.first_moments();
            auto& second_moments = state.second_moments();
            auto& param_steps    = state.param_steps();

            auto beta1 = std::get<0>(options.betas());
            auto beta2 = std::get<1>(options.betas());



            float lower_lr_bound = 0;
            float upper_lr_bound = std::numeric_limits<float>::max();
            // AdaBound paper: https://openreview.net/pdf?id=Bkg3g2R9FX
            if (options.adabound())
            {
                lower_lr_bound = 0.1f - 0.1f / ((1 - beta2) * (float)state.current_step() + 1);
                upper_lr_bound = 0.1f + 0.1f / ((1 - beta2) * (float)state.current_step());
            }

            int n_weights_to_optimize = p.numel();

            constexpr uint32_t n_threads_linear = 128;
            const uint32_t n_blocks             = n_blocks_linear(n_weights_to_optimize);

            float loss_scale = 1.f;

// #define OLD_ADAM
#ifdef OLD_ADAM


            // clang-format off
            if (p.scalar_type() == torch::kDouble)
            {
                adam_step_old<double, double><<<n_blocks, n_threads_linear>>>(
                    n_weights_to_optimize, options.relative_weight_decay(), options.absolute_weight_decay(),
                    options.weight_clipping_magnitude(), loss_scale, options.lr(), beta1, beta2, options.eps(),
                    lower_lr_bound, upper_lr_bound, p.view({-1}), p.view({-1}), grad.view({-1}),
                    first_moments.view({-1}), second_moments.view({-1}), param_steps.view({-1}), state.current_step());
            }
            else if (p.scalar_type() == torch::kFloat)
            {
                adam_step_old<float, float><<<n_blocks, n_threads_linear>>>(
                    n_weights_to_optimize, options.relative_weight_decay(), options.absolute_weight_decay(),
                    options.weight_clipping_magnitude(), loss_scale, options.lr(), beta1, beta2, options.eps(),
                    lower_lr_bound, upper_lr_bound, p.view({-1}), p.view({-1}), grad.view({-1}),
                    first_moments.view({-1}), second_moments.view({-1}), param_steps.view({-1}), state.current_step());
            }
            else
            {
                SAIGA_ASSERT(false);

            }
            // clang-format on
#else
            // clang-format off

            if (p.scalar_type() == torch::kDouble)
            {
                adam_step<double, double><<<n_blocks, n_threads_linear>>>(
                    n_weights_to_optimize, options.relative_weight_decay(), options.absolute_weight_decay(),
                    options.weight_clipping_magnitude(), loss_scale, options.lr(), beta1, beta2, options.eps(),
                    lower_lr_bound, upper_lr_bound, p.view({-1}), //p.view({-1}),
                    grad.view({-1}),
                    first_moments.view({-1}), second_moments.view({-1}), param_steps.view({-1}));
            }
            else if (p.scalar_type() == torch::kFloat)
            {
                adam_step<float, float><<<n_blocks, n_threads_linear>>>(
                    n_weights_to_optimize, options.relative_weight_decay(), options.absolute_weight_decay(),
                    options.weight_clipping_magnitude(), loss_scale, options.lr(), beta1, beta2, options.eps(),
                    lower_lr_bound, upper_lr_bound, p.view({-1}), //p.view({-1}),
                    grad.view({-1}),
                    first_moments.view({-1}), second_moments.view({-1}), param_steps.view({-1}));
            }
            else
            {
                SAIGA_ASSERT(false);
            }
            // clang-format on

#endif
            CUDA_SYNC_CHECK_ERROR();


            /*   (


                   n_weights_to_optimize, m_n_weights_covered_by_matrices, m_relative_weight_decay,
                   m_absolute_weight_decay, m_weight_clipping_magnitude, loss_scale, m_base_learning_rate,
                   m_non_matrix_learning_rate_factor, m_optimize_matrix_params, m_optimize_non_matrix_params, m_beta1,
                   m_beta2, m_epsilon, lower_lr_bound, upper_lr_bound, m_l2_reg, weights_full_precision, weights,
                   gradients, m_first_moments.data(), m_second_moments.data(), m_param_steps.data());*/

            // linear_kernel(adam_step<T>, 0, stream, n_weights_to_optimize, m_n_weights_covered_by_matrices,
            //               m_relative_weight_decay, m_absolute_weight_decay, m_weight_clipping_magnitude, loss_scale,
            //               m_base_learning_rate, m_non_matrix_learning_rate_factor, m_optimize_matrix_params,
            //               m_optimize_non_matrix_params, m_beta1, m_beta2, m_epsilon, lower_lr_bound, upper_lr_bound,
            //               m_l2_reg, weights_full_precision, weights, gradients, m_first_moments.data(),
            //               m_second_moments.data(), m_param_steps.data());
            /*
                        state.step(state.step() + 1);
                        auto beta1 = std::get<0>(options.betas());
                        auto beta2 = std::get<1>(options.betas());

                        auto bias_correction1 = 1 - std::pow(beta1, state.step());
                        auto bias_correction2 = 1 - std::pow(beta2, state.step());

                        if (options.weight_decay() != 0)
                        {
                            grad = grad.add(p, options.weight_decay());
                        }

                        // std::cout << Saiga::TensorInfo(grad) << " " << Saiga::TensorInfo(exp_avg) << " " << beta1 <<
               " "
                        //           << 1 - beta1 << std::endl;
                        //  Decay the first and second moment running average coefficient
                        exp_avg.mul_(beta1).add_(grad, 1 - beta1);
                        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, 1 - beta2);

                        Tensor denom;
                        if (options.amsgrad())
                        {
                            // Maintains the maximum of all 2nd moment running avg. till now
                            torch::max_out(max_exp_avg_sq, exp_avg_sq, max_exp_avg_sq);
                            // Use the max. for normalizing running avg. of gradient
                            denom = (max_exp_avg_sq.sqrt() / sqrt(bias_correction2)).add_(options.eps());
                        }
                        else
                        {
                            denom = (exp_avg_sq.sqrt() / sqrt(bias_correction2)).add_(options.eps());
                        }

                        auto step_size = options.lr() / bias_correction1;
                        p.addcdiv_(exp_avg, denom, -step_size);
                        */
        }
    }
    return loss;
}
void MyAdam::shrinkInternalState(int param_group_index, torch::Tensor indices_to_keep)
{
    SAIGA_ASSERT(param_group_index < param_groups_.size());
    auto& group = param_groups_[param_group_index];

    {
        for (auto& p : group.params())
        {
            auto param_state = state_.find(c10::guts::to_string(p.unsafeGetTensorImpl()));
            // not created yet -> will we initialized later
            if (param_state == state_.end()) continue;
            auto& state = static_cast<MyAdamParamState&>(*state_[c10::guts::to_string(p.unsafeGetTensorImpl())]);
            auto& first_moments  = state.first_moments();
            auto& second_moments = state.second_moments();
            auto& param_steps    = state.param_steps();

            auto remove_selected = [&](torch::Tensor& t)
            {
                auto values_keep = t.index_select(t.sizes().size() - 1, indices_to_keep.squeeze().to(t.device()));
                t.set_(values_keep);
            };

            if (first_moments.defined()) remove_selected(first_moments);
            if (second_moments.defined()) remove_selected(second_moments);
            if (param_steps.defined()) remove_selected(param_steps);
        }
    }
}


void MyAdam::appendToInternalState(int param_group_index, int new_size)
{
    SAIGA_ASSERT(param_group_index < param_groups_.size());
    auto& group = param_groups_[param_group_index];
    {
        for (auto& p : group.params())
        {
            auto param_state = state_.find(c10::guts::to_string(p.unsafeGetTensorImpl()));
            // not created yet -> will we initialized later
            if (param_state == state_.end()) continue;
            auto& state = static_cast<MyAdamParamState&>(*state_[c10::guts::to_string(p.unsafeGetTensorImpl())]);
            auto& first_moments  = state.first_moments();
            auto& second_moments = state.second_moments();
            auto& param_steps    = state.param_steps();

            auto add_selected = [&](torch::Tensor& t)
            {
                torch::Tensor new_vals;
                int new_point_size = new_size - t.size(-1);
#ifdef __GNUC__
                std::vector<long> sizes_tensor(t.sizes().size());
                for (int i = 0; i < t.sizes().size(); ++i) sizes_tensor[i] = t.sizes()[i];
                sizes_tensor[sizes_tensor.size() - 1] = new_point_size;

                new_vals = torch::zeros(sizes_tensor, t.options());
#else
                std::vector<int64_t> sizes_tensor(t.sizes().size());
                for (int i = 0; i < t.sizes().size(); ++i) sizes_tensor[i] = t.sizes()[i];
                sizes_tensor[sizes_tensor.size() - 1] = new_point_size;
                new_vals                              = torch::zeros(sizes_tensor, t.options());
#endif
                auto t_n = torch::cat({t.clone(), new_vals}, -1);
                t.set_(t_n);
            };
            if (first_moments.defined()) add_selected(first_moments);
            if (second_moments.defined()) add_selected(second_moments);
            if (param_steps.defined()) add_selected(param_steps);
        }
    }
}

void MyAdam::save(serialize::OutputArchive& archive) const
{
    SAIGA_ASSERT(false);
}

void MyAdam::load(serialize::InputArchive& archive)
{
    SAIGA_ASSERT(false);
}
}  // namespace optim
}  // namespace torch