/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#pragma once
#include "saiga/core/util/ini/Params.h"
#include "saiga/core/util/ini/ini.h"
#include "saiga/vision/torch/ImageTensor.h"
#include "saiga/vision/torch/PartialConv.h"
#include "saiga/vision/torch/TorchHelper.h"

#include "models/Networks.h"
#include "models/mlp.h"

#ifdef DYN_REF

#    include "tiny-cuda-nn/torch_wrapper.h"

#endif
namespace Saiga
{


class DynamicRefinementMLPImpl : public torch::nn::Module
{
   public:
    DynamicRefinementMLPImpl()
    {
#ifdef DYN_REF

        nlohmann::json config = {
            {"encoding",
             {
                 {"otype", "Frequency"},
                 {"n_frequencies", 8},
             }},
        };
        int n_input_dims = 4;
        // int n_output_dims            = 3;
        nlohmann::json encoding_opts = config.value("encoding", nlohmann::json::object());
        std::cout << encoding_opts << std::endl;

        auto precision                  = tcnn::cpp::EPrecision::Fp32;
        tcnn::cpp::Module* enc_module   = tcnn::cpp::create_encoding(n_input_dims, encoding_opts, precision);
        TcnnTorchModule encoding_module = nullptr;
        encoding_module                 = TcnnTorchModule(TorchTcnnWrapperModule(enc_module));
        encoding->push_back(encoding_module);
#endif
        FCBlock mlp(68, 3, 3, 256);
        model->push_back(mlp);

        register_module("model", model);
        register_module("encoding", encoding);
    }

    torch::Tensor forward(torch::Tensor inputs)
    {
        int size_org = inputs.size(0);
#ifdef DYN_REF
        int to_pad  = tcnn::cpp::batch_size_granularity() - (size_org % tcnn::cpp::batch_size_granularity());
        auto pad_in = torch::nn::ConstantPad1d(torch::nn::ConstantPad1dOptions({0, to_pad}, 0));
        inputs      = pad_in(inputs.transpose(1, 0)).transpose(1, 0).contiguous();
#endif
        auto x_enc = encoding->forward(inputs);
        x_enc      = torch::cat({inputs, x_enc}, 1);
        auto x     = model->forward(x_enc);

        return x.slice(0, 0, size_org);
    }

    torch::nn::Sequential encoding;
    torch::nn::Sequential model;
};

TORCH_MODULE(DynamicRefinementMLP);

class RefinementNetImpl : public torch::nn::Module
{
   public:
    RefinementNetImpl(MultiScaleUnet2dParams params) : params(params)
    {
        std::cout << "USING RefinementNet ! " << std::endl;
        std::vector<int> hidden_states = {16, 16, 16, 16, 16};
        std::string activation         = "relu";
        int additional_keys            = 4;  // 3 camdir, 1 layer

        model->push_back(torch::nn::Conv2d(
            torch::nn::Conv2dOptions(params.num_input_channels + additional_keys, hidden_states[0], 1).bias(false)));
        model->push_back(ActivationFromString(activation));
        for (int i = 0; i < hidden_states.size() - 1; ++i)
        {
            model->push_back(
                torch::nn::Conv2d(torch::nn::Conv2dOptions(hidden_states[i], hidden_states[i + 1], 1).bias(false)));
            model->push_back(ActivationFromString(activation));
        }
        model->push_back(torch::nn::Conv2d(
            torch::nn::Conv2dOptions(hidden_states.back(), params.num_input_channels, 1).bias(false)));
        model->push_back(ActivationFromString(activation));

        register_module("model", model);


        if (params.half_float)
        {
            this->to(torch::kFloat16);
        }
    }

    torch::Tensor forward(torch::Tensor inputs) { return (model->forward(inputs)); }

    MultiScaleUnet2dParams params;

    torch::nn::Sequential model;
};

TORCH_MODULE(RefinementNet);
}  // namespace Saiga
