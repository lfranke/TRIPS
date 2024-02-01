/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#pragma once
#include "saiga/core/image/image.h"
#include "saiga/core/math/math.h"
#include "saiga/core/util/assert.h"
#include "saiga/vision/torch/TorchHelper.h"

#include <torch/torch.h>
class FCBlockImpl : public torch::nn::Module
{
   public:
    FCBlockImpl(int in_features, int out_features, int num_hidden_layers, float hidden_features,
                std::string last_activation = "id", std::string non_linearity = "relu", bool use_bias = true)
        : in_features(in_features), out_features(out_features)
    {
        auto make_lin = [&](int in, int out)
        {
            auto lin = torch::nn::Linear(torch::nn::LinearOptions(in, out).bias(use_bias));
            torch::nn::init::kaiming_normal_(lin->weight, 0, torch::kFanIn, torch::kReLU);
            std::cout << "(lin " << in << "->" << out << ") ";
            return lin;
        };

        seq->push_back(make_lin(in_features, hidden_features));
        seq->push_back(Saiga::ActivationFromString(non_linearity));
        std::cout << "(" << non_linearity << ") ";

        for (int i = 0; i < num_hidden_layers; ++i)
        {
            seq->push_back(make_lin(hidden_features, hidden_features));
            seq->push_back(Saiga::ActivationFromString(non_linearity));
            std::cout << "(" << non_linearity << ") ";
        }

        seq->push_back(make_lin(hidden_features, out_features));
        seq->push_back(Saiga::ActivationFromString(last_activation));
        std::cout << "(" << last_activation << ") ";

        register_module("seq", seq);

        int num_params = 0;
        for (auto& t : this->parameters())
        {
            num_params += t.numel();
        }
        std::cout << "  |  #Params " << num_params;
        std::cout << std::endl;
    }

    at::Tensor forward(at::Tensor x)
    {
        CHECK_EQ(in_features, x.size(-1)) << Saiga::TensorInfo(x);
        x = seq->forward(x);
        CHECK_EQ(out_features, x.size(-1)) << Saiga::TensorInfo(x);
        return x;
    }

    int in_features, out_features;
    torch::nn::Sequential seq;
};

TORCH_MODULE(FCBlock);
