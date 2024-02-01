/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#pragma once

#include "saiga/vision/torch/PartialConvUnet2d.h"

#include "spherical_harmonics.h"

extern CUDA::CudaTimerSystem* timer_for_nets;
/// NEW

class MultiStartBlockImpl : public UnetBlockImpl
{
   public:
    using UnetBlockImpl::forward;

    MultiStartBlockImpl(int in_channels, int out_channels, std::string conv_block, std::string norm_str,
                        std::string pooling_str, std::string activation_str)
    {
        SAIGA_ASSERT(in_channels > 0);
        SAIGA_ASSERT(out_channels > 0);
        conv = UnetBlockFromString(conv_block, in_channels, out_channels, 3, 1, 1, norm_str, activation_str);

        register_module("conv", conv.ptr());
    }

    std::pair<at::Tensor, at::Tensor> forward(at::Tensor x, at::Tensor mask = {}) override
    {
        return conv.forward<std::pair<at::Tensor, at::Tensor>>(x, mask);
    }

    torch::nn::AnyModule conv;
};

TORCH_MODULE(MultiStartBlock);



class MultiScaleUnet2dSlimImpl : public torch::nn::Module
{
   public:
    MultiScaleUnet2dSlimImpl(MultiScaleUnet2dParams params) : params(params)
    {
        std::cout << "Using MultiScaleUnet2dSlim " << std::endl;
        std::vector<int> num_input_channels_per_layer;
        // std::vector<int> filters = {4, 8, 16, 16, 16};
        std::vector<int> filters = params.filters_network;

        std::vector<int> num_input_channels(params.num_input_layers, params.num_input_channels);
        for (int i = params.num_input_layers; i < 5; ++i)
        {
            num_input_channels.push_back(0);
        }
        for (int i = 0; i < 5; ++i)
        {
            auto& f = filters[i];
            f       = f * params.feature_factor;
            if (params.add_input_to_filters && i >= 1)
            {
                f += num_input_channels[i];
            }

            if (i >= 1)
            {
                SAIGA_ASSERT(f >= num_input_channels[0]);
            }
        }


        SAIGA_ASSERT(num_input_channels.size() == filters.size());

        //  start = UnetBlockFromString(params.conv_block, num_input_channels[0], filters[0], 3, 1, 1, "id");
        //  register_module("start", start.ptr());



        final->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(filters[0], params.num_output_channels, 1)));
        final->push_back(ActivationFromString(params.last_act));
        register_module("final", final);

        for (int i = 0; i < params.num_layers; ++i)
        {
            // Down[i] transforms from layer (i) -> (i+1)
            int multistart_in  = params.num_input_channels;  // filters[i];
            int multistart_out = filters[i];
            multistart[i] = MultiStartBlock(multistart_in, multistart_out, params.conv_block, params.norm_layer_down,
                                            params.pooling, params.activation);
            register_module("multistart" + std::to_string(i + 1), multistart[i]);
        }
        for (int i = 0; i < params.num_layers - 1; ++i)
        // for (int i = params.num_layers - 1; i >= 1; --i)
        {
            // Up[i] transforms from layer (i+1) -> (i)
            int up_in  = filters[i + 1];
            int up_out = filters[i];
            up[i]      = UpsampleBlock(up_in, up_out, params.conv_block_up, params.upsample_mode, params.norm_layer_up,
                                       params.activation);
            register_module("up" + std::to_string(i + 1), up[i]);
        }
        // multistart[params.num_layers - 1] = MultiStartBlock(params.num_input_channels, filters[params.num_layers -
        // 1], params.conv_block,
        //     params.norm_layer_down, params.pooling, params.activation);

        multi_channel_masks = params.conv_block == "partial_multi";
        need_up_masks       = params.conv_block_up == "partial_multi";
        if (params.half_float)
        {
            this->to(torch::kFloat16);
        }
    }

    at::Tensor forward(ArrayView<torch::Tensor> inputs)
    {
        std::vector<torch::Tensor> masks(inputs.size());
        return forward(inputs, masks);
    }

    at::Tensor forward(ArrayView<torch::Tensor> inputs, ArrayView<torch::Tensor> masks)
    {
        SAIGA_ASSERT(inputs.size() == params.num_input_layers);
        SAIGA_ASSERT(masks.size() == params.num_input_layers);
        // The downsampling should not happen on uneven image sizes!
        // SAIGA_ASSERT(inputs.front().size(2) % (1 << params.num_layers) == 0);
        // SAIGA_ASSERT(inputs.front().size(3) % (1 << params.num_layers) == 0);
        // debug check if input has correct format
        for (int i = 0; i < inputs.size(); ++i)
        {
            if (params.num_input_layers > i)
            {
                SAIGA_ASSERT(inputs.size() > i);
                SAIGA_ASSERT(inputs[i].defined());
                SAIGA_ASSERT(params.num_input_channels == inputs[i].size(1));
            }
            SAIGA_ASSERT(masks[i].requires_grad() == false);
        }

        if (multi_channel_masks)
        {
            torch::NoGradGuard ngg;
            // multi channel partial convolution needs a mask value for each channel.
            // Here, we just repeat the masks along the channel dimension.
            for (int i = 0; i < inputs.size(); ++i)
            {
                auto& ma = masks[i];
                auto& in = inputs[i];
                if (ma.size(1) == 1 && in.size(1) > 1)
                {
                    ma = ma.repeat({1, in.size(1), 1, 1});
                }
            }
        }

        std::pair<torch::Tensor, torch::Tensor> d[MultiScaleUnet2dParams::max_layers - 1];

        //!
        //        d[0] = multistart[0].forward<std::pair<torch::Tensor, torch::Tensor>>(inputs[0], masks[0]);

        // Loops Range: [1,2, ... , layers-1]
        // At 5 layers we have only 4 stages
        for (int i = 0; i < params.num_layers; ++i)
        {
            d[i] = multistart[i]->forward(inputs[i]);
        }

        if (!need_up_masks)
        {
            for (int i = 0; i < params.num_layers; ++i)
            {
                d[i].second = torch::Tensor();
            }
        }

        // Loops Range: [layers-1, ... , 2, 1]
        for (int i = params.num_layers - 1; i >= 1; --i)
        {
            d[i - 1] = up[i - 1]->forward(d[i], d[i - 1]);
        }
        return final->forward(d[0].first);
    }

    MultiScaleUnet2dParams params;
    bool multi_channel_masks = false;
    bool need_up_masks       = false;

    //  torch::nn::AnyModule start;
    torch::nn::Sequential final;

    MultiStartBlock multistart[MultiScaleUnet2dParams::max_layers - 1] = {nullptr, nullptr, nullptr, nullptr};
    UpsampleBlock up[MultiScaleUnet2dParams::max_layers - 1]           = {nullptr, nullptr, nullptr, nullptr};
};

TORCH_MODULE(MultiScaleUnet2dSlim);



class UpsampleUltraBlockImpl : public torch::nn::Module
{
   public:
    UpsampleUltraBlockImpl(int in_channels, int out_channels, int num_input_channels, std::string conv_block,
                           std::string upsample_mode = "deconv", std::string norm_str = "id",
                           std::string activation = "id")
    {
        SAIGA_ASSERT(in_channels > 0);
        SAIGA_ASSERT(out_channels > 0);

        std::vector<double> scale = {2.0, 2.0};

        // conv = GatedBlock(in_channels, out_channels);
        if (upsample_mode == "deconv")
        {
            up->push_back(torch::nn::ConvTranspose2d(
                torch::nn::ConvTranspose2dOptions(in_channels, out_channels, 4).stride(2).padding(1)));
        }
        else if (upsample_mode == "bilinear")
        {
            up->push_back(torch::nn::Upsample(
                torch::nn::UpsampleOptions().scale_factor(scale).mode(torch::kBilinear).align_corners(false)));
        }
        else if (upsample_mode == "nearest")
        {
            up->push_back(torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(scale).mode(torch::kNearest)));
            // up->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 3).padding(1)));
        }

        if (upsample_mode != "deconv")
        {
            if (conv_block == "partial_multi")
            {
                conv1 = torch::nn::AnyModule(
                    PartialConv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 3).padding(1), true));
            }
            else
            {
                conv1 = torch::nn::AnyModule(GatedBlock(in_channels, out_channels, 3, 1, 1, "id", "id"));
            }
        }
        // conv = GatedBlock(out_channels * 2, out_channels, 3, 1, 1, norm_str);
        conv2 = UnetBlockFromString(conv_block, out_channels + num_input_channels, out_channels, 3, 1, 1, norm_str,
                                    activation);


        up_mask = torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(scale).mode(torch::kNearest));

        register_module("up", up);
        register_module("up_mask", up_mask);

        if (!conv1.is_empty())
        {
            register_module("conv1", conv1.ptr());
        }
        register_module("conv2", conv2.ptr());
    }

    // Combines the upsampled tensor (below) with the skip connection (skip)
    // Usually this can be done with a simple cat however if the size does not match we crop
    torch::Tensor CombineBridge(torch::Tensor below, torch::Tensor skip)
    {
        // SAIGA_ASSERT(skip.first.size(2) == same_layer_as_skip.first.size(2) &&
        //              skip.first.size(3) == same_layer_as_skip.first.size(3));
        if (below.size(2) == skip.size(2) && below.size(3) == skip.size(3))
        {
            return torch::cat({below, skip}, 1);
        }
        else
        {
            return torch::cat(
                {below, CenterCrop2D(skip, std::vector<int64_t>(below.sizes().begin(), below.sizes().end()))}, 1);
        }
    }

    std::pair<at::Tensor, at::Tensor> forward(std::pair<at::Tensor, at::Tensor> layer_below,
                                              std::pair<at::Tensor, at::Tensor> skip)
    {
        SAIGA_ASSERT(layer_below.first.defined());
        SAIGA_ASSERT(skip.first.defined());

        // Upsample the layer from below
        std::pair<at::Tensor, at::Tensor> same_layer_as_skip;
        same_layer_as_skip.first = up->forward(layer_below.first);

        if (layer_below.second.defined())
        {
            same_layer_as_skip.second = up_mask->forward(layer_below.second);
            // SAIGA_ASSERT(skip.second.size(2) == same_layer_as_skip.second.size(2) &&
            //              skip.second.size(3) == same_layer_as_skip.second.size(3));
        }

        if (!conv1.is_empty())
        {
            same_layer_as_skip =
                conv1.forward<std::pair<at::Tensor, at::Tensor>>(same_layer_as_skip.first, same_layer_as_skip.second);
        }


        std::pair<at::Tensor, at::Tensor> output;
        // [b, c, h, w]
        // output.first = torch::cat({same_layer_as_skip.first, skip.first}, 1);
        output.first = CombineBridge(same_layer_as_skip.first, skip.first);

        if (layer_below.second.defined())
        {
            // output.second = torch::cat({same_layer_as_skip.second, skip.second}, 1);
            output.second = CombineBridge(same_layer_as_skip.second, skip.second);
        }

        return conv2.forward<std::pair<at::Tensor, at::Tensor>>(output.first, output.second);
    }

    torch::nn::Sequential up;
    torch::nn::Upsample up_mask = nullptr;
    torch::nn::AnyModule conv1;
    torch::nn::AnyModule conv2;
};

TORCH_MODULE(UpsampleUltraBlock);



class MultiScaleUnet2dUltraSlimImpl : public torch::nn::Module
{
   public:
    MultiScaleUnet2dUltraSlimImpl(MultiScaleUnet2dParams params) : params(params)
    {
        std::cout << "Using MultiScaleUnet2dUltraSlim " << std::endl;
        std::vector<int> num_input_channels_per_layer;
        // std::vector<int> filters = {4, 8, 16, 16, 16};
        std::vector<int> filters = params.filters_network;

        std::vector<int> num_input_channels(params.num_input_layers, params.num_input_channels);
        for (int i = params.num_input_layers; i < 5; ++i)
        {
            num_input_channels.push_back(0);
        }
        for (int i = 0; i < 5; ++i)
        {
            auto& f = filters[i];
            f       = f * params.feature_factor;
            if (params.add_input_to_filters && i >= 1)
            {
                f += num_input_channels[i];
            }

            if (i >= 1)
            {
                SAIGA_ASSERT(f >= num_input_channels[0]);
            }
        }


        SAIGA_ASSERT(num_input_channels.size() == filters.size());

        //  start = UnetBlockFromString(params.conv_block, num_input_channels[0], filters[0], 3, 1, 1, "id");
        //  register_module("start", start.ptr());



        final->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(filters[0], params.num_output_channels, 1)));
        final->push_back(ActivationFromString(params.last_act));
        register_module("final", final);

        for (int i = params.num_layers - 1; i < params.num_layers; ++i)
        {
            // Down[i] transforms from layer (i) -> (i+1)
            int multistart_in  = params.num_input_channels;  // filters[i];
            int multistart_out = filters[i];
            multistart[i] = MultiStartBlock(multistart_in, multistart_out, params.conv_block, params.norm_layer_down,
                                            params.pooling, params.activation);
            register_module("multistart" + std::to_string(i + 1), multistart[i]);
        }
        //  start      = MultiStartBlock(params.num_input_channels, filters[params.num_layers-1], params.conv_block,
        //  params.norm_layer_down, params.pooling,
        //                             params.activation);
        // register_module("start", start.ptr());

        for (int i = 0; i < params.num_layers - 1; ++i)
        // for (int i = params.num_layers - 1; i >= 1; --i)
        {
            // Up[i] transforms from layer (i+1) -> (i)
            int up_in  = filters[i + 1];
            int up_out = filters[i];
            up[i]      = UpsampleUltraBlock(up_in, up_out, params.num_input_channels, params.conv_block_up,
                                            params.upsample_mode, params.norm_layer_up, params.activation);
            register_module("up" + std::to_string(i + 1), up[i]);
        }
        // multistart[params.num_layers - 1] = MultiStartBlock(params.num_input_channels, filters[params.num_layers -
        // 1], params.conv_block,
        //     params.norm_layer_down, params.pooling, params.activation);

        multi_channel_masks = params.conv_block == "partial_multi";
        need_up_masks       = params.conv_block_up == "partial_multi";
        if (params.half_float)
        {
            this->to(torch::kFloat16);
        }
    }

    at::Tensor forward(ArrayView<torch::Tensor> inputs)
    {
        std::vector<torch::Tensor> masks(inputs.size());
        return forward(inputs, masks);
    }

    at::Tensor forward(ArrayView<torch::Tensor> inputs, ArrayView<torch::Tensor> masks)
    {
        SAIGA_ASSERT(inputs.size() == params.num_input_layers);
        SAIGA_ASSERT(masks.size() == params.num_input_layers);
        // The downsampling should not happen on uneven image sizes!
        // SAIGA_ASSERT(inputs.front().size(2) % (1 << params.num_layers) == 0);
        // SAIGA_ASSERT(inputs.front().size(3) % (1 << params.num_layers) == 0);
        // debug check if input has correct format
        for (int i = 0; i < inputs.size(); ++i)
        {
            if (params.num_input_layers > i)
            {
                SAIGA_ASSERT(inputs.size() > i);
                SAIGA_ASSERT(inputs[i].defined());
                SAIGA_ASSERT(params.num_input_channels == inputs[i].size(1));
            }
            SAIGA_ASSERT(masks[i].requires_grad() == false);
        }

        if (multi_channel_masks)
        {
            torch::NoGradGuard ngg;
            // multi channel partial convolution needs a mask value for each channel.
            // Here, we just repeat the masks along the channel dimension.
            for (int i = 0; i < inputs.size(); ++i)
            {
                auto& ma = masks[i];
                auto& in = inputs[i];
                if (ma.size(1) == 1 && in.size(1) > 1)
                {
                    ma = ma.repeat({1, in.size(1), 1, 1});
                }
            }
        }

        std::pair<torch::Tensor, torch::Tensor> d[MultiScaleUnet2dParams::max_layers - 1];

        //!
        //        d[0] = multistart[0].forward<std::pair<torch::Tensor, torch::Tensor>>(inputs[0], masks[0]);

        // Loops Range: [1,2, ... , layers-1]
        // At 5 layers we have only 4 stages

        for (int i = 0; i < params.num_layers; ++i)
        {
            d[i] = std::pair<at::Tensor, at::Tensor>(inputs[i], masks[i]);  // multistart[i]->forward(inputs[i]);
        }
        for (int i = params.num_layers - 1; i < params.num_layers; ++i)
        {
            d[i] = multistart[i]->forward(inputs[i]);
        }
        // d[params.num_layers-1] = start.forward<std::pair<torch::Tensor, torch::Tensor>>(inputs[params.num_layers-1]);
        //   d[params.num_layers-1] = start.forward<std::pair<torch::Tensor,
        //   torch::Tensor>>(inputs[params.num_layers-1], masks[params.num_layers-1]);


        if (!need_up_masks)
        {
            for (int i = 0; i < params.num_layers; ++i)
            {
                d[i].second = torch::Tensor();
            }
        }

        // Loops Range: [layers-1, ... , 2, 1]
        for (int i = params.num_layers - 1; i >= 1; --i)
        {
            d[i - 1] = up[i - 1]->forward(d[i], d[i - 1]);
        }
        return final->forward(d[0].first);
    }

    MultiScaleUnet2dParams params;
    bool multi_channel_masks = false;
    bool need_up_masks       = false;

    // torch::nn::AnyModule start;
    torch::nn::Sequential final;

    MultiStartBlock multistart[MultiScaleUnet2dParams::max_layers - 1] = {nullptr, nullptr, nullptr, nullptr};
    //  MultiStartBlock multistarts[1] = {nullptr};
    UpsampleUltraBlock up[MultiScaleUnet2dParams::max_layers - 1] = {nullptr, nullptr, nullptr, nullptr};
};

TORCH_MODULE(MultiScaleUnet2dUltraSlim);

class UpsampleDecOnlyBlockImpl : public torch::nn::Module
{
   public:
    UpsampleDecOnlyBlockImpl(int in_channels, int out_channels, int num_input_channels, std::string conv_block,
                             std::string upsample_mode = "deconv", std::string norm_str = "id",
                             std::string activation = "id")
    {
        SAIGA_ASSERT(in_channels > 0);
        SAIGA_ASSERT(out_channels > 0);

        std::vector<double> scale = {2.0, 2.0};

        // conv = GatedBlock(in_channels, out_channels);
        if (upsample_mode == "deconv")
        {
            up->push_back(torch::nn::ConvTranspose2d(
                torch::nn::ConvTranspose2dOptions(in_channels, out_channels, 4).stride(2).padding(1)));
        }
        else if (upsample_mode == "bilinear")
        {
            up->push_back(torch::nn::Upsample(
                torch::nn::UpsampleOptions().scale_factor(scale).mode(torch::kBilinear).align_corners(false)));
        }
        else if (upsample_mode == "nearest")
        {
            up->push_back(torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(scale).mode(torch::kNearest)));
            // up->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 3).padding(1)));
        }

        // conv1 = torch::nn::AnyModule(GatedBlock(in_channels, out_channels, 3, 1, 1, "id", "id"));
        //  conv = GatedBlock(out_channels * 2, out_channels, 3, 1, 1, norm_str);
        conv2 = UnetBlockFromString(conv_block, in_channels + num_input_channels, out_channels, 3, 1, 1, norm_str,
                                    activation);


        up_mask = torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(scale).mode(torch::kNearest));

        register_module("up", up);
        register_module("up_mask", up_mask);

        register_module("conv2", conv2.ptr());
    }

    // Combines the upsampled tensor (below) with the skip connection (skip)
    // Usually this can be done with a simple cat however if the size does not match we crop
    torch::Tensor CombineBridge(torch::Tensor below, torch::Tensor skip)
    {
        // SAIGA_ASSERT(skip.first.size(2) == same_layer_as_skip.first.size(2) &&
        //              skip.first.size(3) == same_layer_as_skip.first.size(3));
        if (below.size(2) == skip.size(2) && below.size(3) == skip.size(3))
        {
            return torch::cat({below, skip}, 1);
        }
        else
        {
            return torch::cat(
                {below, CenterCrop2D(skip, std::vector<int64_t>(below.sizes().begin(), below.sizes().end()))}, 1);
        }
    }

    std::pair<at::Tensor, at::Tensor> forward(std::pair<at::Tensor, at::Tensor> layer_below,
                                              std::pair<at::Tensor, at::Tensor> skip)
    {
        SAIGA_ASSERT(layer_below.first.defined());
        SAIGA_ASSERT(skip.first.defined());

        // Upsample the layer from below
        std::pair<at::Tensor, at::Tensor> same_layer_as_skip;
        same_layer_as_skip.first = up->forward(layer_below.first);

        if (layer_below.second.defined())
        {
            same_layer_as_skip.second = up_mask->forward(layer_below.second);
            // SAIGA_ASSERT(skip.second.size(2) == same_layer_as_skip.second.size(2) &&
            //              skip.second.size(3) == same_layer_as_skip.second.size(3));
        }



        std::pair<at::Tensor, at::Tensor> output;
        // [b, c, h, w]
        // output.first = torch::cat({same_layer_as_skip.first, skip.first}, 1);
        output.first = CombineBridge(same_layer_as_skip.first, skip.first);

        if (layer_below.second.defined())
        {
            // output.second = torch::cat({same_layer_as_skip.second, skip.second}, 1);
            output.second = CombineBridge(same_layer_as_skip.second, skip.second);
        }

        return conv2.forward<std::pair<at::Tensor, at::Tensor>>(output.first, output.second);
    }

    torch::nn::Sequential up;
    torch::nn::Upsample up_mask = nullptr;
    torch::nn::AnyModule conv2;
};

TORCH_MODULE(UpsampleDecOnlyBlock);

class MultiScaleUnet2dDecOnlyImpl : public torch::nn::Module
{
   public:
    MultiScaleUnet2dDecOnlyImpl(MultiScaleUnet2dParams params) : params(params)
    {
        std::cout << "Using MultiScaleUnet2dDecOnly " << std::endl;
        std::vector<int> num_input_channels_per_layer;
        // std::vector<int> filters = {4, 8, 16, 16, 16};
        std::vector<int> filters = params.filters_network;

        std::vector<int> num_input_channels(params.num_input_layers, params.num_input_channels);
        for (int i = params.num_input_layers; i < 5; ++i)
        {
            num_input_channels.push_back(0);
        }
        for (int i = 0; i < 5; ++i)
        {
            auto& f = filters[i];
            f       = f * params.feature_factor;
            if (params.add_input_to_filters && i >= 1)
            {
                f += num_input_channels[i];
            }

            if (i >= 1)
            {
                SAIGA_ASSERT(f >= num_input_channels[0]);
            }
        }


        SAIGA_ASSERT(num_input_channels.size() == filters.size());

        //  start = UnetBlockFromString(params.conv_block, num_input_channels[0], filters[0], 3, 1, 1, "id");
        //  register_module("start", start.ptr());



        final->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(filters[0], params.num_output_channels, 1)));
        final->push_back(ActivationFromString(params.last_act));
        register_module("final", final);

        for (int i = params.num_layers - 1; i < params.num_layers; ++i)
        {
            // Down[i] transforms from layer (i) -> (i+1)
            int multistart_in  = params.num_input_channels;  // filters[i];
            int multistart_out = filters[i];
            multistart[i] = MultiStartBlock(multistart_in, multistart_out, params.conv_block, params.norm_layer_down,
                                            params.pooling, params.activation);
            register_module("multistart" + std::to_string(i + 1), multistart[i]);
        }
        //  start      = MultiStartBlock(params.num_input_channels, filters[params.num_layers-1], params.conv_block,
        //  params.norm_layer_down, params.pooling,
        //                             params.activation);
        // register_module("start", start.ptr());

        for (int i = 0; i < params.num_layers - 1; ++i)
        // for (int i = params.num_layers - 1; i >= 1; --i)
        {
            // Up[i] transforms from layer (i+1) -> (i)
            int up_in  = filters[i + 1];
            int up_out = filters[i];
            up[i]      = UpsampleDecOnlyBlock(up_in, up_out, params.num_input_channels, params.conv_block_up,
                                              params.upsample_mode, params.norm_layer_up, params.activation);
            register_module("up" + std::to_string(i + 1), up[i]);
        }
        // multistart[params.num_layers - 1] = MultiStartBlock(params.num_input_channels, filters[params.num_layers -
        // 1], params.conv_block,
        //     params.norm_layer_down, params.pooling, params.activation);

        multi_channel_masks = params.conv_block == "partial_multi";
        need_up_masks       = params.conv_block_up == "partial_multi";
        if (params.half_float)
        {
            this->to(torch::kFloat16);
        }
    }

    at::Tensor forward(ArrayView<torch::Tensor> inputs)
    {
        std::vector<torch::Tensor> masks(inputs.size());
        return forward(inputs, masks);
    }

    at::Tensor forward(ArrayView<torch::Tensor> inputs, ArrayView<torch::Tensor> masks)
    {
        SAIGA_ASSERT(inputs.size() == params.num_input_layers);
        SAIGA_ASSERT(masks.size() == params.num_input_layers);
        // The downsampling should not happen on uneven image sizes!
        // SAIGA_ASSERT(inputs.front().size(2) % (1 << params.num_layers) == 0);
        // SAIGA_ASSERT(inputs.front().size(3) % (1 << params.num_layers) == 0);
        // debug check if input has correct format
        for (int i = 0; i < inputs.size(); ++i)
        {
            if (params.num_input_layers > i)
            {
                SAIGA_ASSERT(inputs.size() > i);
                SAIGA_ASSERT(inputs[i].defined());
                SAIGA_ASSERT(params.num_input_channels == inputs[i].size(1));
            }
            SAIGA_ASSERT(masks[i].requires_grad() == false);
        }

        std::pair<torch::Tensor, torch::Tensor> d[MultiScaleUnet2dParams::max_layers - 1];
        //!
        //        d[0] = multistart[0].forward<std::pair<torch::Tensor, torch::Tensor>>(inputs[0], masks[0]);

        // Loops Range: [1,2, ... , layers-1]
        // At 5 layers we have only 4 stages

        for (int i = 0; i < params.num_layers; ++i)
        {
            d[i] = std::pair<at::Tensor, at::Tensor>(inputs[i], masks[i]);  // multistart[i]->forward(inputs[i]);
        }
        for (int i = params.num_layers - 1; i < params.num_layers; ++i)
        {
            d[i] = multistart[i]->forward(inputs[i]);
        }
        // d[params.num_layers-1] = start.forward<std::pair<torch::Tensor, torch::Tensor>>(inputs[params.num_layers-1]);
        //   d[params.num_layers-1] = start.forward<std::pair<torch::Tensor,
        //   torch::Tensor>>(inputs[params.num_layers-1], masks[params.num_layers-1]);


        if (!need_up_masks)
        {
            for (int i = 0; i < params.num_layers; ++i)
            {
                d[i].second = torch::Tensor();
            }
        }

        // Loops Range: [layers-1, ... , 2, 1]
        for (int i = params.num_layers - 1; i >= 1; --i)
        {
            d[i - 1] = up[i - 1]->forward(d[i], d[i - 1]);
        }
        return final->forward(d[0].first);
    }

    MultiScaleUnet2dParams params;
    bool multi_channel_masks = false;
    bool need_up_masks       = false;

    // torch::nn::AnyModule start;
    torch::nn::Sequential final;

    MultiStartBlock multistart[MultiScaleUnet2dParams::max_layers - 1] = {nullptr, nullptr, nullptr, nullptr};
    //  MultiStartBlock multistarts[1] = {nullptr};
    UpsampleDecOnlyBlock up[MultiScaleUnet2dParams::max_layers - 1] = {nullptr, nullptr, nullptr, nullptr};
};

TORCH_MODULE(MultiScaleUnet2dDecOnly);


////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////

class SmallDecStartBlockImpl : public UnetBlockImpl
{
   public:
    using UnetBlockImpl::forward;

    SmallDecStartBlockImpl(int in_channels, int out_channels, std::string conv_block, std::string norm_str,
                           std::string activation_str)
    {
        SAIGA_ASSERT(in_channels > 0);
        SAIGA_ASSERT(out_channels > 0);
        conv = UnetBlockFromString(conv_block, in_channels, out_channels, 3, 1, 1, norm_str, activation_str);

        register_module("conv", conv.ptr());
    }

    torch::Tensor CombineBridge(torch::Tensor below, torch::Tensor skip)
    {
        if (below.size(2) == skip.size(2) && below.size(3) == skip.size(3))
            return torch::cat({below, skip}, 1);
        else
            return torch::cat(
                {below, CenterCrop2D(skip, std::vector<int64_t>(below.sizes().begin(), below.sizes().end()))}, 1);
    }

    std::pair<at::Tensor, at::Tensor> forward(at::Tensor x, at::Tensor mask = {}) override
    {
        std::pair<at::Tensor, at::Tensor> output;
        output = conv.forward<std::pair<at::Tensor, at::Tensor>>(x, mask);

        output.first = CombineBridge(x, output.first);
        return output;
    }

    torch::nn::AnyModule conv;
};

TORCH_MODULE(SmallDecStartBlock);


// small upsample block:
//  [up 28->28],
//  cat Features
//  Gconv 32 ->24
//  cat Features (bypass skip)
class UpsampleDecOnlySmallBlockImpl : public torch::nn::Module
{
   public:
    UpsampleDecOnlySmallBlockImpl(int in_channels, int out_channels, int num_input_channels, std::string conv_block,
                                  bool last, std::string upsample_mode = "deconv", std::string norm_str = "id",
                                  std::string activation = "id")
    {
        SAIGA_ASSERT(in_channels > 0);
        SAIGA_ASSERT(out_channels > 0);

        std::vector<double> scale = {2.0, 2.0};

        // conv = GatedBlock(in_channels, out_channels);
        if (upsample_mode == "deconv")
        {
            up->push_back(torch::nn::ConvTranspose2d(
                torch::nn::ConvTranspose2dOptions(in_channels, out_channels, 4).stride(2).padding(1)));
        }
        else if (upsample_mode == "bilinear")
        {
            up->push_back(torch::nn::Upsample(
                torch::nn::UpsampleOptions().scale_factor(scale).mode(torch::kBilinear).align_corners(false)));
        }
        else if (upsample_mode == "nearest")
        {
            up->push_back(torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(scale).mode(torch::kNearest)));
            // up->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 3).padding(1)));
        }

        SAIGA_ASSERT(conv_block != "partial_multi");
        // conv1 = torch::nn::AnyModule(GatedBlock(in_channels, out_channels, 3, 1, 1, "id", "id"));
        //  conv = GatedBlock(out_channels * 2, out_channels, 3, 1, 1, norm_str);

        // last layer can output more channels, as no additionally input is added in final block
        const int conv_num_output_channels =
            (!last) ? out_channels - 2 * num_input_channels : out_channels - num_input_channels;
        convolution =
            UnetBlockFromString(conv_block, in_channels, conv_num_output_channels, 3, 1, 1, norm_str, activation);

        register_module("up", up);
        register_module("convolution", convolution.ptr());
    }

    // Combines the upsampled tensor (below) with the skip connection (skip)
    // Usually this can be done with a simple cat however if the size does not match we crop
    torch::Tensor CombineBridge(torch::Tensor below, torch::Tensor skip)
    {
        if (below.size(2) == skip.size(2) && below.size(3) == skip.size(3))
            return torch::cat({below, skip}, 1);
        else
        {
            return torch::cat({below, torch::nn::functional::pad(
                                          skip, torch::nn::functional::PadFuncOptions(
                                                    {0, 0, below.size(2) - skip.size(2), below.size(3) - skip.size(3)})
                                                    .mode(torch::kConstant))},
                              1);
            // return torch::cat(
            //     {CenterCrop2D(below, std::vector<int64_t>(skip.sizes().begin(), skip.sizes().end())), skip}, 1);
        }
    }

    std::pair<at::Tensor, at::Tensor> forward(std::pair<at::Tensor, at::Tensor> layer_below,
                                              std::pair<at::Tensor, at::Tensor> features_input)
    {
        SAIGA_ASSERT(layer_below.first.defined());
        SAIGA_ASSERT(features_input.first.defined());

        // Upsample the layer from below
        std::pair<at::Tensor, at::Tensor> upsample_input;
        upsample_input.first = up->forward(layer_below.first);

        std::pair<at::Tensor, at::Tensor> combined_tensors;

        // [b, c, h, w]
        // output.first = torch::cat({same_layer_as_skip.first, skip.first}, 1);
        combined_tensors.first = CombineBridge(features_input.first, upsample_input.first);

        std::pair<at::Tensor, at::Tensor> output;
        output =
            convolution.forward<std::pair<at::Tensor, at::Tensor>>(combined_tensors.first, combined_tensors.second);

        output.first = CombineBridge(features_input.first, output.first);

        return output;
    }

    torch::nn::Sequential up;
    torch::nn::AnyModule convolution;
};

TORCH_MODULE(UpsampleDecOnlySmallBlock);


class MultiScaleUnet2dDecOnlySmallImpl : public torch::nn::Module
{
   public:
    MultiScaleUnet2dDecOnlySmallImpl(MultiScaleUnet2dParams params) : params(params)
    {
        std::cout << "Using MultiScaleUnet2dDecOnlySmall with filters: " << std::endl;

        // std::vector<int> filters = {32,32,32,32,32};
        std::vector<int> filters = params.filters_network;
        for (int i = 0; i < filters.size(); ++i)
        {
            std::cout << filters[i] << " ";
        }
        std::cout << std::endl;


        // output of block is [filter-num_input] (conv output [filter-2*num_input])
        const int out_first_conv = filters[params.num_layers - 1] - 2 * params.num_input_channels;
        start                    = SmallDecStartBlock(params.num_input_channels, out_first_conv, params.conv_block_up,
                                                      params.norm_layer_up, params.activation);
        register_module("start", start);

        // for (int i = 0; i < params.num_layers - 1; ++i)
        for (int i = params.num_layers - 2; i >= 0; --i)
        {
            // Up[i] transforms from layer (i+1) -> (i)
            int up_in  = filters[i + 1];
            int up_out = filters[i];
            up[i] = UpsampleDecOnlySmallBlock(up_in, up_out, params.num_input_channels, params.conv_block_up, (i == 0),
                                              params.upsample_mode, params.norm_layer_up, params.activation);
            register_module("up" + std::to_string(i + 1), up[i]);
        }

        final->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(filters[0], params.num_output_channels, 1)));
        final->push_back(ActivationFromString(params.last_act));
        register_module("final", final);



        if (params.half_float)
        {
            this->to(torch::kFloat16);
        }
    }

    at::Tensor forward(ArrayView<torch::Tensor> inputs)
    {
        std::vector<torch::Tensor> masks(inputs.size());
        return forward(inputs, masks);
    }

    at::Tensor forward(ArrayView<torch::Tensor> inputs, ArrayView<torch::Tensor> masks)
    {
        SAIGA_ASSERT(inputs.size() == params.num_input_layers);
        SAIGA_ASSERT(masks.size() == params.num_input_layers);
        // The downsampling should not happen on uneven image sizes!
        // SAIGA_ASSERT(inputs.front().size(2) % (1 << params.num_layers) == 0);
        // SAIGA_ASSERT(inputs.front().size(3) % (1 << params.num_layers) == 0);
        // debug check if input has correct format
        for (int i = 0; i < inputs.size(); ++i)
        {
            if (params.num_input_layers > i)
            {
                SAIGA_ASSERT(inputs.size() > i);
                SAIGA_ASSERT(inputs[i].defined());
                SAIGA_ASSERT(params.num_input_channels == inputs[i].size(1));
            }
            SAIGA_ASSERT(masks[i].requires_grad() == false);
        }

        std::pair<torch::Tensor, torch::Tensor> start_tensors =
            std::pair<at::Tensor, at::Tensor>(inputs[params.num_layers - 1], masks[params.num_layers - 1]);

        std::pair<torch::Tensor, torch::Tensor> iterative_out;
        {
            SAIGA_OPTIONAL_TIME_MEASURE("Start", timer_for_nets);

            iterative_out = start->forward(start_tensors.first, start_tensors.second);
        }
        // Loops Range: [layers-1, ... , 2, 1]
        for (int i = params.num_layers - 2; i >= 0; --i)
        {
            SAIGA_OPTIONAL_TIME_MEASURE("Layer " + std::to_string(i), timer_for_nets);

            std::pair<torch::Tensor, torch::Tensor> rendered_feature_maps_for_layer =
                std::pair<at::Tensor, at::Tensor>(inputs[i], masks[i]);
            iterative_out = up[i]->forward(iterative_out, rendered_feature_maps_for_layer);
        }

        at::Tensor result;
        {
            SAIGA_OPTIONAL_TIME_MEASURE("Final", timer_for_nets);

            result = final->forward(iterative_out.first);
        }
        return result;
    }

    MultiScaleUnet2dParams params;

    torch::nn::Sequential final;
    // SmallDecStartBlock
    SmallDecStartBlock start = nullptr;

    UpsampleDecOnlySmallBlock up[8 - 1] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
};

TORCH_MODULE(MultiScaleUnet2dDecOnlySmall);



class UpsampleDecOnlySmallBlockFixedImpl : public torch::nn::Module
{
   public:
    UpsampleDecOnlySmallBlockFixedImpl(int in_channels, int out_channels, int num_input_channels,
                                       std::string conv_block, bool last, std::string upsample_mode = "deconv",
                                       std::string norm_str = "id", std::string activation = "id")
    {
        SAIGA_ASSERT(in_channels > 0);
        SAIGA_ASSERT(out_channels > 0);

        std::vector<double> scale = {2.0, 2.0};

        // conv = GatedBlock(in_channels, out_channels);
        if (upsample_mode == "deconv")
        {
            up->push_back(torch::nn::ConvTranspose2d(
                torch::nn::ConvTranspose2dOptions(in_channels, out_channels, 4).stride(2).padding(1)));
        }
        else if (upsample_mode == "bilinear")
        {
            up->push_back(torch::nn::Upsample(
                torch::nn::UpsampleOptions().scale_factor(scale).mode(torch::kBilinear).align_corners(false)));
        }
        else if (upsample_mode == "nearest")
        {
            up->push_back(torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(scale).mode(torch::kNearest)));
            // up->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 3).padding(1)));
        }

        SAIGA_ASSERT(conv_block != "partial_multi");
        // conv1 = torch::nn::AnyModule(GatedBlock(in_channels, out_channels, 3, 1, 1, "id", "id"));
        //  conv = GatedBlock(out_channels * 2, out_channels, 3, 1, 1, norm_str);

        // last layer can output more channels, as no additionally input is added in final block
        const int conv_num_output_channels =
            (!last) ? out_channels - 2 * num_input_channels : out_channels - num_input_channels;
        convolution =
            UnetBlockFromString(conv_block, in_channels, conv_num_output_channels, 3, 1, 1, norm_str, activation);

        register_module("up", up);
        register_module("convolution", convolution.ptr());
    }

    // Combines the upsampled tensor (below) with the skip connection (skip)
    // Usually this can be done with a simple cat however if the size does not match we crop
    // torch::Tensor CombineBridge(torch::Tensor below, torch::Tensor skip)
    // {
    //     if (below.size(2) == skip.size(2) && below.size(3) == skip.size(3))
    //         return torch::cat({below, skip}, 1);
    //     else
    //     {
    //         return torch::cat({torch::nn::functional::pad(
    //                                below, torch::nn::functional::PadFuncOptions(
    //                                           {0, skip.size(3) - below.size(3), 0, skip.size(2) - below.size(2)})
    //                                           .mode(torch::kConstant)),
    //                            skip},
    //                           1);
    //         // return torch::cat(
    //         //     {CenterCrop2D(below, std::vector<int64_t>(skip.sizes().begin(), skip.sizes().end())), skip}, 1);
    //     }
    // }
    torch::Tensor CombineBridge(torch::Tensor below, torch::Tensor skip)
    {
        if (below.size(2) == skip.size(2) && below.size(3) == skip.size(3))
            return torch::cat({below, skip}, 1);
        else
            return torch::cat(
                {below, CenterCrop2D(skip, std::vector<int64_t>(below.sizes().begin(), below.sizes().end()))}, 1);
    }
    std::pair<at::Tensor, at::Tensor> forward(std::pair<at::Tensor, at::Tensor> layer_below,
                                              std::pair<at::Tensor, at::Tensor> features_input)
    {
        SAIGA_ASSERT(layer_below.first.defined());
        SAIGA_ASSERT(features_input.first.defined());

        // Upsample the layer from below
        std::pair<at::Tensor, at::Tensor> upsample_input;
        upsample_input.first = up->forward(layer_below.first);

        std::pair<at::Tensor, at::Tensor> combined_tensors;

        // [b, c, h, w]
        // output.first = torch::cat({same_layer_as_skip.first, skip.first}, 1);
        combined_tensors.first = CombineBridge(features_input.first, upsample_input.first);

        std::pair<at::Tensor, at::Tensor> output;
        output =
            convolution.forward<std::pair<at::Tensor, at::Tensor>>(combined_tensors.first, combined_tensors.second);

        output.first = CombineBridge(features_input.first, output.first);

        return output;
    }

    torch::nn::Sequential up;
    torch::nn::AnyModule convolution;
};

TORCH_MODULE(UpsampleDecOnlySmallBlockFixed);


class MultiScaleUnet2dDecOnlySmallFixedImpl : public torch::nn::Module
{
   public:
    MultiScaleUnet2dDecOnlySmallFixedImpl(MultiScaleUnet2dParams params) : params(params)
    {
        std::cout << "Using MultiScaleUnet2dDecOnlySmall with filters: " << std::endl;

        // std::vector<int> filters = {32,32,32,32,32};
        std::vector<int> filters = params.filters_network;
        for (int i = 0; i < filters.size(); ++i)
        {
            std::cout << filters[i] << " ";
        }
        std::cout << std::endl;


        // output of block is [filter-num_input] (conv output [filter-2*num_input])
        const int out_first_conv = filters[params.num_layers - 1] - 2 * params.num_input_channels;
        start                    = SmallDecStartBlock(params.num_input_channels, out_first_conv, params.conv_block_up,
                                                      params.norm_layer_up, params.activation);
        register_module("start", start);

        // for (int i = 0; i < params.num_layers - 1; ++i)
        for (int i = params.num_layers - 2; i >= 0; --i)
        {
            // Up[i] transforms from layer (i+1) -> (i)
            int up_in  = filters[i + 1];
            int up_out = filters[i];
            up[i] =
                UpsampleDecOnlySmallBlockFixed(up_in, up_out, params.num_input_channels, params.conv_block_up, (i == 0),
                                               params.upsample_mode, params.norm_layer_up, params.activation);
            register_module("up" + std::to_string(i + 1), up[i]);
        }

        final->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(filters[0], params.num_output_channels, 1)));
        final->push_back(ActivationFromString(params.last_act));
        register_module("final", final);



        if (params.half_float)
        {
            this->to(torch::kFloat16);
        }
    }

    at::Tensor forward(ArrayView<torch::Tensor> inputs)
    {
        std::vector<torch::Tensor> masks(inputs.size());
        return forward(inputs, masks);
    }

    at::Tensor forward(ArrayView<torch::Tensor> inputs, ArrayView<torch::Tensor> masks)
    {
        SAIGA_ASSERT(inputs.size() == params.num_input_layers);
        SAIGA_ASSERT(masks.size() == params.num_input_layers);
        // The downsampling should not happen on uneven image sizes!
        // SAIGA_ASSERT(inputs.front().size(2) % (1 << params.num_layers) == 0);
        // SAIGA_ASSERT(inputs.front().size(3) % (1 << params.num_layers) == 0);
        // debug check if input has correct format
        for (int i = 0; i < inputs.size(); ++i)
        {
            if (params.num_input_layers > i)
            {
                SAIGA_ASSERT(inputs.size() > i);
                SAIGA_ASSERT(inputs[i].defined());
                SAIGA_ASSERT(params.num_input_channels == inputs[i].size(1));
            }
            SAIGA_ASSERT(masks[i].requires_grad() == false);
        }

        std::pair<torch::Tensor, torch::Tensor> start_tensors =
            std::pair<at::Tensor, at::Tensor>(inputs[params.num_layers - 1], masks[params.num_layers - 1]);

        std::pair<torch::Tensor, torch::Tensor> iterative_out;
        {
            SAIGA_OPTIONAL_TIME_MEASURE("Start", timer_for_nets);

            iterative_out = start->forward(start_tensors.first, start_tensors.second);
        }
        // Loops Range: [layers-1, ... , 2, 1]
        for (int i = params.num_layers - 2; i >= 0; --i)
        {
            SAIGA_OPTIONAL_TIME_MEASURE("Layer " + std::to_string(i), timer_for_nets);

            std::pair<torch::Tensor, torch::Tensor> rendered_feature_maps_for_layer =
                std::pair<at::Tensor, at::Tensor>(inputs[i], masks[i]);
            iterative_out = up[i]->forward(iterative_out, rendered_feature_maps_for_layer);
        }

        at::Tensor result;
        {
            SAIGA_OPTIONAL_TIME_MEASURE("Final", timer_for_nets);

            result = final->forward(iterative_out.first);
        }
        return result;
    }

    MultiScaleUnet2dParams params;

    torch::nn::Sequential final;
    // SmallDecStartBlock
    SmallDecStartBlock start = nullptr;

    UpsampleDecOnlySmallBlockFixed up[8 - 1] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
};

TORCH_MODULE(MultiScaleUnet2dDecOnlySmallFixed);



///////////////////////////////////////////////////////////////////////////

class UpsampleDecOnlySmallBlockVarOutputFixedImpl : public torch::nn::Module
{
   public:
    UpsampleDecOnlySmallBlockVarOutputFixedImpl(int in_channels, int out_channels, int num_input_channels,
                                                std::string conv_block, bool last, int sh_bands,
                                                std::string upsample_mode = "deconv", std::string norm_str = "id",
                                                std::string activation = "id")
        : last(last)
    {
        SAIGA_ASSERT(in_channels > 0);
        SAIGA_ASSERT(out_channels > 0);

        std::vector<double> scale = {2.0, 2.0};

        // conv = GatedBlock(in_channels, out_channels);
        if (upsample_mode == "deconv")
        {
            up->push_back(torch::nn::ConvTranspose2d(
                torch::nn::ConvTranspose2dOptions(in_channels, out_channels, 4).stride(2).padding(1)));
        }
        else if (upsample_mode == "bilinear")
        {
            up->push_back(torch::nn::Upsample(
                torch::nn::UpsampleOptions().scale_factor(scale).mode(torch::kBilinear).align_corners(false)));
        }
        else if (upsample_mode == "nearest")
        {
            up->push_back(torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(scale).mode(torch::kNearest)));
            // up->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 3).padding(1)));
        }

        SAIGA_ASSERT(conv_block != "partial_multi");
        // conv1 = torch::nn::AnyModule(GatedBlock(in_channels, out_channels, 3, 1, 1, "id", "id"));
        //  conv = GatedBlock(out_channels * 2, out_channels, 3, 1, 1, norm_str);

        // last layer can output more channels, as no additionally input is added in final block
        const int conv_num_output_channels = (!last) ? out_channels - 2 * num_input_channels : 3 * sh_bands * sh_bands;
        convolution =
            UnetBlockFromString(conv_block, in_channels, conv_num_output_channels, 3, 1, 1, norm_str, activation);

        register_module("up", up);
        register_module("convolution", convolution.ptr());
    }

    // Combines the upsampled tensor (below) with the skip connection (skip)
    // Usually this can be done with a simple cat however if the size does not match we crop
    // torch::Tensor CombineBridge(torch::Tensor below, torch::Tensor skip)
    //{
    //    if (below.size(2) == skip.size(2) && below.size(3) == skip.size(3))
    //        return torch::cat({below, skip}, 1);
    //    else
    //    {
    //        return torch::cat({below, torch::nn::functional::pad(
    //                                      skip, torch::nn::functional::PadFuncOptions(
    //                                                {0, 0, below.size(2) - skip.size(2), below.size(3) -
    //                                                skip.size(3)}) .mode(torch::kConstant))},
    //                          1);
    //        // return torch::cat(
    //        //     {CenterCrop2D(below, std::vector<int64_t>(skip.sizes().begin(), skip.sizes().end())), skip}, 1);
    //    }
    //}

    torch::Tensor CombineBridge(torch::Tensor below, torch::Tensor skip)
    {
        if (below.size(2) == skip.size(2) && below.size(3) == skip.size(3))
            return torch::cat({below, skip}, 1);
        else
            return torch::cat(
                {below, CenterCrop2D(skip, std::vector<int64_t>(below.sizes().begin(), below.sizes().end()))}, 1);
    }

    std::pair<at::Tensor, at::Tensor> forward(std::pair<at::Tensor, at::Tensor> layer_below,
                                              std::pair<at::Tensor, at::Tensor> features_input)
    {
        SAIGA_ASSERT(layer_below.first.defined());
        SAIGA_ASSERT(features_input.first.defined());

        // Upsample the layer from below
        std::pair<at::Tensor, at::Tensor> upsample_input;
        upsample_input.first = up->forward(layer_below.first);

        std::pair<at::Tensor, at::Tensor> combined_tensors;

        // [b, c, h, w]
        // output.first = torch::cat({same_layer_as_skip.first, skip.first}, 1);
        combined_tensors.first = CombineBridge(features_input.first, upsample_input.first);

        std::pair<at::Tensor, at::Tensor> output;
        output =
            convolution.forward<std::pair<at::Tensor, at::Tensor>>(combined_tensors.first, combined_tensors.second);

        if (!last) output.first = CombineBridge(features_input.first, output.first);

        return output;
    }

    torch::nn::Sequential up;
    torch::nn::AnyModule convolution;
    bool last;
};

TORCH_MODULE(UpsampleDecOnlySmallBlockVarOutputFixed);

class MultiScaleUnet2dSphericalHarmonicsFixedImpl : public torch::nn::Module
{
   public:
    MultiScaleUnet2dSphericalHarmonicsFixedImpl(MultiScaleUnet2dParams params)
        : params(params), spherical_harmonics(params.sh_bands)
    {
        std::cout << "Using MultiScaleUnet2dSphericalHarmonicsFixed with filters: " << std::endl;

        // std::vector<int> filters = {32,32,32,32,32};
        std::vector<int> filters = params.filters_network;
        for (int i = 0; i < filters.size(); ++i)
        {
            std::cout << filters[i] << " ";
        }
        std::cout << std::endl;


        // output of block is [filter-num_input] (conv output [filter-2*num_input])
        const int out_first_conv = filters[params.num_layers - 1] - 2 * params.num_input_channels;
        start                    = SmallDecStartBlock(params.num_input_channels, out_first_conv, params.conv_block_up,
                                                      params.norm_layer_up, params.activation);
        register_module("start", start);

        // for (int i = 0; i < params.num_layers - 1; ++i)
        for (int i = params.num_layers - 2; i >= 0; --i)
        {
            // Up[i] transforms from layer (i+1) -> (i)
            int up_in  = filters[i + 1];
            int up_out = filters[i];
            up[i]      = UpsampleDecOnlySmallBlockVarOutputFixed(
                up_in, up_out, params.num_input_channels, params.conv_block_up, (i == 0), params.sh_bands,
                params.upsample_mode, params.norm_layer_up, params.activation);
            register_module("up" + std::to_string(i + 1), up[i]);
        }

        final->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(3, params.num_output_channels, 1)));
        final->push_back(ActivationFromString(params.last_act));
        register_module("final", final);


        // register_module("spherical_harmonics", spherical_harmonics);


        if (params.half_float)
        {
            this->to(torch::kFloat16);
        }
    }

    at::Tensor forward(ArrayView<torch::Tensor> inputs)
    {
        std::vector<torch::Tensor> masks(inputs.size());
        return forward(inputs, masks);
    }

    at::Tensor forward(ArrayView<torch::Tensor> inputs, ArrayView<torch::Tensor> masks)
    {
        SAIGA_ASSERT(inputs.size() == params.num_input_layers + 1);
        SAIGA_ASSERT(masks.size() == params.num_input_layers + 1);
        // The downsampling should not happen on uneven image sizes!
        // SAIGA_ASSERT(inputs.front().size(2) % (1 << params.num_layers) == 0);
        // SAIGA_ASSERT(inputs.front().size(3) % (1 << params.num_layers) == 0);
        // debug check if input has correct format
        // for (int i = 0; i < inputs.size(); ++i)
        //{
        //    if (params.num_input_layers > i)
        //    {
        //        SAIGA_ASSERT(inputs.size() > i);
        //        SAIGA_ASSERT(inputs[i].defined());
        //        SAIGA_ASSERT(params.num_input_channels == inputs[i].size(1));
        //    }
        //    SAIGA_ASSERT(masks[i].requires_grad() == false);
        //}



        std::pair<torch::Tensor, torch::Tensor> start_tensors =
            std::pair<at::Tensor, at::Tensor>(inputs[params.num_layers - 1], masks[params.num_layers - 1]);

        std::pair<torch::Tensor, torch::Tensor> iterative_out;
        {
            SAIGA_OPTIONAL_TIME_MEASURE("Start", timer_for_nets);

            iterative_out = start->forward(start_tensors.first, start_tensors.second);
        }
        // Loops Range: [layers-1, ... , 2, 1]
        for (int i = params.num_layers - 2; i >= 0; --i)
        {
            SAIGA_OPTIONAL_TIME_MEASURE("Layer " + std::to_string(i), timer_for_nets);

            std::pair<torch::Tensor, torch::Tensor> rendered_feature_maps_for_layer =
                std::pair<at::Tensor, at::Tensor>(inputs[i], masks[i]);
            iterative_out = up[i]->forward(iterative_out, rendered_feature_maps_for_layer);
        }

        torch::Tensor result;
        {
            SAIGA_OPTIONAL_TIME_MEASURE("Spherical Harmonics", timer_for_nets);

            //   PrintTensorInfo(inputs.back());
            torch::Tensor view_direction = inputs.back().permute({0, 2, 3, 1});

            // output: (b x h x w x 16)
            //  torch::Tensor spherical_encodings = spherical_harmonics.forward(view_direction);
            // PrintTensorInfo(view_direction);
            // PrintTensorInfo(spherical_encodings);
            // PrintTensorInfo(iterative_out.first.permute({0, 2, 3, 1}).slice(3, 0, 16));
            // PrintTensorInfo(iterative_out.first.permute({0, 2, 3, 1}).slice(3, 0, 16) * spherical_encodings);
            torch::Tensor spherical_encodings = spherical_harmonics.forward(view_direction);
            spherical_encodings               = spherical_encodings.to(iterative_out.first.dtype());

            int sh_band_sq = params.sh_bands * params.sh_bands;

            torch::Tensor enc_r =
                (iterative_out.first.permute({0, 2, 3, 1}).slice(3, 0, sh_band_sq) * spherical_encodings)
                    .sum({3})
                    .unsqueeze(3);
            //   PrintTensorInfo(enc_r);
            torch::Tensor enc_g =
                (iterative_out.first.permute({0, 2, 3, 1}).slice(3, sh_band_sq, 2 * sh_band_sq) * spherical_encodings)
                    .sum({3})
                    .unsqueeze(3);
            //    PrintTensorInfo(enc_g);

            torch::Tensor enc_b = (iterative_out.first.permute({0, 2, 3, 1}).slice(3, sh_band_sq * 2, sh_band_sq * 3) *
                                   spherical_encodings)
                                      .sum({3})
                                      .unsqueeze(3);

            //  enc_r = iterative_out.first.permute({0, 2, 3, 1}).slice(3, 0, 16).sum({3}).unsqueeze(3);
            //  enc_g = iterative_out.first.permute({0, 2, 3, 1}).slice(3, 16, 32).sum({3}).unsqueeze(3);
            //  enc_b = iterative_out.first.permute({0, 2, 3, 1}).slice(3, 32, 48).sum({3}).unsqueeze(3);

            //        PrintTensorInfo(enc_b);


            result = torch::cat({enc_r, enc_g, enc_b}, 3).permute({0, 3, 1, 2});
        }
        {
            SAIGA_OPTIONAL_TIME_MEASURE("Final", timer_for_nets);

            result = final->forward(result);
        }
        //   PrintTensorInfo(result);

        return result;
    }

    MultiScaleUnet2dParams params;

    torch::nn::Sequential final;
    // SmallDecStartBlock
    SmallDecStartBlock start = nullptr;

    SphericalHarmonicsEncoding spherical_harmonics;

    UpsampleDecOnlySmallBlockVarOutputFixed up[8 - 1] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
};

TORCH_MODULE(MultiScaleUnet2dSphericalHarmonicsFixed);



class MultiScaleUnet2dSphericalHarmonicsInL2FixedImpl : public torch::nn::Module
{
   public:
    MultiScaleUnet2dSphericalHarmonicsInL2FixedImpl(MultiScaleUnet2dParams params)
        : params(params), spherical_harmonics(params.sh_bands)
    {
        std::cout << "Using MultiScaleUnet2dSphericalHarmonicsInL2FixedImpl with filters: " << std::endl;

        // std::vector<int> filters = {32,32,32,32,32};
        std::vector<int> filters = params.filters_network;
        for (int i = 0; i < filters.size(); ++i)
        {
            std::cout << filters[i] << " ";
        }
        std::cout << std::endl;


        // output of block is [filter-num_input] (conv output [filter-2*num_input])
        const int out_first_conv = filters[params.num_layers - 1] - 2 * params.num_input_channels;
        start                    = SmallDecStartBlock(params.num_input_channels, out_first_conv, params.conv_block_up,
                                                      params.norm_layer_up, params.activation);
        register_module("start", start);

        // for (int i = 0; i < params.num_layers - 1; ++i)
        for (int i = params.num_layers - 2; i >= 1; --i)
        {
            // Up[i] transforms from layer (i+1) -> (i)
            int up_in  = filters[i + 1];
            int up_out = filters[i];
            up[i]      = UpsampleDecOnlySmallBlockVarOutputFixed(
                up_in, up_out, params.num_input_channels, params.conv_block_up, (i == 1), params.sh_bands,
                params.upsample_mode, params.norm_layer_up, params.activation);
            register_module("up" + std::to_string(i + 1), up[i]);
        }

        int up_in  = 3 + params.num_input_channels;
        int up_out = 3 + 2 * params.num_input_channels;
        up[0] = UpsampleDecOnlySmallBlockVarOutputFixed(up_in, up_out, params.num_input_channels, params.conv_block_up,
                                                        false, params.sh_bands, params.upsample_mode,
                                                        params.norm_layer_up, params.activation);
        register_module("up" + std::to_string(1), up[0]);

        final->push_back(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(3 + params.num_input_channels, params.num_output_channels, 1)));
        final->push_back(ActivationFromString(params.last_act));
        register_module("final", final);


        // register_module("spherical_harmonics", spherical_harmonics);


        if (params.half_float)
        {
            this->to(torch::kFloat16);
        }
    }

    at::Tensor forward(ArrayView<torch::Tensor> inputs)
    {
        std::vector<torch::Tensor> masks(inputs.size());
        return forward(inputs, masks);
    }

    at::Tensor forward(ArrayView<torch::Tensor> inputs, ArrayView<torch::Tensor> masks)
    {
        SAIGA_ASSERT(inputs.size() == params.num_input_layers + 1);
        SAIGA_ASSERT(masks.size() == params.num_input_layers + 1);
        // The downsampling should not happen on uneven image sizes!
        // SAIGA_ASSERT(inputs.front().size(2) % (1 << params.num_layers) == 0);
        // SAIGA_ASSERT(inputs.front().size(3) % (1 << params.num_layers) == 0);
        // debug check if input has correct format
        // for (int i = 0; i < inputs.size(); ++i)
        //{
        //    if (params.num_input_layers > i)
        //    {
        //        SAIGA_ASSERT(inputs.size() > i);
        //        SAIGA_ASSERT(inputs[i].defined());
        //        SAIGA_ASSERT(params.num_input_channels == inputs[i].size(1));
        //    }
        //    SAIGA_ASSERT(masks[i].requires_grad() == false);
        //}



        std::pair<torch::Tensor, torch::Tensor> start_tensors =
            std::pair<at::Tensor, at::Tensor>(inputs[params.num_layers - 1], masks[params.num_layers - 1]);

        std::pair<torch::Tensor, torch::Tensor> iterative_out;
        {
            SAIGA_OPTIONAL_TIME_MEASURE("Start", timer_for_nets);

            iterative_out = start->forward(start_tensors.first, start_tensors.second);
        }
        // Loops Range: [layers-1, ... , 2, 1]
        for (int i = params.num_layers - 2; i >= 1; --i)
        {
            SAIGA_OPTIONAL_TIME_MEASURE("Layer " + std::to_string(i), timer_for_nets);

            std::pair<torch::Tensor, torch::Tensor> rendered_feature_maps_for_layer =
                std::pair<at::Tensor, at::Tensor>(inputs[i], masks[i]);
            iterative_out = up[i]->forward(iterative_out, rendered_feature_maps_for_layer);
        }

        torch::Tensor result;
        {
            SAIGA_OPTIONAL_TIME_MEASURE("Spherical Harmonics", timer_for_nets);

            //   PrintTensorInfo(inputs.back());
            torch::Tensor view_direction =
                torch::nn::functional::avg_pool2d(inputs.back(),
                                                  torch::nn::functional::AvgPool2dFuncOptions({2, 2}).ceil_mode(true))
                    .permute({0, 2, 3, 1});

            // PrintTensorInfo(view_direction);
            // PrintTensorInfo(iterative_out.first);

            // output: (b x h x w x 16)
            //  torch::Tensor spherical_encodings = spherical_harmonics.forward(view_direction);
            // PrintTensorInfo(view_direction);
            // PrintTensorInfo(spherical_encodings);
            // PrintTensorInfo(iterative_out.first.permute({0, 2, 3, 1}).slice(3, 0, 16));
            // PrintTensorInfo(iterative_out.first.permute({0, 2, 3, 1}).slice(3, 0, 16) * spherical_encodings);
            torch::Tensor spherical_encodings = spherical_harmonics.forward(view_direction);
            spherical_encodings               = spherical_encodings.to(iterative_out.first.dtype());

            int sh_band_sq = params.sh_bands * params.sh_bands;

            torch::Tensor enc_r =
                (iterative_out.first.permute({0, 2, 3, 1}).slice(3, 0, sh_band_sq) * spherical_encodings)
                    .sum({3})
                    .unsqueeze(3);
            //   PrintTensorInfo(enc_r);
            torch::Tensor enc_g =
                (iterative_out.first.permute({0, 2, 3, 1}).slice(3, sh_band_sq, 2 * sh_band_sq) * spherical_encodings)
                    .sum({3})
                    .unsqueeze(3);
            //    PrintTensorInfo(enc_g);

            torch::Tensor enc_b = (iterative_out.first.permute({0, 2, 3, 1}).slice(3, sh_band_sq * 2, sh_band_sq * 3) *
                                   spherical_encodings)
                                      .sum({3})
                                      .unsqueeze(3);

            //  enc_r = iterative_out.first.permute({0, 2, 3, 1}).slice(3, 0, 16).sum({3}).unsqueeze(3);
            //  enc_g = iterative_out.first.permute({0, 2, 3, 1}).slice(3, 16, 32).sum({3}).unsqueeze(3);
            //  enc_b = iterative_out.first.permute({0, 2, 3, 1}).slice(3, 32, 48).sum({3}).unsqueeze(3);

            //        PrintTensorInfo(enc_b);


            result = torch::cat({enc_r, enc_g, enc_b}, 3).permute({0, 3, 1, 2});
        }
        {
            SAIGA_OPTIONAL_TIME_MEASURE("Layer " + std::to_string(0), timer_for_nets);

            std::pair<torch::Tensor, torch::Tensor> rendered_feature_maps_for_layer =
                std::pair<at::Tensor, at::Tensor>(inputs[0], masks[0]);
            std::pair<torch::Tensor, torch::Tensor> in_sph = std::pair<at::Tensor, at::Tensor>(result, result);
            in_sph                                         = up[0]->forward(in_sph, rendered_feature_maps_for_layer);
            result                                         = in_sph.first;
        }
        {
            SAIGA_OPTIONAL_TIME_MEASURE("Final", timer_for_nets);

            result = final->forward(result);
        }
        //   PrintTensorInfo(result);

        return result;
    }

    MultiScaleUnet2dParams params;

    torch::nn::Sequential final;
    // SmallDecStartBlock
    SmallDecStartBlock start = nullptr;

    SphericalHarmonicsEncoding spherical_harmonics;

    UpsampleDecOnlySmallBlockVarOutputFixed up[8 - 1] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
};

TORCH_MODULE(MultiScaleUnet2dSphericalHarmonicsInL2Fixed);



#if 0
// small upsample block:
//  [up 28->28],
//  cat Features
//  Gconv 32 ->24
//  cat Features (bypass skip)
class UpsampleDecOnlySmallBlockVarOutputImpl : public torch::nn::Module
{
   public:
    UpsampleDecOnlySmallBlockVarOutputImpl(int in_channels, int out_channels, int num_input_channels,
                                           std::string conv_block, bool last, std::string upsample_mode = "deconv",
                                           std::string norm_str = "id", std::string activation = "id")
        : last(last)
    {
        SAIGA_ASSERT(in_channels > 0);
        SAIGA_ASSERT(out_channels > 0);

        std::vector<double> scale = {2.0, 2.0};

        // conv = GatedBlock(in_channels, out_channels);
        if (upsample_mode == "deconv")
        {
            up->push_back(torch::nn::ConvTranspose2d(
                torch::nn::ConvTranspose2dOptions(in_channels, out_channels, 4).stride(2).padding(1)));
        }
        else if (upsample_mode == "bilinear")
        {
            up->push_back(torch::nn::Upsample(
                torch::nn::UpsampleOptions().scale_factor(scale).mode(torch::kBilinear).align_corners(false)));
        }
        else if (upsample_mode == "nearest")
        {
            up->push_back(torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(scale).mode(torch::kNearest)));
            // up->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 3).padding(1)));
        }

        SAIGA_ASSERT(conv_block != "partial_multi");
        // conv1 = torch::nn::AnyModule(GatedBlock(in_channels, out_channels, 3, 1, 1, "id", "id"));
        //  conv = GatedBlock(out_channels * 2, out_channels, 3, 1, 1, norm_str);

        // last layer can output more channels, as no additionally input is added in final block
        const int conv_num_output_channels = (!last) ? out_channels - 2 * num_input_channels : out_channels;
        convolution =
            UnetBlockFromString(conv_block, in_channels, conv_num_output_channels, 3, 1, 1, norm_str, activation);

        register_module("up", up);
        register_module("convolution", convolution.ptr());
    }

    // Combines the upsampled tensor (below) with the skip connection (skip)
    // Usually this can be done with a simple cat however if the size does not match we crop
    torch::Tensor CombineBridge(torch::Tensor below, torch::Tensor skip)
    {
        if (below.size(2) == skip.size(2) && below.size(3) == skip.size(3))
        {
            return torch::cat({below, skip}, 1);
        }
        else
        {
            // PrintTensorInfo(below);
            // PrintTensorInfo(skip);

            auto r = torch::nn::functional::pad(below,
                                                torch::nn::functional::PadFuncOptions(
                                                    {0, skip.size(3) - below.size(3), 0, skip.size(2) - below.size(2)})
                                                    .mode(torch::kConstant));
            //  PrintTensorInfo(r);

            return torch::cat({r, skip}, 1);
            // return torch::cat(
            //     {CenterCrop2D(below, std::vector<int64_t>(skip.sizes().begin(), skip.sizes().end())), skip}, 1);
        }
    }

    std::pair<at::Tensor, at::Tensor> forward(std::pair<at::Tensor, at::Tensor> layer_below,
                                              std::pair<at::Tensor, at::Tensor> features_input)
    {
        SAIGA_ASSERT(layer_below.first.defined());
        SAIGA_ASSERT(features_input.first.defined());

        // Upsample the layer from below
        std::pair<at::Tensor, at::Tensor> upsample_input;
        upsample_input.first = up->forward(layer_below.first);

        std::pair<at::Tensor, at::Tensor> combined_tensors;

        // [b, c, h, w]
        // output.first = torch::cat({same_layer_as_skip.first, skip.first}, 1);
        combined_tensors.first = CombineBridge(upsample_input.first, features_input.first);

        std::pair<at::Tensor, at::Tensor> output;
        output =
            convolution.forward<std::pair<at::Tensor, at::Tensor>>(combined_tensors.first, combined_tensors.second);

        if (!last) output.first = CombineBridge(features_input.first, output.first);

        return output;
    }

    torch::nn::Sequential up;
    torch::nn::AnyModule convolution;
    bool last;
};

TORCH_MODULE(UpsampleDecOnlySmallBlockVarOutput);

class MultiScaleUnet2dSphericalHarmonicsImpl : public torch::nn::Module
{
   public:
    MultiScaleUnet2dSphericalHarmonicsImpl(MultiScaleUnet2dParams params)
        : params(params), spherical_harmonics(params.sh_bands)
    {
        std::cout << "Using MultiScaleUnet2dSphericalHarmonicsImpl with filters: " << std::endl;

        // std::vector<int> filters = {32,32,32,32,32};
        std::vector<int> filters = params.filters_network;
        for (int i = 0; i < filters.size(); ++i)
        {
            std::cout << filters[i] << " ";
        }
        std::cout << std::endl;


        // output of block is [filter-num_input] (conv output [filter-2*num_input])
        const int out_first_conv = filters[params.num_layers - 1] - 2 * params.num_input_channels;
        start                    = SmallDecStartBlock(params.num_input_channels, out_first_conv, params.conv_block_up,
                                                      params.norm_layer_up, params.activation);
        register_module("start", start);

        // for (int i = 0; i < params.num_layers - 1; ++i)
        for (int i = params.num_layers - 2; i >= 0; --i)
        {
            // Up[i] transforms from layer (i+1) -> (i)
            int up_in  = filters[i + 1];
            int up_out = i == 0 ? (params.sh_bands * params.sh_bands * 3) : filters[i];
            up[i] = UpsampleDecOnlySmallBlockVarOutput(up_in, up_out, params.num_input_channels, params.conv_block_up,
                                                       (i == 0), params.upsample_mode, params.norm_layer_up,
                                                       params.activation);
            register_module("up" + std::to_string(i + 1), up[i]);
        }

        final->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(3, params.num_output_channels, 1)));
        final->push_back(ActivationFromString(params.last_act));
        register_module("final", final);


        //  register_module("spherical_harmonics", spherical_harmonics.module);


        if (params.half_float)
        {
            this->to(torch::kFloat16);
        }
    }

    at::Tensor forward(ArrayView<torch::Tensor> inputs)
    {
        std::vector<torch::Tensor> masks(inputs.size());
        return forward(inputs, masks);
    }

    at::Tensor forward(ArrayView<torch::Tensor> inputs, ArrayView<torch::Tensor> masks)
    {
        SAIGA_ASSERT(inputs.size() == params.num_input_layers + 1);
        SAIGA_ASSERT(masks.size() == params.num_input_layers + 1);
        // The downsampling should not happen on uneven image sizes!
        // SAIGA_ASSERT(inputs.front().size(2) % (1 << params.num_layers) == 0);
        // SAIGA_ASSERT(inputs.front().size(3) % (1 << params.num_layers) == 0);
        // debug check if input has correct format
        // for (int i = 0; i < inputs.size(); ++i)
        //{
        //    if (params.num_input_layers > i)
        //    {
        //        SAIGA_ASSERT(inputs.size() > i);
        //        SAIGA_ASSERT(inputs[i].defined());
        //        SAIGA_ASSERT(params.num_input_channels == inputs[i].size(1));
        //    }
        //    SAIGA_ASSERT(masks[i].requires_grad() == false);
        //}



        std::pair<torch::Tensor, torch::Tensor> start_tensors =
            std::pair<at::Tensor, at::Tensor>(inputs[params.num_layers - 1], masks[params.num_layers - 1]);

        std::pair<torch::Tensor, torch::Tensor> iterative_out;
        {
            SAIGA_OPTIONAL_TIME_MEASURE("Start", timer_for_nets);

            iterative_out = start->forward(start_tensors.first, start_tensors.second);
        }
        // Loops Range: [layers-1, ... , 2, 1]
        for (int i = params.num_layers - 2; i >= 0; --i)
        {
            SAIGA_OPTIONAL_TIME_MEASURE("Layer " + std::to_string(i), timer_for_nets);

            std::pair<torch::Tensor, torch::Tensor> rendered_feature_maps_for_layer =
                std::pair<at::Tensor, at::Tensor>(inputs[i], masks[i]);
            iterative_out = up[i]->forward(iterative_out, rendered_feature_maps_for_layer);
        }

        //   PrintTensorInfo(inputs.back());
        torch::Tensor result;
        {
            SAIGA_OPTIONAL_TIME_MEASURE("Spherical Harmonics", timer_for_nets);

            torch::Tensor view_direction = inputs.back().permute({0, 2, 3, 1});

            // output: (b x h x w x 16)
            torch::Tensor spherical_encodings = spherical_harmonics.forward(view_direction);
            spherical_encodings               = spherical_encodings.to(iterative_out.first.dtype());
            torch::Tensor enc_r =
                (iterative_out.first.permute({0, 2, 3, 1}).slice(3, 0, 16) * spherical_encodings).sum({3}).unsqueeze(3);
            torch::Tensor enc_g = (iterative_out.first.permute({0, 2, 3, 1}).slice(3, 16, 32) * spherical_encodings)
                                      .sum({3})
                                      .unsqueeze(3);
            torch::Tensor enc_b = (iterative_out.first.permute({0, 2, 3, 1}).slice(3, 32, 48) * spherical_encodings)
                                      .sum({3})
                                      .unsqueeze(3);

            //  enc_r = iterative_out.first.permute({0, 2, 3, 1}).slice(3, 0, 16).sum({3}).unsqueeze(3);
            //  enc_g = iterative_out.first.permute({0, 2, 3, 1}).slice(3, 16, 32).sum({3}).unsqueeze(3);
            //  enc_b = iterative_out.first.permute({0, 2, 3, 1}).slice(3, 32, 48).sum({3}).unsqueeze(3);
            result = torch::cat({enc_r, enc_g, enc_b}, 3).permute({0, 3, 1, 2});
        }


        {
            SAIGA_OPTIONAL_TIME_MEASURE("Final", timer_for_nets);

            result = final->forward(result);
        }
        // PrintTensorInfo(result);
        result = CenterCrop2D(result, inputs[0].sizes());
        // PrintTensorInfo(result);
        return result;
    }

    MultiScaleUnet2dParams params;

    torch::nn::Sequential final;
    // SmallDecStartBlock
    SmallDecStartBlock start = nullptr;

    SphericalHarmonicsEncoding spherical_harmonics;

    UpsampleDecOnlySmallBlockVarOutput up[8 - 1] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
};

TORCH_MODULE(MultiScaleUnet2dSphericalHarmonics);
#else
class UpsampleDecOnlySmallBlockVarOutputImpl : public torch::nn::Module
{
   public:
    UpsampleDecOnlySmallBlockVarOutputImpl(int in_channels, int out_channels, int num_input_channels,
                                           std::string conv_block, bool last, std::string upsample_mode = "deconv",
                                           std::string norm_str = "id", std::string activation = "id")
        : last(last)
    {
        SAIGA_ASSERT(in_channels > 0);
        SAIGA_ASSERT(out_channels > 0);

        std::vector<double> scale = {2.0, 2.0};

        // conv = GatedBlock(in_channels, out_channels);
        if (upsample_mode == "deconv")
        {
            up->push_back(torch::nn::ConvTranspose2d(
                torch::nn::ConvTranspose2dOptions(in_channels, out_channels, 4).stride(2).padding(1)));
        }
        else if (upsample_mode == "bilinear")
        {
            up->push_back(torch::nn::Upsample(
                torch::nn::UpsampleOptions().scale_factor(scale).mode(torch::kBilinear).align_corners(false)));
        }
        else if (upsample_mode == "nearest")
        {
            up->push_back(torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(scale).mode(torch::kNearest)));
            // up->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 3).padding(1)));
        }

        SAIGA_ASSERT(conv_block != "partial_multi");
        // conv1 = torch::nn::AnyModule(GatedBlock(in_channels, out_channels, 3, 1, 1, "id", "id"));
        //  conv = GatedBlock(out_channels * 2, out_channels, 3, 1, 1, norm_str);

        // last layer can output more channels, as no additionally input is added in final block
        const int conv_num_output_channels = (!last) ? out_channels - 2 * num_input_channels : 48;
        convolution =
            UnetBlockFromString(conv_block, in_channels, conv_num_output_channels, 3, 1, 1, norm_str, activation);

        register_module("up", up);
        register_module("convolution", convolution.ptr());
    }

    // Combines the upsampled tensor (below) with the skip connection (skip)
    // Usually this can be done with a simple cat however if the size does not match we crop
    torch::Tensor CombineBridge(torch::Tensor below, torch::Tensor skip)
    {
        if (below.size(2) == skip.size(2) && below.size(3) == skip.size(3))
            return torch::cat({below, skip}, 1);
        else
        {
            return torch::cat({below, torch::nn::functional::pad(
                                          skip, torch::nn::functional::PadFuncOptions(
                                                    {0, 0, below.size(2) - skip.size(2), below.size(3) - skip.size(3)})
                                                    .mode(torch::kConstant))},
                              1);
            // return torch::cat(
            //     {CenterCrop2D(below, std::vector<int64_t>(skip.sizes().begin(), skip.sizes().end())), skip}, 1);
        }
    }

    std::pair<at::Tensor, at::Tensor> forward(std::pair<at::Tensor, at::Tensor> layer_below,
                                              std::pair<at::Tensor, at::Tensor> features_input)
    {
        SAIGA_ASSERT(layer_below.first.defined());
        SAIGA_ASSERT(features_input.first.defined());

        // Upsample the layer from below
        std::pair<at::Tensor, at::Tensor> upsample_input;
        upsample_input.first = up->forward(layer_below.first);

        std::pair<at::Tensor, at::Tensor> combined_tensors;

        // [b, c, h, w]
        // output.first = torch::cat({same_layer_as_skip.first, skip.first}, 1);
        combined_tensors.first = CombineBridge(features_input.first, upsample_input.first);

        std::pair<at::Tensor, at::Tensor> output;
        output =
            convolution.forward<std::pair<at::Tensor, at::Tensor>>(combined_tensors.first, combined_tensors.second);

        if (!last) output.first = CombineBridge(features_input.first, output.first);

        return output;
    }

    torch::nn::Sequential up;
    torch::nn::AnyModule convolution;
    bool last;
};

TORCH_MODULE(UpsampleDecOnlySmallBlockVarOutput);

class MultiScaleUnet2dSphericalHarmonicsImpl : public torch::nn::Module
{
   public:
    MultiScaleUnet2dSphericalHarmonicsImpl(MultiScaleUnet2dParams params) : params(params), spherical_harmonics(4)
    {
        std::cout << "Using MultiScaleUnet2dSphericalHarmonicsImpl with filters: " << std::endl;

        // std::vector<int> filters = {32,32,32,32,32};
        std::vector<int> filters = params.filters_network;
        for (int i = 0; i < filters.size(); ++i)
        {
            std::cout << filters[i] << " ";
        }
        std::cout << std::endl;


        // output of block is [filter-num_input] (conv output [filter-2*num_input])
        const int out_first_conv = filters[params.num_layers - 1] - 2 * params.num_input_channels;
        start                    = SmallDecStartBlock(params.num_input_channels, out_first_conv, params.conv_block_up,
                                                      params.norm_layer_up, params.activation);
        register_module("start", start);

        // for (int i = 0; i < params.num_layers - 1; ++i)
        for (int i = params.num_layers - 2; i >= 0; --i)
        {
            // Up[i] transforms from layer (i+1) -> (i)
            int up_in  = filters[i + 1];
            int up_out = filters[i];
            up[i] = UpsampleDecOnlySmallBlockVarOutput(up_in, up_out, params.num_input_channels, params.conv_block_up,
                                                       (i == 0), params.upsample_mode, params.norm_layer_up,
                                                       params.activation);
            register_module("up" + std::to_string(i + 1), up[i]);
        }

        final->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(3, params.num_output_channels, 1)));
        final->push_back(ActivationFromString(params.last_act));
        register_module("final", final);


        // register_module("spherical_harmonics", spherical_harmonics);


        if (params.half_float)
        {
            this->to(torch::kFloat16);
        }
    }

    at::Tensor forward(ArrayView<torch::Tensor> inputs)
    {
        std::vector<torch::Tensor> masks(inputs.size());
        return forward(inputs, masks);
    }

    at::Tensor forward(ArrayView<torch::Tensor> inputs, ArrayView<torch::Tensor> masks)
    {
        SAIGA_ASSERT(inputs.size() == params.num_input_layers + 1);
        SAIGA_ASSERT(masks.size() == params.num_input_layers + 1);
        // The downsampling should not happen on uneven image sizes!
        // SAIGA_ASSERT(inputs.front().size(2) % (1 << params.num_layers) == 0);
        // SAIGA_ASSERT(inputs.front().size(3) % (1 << params.num_layers) == 0);
        // debug check if input has correct format
        // for (int i = 0; i < inputs.size(); ++i)
        //{
        //    if (params.num_input_layers > i)
        //    {
        //        SAIGA_ASSERT(inputs.size() > i);
        //        SAIGA_ASSERT(inputs[i].defined());
        //        SAIGA_ASSERT(params.num_input_channels == inputs[i].size(1));
        //    }
        //    SAIGA_ASSERT(masks[i].requires_grad() == false);
        //}



        std::pair<torch::Tensor, torch::Tensor> start_tensors =
            std::pair<at::Tensor, at::Tensor>(inputs[params.num_layers - 1], masks[params.num_layers - 1]);

        std::pair<torch::Tensor, torch::Tensor> iterative_out;
        {
            SAIGA_OPTIONAL_TIME_MEASURE("Start", timer_for_nets);

            iterative_out = start->forward(start_tensors.first, start_tensors.second);
        }
        // Loops Range: [layers-1, ... , 2, 1]
        for (int i = params.num_layers - 2; i >= 0; --i)
        {
            SAIGA_OPTIONAL_TIME_MEASURE("Layer " + std::to_string(i), timer_for_nets);

            std::pair<torch::Tensor, torch::Tensor> rendered_feature_maps_for_layer =
                std::pair<at::Tensor, at::Tensor>(inputs[i], masks[i]);
            iterative_out = up[i]->forward(iterative_out, rendered_feature_maps_for_layer);
        }

        //   PrintTensorInfo(inputs.back());
        torch::Tensor view_direction = inputs.back().permute({0, 2, 3, 1});

        // output: (b x h x w x 16)
        //  torch::Tensor spherical_encodings = spherical_harmonics.forward(view_direction);
        // PrintTensorInfo(view_direction);
        // PrintTensorInfo(spherical_encodings);
        // PrintTensorInfo(iterative_out.first.permute({0, 2, 3, 1}).slice(3, 0, 16));
        // PrintTensorInfo(iterative_out.first.permute({0, 2, 3, 1}).slice(3, 0, 16) * spherical_encodings);
        torch::Tensor spherical_encodings = spherical_harmonics.forward(view_direction);
        spherical_encodings               = spherical_encodings.to(iterative_out.first.dtype());

        torch::Tensor enc_r =
            (iterative_out.first.permute({0, 2, 3, 1}).slice(3, 0, 16) * spherical_encodings).sum({3}).unsqueeze(3);
        //   PrintTensorInfo(enc_r);
        torch::Tensor enc_g =
            (iterative_out.first.permute({0, 2, 3, 1}).slice(3, 16, 32) * spherical_encodings).sum({3}).unsqueeze(3);
        //    PrintTensorInfo(enc_g);

        torch::Tensor enc_b =
            (iterative_out.first.permute({0, 2, 3, 1}).slice(3, 32, 48) * spherical_encodings).sum({3}).unsqueeze(3);

        //  enc_r = iterative_out.first.permute({0, 2, 3, 1}).slice(3, 0, 16).sum({3}).unsqueeze(3);
        //  enc_g = iterative_out.first.permute({0, 2, 3, 1}).slice(3, 16, 32).sum({3}).unsqueeze(3);
        //  enc_b = iterative_out.first.permute({0, 2, 3, 1}).slice(3, 32, 48).sum({3}).unsqueeze(3);

        //        PrintTensorInfo(enc_b);


        torch::Tensor result = torch::cat({enc_r, enc_g, enc_b}, 3).permute({0, 3, 1, 2});
        {
            SAIGA_OPTIONAL_TIME_MEASURE("Final", timer_for_nets);

            result = final->forward(result);
        }
        //   PrintTensorInfo(result);

        return result;
    }

    MultiScaleUnet2dParams params;

    torch::nn::Sequential final;
    // SmallDecStartBlock
    SmallDecStartBlock start = nullptr;

    SphericalHarmonicsEncoding spherical_harmonics;

    UpsampleDecOnlySmallBlockVarOutput up[8 - 1] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
};

TORCH_MODULE(MultiScaleUnet2dSphericalHarmonics);

#endif


class UpsampleMultiSHImpl : public torch::nn::Module
{
   public:
    UpsampleMultiSHImpl(int in_channels, int out_channels, int num_input_channels, std::string conv_block, bool last,
                        std::string upsample_mode = "deconv", std::string norm_str = "id",
                        std::string activation = "id")
        : last(last)
    {
        SAIGA_ASSERT(in_channels > 0);
        SAIGA_ASSERT(out_channels > 0);



        std::vector<double> scale = {2.0, 2.0};

        // conv = GatedBlock(in_channels, out_channels);
        if (upsample_mode == "deconv")
        {
            up->push_back(torch::nn::ConvTranspose2d(
                torch::nn::ConvTranspose2dOptions(in_channels, out_channels, 4).stride(2).padding(1)));
        }
        else if (upsample_mode == "bilinear")
        {
            up->push_back(torch::nn::Upsample(
                torch::nn::UpsampleOptions().scale_factor(scale).mode(torch::kBilinear).align_corners(false)));
        }
        else if (upsample_mode == "nearest")
        {
            up->push_back(torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(scale).mode(torch::kNearest)));
            // up->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 3).padding(1)));
        }

        SAIGA_ASSERT(conv_block != "partial_multi");

        convolution = UnetBlockFromString(conv_block, in_channels, out_channels, 3, 1, 1, norm_str, activation);

        reorder_1x1_conv->push_back(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels + in_channels, out_channels + in_channels, 1)));


        register_module("up", up);
        register_module("convolution", convolution.ptr());
        register_module("reorder_1x1_conv", reorder_1x1_conv);
    }

    // Combines the upsampled tensor (below) with the skip connection (skip)
    // Usually this can be done with a simple cat however if the size does not match we crop
    torch::Tensor CombineBridge(torch::Tensor below, torch::Tensor skip)
    {
        if (below.size(2) == skip.size(2) && below.size(3) == skip.size(3))
            return torch::cat({below, skip}, 1);
        else
        {
            return torch::cat({below, torch::nn::functional::pad(
                                          skip, torch::nn::functional::PadFuncOptions(
                                                    {0, 0, below.size(2) - skip.size(2), below.size(3) - skip.size(3)})
                                                    .mode(torch::kConstant))},
                              1);
            // return torch::cat(
            //     {CenterCrop2D(below, std::vector<int64_t>(skip.sizes().begin(), skip.sizes().end())), skip}, 1);
        }
    }



    std::pair<at::Tensor, at::Tensor> forward(std::pair<at::Tensor, at::Tensor> layer_below,
                                              std::pair<at::Tensor, at::Tensor> features_input)
    {
        SAIGA_ASSERT(layer_below.first.defined());
        SAIGA_ASSERT(features_input.first.defined());

        // Upsample the layer from below
        std::pair<at::Tensor, at::Tensor> upsample_input;
        upsample_input.first = up->forward(layer_below.first);

        std::pair<at::Tensor, at::Tensor> combined_tensors;

        // [b, c, h, w]
        // output.first = torch::cat({same_layer_as_skip.first, skip.first}, 1);
        combined_tensors.first = CombineBridge(features_input.first, upsample_input.first);

        std::pair<at::Tensor, at::Tensor> conv_output;
        PrintTensorInfo(combined_tensors.first);

        conv_output =
            convolution.forward<std::pair<at::Tensor, at::Tensor>>(combined_tensors.first, combined_tensors.second);

        conv_output.first = CombineBridge(combined_tensors.first, conv_output.first);

        conv_output.first = reorder_1x1_conv->forward(conv_output.first);

        return conv_output;
    }

    torch::nn::Sequential up;
    torch::nn::AnyModule convolution;
    torch::nn::Sequential reorder_1x1_conv;


    bool last;
};

TORCH_MODULE(UpsampleMultiSH);

class MultiSphericalHarmonicsImpl : public torch::nn::Module
{
   public:
    MultiSphericalHarmonicsImpl(MultiScaleUnet2dParams params) : params(params)
    {
        std::cout << "Using MultiSphericalHarmonics with filters: " << std::endl;

        for (int i = 2; i < params.sh_bands; ++i)
        {
            spherical_harmonics.emplace_back(i);
        }

        // std::vector<int> filters = {32,32,32,32,32};
        std::vector<int> filters = params.filters_network;
        for (int i = 0; i < filters.size(); ++i)
        {
            std::cout << filters[i] << " ";
        }
        std::cout << std::endl;


        // output of block is [filter-num_input] (conv output [filter-2*num_input])
        const int out_first_conv = filters[params.num_layers - 1] - 2 * params.num_input_channels;
        start                    = SmallDecStartBlock(params.num_input_channels, out_first_conv, params.conv_block_up,
                                                      params.norm_layer_up, params.activation);
        register_module("start", start);


        for (int i = params.num_layers - 2; i > params.sh_bands - 2; --i)
        {
            // Up[i] transforms from layer (i+1) -> (i)
            int up_in  = filters[i + 1];
            int up_out = filters[i];
            up[i] = UpsampleDecOnlySmallBlockVarOutput(up_in, up_out, params.num_input_channels, params.conv_block_up,
                                                       false, params.upsample_mode, params.norm_layer_up,
                                                       params.activation);
            register_module("up" + std::to_string(i + 1), up[i]);
        }



        int prev_output = 28;
        for (int i = params.sh_bands - 2; i >= 0; --i)
        {
            int bands = params.sh_bands - i;
            // band 2
            int feat_in_conv     = prev_output + 4;
            int feat_out_of_conv = 32;
            if (bands == 3)
            {
                feat_in_conv     = 16 + 4;
                feat_out_of_conv = 88;
            }
            else if (bands == 4)
            {
                feat_in_conv     = 12 + 4;
                feat_out_of_conv = 32;
            }
            sh_up[i] = UpsampleMultiSH(feat_in_conv, feat_out_of_conv, params.num_input_channels, params.conv_block_up,
                                       false, params.upsample_mode, params.norm_layer_up, params.activation);

            register_module("sh_up" + std::to_string(i + 1), sh_up[i]);
        }


        dir_downsampler->push_back(torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions({2, 2})));

        final->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(3, params.num_output_channels, 1)));
        final->push_back(ActivationFromString(params.last_act));
        register_module("final", final);


        //  register_module("spherical_harmonics", spherical_harmonics.module);


        if (params.half_float)
        {
            this->to(torch::kFloat16);
        }
    }

    at::Tensor forward(ArrayView<torch::Tensor> inputs)
    {
        std::vector<torch::Tensor> masks(inputs.size());
        return forward(inputs, masks);
    }

    at::Tensor forward(ArrayView<torch::Tensor> inputs, ArrayView<torch::Tensor> masks)
    {
        SAIGA_ASSERT(inputs.size() == params.num_input_layers + 1);
        SAIGA_ASSERT(masks.size() == params.num_input_layers + 1);

        std::pair<torch::Tensor, torch::Tensor> start_tensors =
            std::pair<at::Tensor, at::Tensor>(inputs[params.num_layers - 1], masks[params.num_layers - 1]);

        std::pair<torch::Tensor, torch::Tensor> iterative_out;
        {
            SAIGA_OPTIONAL_TIME_MEASURE("Start", timer_for_nets);

            iterative_out = start->forward(start_tensors.first, start_tensors.second);
        }
        // Loops Range: [layers-1, ... , 2, 1]
        for (int i = params.num_layers - 2; i > params.sh_bands - 2; --i)
        {
            SAIGA_OPTIONAL_TIME_MEASURE("Layer " + std::to_string(i), timer_for_nets);
            std::cout << i << TensorInfo(iterative_out.first) << std::endl;
            std::pair<torch::Tensor, torch::Tensor> rendered_feature_maps_for_layer =
                std::pair<at::Tensor, at::Tensor>(inputs[i], masks[i]);
            iterative_out = up[i]->forward(iterative_out, rendered_feature_maps_for_layer);
            std::cout << i << TensorInfo(iterative_out.first) << std::endl;
        }

        // 28 feature channels
        torch::Tensor view_direction = inputs.back();

        std::vector<torch::Tensor> downsampled_dirs;
        downsampled_dirs.push_back(view_direction);
        for (int i = 1; i <= params.sh_bands - 2; ++i)
        {
            downsampled_dirs.push_back(dir_downsampler->forward(downsampled_dirs[i - 1]));
        }

        for (int i = params.sh_bands - 2; i >= 0; --i)
        {
            SAIGA_OPTIONAL_TIME_MEASURE("Layer SH " + std::to_string(i), timer_for_nets);
            std::cout << i << TensorInfo(iterative_out.first) << std::endl;
            std::pair<torch::Tensor, torch::Tensor> rendered_feature_maps_for_layer =
                std::pair<at::Tensor, at::Tensor>(inputs[i], masks[i]);
            iterative_out = sh_up[i]->forward(iterative_out, rendered_feature_maps_for_layer);
            std::cout << i << TensorInfo(iterative_out.first) << std::endl;
            // TODO out: fixed features:
            //  2: 72
            //  1: 108
            //  0: 48
            auto& spherical_harms = spherical_harmonics[params.sh_bands - 2 - i];
            auto& view_dir        = downsampled_dirs[i];
            std::cout << "viewdir" << TensorInfo(view_dir.permute({0, 2, 3, 1})) << std::endl;
            torch::Tensor sh_enc = spherical_harms.forward(view_dir.permute({0, 2, 3, 1}));
            int band_output      = (params.sh_bands - i) * (params.sh_bands - i);

            auto input       = iterative_out.first.permute({0, 2, 3, 1});
            int features_out = input.size(3) / band_output;

            input = input * sh_enc.repeat_interleave(features_out, 3);
            input = input.reshape({input.size(0), input.size(1), input.size(2), features_out, band_output}).sum({-1});

            iterative_out.first = input.permute({0, 3, 1, 2});
            std::cout << i << TensorInfo(iterative_out.first) << std::endl;
        }


        //   PrintTensorInfo(inputs.back());
        torch::Tensor result = iterative_out.first;


        {
            SAIGA_OPTIONAL_TIME_MEASURE("Final", timer_for_nets);

            result = final->forward(result);
        }
        //   PrintTensorInfo(result);

        return result;
    }

    MultiScaleUnet2dParams params;

    torch::nn::Sequential final;
    // SmallDecStartBlock
    SmallDecStartBlock start = nullptr;

    std::vector<SphericalHarmonicsEncoding> spherical_harmonics;

    torch::nn::Sequential dir_downsampler;

    UpsampleDecOnlySmallBlockVarOutput up[8 - 1] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    UpsampleMultiSH sh_up[8 - 1]                 = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
};

TORCH_MODULE(MultiSphericalHarmonics);
