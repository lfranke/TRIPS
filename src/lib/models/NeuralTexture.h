/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/core/model/UnifiedMesh.h"
#include "saiga/core/util/FileSystem.h"
#include "saiga/core/util/file.h"
#include "saiga/cuda/cudaTimer.h"
#include "saiga/vision/torch/TorchHelper.h"

#include "rendering/NeuralPointCloudCuda.h"

using namespace Saiga;

// [num_channels, num_points]

class NeuralPointTextureImpl : public torch::nn::Module
{
   public:
    NeuralPointTextureImpl(int num_channels, int num_points, bool random_init, bool log_texture,
                           int num_spherical_harmonics_bands_per_point = -1, bool use_stability = false);

    NeuralPointTextureImpl(const Saiga::UnifiedMesh& model, int channels = 4, int num_points = -1);

    void SetFirstChannelsAsRGB(const UnifiedMesh& model, int num_points = 0);
    std::vector<float> GetDescriptorSlow(int i)
    {
        std::vector<float> desc;
        for (int j = 0; j < texture_raw.size(0); ++j)
        {
            float f = texture_raw[j][i].item().toFloat();
            desc.push_back(f);
        }
        return desc;
    }

    void PrepareConfidence(float narrowing_param_times_epoch, int num_layers = 4)
    {
        confidence_value_of_point = torch::sigmoid((10.f + narrowing_param_times_epoch) * confidence_raw);
    }

    void PrepareTexture(bool abs)
    {
        if (abs)
        {
            texture          = torch::abs(texture_raw);
            background_color = torch::abs(background_color_raw);
        }
        else
        {
            texture          = texture_raw;
            background_color = background_color_raw;
        }
    }

    std::vector<float> GetBackgroundColor()
    {
        std::vector<float> desc;
        for (int j = 0; j < background_color.size(0); ++j)
        {
            float f = background_color[j].item().toFloat();
            desc.push_back(f);
        }
        return desc;
    }

    void SetBackgroundColor(std::vector<float> col)
    {
        torch::NoGradGuard ngg;
        background_color_raw.set_(
            torch::from_blob(col.data(), {(long)col.size()}).to(background_color_raw.device()).clone());
    }

    int NumPoints() { return texture_raw.size(1); }
    int TextureChannels()
    {
        SAIGA_ASSERT(texture_raw.dim() == 2);
        return texture_raw.size(0);
    }
    torch::Tensor ResizeAndFill(torch::Tensor& t, int new_point_size, bool fill_with_zeros = false);

    void RemoveAndFlatten(torch::Tensor& t, torch::Tensor& indices_to_keep);

    void RemoveAndFlattenTexture(torch::Tensor& indices_to_remove)
    {
        RemoveAndFlatten(texture_raw, indices_to_remove);
        RemoveAndFlatten(confidence_value_of_point, indices_to_remove);
        RemoveAndFlatten(confidence_raw, indices_to_remove);
        std::cout << "Removed and flattened Texture to " << texture.sizes() << std::endl;
    }

    // increases size of all buffers and initializes data
    void EnlargeTexture(int new_num_points)
    {
        ResizeAndFill(texture_raw, new_num_points);

        confidence_value_of_point.reset();
        texture.reset();
        ResizeAndFill(confidence_raw, new_num_points, true);

        std::cout << "Resized Texture to " << texture_raw.sizes() << std::endl;
    }

    void to_except_texture(torch::Device dev)
    {
        confidence_value_of_point = confidence_value_of_point.to(dev);
        background_color          = background_color.to(dev);
    }

    void reorderDimensions()
    {
        auto reorder = [&](torch::Tensor& tex)
        {
            torch::NoGradGuard ngg;
            auto tex_new = tex.permute({1, 0}).contiguous();
            tex_new      = tex_new.permute({1, 0});  //.to(torch::kCUDA);
            tex.set_(tex_new);
        };
        reorder(texture_raw);
        reorder(confidence_raw);

        reorder(confidence_value_of_point);
        reorder(texture);
    }

    void setLayerFromPointCloud(NeuralPointCloudCuda& npcc);

    bool log_texture;
    bool random_init;

    // [channels, points]
    torch::Tensor texture;
    torch::Tensor texture_raw;

    // [1, points]
    torch::Tensor confidence_value_of_point;
    torch::Tensor confidence_raw;

    // [channels]
    torch::Tensor background_color_raw;
    torch::Tensor background_color;
};

TORCH_MODULE(NeuralPointTexture);
