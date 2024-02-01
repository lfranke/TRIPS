/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "NeuralTexture.h"

#include "spherical_harmonics.h"


using namespace Saiga;

NeuralPointTextureImpl::NeuralPointTextureImpl(int num_channels, int num_points, bool random_init, bool log_texture,
                                               int num_spherical_harmonics_bands_per_point, bool use_stability)
    : log_texture(log_texture), random_init(random_init)
{
    float fac_init = 0.25f;
    int sh_bands   = 1;
    if (num_spherical_harmonics_bands_per_point > 0) sh_bands = num_spherical_harmonics_bands_per_point;
    if (random_init)
    {
        // Random init in the range [-factor/2, factor/2]
        float factor = 2 * fac_init;
        texture_raw  = (torch::rand({num_channels * sh_bands * sh_bands, num_points}) * factor);
        texture_raw  = texture_raw - torch::ones_like(texture_raw) * (factor / 2);

        texture_raw = torch::empty({num_channels * sh_bands * sh_bands, num_points});
        texture_raw.uniform_(0, 1);
    }
    else
    {
        texture_raw = torch::ones({num_channels * sh_bands * sh_bands, num_points}) * 0.5f * fac_init;
    }
    texture = texture_raw.clone();

    background_color_raw = torch::ones({num_channels}) * 1 * fac_init;
    // background_color_raw = torch::ones({num_channels}) * 0;

    constexpr int channels_conf = 1;
    confidence_value_of_point   = torch::ones({channels_conf, num_points}) * 0.5;
    confidence_raw              = torch::ones({channels_conf, num_points}) * 0.5;

    if (log_texture)
    {
        std::cout << "initialized first texture channel (before checkpoint loading) with 1" << std::endl;
        texture_raw      = torch::log(torch::ones({num_channels, num_points}) * 0.5 * fac_init);
        background_color = torch::ones({num_channels}) * 1;
    }


    // register_parameter("texture", texture);
    register_parameter("texture", texture_raw);
    register_parameter("background_color", background_color_raw);
    register_parameter("confidence_value_of_point", confidence_raw);


    std::cout << "GPU memory - Texture " << texture_raw.sizes() << " : "
              << (texture_raw.nbytes() + confidence_value_of_point.nbytes() + confidence_raw.nbytes() +
                  background_color_raw.nbytes()) /
                     1000000.0
              << "MB" << std::endl;
}



NeuralPointTextureImpl::NeuralPointTextureImpl(const UnifiedMesh& model, int channels, int num_points)
    : NeuralPointTextureImpl(channels, (num_points > 0) ? num_points : model.NumVertices(), false, false)
{
    int num_p = (num_points > 0) ? num_points : model.NumVertices();
    if (channels == 3)
    {
        std::vector<vec3> colors;
        for (auto c : model.color)
        {
            if (colors.size() < num_p) colors.push_back(c.head<3>());
        }
        for (int i = colors.size(); i < num_p; ++i)
        {
            colors.push_back(vec3(1, 1, 1));
        }

        auto t =
            torch::from_blob(colors.data(), {(long)colors.size(), 3}, torch::TensorOptions().dtype(torch::kFloat32))
                .to(texture_raw.device())
                .permute({1, 0})
                .contiguous();
        SAIGA_ASSERT(t.sizes() == texture_raw.sizes());

        {
            torch::NoGradGuard ngg;
            texture_raw.set_(t);
        }

        SetBackgroundColor({0, 0, 0});
    }
    else if (channels == 4)
    {
        std::vector<vec4> colors;
        for (auto c : model.color)
        {
            c(3) = 1;
            if (colors.size() < num_p) colors.push_back(c.head<4>());
        }
        for (int i = model.color.size(); i < num_p; ++i)
        {
            colors.push_back(vec4(1, 1, 1, 0));
        }


        auto t =
            torch::from_blob(colors.data(), {(long)colors.size(), 4}, torch::TensorOptions().dtype(torch::kFloat32))
                .to(texture_raw.device())
                .permute({1, 0})
                .contiguous();
        SAIGA_ASSERT(t.sizes() == texture_raw.sizes());

        {
            torch::NoGradGuard ngg;
            texture_raw.set_(t);
        }

        SetBackgroundColor({0, 0, 0, 1});
    }
}
void NeuralPointTextureImpl::SetFirstChannelsAsRGB(const UnifiedMesh& model, int num_points)
{
    int num_p = (num_points > 0) ? num_points : model.NumVertices();

    std::vector<vec3> colors;
    for (auto c : model.color)
    {
        if (colors.size() < num_p) colors.push_back(c.head<3>());
    }
    for (int i = colors.size(); i < num_p; ++i)
    {
        colors.push_back(vec3(1, 1, 1));
    }

    auto t = torch::from_blob(colors.data(), {(long)colors.size(), 3}, torch::TensorOptions().dtype(torch::kFloat32))
                 .to(texture_raw.device())
                 .permute({1, 0})
                 .contiguous();
    // SAIGA_ASSERT(t.sizes() == texture_raw.sizes());

    {
        torch::NoGradGuard ngg;
        using namespace torch::indexing;
        texture_raw.index_put_({Slice(0, 3)}, t);
    }
}

torch::Tensor NeuralPointTextureImpl::ResizeAndFill(torch::Tensor& t, int new_point_size, bool fill_with_zeros)
{
    using namespace torch::indexing;
    SAIGA_ASSERT(t.sizes().size() == 2 || t.sizes().size() == 3);

    torch::Tensor new_vals;
    if (t.sizes().size() == 2)
    {
        new_vals = torch::ones({t.sizes()[0], long(new_point_size) - t.sizes()[1]}, t.options()) * 0.5;
    }
    else if (t.sizes().size() == 3)
    {
        new_vals = torch::ones({t.sizes()[0], t.sizes()[1], new_point_size - t.sizes()[2]}, t.options()) * 0.5;
    }
    if (fill_with_zeros)
    {
        new_vals *= 0;
    }
    else if (random_init)
    {
        float factor = 2;
        new_vals     = (torch::rand_like(new_vals) * factor) - (new_vals * (2 * (factor / 2)));
    }

    torch::Tensor t_n;
    if (t.sizes().size() == 2)
    {
        t_n = torch::cat({t.clone(), new_vals}, 1);
    }
    else
    {
        t_n = torch::cat({t.clone(), new_vals}, 2);
    }

    {
        torch::NoGradGuard ngg;
        t.set_(t_n);
    }

    return new_vals;
}

void NeuralPointTextureImpl::RemoveAndFlatten(torch::Tensor& t, torch::Tensor& indices_to_keep)
{
    using namespace torch::indexing;
    {
        torch::NoGradGuard ngg;
        // std::cout << TensorInfo(indices_to_keep.squeeze().unsqueeze(0)) << std::endl;
        // std::cout << TensorInfo(t) << std::endl;
        // auto values_keep = t.index({indices_to_keep.squeeze().unsqueeze(0) });
        auto values_keep = t.index_select(t.sizes().size() - 1, indices_to_keep.squeeze().to(t.device()));
        // std::cout << TensorInfo(values_keep) << std::endl;

        t.set_(values_keep);
    }
}
