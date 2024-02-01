/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "RenderModule.h"

#include <torch/torch.h>
PointRenderModuleImpl::PointRenderModuleImpl(std::shared_ptr<CombinedParams> params)
    : params(params), num_layers(params->net_params.num_input_layers)
{
    cache = std::make_shared<PointRendererCache>();
}

std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>> PointRenderModuleImpl::forward(
    NeuralScene& scene, const std::vector<NeuralTrainData>& batch, int current_epoch,
    CUDA::CudaTimerSystem* timer_system)
{
    NeuralRenderInfo render_data;
    render_data.scene         = &scene;
    render_data.num_layers    = num_layers;
    render_data.params        = params->render_params;
    render_data.timer_system  = timer_system;
    render_data.current_epoch = current_epoch;

    if (!this->is_training())
    {
        render_data.params.dropout = 0;
        render_data.train          = false;
    }
    else
    {
        render_data.train = true;
    }

    for (auto& b : batch)
    {
        render_data.images.push_back(b->img);
    }

    return forward(&render_data);
}

std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>> PointRenderModuleImpl::forward(NeuralRenderInfo* nri)
{
    if (0)
    {
        auto poses = nri->scene->poses->Download();
        auto ks    = nri->scene->intrinsics->DownloadK();
        for (auto i : nri->images)
        {
            std::cout << "Render (" << i.camera_index << ", " << i.image_index << ") Pose: " << poses[i.image_index]
                      << " K: " << ks[i.camera_index] << std::endl;
        }
    }

    nri->cache = cache.get();

    if (params->render_params.super_sampling)
    {
        for (auto& i : nri->images)
        {
            i.w *= 2;
            i.h *= 2;
            i.crop_transform = i.crop_transform.scale(2);
        }
    }


    auto combined_images_masks = BlendPointCloud(nri);


    std::vector<torch::Tensor> images(combined_images_masks.begin(), combined_images_masks.begin() + nri->num_layers);

    std::vector<torch::Tensor> depths;
    if (params->render_params.add_depth_to_network)
        depths = std::vector<torch::Tensor>(combined_images_masks.begin() + nri->num_layers,
                                            combined_images_masks.begin() + 2 * nri->num_layers);

    std::vector<torch::Tensor> blendPointMasks;

    if (params->render_params.output_background_mask)
    {
        if (params->render_params.add_depth_to_network)
        {
            SAIGA_ASSERT(combined_images_masks.size() == nri->num_layers * 3);
            blendPointMasks = std::vector<torch::Tensor>(combined_images_masks.begin() + 2 * nri->num_layers,
                                                         combined_images_masks.end());
        }
        else
        {
            SAIGA_ASSERT(combined_images_masks.size() == nri->num_layers * 2);
            blendPointMasks = std::vector<torch::Tensor>(combined_images_masks.begin() + nri->num_layers,
                                                         combined_images_masks.end());
        }
        for (auto& m : blendPointMasks)
        {
            m.detach_();
        }
        SAIGA_ASSERT(!blendPointMasks.front().requires_grad());
    }

    if (params->render_params.super_sampling)
    {
        for (auto& img : images)
        {
            img = torch::avg_pool2d(img, {2, 2});
        }

        for (auto& img : depths)
        {
            img = torch::avg_pool2d(img, {2, 2});
            std::cout << "think about this: " << __LINE__ << std::endl;
        }

        for (auto& img : blendPointMasks)
        {
            img = torch::avg_pool2d(img, {2, 2});
        }

        for (auto& i : nri->images)
        {
            i.w /= 2;
            i.h /= 2;
            i.crop_transform = i.crop_transform.scale(0.5);
        }
    }

    if (nri->scene->environment_map && params->pipeline_params.enable_environment_map &&
        !params->pipeline_params.environment_map_params.use_points_for_env_map)
    {
        //  SAIGA_OPTIONAL_TIME_MEASURE("Environment Map", nri->timer_system);
        std::vector<std::vector<torch::Tensor>> layers(cache->layers_cuda.size());
        for (int i = 0; i < cache->layers_cuda.size(); ++i)
        {
            layers[i] = std::vector<torch::Tensor>(cache->num_batches);
            std::vector<torch::Tensor> dt_masking_tensors;
            for (int b = 0; b < cache->num_batches; ++b)
            {
                layers[i][b] = cache->layers_cuda[i].weight.slice(0, b, b + 1).clone();

                // for depth testing, either weight is zero, thus alpha_dest should be 1 or weight is >0 thus
                // accumulation is complete; also mask out background color
                if (cache->render_mode == PointRendererCache::RenderMode::FUZZY_DT)
                {
                    dt_masking_tensors.push_back(torch::where(layers[i][b] > 0.5, torch::ones_like(layers[i][b]),
                                                              torch::zeros_like(layers[i][b])));
                    layers[i][b] = torch::where(layers[i][b] > 0.5, torch::zeros_like(layers[i][b]),
                                                torch::ones_like(layers[i][b]));
                }
            }
            if (cache->render_mode == PointRendererCache::RenderMode::FUZZY_DT)
                images[i] = images[i] * torch::cat(dt_masking_tensors, 0);
        }

        SAIGA_ASSERT(params->render_params.output_background_mask);

        static bool use_new_sampling = true;
        // ImGui::Checkbox("use new sampling", &use_new_sampling);

        std::vector<torch::Tensor> env_maps;
        if (use_new_sampling)
        {
            env_maps = nri->scene->environment_map->Sample2(
                nri->scene->poses->poses_se3, nri->scene->intrinsics->intrinsics, nri->images, nri->num_layers,
                nri->scene->scene, layers, nri->timer_system);
        }
        else
        {
            env_maps = nri->scene->environment_map->Sample(
                nri->scene->poses->poses_se3, nri->scene->intrinsics->intrinsics, nri->images, nri->num_layers,
                nri->scene->scene, layers, nri->timer_system);
        }

        if (params->pipeline_params.environment_map_params.cat_env_to_color)
        {
            for (int i = 0; i < nri->num_layers; ++i)
            {
                images[i] = torch::cat({images[i], env_maps[i]}, 1);
            }
        }
        else
        {
            for (int i = 0; i < nri->num_layers; ++i)
            {
                torch::Tensor env_map_masking = torch::ones_like(env_maps[i]);
                if (params->render_params.no_envmap_at_points)
                {
                    std::vector<torch::Tensor> masking;
                    for (int b = 0; b < cache->num_batches; ++b)
                    {
                        layers[i][b] = cache->layers_cuda[i].weight.slice(0, b, b + 1).clone();

                        masking.push_back(torch::where(layers[i][b] > 0.001, torch::zeros_like(layers[i][b]),
                                                       torch::ones_like(layers[i][b])));
                    }
                    env_map_masking =
                        torch::cat(masking, 0).repeat({1, params->render_params.num_texture_channels, 1, 1});
                }
                images[i] = images[i] + env_map_masking * env_maps[i];
            }
        }
    }

    if (params->pipeline_params.cat_masks_to_color)
    {
        for (int i = 0; i < nri->num_layers; ++i)
        {
            images[i] = torch::cat({images[i], blendPointMasks[i]}, 1);
        }
    }

#if 0
        for (int i = 0; i < nri->num_layers; ++i)
        {
            using namespace torch::indexing;
            std::cout << images[i].sizes() << ", " << depths[i].sizes() << std::endl;
            auto img = Saiga::TensorToImage<ucvec3>(images[i].index({0,Slice(0,3),Slice(),Slice()}));
            img.save("./img_"+ std::to_string(i) + ".png");
        }
#endif

    if (params->render_params.add_depth_to_network)
    {
        for (int i = 0; i < nri->num_layers; ++i)
        {
#if 0

            auto dep = Saiga::TensorToImage<ucvec3>(depths[i].expand({-1,3,-1,-1})/10.0);
            dep.save("./dep_"+ std::to_string(i) + ".png");
#endif
            images[i] = torch::cat({images[i], depths[i]}, 1);
        }
    }



    // SAIGA_ASSERT(images.front().size(0) == nri->images.size());
    SAIGA_ASSERT(images.front().size(2) == nri->images.front().h);
    SAIGA_ASSERT(images.front().size(3) == nri->images.front().w);



    return {images, blendPointMasks};
}
