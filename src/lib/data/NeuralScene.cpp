/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "NeuralScene.h"

#include "saiga/core/util/ProgressBar.h"

#include "models/MyAdam.h"



NeuralScene::NeuralScene(std::shared_ptr<SceneData> scene, std::shared_ptr<CombinedParams> _params, bool eval_only)
    : scene(scene), params(_params)
{
    params->Check();

    {
        params->render_params.dist_cutoff = scene->scene_cameras.front().distortion.MonotonicThreshold();
    }


    SAIGA_ASSERT(scene);

    // ========== Create Modules ==========

    AABB custom_aabb = scene->dataset_params.aabb;

    point_cloud_cuda = NeuralPointCloudCuda(scene->point_cloud, params->points_adding_params.use_grid_loss,
                                            params->points_adding_params.cells_worldspace_size, custom_aabb, true);

    SAIGA_ASSERT(point_cloud_cuda->t_normal.defined() || !params->render_params.check_normal);


    std::vector<float> exposures;
    for (auto& f : scene->frames) exposures.push_back(f.exposure_value - scene->dataset_params.scene_exposure_value);

    std::vector<vec3> wbs;
    for (auto& f : scene->frames) wbs.push_back(f.white_balance);

    poses      = PoseModule(scene);
    intrinsics = IntrinsicsModule(scene);
    camera = NeuralCamera(ivec2(scene->scene_cameras.front().w, scene->scene_cameras.front().h), params->camera_params,
                          scene->frames.size(), exposures, wbs);

    if (params->pipeline_params.enable_environment_map &&
        !params->pipeline_params.environment_map_params.use_points_for_env_map)
    {
        environment_map = EnvironmentMap(params->pipeline_params.environment_map_params.env_map_channels,
                                         params->pipeline_params.environment_map_params.env_map_resolution,
                                         params->pipeline_params.environment_map_params.env_map_resolution,
                                         params->pipeline_params.log_texture, 0,
                                         params->pipeline_params.environment_map_params.env_spheres,
                                         params->pipeline_params.environment_map_params.env_inner_radius,
                                         params->pipeline_params.environment_map_params.env_radius_factor,
                                         params->pipeline_params.non_subzero_texture);
    }



    SAIGA_ASSERT(scene->point_cloud.NumVertices() > 0);

    texture = NeuralPointTexture(params->pipeline_params.num_texture_channels, scene->point_cloud.NumVertices(),
                                 params->train_params.texture_random_init, params->pipeline_params.log_texture,
                                 params->pipeline_params.num_spherical_harmonics_bands_per_point,
                                 (params->pipeline_params.use_point_adding_and_removing_module));

    if (params->train_params.texture_color_init)
    {
        std::cout << "Using point color as texture" << std::endl;
        texture->SetFirstChannelsAsRGB(scene->point_cloud);
    }

    if (params->pipeline_params.enable_environment_map &&
        params->pipeline_params.environment_map_params.use_points_for_env_map &&
        params->pipeline_params.environment_map_params.start_with_environment_points && !eval_only)
    //||        std::filesystem::exists(params->train_params.checkpoint_directory + "/texture.pth"))
    {
        {
            AddNewRandomForEnvSphere(params->pipeline_params.environment_map_params.env_spheres,
                                     params->pipeline_params.environment_map_params.env_inner_radius,
                                     params->pipeline_params.environment_map_params.env_radius_factor,
                                     params->pipeline_params.environment_map_params.env_num_points, false);
        }
    }

    std::string checkpoint_prefix = params->train_params.checkpoint_directory + "/scene_" + scene->scene_name + "_";
    if (point_cloud_cuda && std::filesystem::exists(checkpoint_prefix + "points.pth"))
    {
        try
        {
            std::cout << "Try loading point cloud checkpoint" << std::endl;
            torch::load(point_cloud_cuda, checkpoint_prefix + "points.pth");
        }
        catch (c10::Error e)
        {
            std::cout << "catch load and set point sizes to "
                      << ((!params->render_params.use_layer_point_size) ? "True" : "False") << std::endl;
            point_cloud_cuda =
                NeuralPointCloudCuda(scene->point_cloud, params->points_adding_params.use_grid_loss,
                                     params->points_adding_params.cells_worldspace_size, custom_aabb, false);
        }
    }
    LoadCheckpoint(params->train_params.checkpoint_directory);
    if (!params->pipeline_params.train) texture->reorderDimensions();

    camera->eval();
    camera->to(device);

    if (params->net_params.half_float)
    {
        camera->to(torch::kFloat16);
    }

    // ========== Create Optimizers ==========
    if (!eval_only)
    {
        {
            CreateTextureOptimizer();
        }

        {
            if (params->optimizer_params.use_myadam_everywhere)
            {
                using TexOpt   = torch::optim::MyAdamOptions;
                using TexOptim = torch::optim::MyAdam;
                std::vector<torch::optim::OptimizerParamGroup> g;

                if (params->camera_params.enable_response && !params->optimizer_params.fix_response)
                {
                    std::cout << "optimizing response with lr " << params->optimizer_params.lr_response << std::endl;
                    auto opt = std::make_unique<TexOpt>(params->optimizer_params.lr_response);
                    g.emplace_back(camera->camera_response->parameters(), std::move(opt));
                }
                if (params->camera_params.enable_white_balance && !params->optimizer_params.fix_wb)
                {
                    std::cout << "optimizing white balance with lr " << params->optimizer_params.lr_wb << std::endl;
                    auto opt = std::make_unique<TexOpt>(params->optimizer_params.lr_wb);
                    std::vector<torch::Tensor> ts;
                    ts.push_back(camera->white_balance_values);
                    g.emplace_back(ts, std::move(opt));
                }

                if (params->camera_params.enable_exposure && !params->optimizer_params.fix_exposure)
                {
                    std::cout << "optimizing exposure with lr " << params->optimizer_params.lr_exposure << std::endl;
                    auto opt = std::make_unique<TexOpt>(params->optimizer_params.lr_exposure);
                    std::vector<torch::Tensor> ts;
                    ts.push_back(camera->exposures_values);
                    g.emplace_back(ts, std::move(opt));
                }


                if (params->camera_params.enable_vignette && !params->optimizer_params.fix_vignette)
                {
                    std::cout << "optimizing vignette with lr " << params->optimizer_params.lr_vignette << std::endl;
                    auto opt = std::make_unique<TexOpt>(params->optimizer_params.lr_vignette);
                    g.emplace_back(camera->vignette_net->parameters(), std::move(opt));
                }

                if (params->camera_params.enable_rolling_shutter && !params->optimizer_params.fix_rolling_shutter)
                {
                    std::cout << "optimizing rolling shutter with lr " << params->optimizer_params.lr_rolling_shutter
                              << std::endl;
                    auto opt = std::make_unique<TexOpt>(params->optimizer_params.lr_rolling_shutter);
                    g.emplace_back(camera->rolling_shutter->parameters(), std::move(opt));
                }
                camera_adam_optimizer = std::make_shared<TexOptim>(g, TexOpt(1));
            }
            else
            {
                std::vector<torch::optim::OptimizerParamGroup> g_cam_adam, g_cam_sgd;

                if (params->camera_params.enable_response && !params->optimizer_params.fix_response)
                {
                    std::cout << "optimizing response with lr " << params->optimizer_params.lr_response << std::endl;
                    auto opt = std::make_unique<torch::optim::AdamOptions>(params->optimizer_params.lr_response);
                    g_cam_adam.emplace_back(camera->camera_response->parameters(), std::move(opt));
                }

                if (params->camera_params.enable_white_balance && !params->optimizer_params.fix_wb)
                {
                    std::cout << "optimizing white balance with lr " << params->optimizer_params.lr_wb << std::endl;
                    auto opt = std::make_unique<torch::optim::AdamOptions>(params->optimizer_params.lr_wb);
                    std::vector<torch::Tensor> ts;
                    ts.push_back(camera->white_balance_values);
                    g_cam_adam.emplace_back(ts, std::move(opt));
                }

                if (params->camera_params.enable_exposure && !params->optimizer_params.fix_exposure)
                {
                    std::cout << "optimizing exposure with lr " << params->optimizer_params.lr_exposure << std::endl;
                    auto opt = std::make_unique<torch::optim::SGDOptions>(params->optimizer_params.lr_exposure);
                    std::vector<torch::Tensor> ts;
                    ts.push_back(camera->exposures_values);
                    g_cam_sgd.emplace_back(ts, std::move(opt));
                }


                if (params->camera_params.enable_vignette && !params->optimizer_params.fix_vignette)
                {
                    std::cout << "optimizing vignette with lr " << params->optimizer_params.lr_vignette << std::endl;
                    auto opt = std::make_unique<torch::optim::SGDOptions>(params->optimizer_params.lr_vignette);
                    g_cam_sgd.emplace_back(camera->vignette_net->parameters(), std::move(opt));
                }

                if (params->camera_params.enable_rolling_shutter && !params->optimizer_params.fix_rolling_shutter)
                {
                    std::cout << "optimizing rolling shutter with lr " << params->optimizer_params.lr_rolling_shutter
                              << std::endl;
                    auto opt = std::make_unique<torch::optim::SGDOptions>(params->optimizer_params.lr_rolling_shutter);
                    g_cam_sgd.emplace_back(camera->rolling_shutter->parameters(), std::move(opt));
                }

                if (!g_cam_adam.empty())
                {
                    camera_adam_optimizer =
                        std::make_shared<torch::optim::Adam>(g_cam_adam, torch::optim::AdamOptions(1));
                }
                if (!g_cam_sgd.empty())
                {
                    camera_sgd_optimizer = std::make_shared<torch::optim::SGD>(g_cam_sgd, torch::optim::SGDOptions(1));
                }
            }
        }

        {
            CreateStructureOptimizer();
        }
    }
}


void NeuralScene::CreateStructureOptimizer()
{
    if (params->optimizer_params.use_myadam_everywhere)
    {
        std::cout << "Optimizing with my adam implementation." << std::endl;
        using TexOpt   = torch::optim::MyAdamOptions;
        using TexOptim = torch::optim::MyAdam;
        std::vector<torch::optim::OptimizerParamGroup> g;

        if (!params->optimizer_params.fix_points)
        {
            std::cout << "optimizing 3D points with lr " << params->optimizer_params.lr_points << std::endl;
            auto opt = std::make_unique<TexOpt>(params->optimizer_params.lr_points);
            std::vector<torch::Tensor> ts;
            ts.push_back(point_cloud_cuda->t_position);
            g.emplace_back(ts, std::move(opt));
        }
        else
        {
            point_cloud_cuda->t_position.set_requires_grad(false);
        }
        if (!params->optimizer_params.fix_point_size)
        {
            std::cout << "optimizing point size with lr " << params->optimizer_params.lr_layer << std::endl;

            point_cloud_cuda->t_point_size.set_requires_grad(true);
            auto opt_t = std::make_unique<TexOpt>(params->optimizer_params.lr_layer);
            std::vector<torch::Tensor> ts;
            ts.push_back(point_cloud_cuda->t_point_size);
            g.emplace_back(ts, std::move(opt_t));
        }
        else
        {
            point_cloud_cuda->t_point_size.set_requires_grad(false);
        }

        if (!params->optimizer_params.fix_poses)
        {
            std::cout << "optimizing poses with lr " << params->optimizer_params.lr_poses << std::endl;
            auto opt = std::make_unique<TexOpt>(params->optimizer_params.lr_poses);
            g.emplace_back(poses->parameters(), std::move(opt));
        }
        else
        {
            std::cout << "no pose optimizer" << std::endl;
        }

        if (!params->optimizer_params.fix_intrinsics)
        {
            std::cout << "optimizing pinhole intrinsics with lr " << params->optimizer_params.lr_intrinsics
                      << std::endl;
            auto opt = std::make_unique<TexOpt>(params->optimizer_params.lr_intrinsics);
            std::vector<torch::Tensor> ts;
            ts.push_back(intrinsics->intrinsics);
            g.emplace_back(ts, std::move(opt));
            intrinsics->train(true);
        }
        else
        {
            std::cout << "no intrinsics optimizer" << std::endl;
            intrinsics->train(false);
        }


        if (!g.empty())
        {
            structure_optimizer = std::make_shared<TexOptim>(g, TexOpt(1));
        }
        else
        {
            std::cout << "no structure optimizer" << std::endl;
        }
    }
    else
    {
        std::vector<torch::optim::OptimizerParamGroup> g_struc;
        if (!params->optimizer_params.fix_points)
        {
            std::cout << "optimizing 3D points with lr " << params->optimizer_params.lr_points << std::endl;
            auto opt = std::make_unique<torch::optim::SGDOptions>(params->optimizer_params.lr_points);
            std::vector<torch::Tensor> ts;
            ts.push_back(point_cloud_cuda->t_position);
            g_struc.emplace_back(ts, std::move(opt));
        }
        else
        {
            point_cloud_cuda->t_position.set_requires_grad(false);
        }


        if (!params->optimizer_params.fix_poses)
        {
            std::cout << "optimizing poses with lr " << params->optimizer_params.lr_poses << std::endl;
            auto opt = std::make_unique<torch::optim::SGDOptions>(params->optimizer_params.lr_poses);
            g_struc.emplace_back(poses->parameters(), std::move(opt));
        }
        else
        {
            std::cout << "no pose optimizer" << std::endl;
        }

        if (!params->optimizer_params.fix_intrinsics)
        {
            std::cout << "optimizing pinhole intrinsics with lr " << params->optimizer_params.lr_intrinsics
                      << std::endl;
            auto opt = std::make_unique<torch::optim::SGDOptions>(params->optimizer_params.lr_intrinsics);
            std::vector<torch::Tensor> ts;
            ts.push_back(intrinsics->intrinsics);
            g_struc.emplace_back(ts, std::move(opt));
        }
        else
        {
            std::cout << "no intrinsics optimizer" << std::endl;
        }

        if (!g_struc.empty())
        {
            structure_optimizer = std::make_shared<torch::optim::SGD>(g_struc, torch::optim::SGDOptions(1e-10));
        }
        else
        {
            std::cout << "no structure optimizer" << std::endl;
        }
    }
}

void NeuralScene::ShrinkTextureOptimizer(torch::Tensor indices_to_keep)
{
    if (params->optimizer_params.texture_optimizer == "adam" || params->optimizer_params.use_myadam_everywhere)
    {
        torch::optim::MyAdam* adam = static_cast<torch::optim::MyAdam*>(texture_optimizer.get());
        adam->shrinkInternalState(0, indices_to_keep);
        adam->shrinkInternalState(2, indices_to_keep);
    }
}

void NeuralScene::AppendToTextureOptimizer(int new_size)
{
    if (params->optimizer_params.texture_optimizer == "adam" || params->optimizer_params.use_myadam_everywhere)
    {
        torch::optim::MyAdam* adam = static_cast<torch::optim::MyAdam*>(texture_optimizer.get());
        adam->appendToInternalState(0, new_size);
        adam->appendToInternalState(2, new_size);
    }
    else
    {
        SAIGA_ASSERT(false);
    }
}

void NeuralScene::CreateTextureOptimizer()
{
    std::vector<torch::optim::OptimizerParamGroup> g;

    if (params->optimizer_params.use_myadam_everywhere)
    {
        using TexOpt   = torch::optim::MyAdamOptions;
        using TexOptim = torch::optim::MyAdam;

        if (!params->optimizer_params.fix_texture)
        {
            std::cout << "Using Mm Adam texture optimizer" << std::endl;
            std::cout << "optimizing texture with lr " << params->optimizer_params.lr_texture << "/"
                      << params->optimizer_params.lr_background_color << std::endl;
            {
                auto opt_t = std::make_unique<TexOpt>(params->optimizer_params.lr_texture);
                std::vector<torch::Tensor> ts;
                ts.push_back(texture->texture_raw);
                g.emplace_back(ts, std::move(opt_t));
            }
            {
                auto opt_t = std::make_unique<TexOpt>(params->optimizer_params.lr_background_color);
                std::vector<torch::Tensor> ts;
                ts.push_back(texture->background_color_raw);
                g.emplace_back(ts, std::move(opt_t));
            }
            {
                auto opt_t = std::make_unique<TexOpt>(params->optimizer_params.lr_confidence);
                std::vector<torch::Tensor> ts;
                ts.push_back(texture->confidence_raw);
                g.emplace_back(ts, std::move(opt_t));
            }
        }

        if (environment_map && !params->optimizer_params.fix_environment_map)
        {
            std::cout << "Tensors:" << std::endl;
            for (auto f : environment_map->parameters())
            {
                PrintTensorInfo(f);
            }
            environment_map->CreateEnvMapOptimizer(params->optimizer_params.lr_environment_map,
                                                   params->optimizer_params.lr_environment_map_density);
        }
        texture_optimizer = std::make_shared<TexOptim>(g, TexOpt(1));
    }
    else if (params->optimizer_params.texture_optimizer == "adam")
    {
        using TexOpt   = torch::optim::AdamOptions;
        using TexOptim = torch::optim::Adam;

        if (!params->optimizer_params.fix_texture)
        {
            std::cout << "Using Adam texture optimzier" << std::endl;
            std::cout << "optimizing texture with lr " << params->optimizer_params.lr_texture << "/"
                      << params->optimizer_params.lr_background_color << std::endl;
            {
                auto opt_t = std::make_unique<TexOpt>(params->optimizer_params.lr_texture);
                std::vector<torch::Tensor> ts;
                ts.push_back(texture->texture_raw);
                g.emplace_back(ts, std::move(opt_t));
            }
            {
                auto opt_t = std::make_unique<TexOpt>(params->optimizer_params.lr_background_color);
                std::vector<torch::Tensor> ts;
                ts.push_back(texture->background_color_raw);
                g.emplace_back(ts, std::move(opt_t));
            }
            {
                auto opt_t = std::make_unique<TexOpt>(params->optimizer_params.lr_confidence);
                std::vector<torch::Tensor> ts;
                ts.push_back(texture->confidence_raw);
                g.emplace_back(ts, std::move(opt_t));
            }
        }

        if (environment_map && !params->optimizer_params.fix_environment_map)
        {
            std::cout << "Tensors:" << std::endl;
            for (auto f : environment_map->parameters())
            {
                PrintTensorInfo(f);
            }
            environment_map->CreateEnvMapOptimizer(params->optimizer_params.lr_environment_map,
                                                   params->optimizer_params.lr_environment_map_density);
        }
        texture_optimizer = std::make_shared<TexOptim>(g, TexOpt(1));
    }
    else if (params->optimizer_params.texture_optimizer == "sgd")
    {
        using TexOpt   = torch::optim::SGDOptions;
        using TexOptim = torch::optim::SGD;


        if (!params->optimizer_params.fix_texture)
        {
            std::cout << "Using SGD texture optimzier" << std::endl;
            std::cout << "optimizing texture with lr " << params->optimizer_params.lr_texture << "/"
                      << params->optimizer_params.lr_background_color << std::endl;
            {
                auto opt_t = std::make_unique<TexOpt>(params->optimizer_params.lr_texture);
                std::vector<torch::Tensor> ts;
                ts.push_back(texture->texture_raw);
                g.emplace_back(ts, std::move(opt_t));
            }
            {
                auto opt_t = std::make_unique<TexOpt>(params->optimizer_params.lr_background_color);
                std::vector<torch::Tensor> ts;
                ts.push_back(texture->background_color);
                g.emplace_back(ts, std::move(opt_t));
            }
            {
                auto opt_t = std::make_unique<TexOpt>(params->optimizer_params.lr_texture);
                std::vector<torch::Tensor> ts;
                ts.push_back(texture->confidence_raw);
                g.emplace_back(ts, std::move(opt_t));
            }
        }

        if (environment_map && !params->optimizer_params.fix_environment_map)
        {
            std::cout << "optimizing environment_map with lr " << params->optimizer_params.lr_environment_map
                      << std::endl;
            auto opt_t = std::make_unique<TexOpt>(params->optimizer_params.lr_environment_map);

            std::cout << "Tensors:" << std::endl;
            for (auto f : environment_map->parameters())
            {
                PrintTensorInfo(f);
            }
        }
        texture_optimizer = std::make_shared<TexOptim>(g, TexOpt(1));
    }
    else
    {
        SAIGA_EXIT_ERROR("unknown optimizer");
    }
}

void NeuralScene::BuildOutlierCloud(int n)
{
    outlier_point_cloud_cuda =
        NeuralPointCloudCuda(scene->OutlierPointCloud(n, 0.1), params->points_adding_params.use_grid_loss,
                             params->points_adding_params.cells_worldspace_size);
    outlier_point_cloud_cuda->MakeOutlier(texture->NumPoints() - 1);
}

void NeuralScene::LoadCheckpoint(const std::string& checkpoint_dir)
{
    std::string checkpoint_prefix = checkpoint_dir + "/scene_" + scene->scene_name + "_";

    if (point_cloud_cuda && std::filesystem::exists(checkpoint_prefix + "points.pth"))
    {
        torch::load(point_cloud_cuda, checkpoint_prefix + "points.pth");
        // point_cloud_cuda->UpdateCellStructure();
        std::cout << "Loaded Checkpoint points " << point_cloud_cuda->t_position.size(0)
                  << " max uv: " << point_cloud_cuda->t_index.max().item().toInt() << std::endl;
        std::cout << "Positions: " << TensorInfo(point_cloud_cuda->t_position) << std::endl;
        std::cout << "Indices: " << TensorInfo(point_cloud_cuda->t_index) << std::endl;
        //    point_cloud_cuda->t_original_index = point_cloud_cuda->t_index.clone();

        SAIGA_ASSERT(point_cloud_cuda->t_position.dtype() == torch::kFloat);
        SAIGA_ASSERT(point_cloud_cuda->t_index.dtype() == torch::kInt32);

        SAIGA_ASSERT(point_cloud_cuda->t_position.size(0) == point_cloud_cuda->t_index.size(0));
    }

    if (texture && std::filesystem::exists(checkpoint_prefix + "texture.pth"))
    {
        std::cout << "Load Texture." << std::endl;
        torch::load(texture, checkpoint_prefix + "texture.pth", torch::kCPU);
        std::cout << "Loaded Checkpoint texture. Texels: " << texture->NumPoints()
                  << " Channels: " << texture->TextureChannels() << std::endl;
        std::cout << "Texture:" << TensorInfo(texture->texture) << std::endl;
        std::cout << "Background: " << texture->background_color_raw << std::endl;
        SAIGA_ASSERT(texture->NumPoints() == point_cloud_cuda->Size());

        //        SAIGA_ASSERT(texture->TextureChannels() == params->pipeline_params.num_texture_channels);
    }

    SAIGA_ASSERT(point_cloud_cuda->t_index.max().item().toInt() <= texture->NumPoints());

    if (std::filesystem::exists(checkpoint_prefix + "poses.pth"))
    {
        std::cout << "Load Checkpoint pose" << std::endl;

        std::cout << "First pose before " << poses->Download().front() << std::endl;

        torch::load(poses, checkpoint_prefix + "poses.pth");

        SAIGA_ASSERT(poses->poses_se3.size(0) == scene->frames.size());
        SAIGA_ASSERT(poses->poses_se3.dtype() == torch::kDouble);
        SAIGA_ASSERT(poses->tangent_poses.dtype() == torch::kDouble);

        std::cout << "First pose after " << poses->Download().front() << std::endl;

        DownloadPoses();
    }

    if (std::filesystem::exists(checkpoint_prefix + "intrinsics.pth"))
    {
        std::cout << "Load Checkpoint intrinsics" << std::endl;
        torch::load(intrinsics, checkpoint_prefix + "intrinsics.pth");
        DownloadIntrinsics();
    }


    if (environment_map && std::filesystem::exists(checkpoint_prefix + "env.pth"))
    {
        std::cout << "Load Checkpoint environment_map" << std::endl;
        torch::load(environment_map, checkpoint_prefix + "env.pth");
    }

    camera->LoadCheckpoint(checkpoint_prefix);
}

void NeuralScene::SaveCheckpoint(const std::string& checkpoint_dir, bool reduced)
{
    std::string checkpoint_prefix = checkpoint_dir + "/scene_" + scene->scene_name + "_";


    if (!reduced)
    {
        // These variables are very large in memory so you can disable the checkpoin write here.
        torch::save(texture, checkpoint_prefix + "texture.pth");

        if (environment_map)
        {
            torch::save(environment_map, checkpoint_prefix + "env.pth");
        }
        torch::save(point_cloud_cuda, checkpoint_prefix + "points.pth");
    }

    {
        torch::save(poses, checkpoint_prefix + "poses.pth");
        torch::save(intrinsics, checkpoint_prefix + "intrinsics.pth");

        auto all_poses = poses->Download();
        SceneData::SavePoses(all_poses, checkpoint_prefix + "poses.txt");
    }

    camera->SaveCheckpoint(checkpoint_prefix);
}

void NeuralScene::Log(const std::string& log_dir)
{
    std::cout << "Scene Log - Texture: ";
    // PrintTensorInfo(texture->texture);
    PrintTensorInfo(texture->texture_raw);
    {
        auto bg = texture->GetBackgroundColor();
        std::cout << "  Background Desc:  ";
        for (auto b : bg)
        {
            std::cout << b << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "  Confidence per point: ";
    PrintTensorInfo(texture->confidence_value_of_point.slice(0, 0, 1));
    std::cout << "  Confidences under " << params->points_adding_params.removal_confidence_cutoff << ": "
              << torch::count_nonzero(
                     torch::where(texture->confidence_value_of_point.slice(0, 0, 1) <
                                      params->points_adding_params.removal_confidence_cutoff,
                                  torch::ones_like(texture->confidence_value_of_point.slice(0, 0, 1)),
                                  torch::zeros_like(texture->confidence_value_of_point.slice(0, 0, 1))))
                     .item<int>()
              << std::endl;

    if (!params->optimizer_params.fix_point_size)
    {
        std::cout << "  LayerBuf per point: ";
        //   PrintTensorInfo(texture->layer_value_of_point);
        //   const float threshold_layerbuf = 0.0001f;
        //   std::cout << "  LayerBuf under " << threshold_layerbuf << ": "
        //             << torch::count_nonzero(torch::where(texture->layer_value_of_point < threshold_layerbuf,
        //                                                  torch::ones_like(texture->layer_value_of_point),
        //                                                  torch::zeros_like(texture->layer_value_of_point)))
        //                    .item<int>()
        //             << std::endl;
        PrintTensorInfo(point_cloud_cuda->t_point_size);
        std::cout << "      softplus: ";
        PrintTensorInfo(torch::nn::functional::softplus(point_cloud_cuda->t_point_size));
    }

    if (environment_map)
    {
        std::cout << "  Environment map: ";
        std::cout << "  color ";
        PrintTensorInfo(environment_map->color);

        std::cout << "\t\t  Density ";
        PrintTensorInfo(environment_map->density);
    }

    if (!params->optimizer_params.fix_intrinsics)
    {
        for (auto& cam : scene->scene_cameras)
        {
            if (cam.camera_model_type == CameraModel::PINHOLE_DISTORTION)
            {
                std::cout << "  K:    " << cam.K << std::endl;
                std::cout << "  Dist: " << cam.distortion.Coeffs().transpose() << std::endl;
            }
        }
    }

    if (!params->optimizer_params.fix_poses)
    {
        std::cout << "  Poses: ";
        PrintTensorInfo(poses->poses_se3);
    }

    if (!params->optimizer_params.fix_points)
    {
        std::cout << "  Point Position: ";
        PrintTensorInfo(point_cloud_cuda->t_position);
    }

    if (camera->vignette_net && !params->optimizer_params.fix_vignette)
    {
        camera->vignette_net->PrintParams(log_dir, scene->scene_name);
    }
}

void NeuralScene::AddNewPoints(std::vector<vec3> new_positions, std::vector<vec3> new_normal)
{
    SAIGA_ASSERT(false, "NOT IMPLEMENTED");

    SAIGA_ASSERT(new_positions.size() == new_normal.size());
    auto mesh = point_cloud_cuda->Mesh();
    // add points to mesh
    mesh.position.insert(mesh.position.end(), new_positions.begin(), new_positions.end());
    mesh.normal.insert(mesh.normal.end(), new_normal.begin(), new_normal.end());

    // recompute radius
    SceneData::ComputeRadius(mesh);

    AABB custom_aabb = scene->dataset_params.aabb;

    // make mesh to neural point cloud
    point_cloud_cuda = NeuralPointCloudCuda(mesh, params->points_adding_params.use_grid_loss,
                                            params->points_adding_params.cells_worldspace_size, custom_aabb);

    // update texture
    texture->EnlargeTexture(point_cloud_cuda->Size());

    {
        //    torch::NoGradGuard ngg;
        // CreateTextureOptimizer();
        // std::cout << "Texture pptimizer was reset. There may be a better implementation, but this would require "
        //              "resizing the optimizer's internal states"
        //          << std::endl;

        std::vector<torch::Tensor>& params = texture_optimizer->parameters();
        for (auto t : params)
        {
            // resize and reset texture gradients
            t.mutable_grad() = torch::zeros_like(t).detach_();  // = torch::NullType::singleton();
        }
    }
    texture_optimizer->zero_grad();
}

void NeuralScene::AddNewRandomPoints(float factor)
{
    auto mesh = point_cloud_cuda->Mesh();

    SceneData::AddRandomPoints(mesh, factor);
    AABB custom_aabb = scene->dataset_params.aabb;
    // make mesh to neural point cloud
    point_cloud_cuda = NeuralPointCloudCuda(mesh, params->points_adding_params.use_grid_loss,
                                            params->points_adding_params.cells_worldspace_size, custom_aabb);

    // update texture
    texture->EnlargeTexture(point_cloud_cuda->Size());

    {
        //    torch::NoGradGuard ngg;
        // CreateTextureOptimizer();

        // std::cout << "Texture pptimizer was reset. There may be a better implementation, but this would require "
        //             "resizing the optimizer's internal states"
        //          << std::endl;
        {
            std::vector<torch::Tensor>& params_to = texture_optimizer->parameters();
            for (auto t : params_to)
            {
                // resize and reset texture gradients
                t.mutable_grad() = torch::zeros_like(t).detach_();  // = torch::NullType::singleton();
            }
        }
        if (!params->optimizer_params.fix_points)
        {
            //  CreateStructureOptimizer();
            std::vector<torch::Tensor>& params_to = structure_optimizer->parameters();

            for (auto t : params_to)
            {
                // resize and reset texture gradients
                t.mutable_grad() = torch::zeros_like(t).detach_();  // = torch::NullType::singleton();
            }
        }
    }
    texture_optimizer->zero_grad();
}

const double SQRT_5           = 2.2360679774997896964091736687312762354406183596115257;
const double GOLDEN_RATIO_PHI = ((SQRT_5 + 1.0) * 0.5);
const double INV_PHI          = (GOLDEN_RATIO_PHI - 1.0);  // property that Phi - 1 = Phi^(-1)
// see https://github.com/lorafib/LumiPath/blob/master/PathTracer/shader_common.h
static float fractOfProduct(const float a, const float b)
{
    return fmaf(a, b, -truncf(a * b)) /*a*b - floorf(a*b)*/;
}

static vec3 convertToCartesian(const float theta, const float phi)
{
    return vec3(sinf(theta) * cosf(phi), cosf(theta), sinf(phi) * sinf(theta));
}

static vec2 getPhiAndZ(unsigned int n, float numFibPoints)
{
    // phi, z by eq. (2): phi = 2*pi*[i / Phi], z = 1 - (2*i + 1) / n
    float z0 = 1.f - 1.f / numFibPoints;
    return vec2(2 * M_PI * fractOfProduct((float)n, INV_PHI), z0 - 2.f * float(n) / numFibPoints);
}

static vec2 getPolar(unsigned int n, float numFibPoints)
{
    vec2 polar  = getPhiAndZ(n, numFibPoints);
    float theta = acosf(polar.y());
    polar.y()   = polar.x();
    polar.x()   = theta;
    return polar;
}

static vec3 getCartesian(unsigned int n, float numFibPoints)
{
    vec2 polar = getPolar(n, numFibPoints);
    return convertToCartesian(polar.x(), polar.y());
}


void NeuralScene::AddNewRandomForEnvSphere(int num_spheres, float inner_radius, float env_radius_factor, int num_points,
                                           bool update_optimizer)
{
    std::vector<vec3> env_points;

    for (int i_s = 0; i_s < num_spheres; ++i_s)
    {
        quat rot =
            angleAxis(float(Saiga::linearRand(-M_PI, M_PI)),
                      vec3(Saiga::linearRand(-1, 1), Saiga::linearRand(-1, 1), Saiga::linearRand(-1, 1)).normalized());
        float rad = inner_radius * pow(env_radius_factor, i_s);
        for (int i = 0; i < num_points; ++i)
        {
            auto getFibPoint = [&]()
            {
                vec3 fib_p = getCartesian(i, num_points).normalized();

                fib_p = rot * fib_p;
                fib_p *= rad;
                return fib_p;
            };

            vec3 p = getFibPoint();
            env_points.push_back(p);
        }
    }
    AddNewPoints(env_points, update_optimizer);
}

void NeuralScene::AddNewRandomPointsFromCTHdr(torch::Tensor hdr_img_stack, int max_num_points_to_add,
                                              float ct_volume_scale, vec3 ct_volume_translation, AABB scene_aabb)
{
    const int volume_cut_off_amount = 0;

    hdr_img_stack -= hdr_img_stack.min();
    if (hdr_img_stack.max().cpu().item<float>() > 0) hdr_img_stack /= hdr_img_stack.max();

        // torch::Tensor random_values;
#if 0
    {
        std::cout << "Normalized Stack: " << TensorInfo(hdr_img_stack) << std::endl;
        hdr_img_stack = hdr_img_stack.squeeze().permute({2, 1, 0});

        auto volume_sizes = hdr_img_stack.sizes();
        torch::NoGradGuard ngg;
        torch::Tensor num_points_per_cell =
            torch::floor(hdr_img_stack * max_num_points_to_add).to(torch::kInt32).to(torch::kCUDA);

        // vec3 one_over_len_of_volume =
        //     vec3(1, 1, 1).array() / vec3(volume_sizes[1], volume_sizes[2], volume_sizes[0]).array();
        //
        // torch::Tensor length_of_cell =
        //    torch::from_blob(one_over_len_of_volume.data(), {3},
        //    torch::TensorOptions().dtype(torch::kFloat32)).clone();
        // length_of_cell *= ct_volume_scale * 2.f;

        std::cout << ct_volume_scale * 2.f << std::endl;
        std::cout << volume_sizes << std::endl;

        auto lin_x = ((torch::arange(0, volume_sizes[0]).to(torch::kFloat32) / float(volume_sizes[0])) - 0.5) *
                         ct_volume_scale * 2.f +
                     ct_volume_translation.x();
        auto lin_y = ((torch::arange(0, volume_sizes[1]).to(torch::kFloat32) / float(volume_sizes[1])) - 0.5) *
                         ct_volume_scale * 2.f +
                     ct_volume_translation.y();
        auto lin_i = ((torch::arange(0, volume_sizes[2]).to(torch::kFloat32) / float(volume_sizes[2])) - 0.5) *
                         ct_volume_scale * 2.f +
                     ct_volume_translation.z();

        auto grid_v = torch::meshgrid({lin_x, lin_y, lin_i});
        for (auto& t : grid_v) t.unsqueeze_(0);
        auto grid = torch::cat(grid_v, 0).to(torch::kCUDA);

        //  std::cout << TensorInfo(grid) << std::endl;
        //  std::cout << grid << std::endl;
        // std::cout << TensorInfo(grid_v[0]) << std::endl;
        // std::cout << TensorInfo(grid_v[1]) << std::endl;
        // std::cout << TensorInfo(grid_v[2]) << std::endl;
        // std::cout << TensorInfo(lin_i) << std::endl;
        // std::cout << TensorInfo(lin_y) << std::endl;
        // std::cout << TensorInfo(lin_x) << std::endl;

        std::vector<torch::Tensor> points_to_add;
        Saiga::ProgressBar bar(std::cout, "Add Points  |", max_num_points_to_add, 30, false, 1000);
        for (int i = 0; i < max_num_points_to_add; ++i)
        {
            torch::Tensor random_vals =
                torch::rand({3, volume_sizes[0], volume_sizes[1], volume_sizes[2]}).to(torch::kCUDA);

            random_vals.slice(0, 0, 1) /= (volume_sizes[0]);
            random_vals.slice(0, 1, 2) /= (volume_sizes[1]);
            random_vals.slice(0, 2, 3) /= (volume_sizes[2]);
            random_vals *= 2.f * ct_volume_scale;

            random_vals += grid;
            random_vals.contiguous();
            // std::cout << "random vals " << TensorInfo(random_vals) << std::endl;
            random_vals = random_vals.reshape({3, -1});
            // std::cout << "random vals " << TensorInfo(random_vals) << std::endl;
            auto ind = (num_points_per_cell.reshape({-1}) > i).nonzero().contiguous().squeeze(1);
            //  std::cout << "ind " << TensorInfo(ind) << std::endl;
            if (ind.size(0) <= 0)
            {
                bar.addProgress(1);
                continue;
            }

            auto vals = torch::index_select(random_vals, 1, ind);
            //  std::cout << "vals " << TensorInfo(vals) << std::endl;
            //  std::cout << "vals.slice(0, 0, 1) > scene_aabb.min.x() "
            //<< TensorInfo((vals.slice(0, 0, 1) > scene_aabb.min.x()).squeeze(0)) << std::endl;
            if (vals.size(1) == 1)
            {
                ind = ((vals.slice(0, 0, 1) > scene_aabb.min.x()) & (vals.slice(0, 1, 2) > scene_aabb.min.y()) &
                       (vals.slice(0, 2, 3) > scene_aabb.min.z()) & (vals.slice(0, 0, 1) < scene_aabb.max.x()) &
                       (vals.slice(0, 1, 2) < scene_aabb.max.y()) & (vals.slice(0, 2, 3) < scene_aabb.max.z()))
                          .nonzero()
                          .contiguous()
                          .squeeze();
            }
            else
            {
                // remove outside of aabb
                ind = ((vals.slice(0, 0, 1) > scene_aabb.min.x()).squeeze(0) &
                       (vals.slice(0, 1, 2) > scene_aabb.min.y()).squeeze(0) &
                       (vals.slice(0, 2, 3) > scene_aabb.min.z()).squeeze(0) &
                       (vals.slice(0, 0, 1) < scene_aabb.max.x()).squeeze(0) &
                       (vals.slice(0, 1, 2) < scene_aabb.max.y()).squeeze(0) &
                       (vals.slice(0, 2, 3) < scene_aabb.max.z()).squeeze(0))
                          .nonzero()
                          .contiguous()
                          .squeeze();
            }
            if (ind.size(0) > 0)
            {
                // std::cout << "ind " << TensorInfo(ind) << std::endl;

                vals = torch::index_select(vals, 1, ind);
                //   std::cout << "vals " << TensorInfo(vals) << std::endl;
            }
            points_to_add.push_back(vals);
            bar.addProgress(1);
        }

        if (points_to_add.size() > 0)
        {
            torch::Tensor vals_to_add = torch::cat(points_to_add, 1).contiguous().clone();
            vals_to_add               = vals_to_add.permute({1, 0}).contiguous();
            std::cout << "vals_to_add " << TensorInfo(vals_to_add) << std::endl;

            AddNewPoints(vals_to_add);
        }
    }
    // exit(1);
    /*
    for (int x = volume_cut_off_amount; x < volume_sizes[2] - volume_cut_off_amount; ++x)
    {
        for (int y = volume_cut_off_amount; y < volume_sizes[1] - volume_cut_off_amount; ++y)
        {
            for (int i = volume_cut_off_amount; i < volume_sizes[0] - volume_cut_off_amount; ++i)
            {

                float val            = hdr_img_stack.index({i, y, x}).item<float>();
                int points_this_cell = int(val * max_num_points_to_add);
                if (points_this_cell > 0)
                {
                    //[-0.5,0.5]
                    vec3 p = (vec3(x, y, i).array() * one_over_len_of_volume.array()) - vec3(0.5, 0.5, 0.5).array();

                    //[aabbmin, aabbmax]
                    vec3 offset_pos_v = p * ct_volume_scale * 2.f + ct_volume_translation;
                    if (!scene_aabb.contains(offset_pos_v)) continue;

                    for (int n = 0; n < points_this_cell; ++n) offset_p[running_count + n] = (offset_pos_v);

                    running_count += points_this_cell;
                }
            }
                    bar.addProgress(volume_sizes[0] - volume_cut_off_amount);
        }
    }
     */

//    std::cout << "Actual Points added:" << running_count << std::endl;
//    random_values = random_values.slice(0, 0, running_count) * length_of_cell + mult_t;
//    std::cout << "Random Values to be added: " << TensorInfo(random_values) << std::endl;
//
//    AddNewPoints(random_values);
#else
    hdr_img_stack = hdr_img_stack.squeeze_().cpu();
    std::cout << "Normalized Stack: " << TensorInfo(hdr_img_stack) << std::endl;
    auto volume_sizes = hdr_img_stack.sizes();

    int num_points_max = int(std::ceil((hdr_img_stack.sum() * float(max_num_points_to_add)).item<float>()));
    std::cout << "Max Points added:" << num_points_max << std::endl;
    torch::Tensor random_values = torch::rand({int(num_points_max), 3});

    // auto lin_i = torch::arange(0,volume_sizes[0]);
    // auto lin_y = torch::arange(0,volume_sizes[1]);
    // auto lin_x = torch::arange(0,volume_sizes[2]);
    // auto grid_v = torch::meshgrid({lin_i,lin_y,lin_x});
    // auto grid = torch::cat(grid_v,0);
    //
    // std::cout << TensorInfo(grid) <<  TensorInfo(grid_v[0]) <<  TensorInfo(grid_v[1]) <<  TensorInfo(grid_v[2]) <<
    // TensorInfo(lin_i) << TensorInfo(lin_y) <<TensorInfo( lin_x) << std::endl;


    long long running_count = 0;
    vec3 one_over_len_of_volume =
        vec3(1, 1, 1).array() / vec3(volume_sizes[1], volume_sizes[2], volume_sizes[0]).array();
    torch::Tensor length_of_cell =
        torch::from_blob(one_over_len_of_volume.data(), {3}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
    length_of_cell *= ct_volume_scale * 2.f;

    Saiga::ProgressBar bar(std::cout, "Add Points  |", volume_sizes[0] * volume_sizes[1] * volume_sizes[2], 30, false,
                           5000);
    /*
        // #pragma omp parallel for
        for (int i = volume_cut_off_amount; i < volume_sizes[0] - volume_cut_off_amount; ++i)
        {
            // auto img_z_sliced = hdr_img_stack.slice(0,i,i+1).squeeze();
            for (int y = volume_cut_off_amount; y < volume_sizes[1] - volume_cut_off_amount; ++y)
            {
                //   auto img_yz_sliced = hdr_img_stack.slice(0,i,i+1).squeeze().slice(0,y,y+1).squeeze();
                for (int x = volume_cut_off_amount; x < volume_sizes[2] - volume_cut_off_amount; ++x)
                {
                    using namespace torch::indexing;
                    //  auto img_xyz_sliced =
                    //  hdr_img_stack.slice(0,i,i+1).squeeze().slice(0,y,y+1).squeeze().slice(0,x,x+1).squeeze();
                    float val = hdr_img_stack.index({Slice(i, i + 1), Slice(y, y + 1), Slice(x, x + 1)}).item<float>();
                    int points_this_cell = int(val * max_num_points_to_add);
                    if (points_this_cell > 0)
                    {
                        //[-0.5,0.5]
                        vec3 p = (vec3(x, y, i).array() * one_over_len_of_volume.array()) - vec3(0.5, 0.5, 0.5).array();

                        //[aabbmin, aabbmax]
                        vec3 offset_pos_v = p * ct_volume_scale * 2.f + ct_volume_translation;

                        if (!scene_aabb.contains(offset_pos_v)) continue;
                        torch::Tensor offset_pos_min =
                            torch::from_blob(offset_pos_v.data(), {3}, torch::TensorOptions().dtype(torch::kFloat32))
                                .clone();

                        random_values.slice(0, running_count, running_count + points_this_cell) *= length_of_cell;
                        random_values.slice(0, running_count, running_count + points_this_cell) += offset_pos_min;

                        running_count += points_this_cell;
                    }
                }
                bar.addProgress(volume_sizes[0] - volume_cut_off_amount);
            }
        }
        std::cout << "Actual Points added:" << running_count << std::endl;
        random_values = random_values.slice(0, 0, running_count);
        std::cout << "Random Values to be added: " << TensorInfo(random_values) << std::endl;

      */
    std::vector<vec3> offset_p(num_points_max);
    for (int x = volume_cut_off_amount; x < volume_sizes[2] - volume_cut_off_amount; ++x)
    {
        for (int y = volume_cut_off_amount; y < volume_sizes[1] - volume_cut_off_amount; ++y)
        {
            for (int i = volume_cut_off_amount; i < volume_sizes[0] - volume_cut_off_amount; ++i)
            {
                // using namespace torch::indexing;
                //  float val = hdr_img_stack.index({Slice(i, i + 1), Slice(y, y + 1), Slice(x, x +
                // 1)}).item<float>();
                //  float val            = hdr_img_stack[i][y][x].item<float>();
                float val            = hdr_img_stack.index({i, y, x}).item<float>();
                int points_this_cell = int(val * max_num_points_to_add);
                if (points_this_cell > 0)
                {
                    //[-0.5,0.5]
                    vec3 p = (vec3(x, y, i).array() * one_over_len_of_volume.array()) - vec3(0.5, 0.5, 0.5).array();

                    //[aabbmin, aabbmax]
                    vec3 offset_pos_v = p * ct_volume_scale * 2.f + ct_volume_translation;
                    if (!scene_aabb.contains(offset_pos_v)) continue;

                    for (int n = 0; n < points_this_cell; ++n) offset_p[running_count + n] = (offset_pos_v);

                    running_count += points_this_cell;
                }
            }
            bar.addProgress(volume_sizes[0] - volume_cut_off_amount);
        }
    }
    torch::Tensor mult_t =
        torch::from_blob(offset_p.data(), {running_count, 3}, torch::TensorOptions().dtype(torch::kFloat32)).clone();

    std::cout << "Actual Points added:" << running_count << std::endl;
    random_values = random_values.slice(0, 0, running_count) * length_of_cell + mult_t;
    std::cout << "Random Values to be added: " << TensorInfo(random_values) << std::endl;

    AddNewPoints(random_values);
#endif
}


void NeuralScene::AddNewRandomPointsFromCTStack(int num_points_to_add_max_per_cell, std::string path,
                                                float ct_volume_scale, vec3 ct_volume_translation)
{
    std::set<std::filesystem::path> png_paths;
    for (const auto& entry : std::filesystem::directory_iterator(path))
    {
        png_paths.insert(entry.path());
    }
    // std::vector<TemplatedImage<ushort>> png_stack;
    // auto getPx = [](TemplatedImage<ushort>& image, int x, int y) {return float(uint(image(x,y)))/float(65535); };

    std::vector<TemplatedImage<uint8_t>> png_stack;
    auto getPx = [](TemplatedImage<uint8_t>& image, int x, int y) { return float(uint32_t(image(x, y))) / float(255); };
    for (auto& filename : png_paths)
    {
        std::cout << filename << std::endl;
        png_stack.emplace_back(filename.string());
    }

    // counting pass
    /*double all_count = 0;
    float max_val = 0;
    for(auto img: png_stack){
        for(int y=0; y<img.h; ++y)
        {
            for (int x = 0; x < img.w; ++x)
            {
                float val = getPx(img,x,y);
                all_count += val;
                max_val = std::max(max_val, val);
            }
        }
    }*/

    int points_to_add = 0;
    for (auto img : png_stack)
    {
        for (int y = 0; y < img.h; ++y)
        {
            for (int x = 0; x < img.w; ++x)
            {
                float val            = getPx(img, x, y);
                int points_this_cell = int(float(num_points_to_add_max_per_cell) * val);
                points_to_add += points_this_cell;
            }
        }
    }

    std::cout << "num_points_to_add" << points_to_add << std::endl;

    torch::Tensor random_values = torch::rand({points_to_add, 3});
    std::cout << "Random Values to be added: " << TensorInfo(random_values) << std::endl;

    uint32_t running_count = 0;
    for (int i = 0; i < png_stack.size(); ++i)
    {
        auto img = png_stack[i];
        for (int y = 0; y < img.h; ++y)
        {
            for (int x = 0; x < img.w; ++x)
            {
                float val            = getPx(img, x, y);
                int points_this_cell = int(float(num_points_to_add_max_per_cell) * val);

                SAIGA_ASSERT(points_this_cell >= 0);
                if (points_this_cell > 0)
                {
                    vec3 one_over_len_of_volume = vec3(1, 1, 1).array() / vec3(img.h, img.w, png_stack.size()).array();

                    //(xy is flipped)
                    vec3 p = vec3(y, x, i).array() * one_over_len_of_volume.array() - vec3(0.5, 0.5, 0.5).array();
                    vec3 offset_pos_v = p * ct_volume_scale + ct_volume_translation;

                    torch::Tensor offset_pos_min =
                        torch::from_blob(offset_pos_v.data(), {3}, torch::TensorOptions().dtype(torch::kFloat32));

                    torch::Tensor length_of_cell = torch::from_blob(one_over_len_of_volume.data(), {3},
                                                                    torch::TensorOptions().dtype(torch::kFloat32));
                    length_of_cell *= ct_volume_scale;

                    random_values.slice(0, running_count, running_count + points_this_cell) =
                        offset_pos_min +
                        random_values.slice(0, running_count, running_count + points_this_cell) * length_of_cell;
                }
                running_count += points_this_cell;
            }
        }
    }
    std::cout << "Random Values to be added: " << TensorInfo(random_values) << std::endl;

    AddNewPoints(random_values);
}


void NeuralScene::AddNewPoints(torch::Tensor random_values, bool update_optimizer)
{
    std::vector<vec3> positions(random_values.size(0));
    random_values = random_values.cpu();
    SAIGA_ASSERT(random_values.dtype() == torch::kFloat32);
    std::memcpy(positions.data(), random_values.data_ptr(), sizeof(vec3) * random_values.size(0));

    AddNewPoints(positions, update_optimizer);
}

void NeuralScene::AddNewPoints(std::vector<vec3>& positions, bool update_optimizer)
{
    torch::NoGradGuard ngg;

    torch::Tensor point_size_save;
    if (point_cloud_cuda->t_point_size.defined()) point_size_save = point_cloud_cuda->t_point_size.clone();

    auto mesh = point_cloud_cuda->Mesh();

    SceneData::AddPoints(mesh, positions, positions);
    AABB custom_aabb = scene->dataset_params.aabb;

    // make mesh to neural point cloud
    point_cloud_cuda = NeuralPointCloudCuda(mesh, params->points_adding_params.use_grid_loss,
                                            params->points_adding_params.cells_worldspace_size, custom_aabb);

    // update texture
    texture->EnlargeTexture(point_cloud_cuda->Size());

    point_cloud_cuda->t_point_size.slice(0, 0, point_size_save.size(0)).set_(point_size_save);


    if (update_optimizer)
    {
        //    torch::NoGradGuard ngg;
        // CreateTextureOptimizer();
        AppendToTextureOptimizer(point_cloud_cuda->Size());
        // std::cout << "Texture optimizer was reset. There may be a better implementation, but this would require "
        //             "resizing the optimizer's internal states"
        //          << std::endl;
        {
            auto& params_groups = texture_optimizer->param_groups();
            for (auto& p_gr : params_groups)
            {
                auto p_t = p_gr.params();
                for (auto& t : p_t)
                {
                    // resize and reset texture gradients
                    t.mutable_grad() = torch::zeros_like(t).detach_();
                }
            }
        }
        CreateStructureOptimizer();
        if (!params->optimizer_params.fix_points || !params->optimizer_params.fix_intrinsics ||
            !params->optimizer_params.fix_poses)
        {
            std::vector<torch::Tensor>& params_to = structure_optimizer->parameters();

            for (auto t : params_to)
            {
                // resize and reset texture gradients
                t.mutable_grad() = torch::zeros_like(t).detach_();  // = torch::NullType::singleton();
                //    std::cout << "GRAD" <<TensorInfo(t.mutable_grad()) << std::endl;
            }
        }
        texture_optimizer->zero_grad();
    }
    texture->PrepareConfidence(0.f);
    texture->PrepareTexture(params->pipeline_params.non_subzero_texture);
}

void NeuralScene::AddPointsViaPointGrowing(int factor, float distance, bool update_optimizer)
{
    auto mesh = point_cloud_cuda->Mesh();
    SceneData::ComputeRadius(mesh);
    SceneData::DuplicatePointsRand(mesh, factor, distance, true);
    AABB custom_aabb = scene->dataset_params.aabb;

    // make mesh to neural point cloud
    point_cloud_cuda = NeuralPointCloudCuda(mesh, params->points_adding_params.use_grid_loss,
                                            params->points_adding_params.cells_worldspace_size, custom_aabb);

    // update texture
    texture->EnlargeTexture(point_cloud_cuda->Size());

    if (update_optimizer)
    {
        AppendToTextureOptimizer(point_cloud_cuda->Size());
        {
            auto& params_groups = texture_optimizer->param_groups();
            for (auto& p_gr : params_groups)
            {
                auto p_t = p_gr.params();
                for (auto& t : p_t)
                {
                    // resize and reset texture gradients
                    t.mutable_grad() = torch::zeros_like(t).detach_();
                }
            }
        }
    }
    texture_optimizer->zero_grad();
    texture->PrepareConfidence(0.f);
}


void NeuralScene::AddNewRandomPointsInValuefilledBB(int num_points_to_add_per_cell_max, float percent_of_boxes)
{
    point_cloud_cuda->NormalizeBBCellValue();

    int num_top_boxes = point_cloud_cuda->t_cell_value.size(0);
    std::cout << "Grid Cell values:" << TensorInfo(point_cloud_cuda->t_cell_value) << std::endl;
    if (percent_of_boxes < 1 && percent_of_boxes > 0)
    {
        num_top_boxes = std::ceil(percent_of_boxes * float(point_cloud_cuda->t_cell_value.size(0)));
    }
    else
    {
        percent_of_boxes = 1.f;
    }
    auto top_k_boxes      = torch::topk(point_cloud_cuda->t_cell_value.squeeze(), num_top_boxes);
    torch::Tensor indices = std::get<1>(top_k_boxes);
    auto cell_bb_min      = point_cloud_cuda->t_cell_bb_min.index({indices});
    auto cell_bb_length   = point_cloud_cuda->t_cell_bb_length.index({indices});
    auto cell_values      = point_cloud_cuda->t_cell_value.index({indices});
    // for equal point each epoch
    int num_max_points_to_add = int(float(num_points_to_add_per_cell_max * cell_values.sum().item<float>()));

    torch::Tensor random_values = torch::rand({num_max_points_to_add, 3});
    std::cout << "Random Values to be added max: " << TensorInfo(random_values) << std::endl;

    int running_number_added = 0;
    for (int i = 0; i < num_top_boxes; ++i)
    {
        int num_this_cell = int(cell_values.slice(0, i, i + 1).item<float>() * float(num_points_to_add_per_cell_max));

        random_values.slice(0, running_number_added, running_number_added + num_this_cell) *=
            cell_bb_length.slice(0, i, i + 1).cpu();
        random_values.slice(0, running_number_added, running_number_added + num_this_cell) *=
            cell_bb_min.slice(0, i, i + 1).cpu();

        // random_values.slice(0,running_number_added, running_number_added+num_this_cell)=
        //     cell_bb_min.slice(0,i,i+1).cpu() + random_values.slice(0,i*point_per_box,
        //     (i+1)*point_per_box)*cell_bb_length.slice(0,i,i+1).cpu();
        running_number_added += num_this_cell;
    }

    AddNewPoints(random_values);
}


void NeuralScene::RemovePoints(torch::Tensor indices_to_remove, bool update_optimizer)
{
    torch::Tensor point_size_save;
    if (point_cloud_cuda->t_point_size.defined()) point_size_save = point_cloud_cuda->t_point_size.clone();

    auto mesh  = point_cloud_cuda->Mesh();
    auto i_cpu = indices_to_remove.cpu().contiguous();
    SAIGA_ASSERT(indices_to_remove.dtype() == torch::kLong);
    std::vector<int64_t> indices_vec((int64_t*)i_cpu.data_ptr(), ((int64_t*)i_cpu.data_ptr()) + i_cpu.numel());
#if 0
    for(int i=0; i<indices_vec.size(); ++i)
    {
        int64_t i_v = indices_vec[i];
        std::cout << "(" <<i_v << " " << i_cpu[i].item().toInt() << ") ";
    }
#endif
    std::sort(indices_vec.begin(), indices_vec.end());
    std::vector<int> indices_to_keep(mesh.NumVertices() - indices_vec.size());

    for (int i = 0, remove_index_it = 0, keep_index_it = 0; i < mesh.NumVertices(); ++i)
    {
        if (indices_vec[remove_index_it] != i)
        {
            indices_to_keep[keep_index_it] = i;
            keep_index_it++;
        }
        else
        {
            remove_index_it++;
        }
    }
    std::cout << "Mesh before rem " << mesh.NumVertices() << std::endl;
    std::vector<int> rem_erase_vec;
    for (auto i_v : indices_vec)
    {
        rem_erase_vec.push_back(i_v);
    }
    // auto rem_erase_vec = std::vector<int>(indices_vec.begin(), indices_vec.end());
    mesh.EraseVertices(rem_erase_vec);
    AABB custom_aabb = scene->dataset_params.aabb;

    std::cout << "Mesh after rem " << mesh.NumVertices() << std::endl;
    point_cloud_cuda = NeuralPointCloudCuda(mesh, params->points_adding_params.use_grid_loss,
                                            params->points_adding_params.cells_worldspace_size, custom_aabb);


    torch::Tensor indices_to_keep_tensor = torch::from_blob(indices_to_keep.data(), {(long)indices_to_keep.size(), 1},
                                                            torch::TensorOptions().dtype(torch::kInt32));
    indices_to_keep_tensor               = indices_to_keep_tensor.to(torch::kLong);
    // update texture
    texture->RemoveAndFlattenTexture(indices_to_keep_tensor);

    if (point_size_save.defined())
    {
        auto values_keep =
            point_size_save.index_select(0, indices_to_keep_tensor.squeeze().to(point_size_save.device()));

        point_cloud_cuda->t_point_size.set_(values_keep);
    }

    if (update_optimizer)
    {
        //    torch::NoGradGuard ngg;
        // CreateTextureOptimizer();
        ShrinkTextureOptimizer(indices_to_keep_tensor);
        // std::cout << "Texture optimizer was reset. There may be a better implementation, but this would require "
        //             "resizing the optimizer's internal states"
        //          << std::endl;
        {
            auto& params_groups = texture_optimizer->param_groups();
            for (auto& p_gr : params_groups)
            {
                auto p_t = p_gr.params();
                for (auto& t : p_t)
                {
                    // resize and reset texture gradients
                    t.mutable_grad() = torch::zeros_like(t).detach_();  // = torch::NullType::singleton();
                    //     std::cout << "GRAD" << TensorInfo(t.mutable_grad()) << std::endl;
                }
            }
        }
        if (!params->optimizer_params.fix_points)
        {
            CreateStructureOptimizer();
            std::vector<torch::Tensor>& params_to = structure_optimizer->parameters();

            for (auto t : params_to)
            {
                // resize and reset texture gradients
                t.mutable_grad() = torch::zeros_like(t).detach_();  // = torch::NullType::singleton();
                //      std::cout << "GRAD" << TensorInfo(t.mutable_grad()) << std::endl;
            }
        }
        texture_optimizer->zero_grad();
    }
}


void NeuralScene::OptimizerStep(int epoch_id, bool structure_only)
{
    if (!structure_only && texture_optimizer)
    {
        // std::cout << "step" << std::endl;
        // PrintTensorInfo(environment_map->color.mutable_grad());
        // PrintTensorInfo(environment_map->density.mutable_grad());
        // PrintTensorInfo(texture->texture.mutable_grad());

        texture_optimizer->step();
        texture_optimizer->zero_grad();
    }
    if (environment_map && !params->optimizer_params.fix_environment_map)
    {
        environment_map->optimizer_adam->step();
        environment_map->optimizer_adam->zero_grad();
    }
    if (epoch_id > params->train_params.lock_camera_params_epochs)
    {
        if (camera_adam_optimizer)
        {
            camera_adam_optimizer->step();
            camera_adam_optimizer->zero_grad();
        }
        if (camera_sgd_optimizer)
        {
            camera_sgd_optimizer->step();
            camera_sgd_optimizer->zero_grad();
        }
        camera->ApplyConstraints();
    }

    if (structure_optimizer && epoch_id > params->train_params.lock_structure_params_epochs)
    {
        // std::cout << TensorInfo(poses->tangent_poses.slice(0, 0, num_images))
        //           << TensorInfo(poses->poses_se3.slice(0, 0, num_images)) << std::endl;
        // std::cout << "step pose" << std::endl;
        structure_optimizer->step();
        // std::cout << TensorInfo(poses->tangent_poses.slice(0, 0, num_images))
        //           << TensorInfo(poses->poses_se3.slice(0, 0, num_images)) << std::endl;
        poses->ApplyTangent();
        // std::cout << "apply pose" << std::endl;
        //
        // std::cout << TensorInfo(poses->tangent_poses.slice(0, 0, num_images))
        //          << TensorInfo(poses->poses_se3.slice(0, 0, num_images)) << std::endl;

        structure_optimizer->zero_grad();
    }

    if (!params->optimizer_params.fix_intrinsics)
    {
        DownloadIntrinsics();
    }
}

void NeuralScene::OptimizerClear(int epoch_id, bool structure_only)
{
    if (!structure_only && texture_optimizer)
    {
        texture_optimizer->zero_grad();
    }

    if (epoch_id > params->train_params.lock_camera_params_epochs)
    {
        if (camera_adam_optimizer)
        {
            camera_adam_optimizer->zero_grad();
        }
        if (camera_sgd_optimizer)
        {
            camera_sgd_optimizer->zero_grad();
        }
        //  camera->ApplyConstraints();
    }

    if (structure_optimizer && epoch_id > params->train_params.lock_structure_params_epochs)
    {
        //        poses->ApplyTangent();
        structure_optimizer->zero_grad();
    }

    if (!params->optimizer_params.fix_intrinsics)
    {
        // DownloadIntrinsics();
    }
}


void NeuralScene::Train(int epoch_id, bool train)
{
    if (texture) texture->train(train);
    if (camera) camera->train(train);


    if (camera_sgd_optimizer) camera_sgd_optimizer->zero_grad();
    if (camera_adam_optimizer) camera_adam_optimizer->zero_grad();
    if (texture_optimizer) texture_optimizer->zero_grad();
    if (structure_optimizer) structure_optimizer->zero_grad();
}

void NeuralScene::UpdateLearningRate(int epoch_id, double factor)
{
    SAIGA_ASSERT(factor > 0);

    double lr_update_adam = factor;

    double lr_update_sgd = factor;

    if (texture_optimizer)
    {
        if (params->optimizer_params.texture_optimizer == "adam")
        {
            UpdateOptimLR(texture_optimizer.get(), lr_update_adam);
        }
        else if (params->optimizer_params.texture_optimizer == "sgd")
        {
            UpdateOptimLR(texture_optimizer.get(), lr_update_sgd);
        }
        else
        {
            SAIGA_EXIT_ERROR("sldg");
        }
    }

    if (epoch_id > params->train_params.lock_camera_params_epochs)
    {
        if (camera_adam_optimizer)
        {
            UpdateOptimLR(camera_adam_optimizer.get(), lr_update_adam);
        }
        if (camera_sgd_optimizer)
        {
            UpdateOptimLR(camera_sgd_optimizer.get(), lr_update_sgd);
        }
    }

    if (structure_optimizer && epoch_id > params->train_params.lock_structure_params_epochs)
    {
        UpdateOptimLR(structure_optimizer.get(), lr_update_sgd);
    }
}

void NeuralScene::DownloadIntrinsics()
{
    // for (auto& cam : scene->scene_cameras)
    //{
    //     if (cam.camera_model_type != CameraModel::PINHOLE_DISTORTION) SAIGA_ASSERT(false, "Not implemented");
    // }
    //  if (scene->dataset_params.camera_model == CameraModel::PINHOLE_DISTORTION)
    {
        auto Ks = intrinsics->DownloadK();
        auto ds = intrinsics->DownloadDistortion();
        SAIGA_ASSERT(Ks.size() == scene->scene_cameras.size());

        for (int i = 0; i < Ks.size(); ++i)
        {
            scene->scene_cameras[i].K          = Ks[i];
            scene->scene_cameras[i].distortion = ds[i];
        }

        // We have do download and update the intrinsic matrix
        // because the cropping has to have the latest version
        params->render_params.dist_cutoff = scene->scene_cameras.front().distortion.MonotonicThreshold();
    }
    // else if (scene->dataset_params.camera_model == CameraModel::OCAM)
    // {
    //     auto Ks = intrinsics->DownloadK();
    //
    // }
}

void NeuralScene::DownloadPoses()
{
    auto new_poses = poses->Download();

    SAIGA_ASSERT(new_poses.size() == scene->frames.size());
    for (int i = 0; i < new_poses.size(); ++i)
    {
        scene->frames[i].pose = new_poses[i].inverse();
    }
}
