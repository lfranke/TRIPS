/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#pragma once

#include "saiga/core/util/ini/ini.h"
#include "saiga/vision/torch/PartialConvUnet2d.h"
#include "saiga/vision/torch/TrainParameters.h"

#include "config.h"

#include <string>


using namespace Saiga;

struct RenderParams : public ParamsBase
{
    SAIGA_PARAM_STRUCT(RenderParams);

    SAIGA_PARAM_STRUCT_FUNCTIONS;


    // only for debugging
    int test_backward_mode = 0;
    float test_refl_x      = 0.f;
    float test_refl_y      = 0.f;
    float test_refl_z      = 0.f;

    bool render_points   = true;
    bool render_outliers = false;
    int outlier_count    = 1000000;
    bool check_normal    = false;

    bool use_layer_point_size = false;

    // double res rendering + average pool
    bool super_sampling = false;

    float dropout            = 0.25;
    float depth_accept       = 0.01;
    float depth_accept_blend = 0.01;

    bool ghost_gradients            = false;
    float drop_out_radius_threshold = 0.6;
    bool drop_out_points_by_radius  = true;

    // Writes the weight into the 4-channel output texture
    bool debug_weight_color              = false;
    bool debug_depth_color               = false;
    float debug_max_weight               = 10;
    bool debug_print_num_rendered_points = false;

    float distortion_gradient_factor = 0.005;
    float K_gradient_factor          = 0.5;

    // == parameters set by the system ==
    int num_texture_channels           = -1;
    float dist_cutoff                  = 20.f;
    bool output_background_mask        = false;
    float output_background_mask_value = 0;


    bool add_depth_to_network                 = false;
    float stability_cutoff_value              = 10000.f;
    bool use_point_adding_and_removing_module = true;
    bool need_point_gradient                  = true;
    bool use_environment_map                  = true;
    int render_mode                           = 1;
    bool viewer_only                          = false;
    int debug_max_list_length                 = -1;

    bool normalize_grads = false;

    bool combine_lists                          = false;
    bool render_points_in_all_lower_resolutions = true;

    bool saturated_alpha_accumulation = false;

    // TODO MAKE MANDATORY TRUE OR FIX ENV MAP ALPHA CONT
    bool no_envmap_at_points = false;

    template <class ParamIterator>
    void Params(ParamIterator* it)
    {
        // SAIGA_PARAM(render_outliers);
        SAIGA_PARAM(check_normal);
        // SAIGA_PARAM(ghost_gradients);
        SAIGA_PARAM(drop_out_points_by_radius);
        // SAIGA_PARAM(outlier_count);
        SAIGA_PARAM(drop_out_radius_threshold);
        SAIGA_PARAM(dropout);
        SAIGA_PARAM(depth_accept);
        SAIGA_PARAM(depth_accept_blend);
        // SAIGA_PARAM(test_backward_mode);
        // SAIGA_PARAM(distortion_gradient_factor);
        // SAIGA_PARAM(K_gradient_factor);
        // SAIGA_PARAM(add_depth_to_network);
        SAIGA_PARAM(no_envmap_at_points);
        SAIGA_PARAM(normalize_grads);
        // SAIGA_PARAM(combine_lists);
        // SAIGA_PARAM(render_points_in_all_lower_resolutions);

        // SAIGA_PARAM(saturated_alpha_accumulation);
    }
};

struct NeuralCameraParams : public ParamsBase
{
    SAIGA_PARAM_STRUCT(NeuralCameraParams);

    SAIGA_PARAM_STRUCT_FUNCTIONS;

    bool enable_vignette = true;
    bool enable_exposure = true;
    bool enable_response = true;

    bool enable_white_balance   = false;
    bool enable_motion_blur     = false;
    bool enable_rolling_shutter = false;

    int response_params        = 25;
    float response_gamma       = 1.0 / 2.2;
    float response_leak_factor = 0.01;

    template <class ParamIterator>
    void Params(ParamIterator* it)
    {
        SAIGA_PARAM(enable_vignette);
        SAIGA_PARAM(enable_exposure);
        SAIGA_PARAM(enable_response);

        SAIGA_PARAM(enable_white_balance);
        // SAIGA_PARAM(enable_motion_blur);
        // SAIGA_PARAM(enable_rolling_shutter);

        SAIGA_PARAM(response_params);
        SAIGA_PARAM(response_gamma);
        SAIGA_PARAM(response_leak_factor);
    }
};

struct OptimizerParams : public ParamsBase
{
    SAIGA_PARAM_STRUCT(OptimizerParams);

    SAIGA_PARAM_STRUCT_FUNCTIONS;

    template <class ParamIterator>
    void Params(ParamIterator* it)
    {
        SAIGA_PARAM(network_checkpoint_directory);

        SAIGA_PARAM(texture_optimizer);
        SAIGA_PARAM(use_myadam_everywhere);

        SAIGA_PARAM(fix_render_network);
        SAIGA_PARAM(fix_texture);
        SAIGA_PARAM(fix_environment_map);

        // structure
        SAIGA_PARAM(fix_points);
        SAIGA_PARAM(fix_poses);
        SAIGA_PARAM(fix_intrinsics);
        SAIGA_PARAM(fix_dynamic_refinement);
        SAIGA_PARAM(fix_point_size);

        // camera
        SAIGA_PARAM(fix_vignette);
        SAIGA_PARAM(fix_response);
        SAIGA_PARAM(fix_wb);
        SAIGA_PARAM(fix_exposure);
        SAIGA_PARAM(fix_motion_blur);
        SAIGA_PARAM(fix_rolling_shutter);

        SAIGA_PARAM(lr_render_network);
        SAIGA_PARAM(lr_texture);
        SAIGA_PARAM(lr_background_color);
        SAIGA_PARAM(lr_dynamic_refinement);
        SAIGA_PARAM(lr_environment_map);
        SAIGA_PARAM(lr_confidence);
        SAIGA_PARAM(lr_layer);
        SAIGA_PARAM(lr_environment_map_density);

        SAIGA_PARAM(lr_points);
        SAIGA_PARAM(lr_poses);
        SAIGA_PARAM(lr_intrinsics);


        SAIGA_PARAM(response_smoothness);
        SAIGA_PARAM(lr_vignette);
        SAIGA_PARAM(lr_response);
        SAIGA_PARAM(lr_wb);
        SAIGA_PARAM(lr_exposure);
        SAIGA_PARAM(lr_motion_blur);
        SAIGA_PARAM(lr_rolling_shutter);
    }

    std::string network_checkpoint_directory = "";

    std::string texture_optimizer = "adam";

    bool use_myadam_everywhere = true;

    bool fix_render_network  = false;
    bool fix_texture         = false;
    bool fix_environment_map = false;

    // structure
    bool fix_points             = true;
    bool fix_poses              = true;
    bool fix_intrinsics         = true;
    bool fix_dynamic_refinement = true;
    bool fix_point_size         = false;

    // camera
    bool fix_vignette        = true;
    bool fix_response        = true;
    bool fix_wb              = true;
    bool fix_exposure        = true;
    bool fix_motion_blur     = true;
    bool fix_rolling_shutter = true;

    double lr_render_network     = 0.0002;
    double lr_texture            = 0.08;   // log_texture: 0.01
    double lr_background_color   = 0.004;  // log_texture 0.0005
    double lr_environment_map    = 0.02;   // log_texture: 0.0025
    double lr_dynamic_refinement = 0.005;

    // structure
    double lr_points     = 0.005;
    double lr_poses      = 0.01;
    double lr_intrinsics = 1;

    double lr_layer = 0.01;

    double lr_confidence              = 0.001;
    double lr_environment_map_density = 0.02;
    // camera
    double lr_vignette         = 5e-6;  // sgd: 5e-7, adam 1e-4
    double lr_response         = 0.001;
    double response_smoothness = 1;
    double lr_wb               = 5e-4;
    double lr_exposure         = 0.0005;  // sgd 5e-4, adam 1e-3
    double lr_motion_blur      = 0.005;
    double lr_rolling_shutter  = 2e-6;
};



struct EnvironmentMapParams
{
    // same as network input, if not cat_env_to_color-ed
    int env_map_channels = 4;

    bool use_points_for_env_map        = true;
    int env_num_points                 = 500000;
    bool only_add_points_once          = true;
    bool start_with_environment_points = false;

    int env_map_resolution  = 1024;
    int env_spheres         = 4;
    float env_inner_radius  = 10.f;
    float env_radius_factor = 5.f;
    int env_axis            = 0;

    // Concats the mask/env_map along the channel dimension
    // This increases the number of input channels of the network
    bool cat_env_to_color = false;
};

struct PipelineParams : public ParamsBase
{
    SAIGA_PARAM_STRUCT(PipelineParams);

    SAIGA_PARAM_STRUCT_FUNCTIONS;

    template <class ParamIterator>
    void Params(ParamIterator* it)
    {
        SAIGA_PARAM(train);

        SAIGA_PARAM(verbose_eval);
        SAIGA_PARAM(log_render);
        SAIGA_PARAM(log_texture);
        SAIGA_PARAM(non_subzero_texture);
        SAIGA_PARAM(skip_neural_render_network);
        SAIGA_PARAM(skip_neural_render_network_but_add_layers);

        SAIGA_PARAM(num_texture_channels);
        SAIGA_PARAM(num_spherical_harmonics_bands_per_point);
        SAIGA_PARAM(cat_masks_to_color);
        // SAIGA_PARAM(use_point_adding_and_removing_module);
        SAIGA_PARAM_LIST(render_modes_start_epochs, ' ');

        SAIGA_PARAM(enable_environment_map);
        SAIGA_PARAM(environment_map_params.use_points_for_env_map);
        SAIGA_PARAM(environment_map_params.env_num_points);
        SAIGA_PARAM(environment_map_params.start_with_environment_points);
        SAIGA_PARAM(environment_map_params.only_add_points_once);

        SAIGA_PARAM(environment_map_params.env_map_resolution);
        SAIGA_PARAM(environment_map_params.env_map_channels);
        SAIGA_PARAM(environment_map_params.env_inner_radius);
        SAIGA_PARAM(environment_map_params.env_radius_factor);
        SAIGA_PARAM(environment_map_params.env_spheres);
        SAIGA_PARAM(environment_map_params.env_axis);
        SAIGA_PARAM(environment_map_params.cat_env_to_color);
    }

    bool train = true;

    bool verbose_eval                              = false;
    bool log_render                                = false;
    bool log_texture                               = false;
    bool skip_neural_render_network                = false;
    bool skip_neural_render_network_but_add_layers = false;
    bool skip_sensor_model                         = false;
    bool non_subzero_texture                       = false;

    int num_spherical_harmonics_bands_per_point = -1;

    bool enable_environment_map = false;
    EnvironmentMapParams environment_map_params;

    int num_texture_channels = 4;

    bool cat_masks_to_color = false;
    // bool add_depth_to_network = false;

    // use point manipulations -> add importance to each point and use it according to the PointAddingParams
    bool use_point_adding_and_removing_module = true;

    std::vector<int> render_modes_start_epochs = {-1, 0, -1, -1, -1};
};



struct PointAddingParams : public ParamsBase
{
    SAIGA_PARAM_STRUCT(PointAddingParams);

    SAIGA_PARAM_STRUCT_FUNCTIONS;

    template <class ParamIterator>
    void Params(ParamIterator* it)
    {
        SAIGA_PARAM(start_adding_points_epoch);
        SAIGA_PARAM(point_adding_epoch_interval);
        SAIGA_PARAM(add_points_amount_max_per_cell);

        SAIGA_PARAM(start_removing_points_epoch);
        SAIGA_PARAM(point_removal_epoch_interval);
        SAIGA_PARAM(removal_confidence_cutoff);

        SAIGA_PARAM(scene_add_initially_random_points);
        SAIGA_PARAM(dont_use_initial_pointcloud);

        // SAIGA_PARAM(cells_worldspace_size);

        // SAIGA_PARAM(fixed_ct_reco_path);
        if (fixed_ct_reco_path != "") SAIGA_ASSERT(std::filesystem::exists(fixed_ct_reco_path));

        // SAIGA_PARAM(switch_to_fuzzy_rendering_after_epoch);
        SAIGA_PARAM(sigmoid_narrowing_factor);


        // SAIGA_PARAM(only_use_point_growing);

        // SAIGA_PARAM(neat_use_as_subprocess_ct_reco);
        // SAIGA_PARAM(full_path_to_neat_executable);
        SAIGA_PARAM(neat_loss_folder_name);

        // SAIGA_PARAM(neat_scene_scale);

        if (neat_use_as_subprocess_ct_reco && full_path_to_neat_executable.empty())
        {
            //    SAIGA_ASSERT(false, "please set NeAT executable path");
        }

        if (fixed_ct_reco_path.empty() && !neat_use_as_subprocess_ct_reco)
        {
            use_grid_loss = true;
            std::cout << "USE GRID BASED LOSS PROJ INSTEAD OF CT" << std::endl;
        }

        SAIGA_PARAM(neat_zmin);
        SAIGA_PARAM(neat_tv_loss);

        // SAIGA_PARAM(debug_max_list_length);
        SAIGA_PARAM(push_point_confidences_down);
    }



    // initial adding of points, -1 deactivated point adding
    int start_adding_points_epoch = 150;
    // repeated adding of points
    int point_adding_epoch_interval = 200;

    // point removal, -1 deactivates point removal
    int start_removing_points_epoch = 200;
    // repeated removal of points: e == start_removing_points_epoch + i*point_removal_epoch_interval
    int point_removal_epoch_interval = 50;

    int add_points_amount_max_per_cell = 20;

    // using a grid structure overlayed, add in this amount of boxes
    float add_only_in_top_x_factor_of_cells = 0.05;

    float cells_worldspace_size           = 1.f;
    int scene_add_initially_random_points = 0;
    bool dont_use_initial_pointcloud      = false;

    float sigmoid_narrowing_factor = 0.f;

    std::string fixed_ct_reco_path = "";

    std::string full_path_to_neat_executable = "build/NeAT/src/NeAT-build/bin/reconstruct";

    float push_point_confidences_down = 0.f;

    float neat_scene_scale = 1.f;

    float removal_confidence_cutoff = 0.3;

    // point growing strategy
    bool only_use_point_growing = false;
    // use NeAT as subprocess
    bool neat_use_as_subprocess_ct_reco = true;
    std::string neat_loss_folder_name   = "l1_loss_grey";

    // use grid based loss projection if Neat or other CT reco is not available
    bool use_grid_loss = false;

    float neat_zmin           = 0.f;
    int debug_max_list_length = -1;
    float neat_tv_loss        = 0.0001;
};

struct MyTrainParams : public TrainParams
{
    MyTrainParams() {}

    MyTrainParams(const std::string file) { Load(file); }

    using ParamStructType = MyTrainParams;

    SAIGA_PARAM_STRUCT_FUNCTIONS;

    template <class ParamIterator>
    void Params(ParamIterator* it)
    {
        TrainParams::Params(it);
        SAIGA_PARAM(train_crop_size);
        SAIGA_PARAM(train_mask_border);
        SAIGA_PARAM(reduced_check_point);
        SAIGA_PARAM(write_images_at_checkpoint);
        SAIGA_PARAM(keep_all_scenes_in_memory);

        SAIGA_PARAM(use_image_masks);
        SAIGA_PARAM(write_test_images);
        SAIGA_PARAM(texture_random_init);
        SAIGA_PARAM(texture_color_init);
        SAIGA_PARAM(train_use_crop);

        SAIGA_PARAM(experiment_dir);
        SAIGA_PARAM(scene_base_dir);
        SAIGA_PARAM_LIST(scene_names, ' ');

        SAIGA_PARAM(loss_vgg);
        SAIGA_PARAM(loss_l1);
        SAIGA_PARAM(loss_mse);
        SAIGA_PARAM(loss_ssim);
        SAIGA_PARAM(loss_lpips);

        SAIGA_PARAM(min_zoom);
        SAIGA_PARAM(max_zoom);
        SAIGA_PARAM(warmup_epochs);
        SAIGA_PARAM(only_start_vgg_after_epochs);

        SAIGA_PARAM(crop_prefere_border);
        SAIGA_PARAM(crop_gaussian_sample);
        SAIGA_PARAM(crop_rotation);
        SAIGA_PARAM(crop_max_dis_center);
        SAIGA_PARAM(optimize_eval_camera);
        SAIGA_PARAM(interpolate_eval_settings);

        SAIGA_PARAM(noise_pose_r);
        SAIGA_PARAM(noise_pose_t);
        SAIGA_PARAM(noise_intr_k);
        SAIGA_PARAM(noise_intr_d);
        SAIGA_PARAM(noise_point);
        SAIGA_PARAM(noise_point_size);

        SAIGA_PARAM(lr_decay_factor);
        SAIGA_PARAM(lr_decay_patience);
        SAIGA_PARAM(lock_camera_params_epochs);
        SAIGA_PARAM(lock_structure_params_epochs);

        SAIGA_PARAM(lock_dynamic_refinement_epochs);

        SAIGA_PARAM(temp_image_dir);

        SAIGA_PARAM(crop_for_loss);
        SAIGA_PARAM(vgg_path);
    }

    bool crop_for_loss   = true;
    std::string vgg_path = "loss/traced_caffe_vgg_optim.pt";

    // transformation
    float noise_pose_r     = 0;  // in Degrees
    float noise_pose_t     = 0;  // in mm
    float noise_intr_k     = 0;
    float noise_intr_d     = 0;
    float noise_point      = 0;
    float noise_point_size = 0;

    float min_zoom           = 0.75;
    float max_zoom           = 1.5f;
    bool crop_prefere_border = true;

    double loss_vgg   = 1.0;
    double loss_l1    = 1.0;
    double loss_mse   = 0.0;
    double loss_ssim  = 0.0;
    double loss_lpips = 0.0;

    int max_eval_size = 200000;

    int only_start_vgg_after_epochs = -1;

    bool use_image_masks      = false;
    int train_crop_size       = 256;
    bool train_use_crop       = true;
    int train_mask_border     = 16;
    bool crop_gaussian_sample = false;
    bool crop_rotation        = false;
    int crop_max_dis_center   = -1;

    bool keep_all_scenes_in_memory  = false;
    bool reduced_check_point        = false;
    bool write_images_at_checkpoint = true;
    bool write_test_images          = false;
    bool texture_random_init        = true;
    bool texture_color_init         = false;

    bool optimize_eval_camera = false;

    int warmup_epochs = -1;


    // Interpolate estimated exposure values for test-images from neighbor frames
    // Assumes that the images were captured sequentially
    bool interpolate_eval_settings = false;

    std::string experiment_dir           = "experiments/";
    std::string scene_base_dir           = "scenes/";
    std::vector<std::string> scene_names = {"bo"};

    std::string temp_image_dir = "";


    // in epoch 1 the lr is x
    // in epoch <max_epoch> the lr is x / 10
    float lr_decay_factor = 0.75;
    int lr_decay_patience = 15;


    // In the first few iterations we do not optimize camera parameters
    // such as vignetting and CRF because the solution is still too far of a reasonable result
    int lock_camera_params_epochs      = 25;
    int lock_structure_params_epochs   = 25;
    int lock_dynamic_refinement_epochs = 50;
};


struct CombinedParams
{
    MyTrainParams train_params;
    RenderParams render_params;
    PipelineParams pipeline_params;
    PointAddingParams points_adding_params;
    OptimizerParams optimizer_params;
    NeuralCameraParams camera_params;
    MultiScaleUnet2dParams net_params;

    CombinedParams() {}

    CombinedParams(const std::string& combined_file)
        : train_params(combined_file),
          render_params(combined_file),
          pipeline_params(combined_file),
          points_adding_params(combined_file),
          optimizer_params(combined_file),
          camera_params(combined_file),
          net_params(combined_file)
    {
    }

    void Save(const std::string file)
    {
        train_params.Save(file);
        render_params.Save(file);
        pipeline_params.Save(file);
        points_adding_params.Save(file);
        optimizer_params.Save(file);
        camera_params.Save(file);
        net_params.Save(file);
    }

    void Load(std::string file)
    {
        train_params.Load(file);
        render_params.Load(file);
        pipeline_params.Load(file);
        optimizer_params.Load(file);
        points_adding_params.Load(file);
        camera_params.Load(file);
        net_params.Load(file);
    }

    void Load(CLI::App& app)
    {
        train_params.Load(app);
        render_params.Load(app);
        pipeline_params.Load(app);
        optimizer_params.Load(app);
        points_adding_params.Load(app);
        camera_params.Load(app);
        net_params.Load(app);
    }


    void Check();

    void imgui();
};
