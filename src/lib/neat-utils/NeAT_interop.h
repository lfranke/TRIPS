/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#pragma once
#include "saiga/core/util/ini/ini.h"


struct CameraBase : public ParamsBase
{
    SAIGA_PARAM_STRUCT(CameraBase);
    SAIGA_PARAM_STRUCT_FUNCTIONS;
    template <class ParamIterator>
    void Params(ParamIterator* it)

    {
        SAIGA_PARAM(w);
        SAIGA_PARAM(h);

        auto vector2string = [](auto vector)
        {
            std::stringstream sstrm;
            sstrm << std::setprecision(15) << std::scientific;
            for (unsigned int i = 0; i < vector.size(); ++i)
            {
                sstrm << vector[i];
                if (i < vector.size() - 1) sstrm << " ";
            }
            return sstrm.str();
        };



        {
            std::vector<std::string> K = split(vector2string(this->K.cast<double>().coeffs()), ' ');
            SAIGA_PARAM_LIST_COMMENT(K, ' ', "# fx fy cx cy s");
            SAIGA_ASSERT(K.size() == 5);

            Vector<double, 5> K_coeffs;
            for (int i = 0; i < 5; ++i)
            {
                double d    = to_double(K[i]);
                K_coeffs(i) = d;
            }
            this->K = IntrinsicsPinholed(K_coeffs);
        }
    }


    int w = 0;
    int h = 0;
    IntrinsicsPinholed K;
};

struct DatasetParams : public ParamsBase
{
    SAIGA_PARAM_STRUCT(DatasetParams);
    SAIGA_PARAM_STRUCT_FUNCTIONS;
    //    virtual void Params(Saiga::SimpleIni* ini, CLI::App* app) override
    template <class ParamIterator>
    void Params(ParamIterator* it)
    {
        SAIGA_PARAM(image_dir);
        SAIGA_PARAM(mask_dir);
        SAIGA_PARAM(projection_factor);
        SAIGA_PARAM(vis_volume_intensity_factor);
        SAIGA_PARAM(scene_scale);
        SAIGA_PARAM(xray_min);
        SAIGA_PARAM(xray_max);
        SAIGA_PARAM(volume_file);
        SAIGA_PARAM(log_space_input);
        SAIGA_PARAM(use_log10_conversion);

        SAIGA_PARAM(z_min);
    }

    // only set if a ground truth volume exists
    std::string volume_file = "";


    // linear multiplier to the projection
    // otherwise it is just transformed by xray/min/max parameters
    double projection_factor = 1;


    // Only for visualization!
    // multiplied to the intensity of the projection (after normalization)
    double vis_volume_intensity_factor = 1;

    // the camera position is multiplied by this factor to "scale" the scene
    double scene_scale = 1;

    std::string image_dir = "";
    std::string mask_dir  = "";

    // "real" raw xray is usually NOT in log space (background is white)
    // if the data is already preprocessed and converted to log space (background is black)
    // set this flag in the dataset
    bool log_space_input = false;

    // pepper:13046, 65535
    // ropeball: 26000, 63600
    double xray_min = 0;
    double xray_max = 65535;

    double z_min = 0;

    // true: log10
    // false: loge
    bool use_log10_conversion = true;
};


struct MyNeATTrainParams : public TrainParams
{
    using ParamStructType = MyNeATTrainParams;
    //    MyTrainParams() {}
    //    MyTrainParams(const std::string file) { Load(file); }

    //    using ParamStructType = MyTrainParams;
    // SAIGA_PARAM_STRUCT(MyTrainParams);

    MyNeATTrainParams() {}
    MyNeATTrainParams(const std::string file) : TrainParams(file) {}


    SAIGA_PARAM_STRUCT_FUNCTIONS;

    //    SAIGA_PARAM_STRUCT_FUNCTIONS;

    //    virtual void Params(Saiga::SimpleIni* ini, CLI::App* app) override
    template <class ParamIterator>
    void Params(ParamIterator* it)
    {
        TrainParams::Params(it);

        SAIGA_PARAM(scene_dir);
        SAIGA_PARAM_LIST2(scene_name, ' ');
        SAIGA_PARAM(split_name);

        SAIGA_PARAM(optimize_structure_every_epochs);
        SAIGA_PARAM(optimize_structure_convergence);
        SAIGA_PARAM(per_node_batch_size);
        SAIGA_PARAM(rays_per_image);
        SAIGA_PARAM(output_volume_size);

        SAIGA_PARAM(lr_exex_grid_adam);
        SAIGA_PARAM(optimize_pose);
        SAIGA_PARAM(optimize_intrinsics);
        SAIGA_PARAM(lr_pose);
        SAIGA_PARAM(lr_intrinsics);
        SAIGA_PARAM(lr_decay_factor);
        SAIGA_PARAM(optimize_structure_after_epochs);
        SAIGA_PARAM(optimize_tree_structure_after_epochs);
        SAIGA_PARAM(optimize_tone_mapper_after_epochs);
        SAIGA_PARAM(init_bias_with_bg);
        SAIGA_PARAM(grid_init);
        SAIGA_PARAM(loss_tv);
        SAIGA_PARAM(loss_edge);
        SAIGA_PARAM(eval_scale);

        SAIGA_PARAM(experiment_name_str);
        SAIGA_PARAM(experiment_dir_override);

        SAIGA_PARAM(temp_image_dir);
    }

    std::string scene_dir               = "";
    std::vector<std::string> scene_name = {"not_set"};
    std::string split_name              = "exp_uniform_50";
    std::string experiment_name_str     = "";
    std::string experiment_dir_override = "";
    std::string temp_image_dir          = "";

    int optimize_structure_every_epochs  = 1;
    float optimize_structure_convergence = 0.95;

    std::string grid_init = "uniform";

    int rays_per_image      = 50000;
    int per_node_batch_size = 256;
    int output_volume_size  = 256;

    double lr_decay_factor = 1.0;

    double eval_scale = 0.25;

    double loss_tv   = 1e-4;
    double loss_edge = 1e-3;


    float lr_exex_grid_adam = 0.04;

    // On each image we compute the median value of a top right corner crop
    // This is used to initialize the tone-mapper's bias value
    bool init_bias_with_bg = false;

    // In the first few epochs we keep the camera pose/model fixed!
    int optimize_tree_structure_after_epochs = 1;
    int optimize_structure_after_epochs      = 5;
    int optimize_tone_mapper_after_epochs    = 1;
    bool optimize_pose                       = true;
    bool optimize_intrinsics                 = true;
    float lr_pose                            = 0.001;
    float lr_intrinsics                      = 100;
};

struct Netparams : public ParamsBase
{
    SAIGA_PARAM_STRUCT(Netparams);
    SAIGA_PARAM_STRUCT_FUNCTIONS;
    //    virtual void Params(Saiga::SimpleIni* ini, CLI::App* app) override
    template <class ParamIterator>
    void Params(ParamIterator* it)
    {
        SAIGA_PARAM(grid_size);
        SAIGA_PARAM(grid_features);
        SAIGA_PARAM(last_activation_function);
        SAIGA_PARAM(softplus_beta);

        SAIGA_PARAM(decoder_skip);
        SAIGA_PARAM(decoder_lr);
        SAIGA_PARAM(decoder_activation);
        SAIGA_PARAM(decoder_hidden_layers);
        SAIGA_PARAM(decoder_hidden_features);
    }

    int grid_size     = 17;
    int grid_features = 4;

    // relu, id, abs
    std::string last_activation_function = "softplus";
    float softplus_beta                  = 2;

    bool decoder_skip              = false;
    float decoder_lr               = 0.00005;
    std::string decoder_activation = "silu";
    int decoder_hidden_layers      = 1;
    int decoder_hidden_features    = 64;
};


struct TreeOptimizerParams
{
    int num_threads       = 16;
    bool use_saved_errors = true;
    int max_active_nodes  = 512;
    bool verbose          = false;

    double error_merge_factor = 1.1;
    double error_split_factor = 0.75;
};
// Params for the HyperTree
struct OctreeParams : public ParamsBase
{
    SAIGA_PARAM_STRUCT(OctreeParams);
    SAIGA_PARAM_STRUCT_FUNCTIONS;
    //    virtual void Params(Saiga::SimpleIni* ini, CLI::App* app) override
    template <class ParamIterator>
    void Params(ParamIterator* it)
    {
        SAIGA_PARAM(start_layer);
        SAIGA_PARAM(tree_depth);
        SAIGA_PARAM(optimize_structure);
        SAIGA_PARAM(max_samples_per_node);
        SAIGA_PARAM(culling_start_epoch);
        SAIGA_PARAM(node_culling);
        SAIGA_PARAM(culling_threshold);


        SAIGA_PARAM(tree_optimizer_params.use_saved_errors);
        SAIGA_PARAM(tree_optimizer_params.max_active_nodes);
        SAIGA_PARAM(tree_optimizer_params.error_merge_factor);
        SAIGA_PARAM(tree_optimizer_params.error_split_factor);
        SAIGA_PARAM(tree_optimizer_params.verbose);
    }

    int start_layer         = 3;
    int tree_depth          = 4;
    bool optimize_structure = true;

    int max_samples_per_node = 32;

    int culling_start_epoch = 4;
    bool node_culling       = true;

    // 0.01 for mean, 0.4 for max
    float culling_threshold = 0.1;


    TreeOptimizerParams tree_optimizer_params;
};

struct PhotometricCalibrationParams : public ParamsBase
{
    SAIGA_PARAM_STRUCT(PhotometricCalibrationParams);
    SAIGA_PARAM_STRUCT_FUNCTIONS;

    //    virtual void Params(Saiga::SimpleIni* ini, CLI::App* app) override
    template <class ParamIterator>
    void Params(ParamIterator* it)
    {
        SAIGA_PARAM(response_enable);
        SAIGA_PARAM(response_range);
        SAIGA_PARAM(response_lr);

        SAIGA_PARAM(exposure_enable);
        SAIGA_PARAM(exposure_mult);
        SAIGA_PARAM(exposure_lr);


        SAIGA_PARAM(sensor_bias_enable);
        SAIGA_PARAM(sensor_bias_size_w);
        SAIGA_PARAM(sensor_bias_lr);
    }

    bool response_enable = true;
    float response_range = 2;
    float response_lr    = 0.1;

    bool exposure_enable = false;
    bool exposure_mult   = false;
    float exposure_lr    = 0.01;

    // the size in h will be computed from the aspect ratio
    bool sensor_bias_enable = false;
    int sensor_bias_size_w  = 32;
    float sensor_bias_lr    = 0.05;
};


struct NeATCombinedParams
{
    MyNeATTrainParams train_params;
    OctreeParams octree_params;
    Netparams net_params;
    PhotometricCalibrationParams photo_calib_params;

    NeATCombinedParams() {}
    NeATCombinedParams(const std::string& combined_file)
        : train_params(combined_file),
          octree_params(combined_file),
          net_params(combined_file),
          photo_calib_params(combined_file)
    {
    }

    void Save(const std::string file)
    {
        train_params.Save(file);
        octree_params.Save(file);
        net_params.Save(file);
        photo_calib_params.Save(file);
    }

    void Load(std::string file)
    {
        train_params.Load(file);
        octree_params.Load(file);
        net_params.Load(file);
        photo_calib_params.Load(file);
    }

    void Load(CLI::App& app)
    {
        train_params.Load(app);
        octree_params.Load(app);
        net_params.Load(app);
        photo_calib_params.Load(app);
    }
};