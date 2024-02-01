/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#pragma once
#include "saiga/cuda/imageProcessing/image.h"
#include "saiga/cuda/imgui_cuda.h"
#include "saiga/vision/torch/CudaHelper.h"

#include "NeuralPointCloudCuda.h"
#include "PointRenderer.h"
#include "RenderConstants.h"
#include "RenderInfo.h"
#include "config.h"
#include "data/Dataset.h"
#include "data/NeuralScene.h"
#include "data/Settings.h"

#include "cooperative_groups.h"
#include <curand_kernel.h>



#ifdef CUDA_DEBUG
#    define CUDA_DEBUG_ASSERT(_x) CUDA_KERNEL_ASSERT(_x)
#else
#    define CUDA_DEBUG_ASSERT(_x)
#endif
using namespace Saiga;
// constants
static constexpr int default_block_size = 128;
static constexpr int POINT_PER_THREAD   = 1;
#define POINT_BATCH_SIZE 10240
#define MAX_DEPTH_CONST 100000.f

using Packtype           = unsigned long long;
constexpr int max_layers = 8;


struct DeviceRenderParams
{
    int num_texture_channels;
    bool check_normal;
    float dropout;
    bool ghost_gradients;
    float dist_cutoff;
    int num_layers;
    float depth_accept;
    float depth_accept_blend;
    float drop_out_radius_threshold;
    bool drop_out_points_by_radius;
    int test_backward_mode;
    float distortion_gradient_factor;
    float K_gradient_factor;

    float debug_test_refl_x;
    float debug_test_refl_y;
    float debug_test_refl_z;
    int current_epoch;
    bool use_point_adding_and_removing_module;
    float stability_cutoff_value;
    // float gradient_spread;
    bool viewer_only;
    int debug_max_list_length;

    bool use_layer_point_size;
    bool combine_lists;
    bool render_points_in_all_lower_resolutions;
    bool saturated_alpha_accumulation;
    // For every layer a batch of images
    // [layers, batches, 1, height_of_layer, width_of_layer]
    StaticDeviceTensor<float, 4> depth[max_layers];
    StaticDeviceTensor<float, 4> weight[max_layers];
    StaticDeviceTensor<float, 4> max_depth[max_layers];

    StaticDeviceTensor<int, 4> counting[max_layers];
    StaticDeviceTensor<int, 4> per_image_atomic_counters[max_layers];

    StaticDeviceTensor<float, 4> gradient_of_forward_pass_x;

    StaticDeviceTensor<float, 3> tmp_projections;

    // for every image one pose
    Sophus::SE3d* _poses;
    curandState* curand_state;

    HD inline Sophus::SE3f Pose(int image_index) { return _poses[image_index].cast<float>(); }

    // [num_cameras, num_model_params]
    StaticDeviceTensor<float, 2> intrinsics;

    HD inline thrust::pair<IntrinsicsPinholef, Distortionf> PinholeIntrinsics(int camera_index)
    {
        float* ptr             = &intrinsics(camera_index, 0);
        IntrinsicsPinholef K   = ((vec5*)ptr)[0];
        Distortionf distortion = ((vec8*)(ptr + 5))[0];

        return {K, distortion};
    }

    HD inline thrust::pair<Vector<float, 5>, ArrayView<const float>> OcamIntrinsics(int camera_index)
    {
        float* ptr = &intrinsics(camera_index, 0);
        int count  = intrinsics.sizes[1];

        Vector<float, 5> aff = ((vec5*)ptr)[0];
        ArrayView<const float> poly((ptr + 5), count - 5);

        return {aff, poly};
    }

    __host__ __device__ DeviceRenderParams() = default;

    DeviceRenderParams(RenderParams params)
    {
        num_texture_channels                   = params.num_texture_channels;
        use_point_adding_and_removing_module   = params.use_point_adding_and_removing_module;
        check_normal                           = params.check_normal;
        dropout                                = params.dropout;
        ghost_gradients                        = params.ghost_gradients;
        dist_cutoff                            = params.dist_cutoff;
        depth_accept                           = params.depth_accept;
        drop_out_points_by_radius              = params.drop_out_points_by_radius;
        drop_out_radius_threshold              = params.drop_out_radius_threshold;
        test_backward_mode                     = params.test_backward_mode;
        distortion_gradient_factor             = params.distortion_gradient_factor;
        K_gradient_factor                      = params.K_gradient_factor;
        debug_test_refl_x                      = params.test_refl_x;
        debug_test_refl_y                      = params.test_refl_y;
        debug_test_refl_z                      = params.test_refl_z;
        stability_cutoff_value                 = params.stability_cutoff_value;
        viewer_only                            = params.viewer_only;
        debug_max_list_length                  = params.debug_max_list_length;
        depth_accept_blend                     = params.depth_accept_blend;
        use_layer_point_size                   = params.use_layer_point_size;
        combine_lists                          = params.combine_lists;
        render_points_in_all_lower_resolutions = params.render_points_in_all_lower_resolutions;
        saturated_alpha_accumulation           = params.saturated_alpha_accumulation;
    }
};

struct DeviceTexture
{
    StaticDeviceTensor<float, 2> in_texture;
    StaticDeviceTensor<float, 2> points_confidence_value;
};

struct DeviceAlphaCompositionParams
{
    StaticDeviceTensor<int, 2> collections[max_layers];
    StaticDeviceTensor<float, 2> per_point_data[max_layers];
    StaticDeviceTensor<float, 2> gradient_sum_backwards[max_layers];
    StaticDeviceTensor<int, 3> scanned_countings[max_layers];
    StaticDeviceTensor<int, 1> ticket_counter[max_layers];
};
// #endif

struct DeviceBilinearAlphaParams
{
    /*(batch,h,w)*/
    StaticDeviceTensor<int32_t, 3> per_pixel_list_heads[max_layers];
    /*(batch,h,w)*/
    StaticDeviceTensor<int32_t, 3> per_pixel_list_lengths[max_layers];
    /*(batch,h,w)*/
    StaticDeviceTensor<int32_t, 3> scanned_countings[max_layers];
    /*(batch,point_cloud_size*4)*/
    StaticDeviceTensor<int32_t, 2> per_pixel_list_links[max_layers];
    /*(batch,tile_h,tile_w,max_num_tile_elements,4)*/
    StaticDeviceTensor<double, 4> per_pixel_sorted_bilin_lists[max_layers];

    /*(batch,h,w,max_num_tile_elements,6)*/
    StaticDeviceTensor<float, 5> bw_sorted_maxed[max_layers];
};

struct DeviceForwardParams
{
    /*(batch,num_desc,h,w)*/
    StaticDeviceTensor<float, 4> neural_out[max_layers];
    // StaticDeviceTensor<float, 4> blend_out[max_layers];
};

struct DeviceBackwardParams
{
    Vec6* out_gradient_pose;
    float* out_gradient_pose_count;

    vec4* out_gradient_points;
    float* out_gradient_points_count;


    // vec4* out_gradient_dynamic_points;
    // float* out_gradient_dynamic_points_count;
    StaticDeviceTensor<float, 3> out_gradient_dynamic_points;
    StaticDeviceTensor<float, 2> out_gradient_dynamic_points_count;

    StaticDeviceTensor<float, 2> out_gradient_intrinsics;
    float* out_gradient_intrinsics_count;

    StaticDeviceTensor<float, 2> out_gradient_texture;
    StaticDeviceTensor<float, 2> out_gradient_confidence;
    StaticDeviceTensor<float, 2> out_gradient_layer;
    StaticDeviceTensor<float, 4> in_gradient_image[max_layers];
};
