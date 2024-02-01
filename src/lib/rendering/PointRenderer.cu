/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// #undef CUDA_DEBUG
// #define CUDA_NDEBUG

// #include "saiga/colorize.h"
#include "saiga/cuda/random.h"
#include "saiga/cuda/reduce.h"
#include "saiga/vision/torch/CudaHelper.h"

#include "PointRenderer.h"
#include "PointRendererHelper.h"
#include "RenderConstants.h"

#include "cooperative_groups.h"
#include <curand_kernel.h>

// curandState* curand_state_h;



__global__ void ProjectDirectionToWSKernel(StaticDeviceTensor<float, 3> in_directions,
                                           StaticDeviceTensor<float, 3> out_directions, Sophus::SE3d* pose)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    if (gx >= in_directions.size(2) || gy >= in_directions.size(1)) return;

    vec3 dir_vs = vec3(in_directions(0, gy, gx), in_directions(1, gy, gx), in_directions(2, gy, gx));
    //  CUDA_KERNEL_ASSERT(dir_vs.z() != 0);
    Sophus::SE3d inv_pose = pose[0].inverse();

    vec3 point_ws = inv_pose.cast<float>() * dir_vs.normalized();

    vec3 dir_ws = point_ws - inv_pose.cast<float>() * vec3(0, 0, 0);
    dir_ws /= length(dir_ws);

    out_directions(0, gy, gx) = dir_ws.x();
    out_directions(1, gy, gx) = dir_ws.y();
    out_directions(2, gy, gx) = dir_ws.z();
}

torch::Tensor ProjectDirectionsToWS(torch::Tensor directions, torch::Tensor pose)
{
    // Sophus::SE3d inv_pose = pose.inverse();
    torch::Tensor result = torch::empty_like(directions);
    // std::cout << "----" << TensorInfo(directions) << std::endl;
    // std::cout << "----" << TensorInfo(result) << std::endl;

    int bx = iDivUp(directions.size(2), 16);
    int by = iDivUp(directions.size(1), 16);
    SAIGA_ASSERT(bx > 0 && by > 0);

    //  std::cout << "----" << bx << " " << by << " " << directions.size(2) << " " << directions.size(1) << " "
    //            << std::endl;
    ::ProjectDirectionToWSKernel<<<dim3(bx, by, 1), dim3(16, 16, 1)>>>(directions, result,
                                                                       (Sophus::SE3d*)pose.data_ptr<double>());
    CUDA_SYNC_CHECK_ERROR();
    // std::cout << "----" << TensorInfo(result) << std::endl;

    return result;
}


__global__ void DebugWeightToColor(ImageView<float> weight, StaticDeviceTensor<float, 3> out_neural_image,
                                   float debug_max_weight)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    if (gx >= weight.width || gy >= weight.height) return;

    auto cou = weight(gy, gx);
    CUDA_DEBUG_ASSERT(out_neural_image.sizes[0] == 4);

    if (cou == 0)
    {
        // copy background into output
        for (int ci = 0; ci < 3; ++ci)
        {
            out_neural_image(ci, gy, gx) = 0;
        }
        out_neural_image(3, gy, gx) = 1;
    }
    else
    {
        float x = cou / debug_max_weight;
        // float t = ::saturate(x);
        // vec3 c  = saturate(vec3(sqrt(t), t * t * t, std::max(sin(3.1415 * 1.75 * t), pow(t, 12.0))));

        vec3 c = colorizeTurbo(x);

        // divide by weight
        for (int ci = 0; ci < 3; ++ci)
        {
            out_neural_image(ci, gy, gx) = c(ci);
        }
        out_neural_image(3, gy, gx) = 1;
    }
}

__global__ void DebugDepthToColor(ImageView<float> depth, StaticDeviceTensor<float, 3> out_neural_image,
                                  float debug_max_weight)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    if (gx >= depth.width || gy >= depth.height) return;

    auto cou = depth(gy, gx);
    CUDA_DEBUG_ASSERT(out_neural_image.sizes[0] == 4);

    if (cou == 0)
    {
        // copy background into output
        for (int ci = 0; ci < 3; ++ci)
        {
            out_neural_image(ci, gy, gx) = 0;
        }
        out_neural_image(3, gy, gx) = 1;
    }
    else
    {
        float x = cou / debug_max_weight;
        vec3 c  = vec3(x, x, x);
        // divide by weight
        for (int ci = 0; ci < 3; ++ci)
        {
            out_neural_image(ci, gy, gx) = c(ci);
        }
        out_neural_image(3, gy, gx) = 1;
    }
}


__global__ void DebugCountingsToColor(ImageView<int> counting, StaticDeviceTensor<float, 3> out_neural_image,
                                      float debug_max_weight)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    if (gx >= counting.width || gy >= counting.height) return;

    auto cou = counting(gy, gx);
    CUDA_DEBUG_ASSERT(out_neural_image.sizes[0] == 4);

    if (cou == 0)
    {
        // copy background into output
        for (int ci = 0; ci < 3; ++ci)
        {
            out_neural_image(ci, gy, gx) = 0;
        }
        out_neural_image(3, gy, gx) = 1;
    }
    else
    {
        float x = cou / debug_max_weight;
        vec3 c  = vec3(x, x, x);

        for (int ci = 0; ci < 3; ++ci)
        {
            out_neural_image(ci, gy, gx) = c(ci);
        }
        out_neural_image(3, gy, gx) = 1;
    }
}


__global__ void CreateMask(StaticDeviceTensor<float, 4> in_weight, StaticDeviceTensor<float, 4> out_mask,
                           float background_value, int b)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;


    if (!in_weight.Image().template inImage(gy, gx)) return;

    auto w = in_weight.At({b, 0, gy, gx});

    if (w == 0)
    {
        out_mask.At({b, 0, gy, gx}) = background_value;
    }
    else
    {
        out_mask(b, 0, gy, gx) = 1;
    }
}


void PointRendererCache::Build(NeuralRenderInfo* info, bool forward)
{
    this->info        = info;
    this->num_batches = info->images.size();



    this->render_mode = (RenderMode)info->params.render_mode;



    SAIGA_OPTIONAL_TIME_MEASURE("Build Cache", info->timer_system);
    static_assert(sizeof(Packtype) == 8);

    SAIGA_ASSERT(num_batches > 0);


    {
        SAIGA_OPTIONAL_TIME_MEASURE("Allocate", info->timer_system);
        Allocate(info, forward);
    }

    if (forward)
    {
        SAIGA_OPTIONAL_TIME_MEASURE("Initialize", info->timer_system);
        InitializeData(forward);
    }
    else
    {
        output_gradient_texture    = torch::zeros_like(info->scene->texture->texture);
        output_gradient_confidence = torch::zeros_like(info->scene->texture->confidence_value_of_point);


        output_gradient_background = torch::zeros_like(info->scene->texture->background_color);

        if (info->scene->point_cloud_cuda->t_point_size.requires_grad())
            output_gradient_layer = torch::zeros_like(info->scene->point_cloud_cuda->t_point_size);

        if (info->scene->point_cloud_cuda->t_position.requires_grad())
        {
            output_gradient_points = torch::zeros_like(info->scene->point_cloud_cuda->t_position);
            // if (render_mode != PointRendererCache::RenderMode::TILED_BILINEAR_BLEND)
            if (info->params.normalize_grads)
                output_gradient_point_count =
                    torch::zeros({output_gradient_points.size(0)}, output_gradient_points.options());
        }
#if 0
        if (info->scene->dynamic_refinement_t.sizes().size() > 1)
        {
            output_gradient_dynamic_points = torch::zeros_like(info->scene->dynamic_refinement_t);
            // if (render_mode != PointRendererCache::RenderMode::TILED_BILINEAR_BLEND)
            if (info->params.normalize_grads)

                output_gradient_dynamic_point_count =
                    torch::zeros({output_gradient_dynamic_points.size(0), output_gradient_dynamic_points.size(1)},
                                 output_gradient_points.options());
        }
#endif
        if (info->scene->poses->tangent_poses.requires_grad())
        {
            output_gradient_pose_tangent = torch::zeros_like(info->scene->poses->tangent_poses);
            //            if (render_mode != PointRendererCache::RenderMode::TILED_BILINEAR_BLEND)
            if (info->params.normalize_grads)
                output_gradient_pose_tangent_count =
                    torch::zeros({info->scene->poses->tangent_poses.size(0)},
                                 info->scene->poses->tangent_poses.options().dtype(torch::kFloat32));
        }

        if (info->scene->intrinsics->is_training())
        {
            output_gradient_intrinsics = torch::zeros_like(info->scene->intrinsics->intrinsics);
            // if (render_mode != PointRendererCache::RenderMode::TILED_BILINEAR_BLEND)
            if (info->params.normalize_grads)
                output_gradient_intrinsics_count = torch::zeros({info->scene->intrinsics->intrinsics.size(0)},
                                                                info->scene->intrinsics->intrinsics.options());
        }
    }
}

void PointRendererCache::Allocate(NeuralRenderInfo* info, bool forward)
{
    auto& fd = info->images.front();
    int h    = fd.h;
    int w    = fd.w;

    SAIGA_ASSERT(info->scene->point_cloud_cuda);
    SAIGA_ASSERT(info->scene->texture);

    std::vector<int> new_cache_size = {(int)info->scene->texture->texture.size(0),
                                       info->scene->point_cloud_cuda->Size(),
                                       info->num_layers,
                                       num_batches,
                                       h,
                                       w,
                                       render_mode};


    bool size_changed = new_cache_size != cache_size;

    if (size_changed)
    {
        cache_has_forward  = false;
        cache_has_backward = false;
    }

    bool need_allocate_forward  = !cache_has_forward && forward;
    bool need_allocate_backward = !cache_has_backward && !forward;

    if (!need_allocate_forward && !need_allocate_backward)
    {
        // std::cout << "skip allocate" << std::endl;
        return;
    }

    // std::cout << "allocate render cache " << need_allocate_forward << " " << need_allocate_backward << " "
    //          << size_changed << std::endl;

    /*    if (curand_state_h == nullptr)
        {
            cudaMalloc(&curand_state_h, sizeof(curandState));
            Saiga::CUDA::initRandom(ArrayView<curandState>(curand_state_h, 1), 0);
        }*/
    if (size_changed)
    {
        layers_cuda.resize(info->num_layers);
    }

    const int MAX_ELEM_TILE_EX   = 8192;
    const int TILE_SIZE          = 16;
    int link_list_tile_extension = w / TILE_SIZE * h / TILE_SIZE * MAX_ELEM_TILE_EX;
    float scale                  = 1;
    for (int i = 0; i < info->num_layers; ++i)
    {
        SAIGA_ASSERT(w > 0 && h > 0);
        auto& l = layers_cuda[i];

        if (need_allocate_forward || need_allocate_backward)
        {
            if (render_mode == PointRendererCache::RenderMode::TILED_BILINEAR_BLEND)
            {
                int size_of_save_buffer = 6;
                size_of_save_buffer     = 7;
                l.bw_sorted_maxed       = torch::empty({num_batches, h, w, max_pixels_per_list, size_of_save_buffer},
                                                       torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat));

                //   l.per_pixel_list_heads =
                //       torch::empty({num_batches, h, w},
                //       torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32));

                l.per_pixel_list_lengths =
                    torch::empty({num_batches, h, w}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32));

                l.scanned_counts =
                    torch::empty({num_batches, h, w}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32));

                // l.per_pixel_list_links =
                //     torch::empty({num_batches, (info->scene->point_cloud_cuda->Size() + link_list_tile_extension) *
                //     4},
                //                  torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32));
            }
            else
            {
                l.counting                  = torch::empty({num_batches, 1, h, w},
                                                           torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32));
                l.scanned_counting          = torch::empty({num_batches, 1, h, w},
                                                           torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32));
                l.per_image_atomic_counters = torch::empty(
                    {num_batches, 1, h, w}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32));

                l.depth     = torch::empty({num_batches, 1, h, w},
                                           torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
                l.weight    = torch::empty({num_batches, 1, h, w},
                                           torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
                l.max_depth = torch::empty({num_batches, 1, h, w},
                                           torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));


                //  l.per_pixel_sorted_bilin_lists =
                //      torch::empty({num_batches, h, w, max_pixels_per_list},
                //                   torch::TensorOptions().device(torch::kCUDA).dtype(torch::kDouble));
            }
        }



        l.size  = {w, h};
        l.scale = scale;

        if (info->scene->params->net_params.network_version != "MultiScaleUnet2d")
        {
            h = std::ceil(float(h) / 2.f);
            w = std::ceil(float(w) / 2.f);
        }
        else
        {
            h = h / 2;
            w = w / 2;
        }

        scale *= 0.5;
    }



    if (need_allocate_forward && info->train && info->params.dropout > 0)
    {
        dropout_points = torch::empty({num_batches, info->scene->point_cloud_cuda->Size()},
                                      torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
    }

    cache_size = new_cache_size;
    if (forward)
    {
        cache_has_forward = true;
    }
    else
    {
        cache_has_backward = true;
    }
}

void PointRendererCache::InitializeData(bool forward)
{
    if (forward)
    {
        for (auto& l : layers_cuda)
        {
            if (render_mode == PointRendererCache::RenderMode::TILED_BILINEAR_BLEND)
            {
                //   l.per_pixel_list_heads.fill_(-1);
                if (info->train)
                {
                    l.bw_sorted_maxed.fill_(-1.f);
                }
                l.per_pixel_list_lengths.zero_();
            }
            else
            {
                l.counting.zero_();
                l.scanned_counting.zero_();
                l.per_image_atomic_counters.zero_();
                l.weight.zero_();

                l.depth.fill_(MAX_DEPTH_CONST);
                l.max_depth.fill_(MAX_DEPTH_CONST);
            }

            // l.per_pixel_list_links.fill_(-1);

            // l.per_pixel_sorted_bilin_lists.fill_(std::numeric_limits<double>::max());
            // PrintTensorInfo(l.per_pixel_sorted_bilin_lists);
        }


        // This is created every frame, because we 'move' it to the output
        output_forward.resize(info->num_layers);
        for (int i = 0; i < info->num_layers; ++i)
        {
            int w                = layers_cuda[i].size(0);
            int h                = layers_cuda[i].size(1);
            int texture_channels = info->params.num_texture_channels;
            if (info->train || render_mode != PointRendererCache::RenderMode::TILED_BILINEAR_BLEND)
            {
                output_forward[i] = torch::ones({num_batches, texture_channels, h, w},
                                                torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
                for (int bg = 0; bg < info->scene->texture->background_color.size(0); ++bg)
                    output_forward[i].slice(1, bg, bg + 1) *=
                        info->scene->texture->background_color.slice(0, bg, bg + 1);
            }
            else
            {
                // output_forward[i] = torch::ones({num_batches, h, w, texture_channels},
                //                                 torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
                // for (int bg = 0; bg < info->scene->texture->background_color.size(0); ++bg)
                //     output_forward[i].slice(3, bg, bg + 1) *=
                //         info->scene->texture->background_color.slice(0, bg, bg + 1);

                output_forward[i] = info->scene->texture->background_color.repeat({num_batches, h, w, 1}).contiguous();
            }


            // else
            //     output_forward[i] = torch::zeros({num_batches, h, w, texture_channels},
            //                                      torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32))
            //                             .permute({0, 3, 1, 2});
        }

        if (info->params.add_depth_to_network)
        {
            output_forward_depthbuffer.resize(info->num_layers);
            for (int i = 0; i < info->num_layers; ++i)
            {
                int w                = layers_cuda[i].size(0);
                int h                = layers_cuda[i].size(1);
                int texture_channels = 1;
                output_forward_depthbuffer[i] =
                    torch::zeros({num_batches, texture_channels, h, w},
                                 torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
            }
        }

        if (info->params.output_background_mask)
        {
            output_forward_blend.resize(info->num_layers);
            for (int i = 0; i < info->num_layers; ++i)
            {
                auto& l = layers_cuda[i];
                output_forward_blend[i] =
                    torch::zeros({num_batches, 1, l.size.y(), l.size.x()},
                                 torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
            }
        }

#if 0
        for (auto& t : output_forward)
        {
            t.zero_();
        }
#endif
        if (info->train)
        {
            if (info->params.dropout > 0)
            {
                dropout_points.bernoulli_(info->params.dropout);
            }
            else
            {
                //   dropout_points.zero_();
            }
        }
    }
}


DeviceRenderParams PointRendererCache::PrepareDeviceRenderParams()
{
    static DeviceRenderParams drp;

    drp = DeviceRenderParams(info->params);
    if (info->scene)
    {
        drp._poses     = (Sophus::SE3d*)info->scene->poses->poses_se3.data_ptr<double>();
        drp.intrinsics = info->scene->intrinsics->intrinsics;
    }
    drp.num_layers = info->num_layers;

    for (int i = 0; i < info->num_layers; ++i)
    {
        if (render_mode != PointRendererCache::RenderMode::TILED_BILINEAR_BLEND)
        {
            drp.depth[i]                     = layers_cuda[i].depth;
            drp.weight[i]                    = layers_cuda[i].weight;
            drp.max_depth[i]                 = layers_cuda[i].max_depth;
            drp.counting[i]                  = layers_cuda[i].counting;
            drp.per_image_atomic_counters[i] = layers_cuda[i].per_image_atomic_counters;
            std::cout << "upload " << TensorInfo(layers_cuda[i].counting) << "--"
                      << TensorInfo(layers_cuda[i].per_image_atomic_counters) << std::endl;
        }
    }

    if (info->params.use_point_adding_and_removing_module)
    {
        if (gradient_of_forward_pass_x.defined())
        {
            drp.gradient_of_forward_pass_x = gradient_of_forward_pass_x;
        }
    }

    // drp.curand_state  = curand_state_h;
    drp.current_epoch = info->current_epoch;

    return drp;
}
DeviceTexture PointRendererCache::PrepareDeviceTexture()
{
    static DeviceTexture d_tex;

    // if (render_mode == RenderMode::TILED_BILINEAR_BLEND)
    //{
    //     d_tex.points_confidence_value = info->scene->texture->confidence_value_of_point.permute({1,
    //     0}).contiguous(); d_tex.in_texture              = info->scene->texture->texture.permute({1,
    //     0}).contiguous();
    // }
    // else
    //{
    d_tex.points_confidence_value = info->scene->texture->confidence_value_of_point;
    // d_tex.points_layer_value      = info->scene->texture->layer_value_of_point;

    d_tex.in_texture = info->scene->texture->texture;
    //}
    // std::cout << TensorInfo(info->scene->texture->texture.contiguous()) << std::endl;

    return d_tex;
}


DeviceBackwardParams PointRendererCache::PrepareDeviceBackwardParams()
{
    DeviceBackwardParams dbp = {0};

    dbp.out_gradient_pose       = nullptr;
    dbp.out_gradient_pose_count = nullptr;
    if (output_gradient_pose_tangent.defined())
    {
        SAIGA_ASSERT(output_gradient_pose_tangent.size(1) == 6);
        dbp.out_gradient_pose = (Vec6*)output_gradient_pose_tangent.data_ptr<double>();
        if (info->params.normalize_grads)
            dbp.out_gradient_pose_count = output_gradient_pose_tangent_count.data_ptr<float>();
    }

    dbp.out_gradient_points       = nullptr;
    dbp.out_gradient_points_count = nullptr;
    if (output_gradient_points.defined())
    {
        SAIGA_ASSERT(output_gradient_points.size(1) == 4);
        dbp.out_gradient_points = (vec4*)output_gradient_points.data_ptr<float>();
        if (info->params.normalize_grads) dbp.out_gradient_points_count = output_gradient_point_count.data_ptr<float>();
    }

    dbp.out_gradient_dynamic_points.data       = nullptr;
    dbp.out_gradient_dynamic_points_count.data = nullptr;
    if (output_gradient_dynamic_points.defined())
    {
        SAIGA_ASSERT(output_gradient_dynamic_points.size(2) == 3);
        dbp.out_gradient_dynamic_points = output_gradient_dynamic_points;
        if (info->params.normalize_grads) dbp.out_gradient_dynamic_points_count = output_gradient_dynamic_point_count;
    }


    dbp.out_gradient_intrinsics.data  = nullptr;
    dbp.out_gradient_intrinsics_count = nullptr;
    if (output_gradient_intrinsics.defined())
    {
        dbp.out_gradient_intrinsics = output_gradient_intrinsics;
        if (info->params.normalize_grads)
            dbp.out_gradient_intrinsics_count = output_gradient_intrinsics_count.data_ptr<float>();
    }

    if (output_gradient_layer.defined())
    {
        dbp.out_gradient_layer = output_gradient_layer;
    }
    else
    {
        dbp.out_gradient_layer.data = nullptr;
    }
    dbp.out_gradient_texture    = output_gradient_texture;
    dbp.out_gradient_confidence = output_gradient_confidence;

    SAIGA_ASSERT(image_gradients.size() == info->num_layers);
    for (int i = 0; i < info->num_layers; ++i)
    {
        SAIGA_ASSERT(image_gradients[i].dim() == 4);
        dbp.in_gradient_image[i] = image_gradients[i];
    }

    return dbp;
}


void PointRendererCache::CreateMask(int batch, float background_value)
{
    SAIGA_ASSERT(output_forward_blend.size() == info->num_layers);
    for (int i = 0; i < info->num_layers; ++i)
    {
        // Allocate result tensor
        auto& l = layers_cuda[i];
        int bx  = iDivUp(l.size.x(), 16);
        int by  = iDivUp(l.size.y(), 16);
        SAIGA_ASSERT(bx > 0 && by > 0);

        SAIGA_ASSERT(output_forward_blend[i].size(2) == l.size.y());
        ::CreateMask<<<dim3(bx, by, 1), dim3(16, 16, 1)>>>(l.weight, output_forward_blend[i], background_value, batch);
    }
    CUDA_SYNC_CHECK_ERROR();
}



std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>> BlendPointCloudForward(
    torch::autograd::AutogradContext* ctx, NeuralRenderInfo* info)
{
    // cache.render_mode = PointRendererCache::RenderMode::TILED_BILINEAR_BLEND;

    // int num_batches     = info->images.size();
    NeuralScene& scene  = *info->scene;
    RenderParams params = info->params;

    PointRendererCache& cache = *info->cache;


    cache.Build(info, true);

    int num_batches = cache.num_batches;

    cache.PushParametersForward();

    if (params.render_outliers)
    {
        if (!scene.outlier_point_cloud_cuda)
        {
            scene.BuildOutlierCloud(params.outlier_count);
        }
    }

    torch::Tensor d2;
    auto add_displacement_tensor_info_to_point_cloud = [&](int b_id)
    {
        // a dummy [1] tensor is used when no displacements are optimized
        if (scene.dynamic_refinement_t.sizes().size() <= 1) return;

        {
            auto displacement = scene.dynamic_refinement_t.slice(0, b_id, b_id + 1).squeeze(0);
            d2                = torch::cat({displacement, torch::empty_like(displacement.slice(1, 0, 1))}, 1);

            scene.point_cloud_cuda->t_position_displacement = d2;
        }
    };

    // std::cout << (int)info->images.front().camera_model_type << std::endl;
    // std::cout << info->images.front().crop_transform << std::endl;
    // std::cout << info->images.front().crop_rotation << std::endl;
    // std::cout << (info->scene->intrinsics->intrinsics.slice(0, 0, 1)) << std::endl;

    // only for blending
    //  tensors are shape [2,max_elems]
    std::vector<std::vector<torch::Tensor>> collection_buffer(num_batches);
    std::vector<std::vector<torch::Tensor>> per_point_data_buffer(num_batches);

    if (cache.render_mode == PointRendererCache::RenderMode::TILED_BILINEAR_BLEND)
    {
        static bool new_impl = true;

        if (new_impl && info->params.use_layer_point_size && info->params.render_points_in_all_lower_resolutions &&
            !info->params.combine_lists)
        {
            static bool once = true;
            if (once)
            {
                std::cout << "USING NEW IMPL" << std::endl;
                once = false;
            }
            cache.UploadLinkedListBuffers();

            for (int b = 0; b < num_batches; ++b)
            {
                cache.RenderFast16(b, scene.point_cloud_cuda, info->train, scene.texture->background_color,
                                   info->timer_system);
            }

            if (!info->train)
                for (int i = 0; i < info->num_layers; ++i)
                    cache.output_forward[i] = cache.output_forward[i].permute({0, 3, 1, 2});
        }
        else
        {
            int h                   = info->images.front().h;
            int w                   = info->images.front().w;
            const int TILE_SIZE     = 16;
            const int NUM_PER_PIXEL = 64;

            // fast integer ceil devision
            int tile_h = (h + (TILE_SIZE - 1)) / TILE_SIZE;
            int tile_w = (w + (TILE_SIZE - 1)) / TILE_SIZE;

            torch::Tensor full_list_buffer;
            torch::Tensor full_list_buffer_data;
            {
                SAIGA_OPTIONAL_TIME_MEASURE("Allocate Tensors", info->timer_system);
#ifdef TILED_REORDER
                const int MAX_ELEM_TILE_EX   = 8192;
                const int TILE_SIZE          = 16;
                int link_list_tile_extension = w / TILE_SIZE * h / TILE_SIZE * MAX_ELEM_TILE_EX;
#else
                int link_list_tile_extension = 0;
#endif
// #define USE_INT64
#ifdef USE_INT64
                // large data buffer: Max size: amount of points
                full_list_buffer =
                    torch::empty({num_batches, scene.point_cloud_cuda->Size() * 4 + 4 * link_list_tile_extension, 1},
                                 torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt64));
#else
                //     full_list_buffer = torch::empty({num_batches, scene.point_cloud_cuda->Size(), 1},
                //                                            torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat64));
#endif
                //      full_list_buffer_data = torch::empty({num_batches, scene.point_cloud_cuda->Size(), 5},
                //                                           torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));



                cache.UploadLinkedListBuffers();
            }

            CUDA_SYNC_CHECK_ERROR();
            // static void* d_temp_storage           = nullptr;
            // static size_t temp_storage_bytes_cont = 0;
            int max_num_elements = 0;

            // std::vector<std::vector<torch::Tensor>> scanned_counts(num_batches);


            {
                std::vector<int> list_lengths;
                std::vector<int> list_lengths_complete;
                // SAIGA_OPTIONAL_TIME_MEASURE("Render", info->timer_system);
                std::vector<torch::Tensor> indices_more_than_X(num_batches);
                std::vector<torch::Tensor> indices_less_than_X(num_batches);

                static int split_value = 4;
                // ImGui::SliderInt("split value", &split_value, 0, 64);
                static int min_value = 0;
                // ImGui::SliderInt("min value", &min_value, 0, 64);

                for (int b = 0; b < num_batches; ++b)
                {
                    if (params.render_points)
                    {
                        {
                            SAIGA_OPTIONAL_TIME_MEASURE("Count Render", info->timer_system);

                            add_displacement_tensor_info_to_point_cloud(b);

                            cache.CountTiled(b, scene.point_cloud_cuda, info->train);
                        }
                        int num_elements = 0;
                        int num_lists    = 0;
                        {
                            // ist_lengths.push_back(std::vector<int>());
                            SAIGA_OPTIONAL_TIME_MEASURE("Scan Counts and division", info->timer_system);

                            void* d_temp_storage                = NULL;
                            size_t temp_storage_bytes_allocated = 0;
                            torch::Tensor tmp_tensor;


                            for (int i = 0; i < info->num_layers; ++i)
                            {
                                auto& l = cache.layers_cuda[i];
                                //     PrintTensorInfo(l.per_pixel_list_lengths);
                                int num_items = l.size.x() * l.size.y();
                                int* d_in =
                                    l.per_pixel_list_lengths.data_ptr<int>() + b * l.per_pixel_list_lengths.stride(0);
                                int* d_out = l.scanned_counts.data_ptr<int>() + b * l.scanned_counts.stride(0);

                                size_t temp_storage_bytes = 0;

                                cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes, d_in, d_out, num_items);
                                CUDA_SYNC_CHECK_ERROR();

                                if (temp_storage_bytes_allocated < temp_storage_bytes)
                                {
                                    temp_storage_bytes_allocated = temp_storage_bytes;

                                    tmp_tensor     = torch::empty({iDivUp((long)temp_storage_bytes, 4L)},
                                                                  torch::TensorOptions(torch::kCUDA));
                                    d_temp_storage = tmp_tensor.data_ptr();
                                }
                                CUDA_SYNC_CHECK_ERROR();

                                cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out,
                                                              num_items);
                                CUDA_SYNC_CHECK_ERROR();

                                //   PrintTensorInfo(scanned_counts[i]);

                                int max_scanned = l.scanned_counts.slice(0, b, b + 1)
                                                      .slice(1, l.size.y() - 1, l.size.y())
                                                      .slice(2, l.size.x() - 1, l.size.x())
                                                      .item<int>();
                                int last_elem = l.per_pixel_list_lengths.slice(0, b, b + 1)
                                                    .slice(1, l.size.y() - 1, l.size.y())
                                                    .slice(2, l.size.x() - 1, l.size.x())
                                                    .item<int>();

                                //  get non_zeros
                                auto lengths_this_layer = l.per_pixel_list_lengths.slice(0, b, b + 1).view({-1});
#if 0

                            //      // shape (num,1)
                            auto non_zero_ = lengths_this_layer.gt(0).squeeze();
                            //      // shape (num,1)
                            auto less_than_X_ = lengths_this_layer.lt(split_value).squeeze();
                            auto less_than_X_indices =
                                (non_zero_ * less_than_X_).nonzero().to(torch::kInt32) + num_lists;

                            // auto more_or_equal_than_X_ = lengths_this_layer.ge(split_value).squeeze();
                            // auto more_or_equal_than_X_indices =
                            //     more_or_equal_than_X_.nonzero().to(torch::kInt32) + num_lists;


                            if (!indices_less_than_X[b].defined())
                            {
                                indices_less_than_X[b] = less_than_X_indices;
                            }
                            else
                            {
                                indices_less_than_X[b] = torch::cat({indices_less_than_X[b], less_than_X_indices}, 0);
                            }
#endif

                                auto more_or_equal_than_X_ = lengths_this_layer.ge(min_value).squeeze();
                                auto more_or_equal_than_X_indices =
                                    more_or_equal_than_X_.nonzero().to(torch::kInt32) + num_lists;
                                if (!indices_more_than_X[b].defined())
                                {
                                    indices_more_than_X[b] = more_or_equal_than_X_indices;
                                }
                                else
                                {
                                    indices_more_than_X[b] =
                                        torch::cat({indices_more_than_X[b], more_or_equal_than_X_indices}, 0);
                                }
                                num_lists += l.size.x() * l.size.y();

                                num_elements += max_scanned + last_elem;
                                list_lengths.push_back(max_scanned + last_elem);

// #define COUNT_LISTS
#ifdef COUNT_LISTS
                                auto count_tensor = lengths_this_layer.ge(32).count_nonzero();
                                //   std::cout << "Larger than 32:" << std::endl;
                                //   std::cout << "layer: " << i << ": " <<
                                //   TensorInfo(count_tensor) << std::endl; std::cout << "layer:
                                //   " << i << ": " << TensorInfo(lengths_this_layer.gt(32)) <<
                                //   std::endl;
                                auto g_c = lengths_this_layer.ge(16).nonzero();
                                //   PrintTensorInfo(g_c);
                                //   PrintTensorInfo(lengths_this_layer);
                                auto g_sc = lengths_this_layer.index({g_c.squeeze()});
                                //   PrintTensorInfo(g_sc);
                                count_tensor = (g_sc.lt(32)).nonzero().count_nonzero();
                                //   std::cout << "Larger than 16:" << std::endl;
                                //   std::cout << "layer: " << i << ": " <<
                                //   TensorInfo(count_tensor) << std::endl;
#endif
                            }
                            //    std::cout << num_elements << std::endl;
                            list_lengths_complete.push_back(num_elements);
                            max_num_elements = max(num_elements, max_num_elements);
                        }
                    }
                }
                CUDA_SYNC_CHECK_ERROR();

                torch::Tensor layer_lengths;
                {
                    SAIGA_OPTIONAL_TIME_MEASURE("Alloc", info->timer_system);
                    layer_lengths = torch::from_blob(list_lengths.data(), {num_batches, info->num_layers},
                                                     torch::TensorOptions().dtype(torch::kInt32))
                                        .contiguous()
                                        .clone()
                                        .cuda();
                    for (int i = 0; i < info->num_layers; ++i)
                    {
                        auto& l = cache.layers_cuda[i];
                        l.per_pixel_list_lengths.zero_();
                    }

                    full_list_buffer = torch::empty({num_batches, max_num_elements, 1},
                                                    torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat64));
                    full_list_buffer_data =
                        torch::empty({num_batches, max_num_elements, info->train ? 5 : 4},
                                     torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
                }
                cache.UploadLinkedListBuffers();

                CUDA_SYNC_CHECK_ERROR();
                if (0)
                {
                    int size_sorting = info->scene->point_cloud_cuda->Size();
                    static torch::Tensor rand_keys;
                    static torch::Tensor rand_vals;
                    static torch::Tensor rand_keys_out;
                    static torch::Tensor rand_vals_out;
                    {
                        SAIGA_OPTIONAL_TIME_MEASURE("ALLOC LARGE SORT", info->timer_system);
                        rand_keys = torch::rand({size_sorting},
                                                torch::TensorOptions().device(torch::kCUDA).dtype(torch::kDouble))
                                        .clone()
                                        .contiguous();
                        rand_keys_out = rand_keys.clone().contiguous();
                        rand_vals     = torch::rand({size_sorting},
                                                    torch::TensorOptions().device(torch::kCUDA).dtype(torch::kDouble))
                                        .clone()
                                        .contiguous();
                        rand_vals_out = rand_vals.clone().contiguous();
                        // PrintTensorInfo(rand_keys_out);
                        // PrintTensorInfo(rand_keys);
                        // PrintTensorInfo(rand_vals_out);
                        // PrintTensorInfo(rand_vals);
                        //  Determine temporary device storage requirements
                    }
                    void* d_temp_storage = NULL;

                    CUDA_SYNC_CHECK_ERROR();

                    {
                        size_t temp_storage_bytes = 0;
                        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                                        rand_keys.data_ptr<double>(), rand_keys_out.data_ptr<double>(),
                                                        rand_vals.data_ptr<double>(), rand_vals_out.data_ptr<double>(),
                                                        size_sorting);
                        // Allocate temporary storage
                        {
                            SAIGA_OPTIONAL_TIME_MEASURE("ALLOCTEMP", info->timer_system);
                            cudaMalloc(&d_temp_storage, temp_storage_bytes);
                        }
                        {
                            SAIGA_OPTIONAL_TIME_MEASURE("TEST LARGE SORT", info->timer_system);
                            // Run sorting operation
                            cub::DeviceRadixSort::SortPairs(
                                d_temp_storage, temp_storage_bytes, rand_keys.data_ptr<double>(),
                                rand_keys_out.data_ptr<double>(), rand_vals.data_ptr<double>(),
                                rand_vals_out.data_ptr<double>(), size_sorting);
                            CUDA_SYNC_CHECK_ERROR();
                        }
                    }
                    cudaFree(d_temp_storage);
                }
                {
                    SAIGA_OPTIONAL_TIME_MEASURE("Collect", info->timer_system);
                    for (int b = 0; b < num_batches; ++b)
                    {
                        {
                            cache.CollectTiled2(b, scene.point_cloud_cuda, full_list_buffer, full_list_buffer_data,
                                                layer_lengths, info->train);
                        }
                    }

                    //     PrintTensorInfo(full_list_buffer_data);
                }

                {
                    SAIGA_OPTIONAL_TIME_MEASURE("Fused Sort and Blend", info->timer_system);
                    {
                        for (int b = 0; b < num_batches; ++b)
                        {
                            if (params.render_points)
                            {
                                cache.FusedSortAndBlend2(
                                    b, full_list_buffer, full_list_buffer_data, scene.texture->background_color,
                                    info->train, info->params.use_environment_map, list_lengths_complete[b],
                                    layer_lengths, indices_more_than_X[b], indices_less_than_X[b], info->timer_system);
                            }
                        }
                    }
                    if (!info->train)
                        for (int i = 0; i < info->num_layers; ++i)
                            cache.output_forward[i] = cache.output_forward[i].permute({0, 3, 1, 2});
                }
            }
        }
    }

#if 0
    {
        SAIGA_OPTIONAL_TIME_MEASURE("DebugShowCounts", info->timer_system);

        if (params.render_points)
        {

            for (int b = 0; b < num_batches; ++b)
            {
                for (int i = 0; i < info->num_layers; ++i)
                {
                    // Allocate result tensor
                    auto& l = cache.layers_cuda[i];
                    int bx  = iDivUp(l.size.x(), 16);
                    int by  = iDivUp(l.size.y(), 16);
                    SAIGA_ASSERT(bx > 0 && by > 0);
                    auto in_out_neural_image = cache.output_forward[i][b];

                    //auto countings = l.BatchViewCounting(b);
                    //::DebugCountingsToColor<<<dim3(bx, by, 1), dim3(16, 16, 1)>>>(countings, in_out_neural_image,
                    //                                                          1);

                    auto scanned_countings                    = l.BatchViewScannedCounting(b);
                    ::DebugCountingsToColor<<<dim3(bx, by, 1), dim3(16, 16, 1)>>>(scanned_countings, in_out_neural_image,
                                                                                  1000);
                }
            }
        }

    }
#endif



    if (info->params.debug_weight_color && info->params.num_texture_channels == 4)
    {
        for (int b = 0; b < num_batches; ++b)
        {
            for (int i = 0; i < info->num_layers; ++i)
            {
                // Allocate result tensor
                auto& l = cache.layers_cuda[i];
                int bx  = iDivUp(l.size.x(), 16);
                int by  = iDivUp(l.size.y(), 16);
                SAIGA_ASSERT(bx > 0 && by > 0);
                auto in_out_neural_image = cache.output_forward[i][b];

                auto weights = l.BatchViewWeights(b);
                ::DebugWeightToColor<<<dim3(bx, by, 1), dim3(16, 16, 1)>>>(weights, in_out_neural_image,
                                                                           info->params.debug_max_weight);
            }
        }
    }

    if (info->params.debug_depth_color && info->params.num_texture_channels == 4)
    {
        for (int b = 0; b < num_batches; ++b)
        {
            for (int i = 0; i < info->num_layers; ++i)
            {
                // Allocate result tensor
                auto& l = cache.layers_cuda[i];
                int bx  = iDivUp(l.size.x(), 16);
                int by  = iDivUp(l.size.y(), 16);
                SAIGA_ASSERT(bx > 0 && by > 0);
                auto in_out_neural_image = cache.output_forward[i][b];

                auto depths = l.BatchViewDepth(b);
                ::DebugDepthToColor<<<dim3(bx, by, 1), dim3(16, 16, 1)>>>(depths, in_out_neural_image,
                                                                          info->params.debug_max_weight);
            }
        }
    }


    if (info->params.debug_print_num_rendered_points)
    {
        double weight_sum = 0;
        for (int i = 0; i < info->num_layers; ++i)
        {
            // Allocate result tensor
            auto& l = cache.layers_cuda[i];
            weight_sum += l.weight.sum().item().toFloat();
        }
        std::cout << "# Rasterized Points = " << (int)weight_sum << std::endl;
    }

    if (ctx)
    {
        SAIGA_OPTIONAL_TIME_MEASURE("Save in Graph", info->timer_system);
        std::vector<torch::Tensor> save_variables;
        if (cache.render_mode == PointRendererCache::RenderMode::TILED_BILINEAR_BLEND)
        {
            for (auto l : cache.layers_cuda)
            {
                save_variables.push_back(l.bw_sorted_maxed);
            }
            save_variables.insert(save_variables.end(), cache.output_forward.begin(), cache.output_forward.end());
        }
        else
        {
            for (auto l : cache.layers_cuda)
            {
                save_variables.push_back(l.depth);
                save_variables.push_back(l.weight);
                save_variables.push_back(l.scanned_counting);
                save_variables.push_back(l.per_image_atomic_counters);
            }
            save_variables.insert(save_variables.end(), cache.output_forward.begin(), cache.output_forward.end());
            for (auto v : collection_buffer)
            {
                for (auto elem : v)
                {
                    save_variables.push_back(elem);
                }
            }

            for (auto vx : per_point_data_buffer)
            {
                for (auto elemx : vx)
                {
                    save_variables.push_back(elemx);
                }
            }
        }

        save_variables.push_back(cache.dropout_points);
        ctx->save_for_backward(save_variables);
        CUDA_SYNC_CHECK_ERROR();
    }

    if (info->params.add_depth_to_network)
    {
        cache.output_forward.insert(cache.output_forward.end(), cache.output_forward_depthbuffer.begin(),
                                    cache.output_forward_depthbuffer.end());
    }

    // cudaDeviceSynchronize();
    return {std::move(cache.output_forward), std::move(cache.output_forward_blend)};
}

template <typename T, int N>
__global__ void NormalizeGradient(Vector<T, N>* tangent, float* tangent_count, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    Vector<T, N> t = tangent[tid];
    float c        = tangent_count[tid];

    if (c > 0)
    {
        // if (N == 6)
        //     for (int i = 0; i < 6; ++i) printf("++%f++ ", float(t(i)));
        tangent[tid] = t / c;
        // tangent[tid] = t / T(c);
        // if (N == 6)
        //    for (int i = 0; i < 6; ++i) printf("##%f## ", float(tangent[tid](i)));
    }
}

template <typename T, int N>
__global__ void NormalizeGradientDevTensor(StaticDeviceTensor<T, 2> tangent, StaticDeviceTensor<float, 1> tangent_count,
                                           int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // Vector<T, N> t = tangent[tid];
    float c = tangent_count(tid);

    if (c > 0)
    {
        for (int i = 0; i < N; ++i)
        {
            tangent(tid, i) = tangent(tid, i) / c;
        }
    }
}

torch::autograd::variable_list BlendPointCloudBackward(torch::autograd::AutogradContext* ctx, NeuralRenderInfo* info,
                                                       torch::autograd::variable_list _image_gradients)
{
    SAIGA_ASSERT(info->cache);
    for (auto& ig : _image_gradients)
    {
        SAIGA_ASSERT(ig.dtype() == torch::kFloat32);
    }

    // int num_batches     = info->images.size();
    NeuralScene& scene  = *info->scene;
    RenderParams params = info->params;

    // PointRendererCache cache;
    PointRendererCache& cache = *info->cache;

    int num_batches = cache.num_batches;

    auto add_displacement_tensor_info_to_point_cloud = [&](int b_id)
    {
        // a dummy [1] tensor is used when no displacements are optimized
        if (scene.dynamic_refinement_t.sizes().size() <= 1) return;

        {
            auto displacement = scene.dynamic_refinement_t.slice(0, b_id, b_id + 1).squeeze(0);
            auto d2           = torch::cat({displacement, torch::empty_like(displacement.slice(1, 0, 1))}, 1);
            scene.point_cloud_cuda->t_position_displacement = d2;
        }
    };

    /*
     *  These buffers are large buffers including space for exactly all pixels collected.
     *  Accessing can be done with the scanned countings list.
     *  there exists one for each batch and layer (i.e. 4 batches, 4 layers = [4][4])
     *  gradient_sum_back_buffer is an intermediate buffer for the Jacobians, with num_tex_parameters + 1 channels
     */
    std::vector<std::vector<torch::Tensor>> collection_buffer(num_batches);
    std::vector<std::vector<torch::Tensor>> per_point_data_buffer(num_batches);
    std::vector<std::vector<torch::Tensor>> gradient_sum_back_buffer(num_batches);


    {
        SAIGA_OPTIONAL_TIME_MEASURE("Prepare Backward", info->timer_system);
        cache.Build(info, false);

        // The first [num_layers] gradients are the actual neural image gradients. After that we get the gradients
        // of the mask which does not help us much
        cache.image_gradients =
            std::vector<torch::Tensor>(_image_gradients.begin(), _image_gradients.begin() + info->num_layers);

        auto save_variables = ctx->get_saved_variables();

        if (cache.render_mode == PointRendererCache::RenderMode::TILED_BILINEAR_BLEND)
        {
            cache.output_forward.resize(info->num_layers);
            int offset_v = 1;
            for (int i = 0; i < info->num_layers; ++i)
            {
                cache.layers_cuda[i].bw_sorted_maxed = save_variables[i * offset_v + 0];
                cache.output_forward[i]              = save_variables[info->num_layers * offset_v + i];
            }
        }
        else
        {
            int offset_v = 4;
            SAIGA_ASSERT(save_variables.size() ==
                         info->num_layers * (offset_v + 1) + 1 + 2 * info->num_layers * num_batches);
            cache.output_forward.resize(info->num_layers);
            for (int i = 0; i < info->num_layers; ++i)
            {
                cache.layers_cuda[i].depth                     = save_variables[i * offset_v + 0];
                cache.layers_cuda[i].weight                    = save_variables[i * offset_v + 1];
                cache.layers_cuda[i].scanned_counting          = save_variables[i * offset_v + 2];
                cache.layers_cuda[i].per_image_atomic_counters = save_variables[i * offset_v + 3];
                cache.output_forward[i]                        = save_variables[info->num_layers * offset_v + i];
            }
            int start_collb = info->num_layers * offset_v + info->num_layers;
            for (int b = 0; b < num_batches; ++b)
            {
                for (int i = 0; i < info->num_layers; ++i)
                {
                    collection_buffer[b].push_back(save_variables[start_collb + b * info->num_layers + i]);
                }
            }
            int start_data = info->num_layers * offset_v + info->num_layers + num_batches * info->num_layers;
            for (int b = 0; b < num_batches; ++b)
            {
                for (int i = 0; i < info->num_layers; ++i)
                {
                    per_point_data_buffer[b].push_back(save_variables[start_data + b * info->num_layers + i]);
                    // intermediate summation buffer, last element is for alpha_dest storing
                    gradient_sum_back_buffer[b].push_back(
                        torch::zeros({params.num_texture_channels + 1, per_point_data_buffer[b].back().size(0)},
                                     torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32)));
                }
            }
        }
        cache.dropout_points = save_variables.back();

        SAIGA_ASSERT(cache.image_gradients.size() == info->num_layers);

        cache.PushParametersBackward();
    }
    if (cache.render_mode == PointRendererCache::RenderMode::TILED_BILINEAR_BLEND)
    {
        cache.UploadCollectionBuffersBackwardsTiled();
        SAIGA_OPTIONAL_TIME_MEASURE("BlendBackwardsBilinearFast", info->timer_system);
        for (int b = 0; b < num_batches; ++b)
        {
            add_displacement_tensor_info_to_point_cloud(b);
            cache.BlendBackwardsBilinearFast(b, scene.point_cloud_cuda, scene.texture->background_color,
                                             info->params.use_environment_map);
        }
    }


    if (info->params.normalize_grads)
    {
        SAIGA_OPTIONAL_TIME_MEASURE("Post Process Gradient", info->timer_system);
        if (cache.output_gradient_pose_tangent.defined())
        {
            // std::cout << "POSE NORMALIZATION" <<
            // TensorInfo(cache.output_gradient_pose_tangent)
            //           << TensorInfo(cache.output_gradient_pose_tangent_count) <<
            //           std::endl
            //           << std::endl;
            //  Average pose gradient over all measurements
            int n = cache.output_gradient_pose_tangent.size(0);
            int c = iDivUp(n, 128);
            // NormalizeGradient<double, 6><<<c,
            // 128>>>((Vec6*)cache.output_gradient_pose_tangent.data_ptr<double>(),
            //                                           cache.output_gradient_pose_tangent_count.data_ptr<float>(),
            // n);

            NormalizeGradientDevTensor<double, 6>
                <<<c, 128>>>(cache.output_gradient_pose_tangent, cache.output_gradient_pose_tangent_count, n);
            CUDA_SYNC_CHECK_ERROR();

            // std::cout << std::endl
            //           << "END POSE NORMALIZATION" <<
            //           TensorInfo(cache.output_gradient_pose_tangent)
            //           << TensorInfo(cache.output_gradient_pose_tangent_count) <<
            //           std::endl;
        }

        if (cache.output_gradient_points.defined())
        {
            // Average point gradient over all measurements
            int n = cache.output_gradient_points.size(0);
            int c = iDivUp(n, 128);
            NormalizeGradient<float, 3><<<c, 128>>>((vec3*)cache.output_gradient_points.data_ptr<float>(),
                                                    cache.output_gradient_point_count.data_ptr<float>(), n);
        }
        if (cache.output_gradient_dynamic_points.defined())
        {
            for (int b = 0; b < num_batches; ++b)
            {
                auto tensor_to_normalize = cache.output_gradient_dynamic_points.slice(0, b, b + 1).squeeze();
                auto count_tensor        = cache.output_gradient_dynamic_point_count.slice(0, b, b + 1).squeeze();
                int n                    = tensor_to_normalize.size(0);
                int c                    = iDivUp(n, 128);
                // NormalizeGradient<double, 6><<<c,
                // 128>>>((Vec6*)cache.output_gradient_pose_tangent.data_ptr<double>(),
                // cache.output_gradient_pose_tangent_count.data_ptr<float>(),
                //                                          n);

                NormalizeGradientDevTensor<float, 3><<<c, 128>>>(tensor_to_normalize, count_tensor, n);
                CUDA_SYNC_CHECK_ERROR();
            }
        }

        if (cache.output_gradient_intrinsics.defined())
        {
            // Average intrinsics/distortion gradient over all measurements
            int n = cache.output_gradient_intrinsics.size(0);
            int c = iDivUp(n, 128);
            NormalizeGradient<float, 13>
                <<<c, 128>>>((Vector<float, 13>*)cache.output_gradient_intrinsics.data_ptr<float>(),
                             cache.output_gradient_intrinsics_count.data_ptr<float>(), n);
        }
    }

    CUDA_SYNC_CHECK_ERROR();

    // gradients for displacement field are equal to point gradients for that batch, as:
    //  point_pos = original_point_pos + displacements
    //  thus:
    //  d_point_pos / d_displacements = 1
    //  d_point_pos / d_original_point_pos = 1


    return {std::move(cache.output_gradient_texture),        std::move(cache.output_gradient_background),
            std::move(cache.output_gradient_points),         std::move(cache.output_gradient_pose_tangent),
            std::move(cache.output_gradient_intrinsics),     std::move(cache.output_gradient_confidence),
            std::move(cache.output_gradient_dynamic_points), std::move(cache.output_gradient_layer)};
}
__global__ void ApplyTangent(Vec6* tangent, Sophus::SE3d* pose, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    Vec6 t = tangent[tid];
    auto p = pose[tid];

    // TODO check magic rotation scaling value
    t.template tail<3>() *= 0.1;
#ifdef _WIN32
    Sophus::SE3d p2(Sophus::se3_expd(t) * p);
    for (int i = 0; i < 7; ++i) pose[tid].data()[i] = p2.data()[i];
#else
    p         = Sophus::se3_expd(t) * p;
    pose[tid] = p;
#endif

    tangent[tid] = Vec6::Zero();
}

void ApplyTangentToPose(torch::Tensor tangent, torch::Tensor pose)
{
    SAIGA_ASSERT(tangent.is_contiguous() && pose.is_contiguous());
    int n = tangent.size(0);
    int c = iDivUp(n, 128);
    ApplyTangent<<<c, 128>>>((Vec6*)tangent.data_ptr<double>(), (Sophus::SE3d*)pose.data_ptr<double>(), n);
    CUDA_SYNC_CHECK_ERROR();
}