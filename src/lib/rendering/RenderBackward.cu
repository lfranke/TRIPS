/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
// functions for backwards rendering
// #include "saiga/colorize.h"
#include "saiga/cuda/random.h"
#include "saiga/cuda/reduce.h"
#include "saiga/vision/torch/CudaHelper.h"

#include "PointBlending.h"
#include "PointRenderer.h"
#include "PointRendererHelper.h"
#include "RenderConstants.h"

#include "cooperative_groups.h"
#include <curand_kernel.h>

__device__ __constant__ DeviceRenderParams d_render_params;
__device__ __constant__ DeviceTexture d_texture;
__device__ __constant__ DeviceForwardParams d_forward_params;
__device__ __constant__ DeviceBackwardParams d_backward_params;
__device__ __constant__ DeviceAlphaCompositionParams d_alpha_comp_params_bw;
__device__ __constant__ DeviceBilinearAlphaParams d_bilinear_alpha_params_bw;


void PointRendererCache::PushParametersBackward()
{
    SAIGA_OPTIONAL_TIME_MEASURE("Param Upload", info->timer_system);

    {
        static DeviceForwardParams dfp;
        for (int i = 0; i < info->num_layers; ++i)
        {
            dfp.neural_out[i] = output_forward[i];
        }
        CHECK_CUDA_ERROR(cudaMemcpyToSymbolAsync(d_forward_params, &dfp, sizeof(dfp)));
    }
    {
        DeviceBackwardParams dbp = PrepareDeviceBackwardParams();
        CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_backward_params, &dbp, sizeof(dbp)));
    }
    {
        static DeviceRenderParams drp;
        drp = PrepareDeviceRenderParams();

        CHECK_CUDA_ERROR(cudaMemcpyToSymbolAsync(d_render_params, &drp, sizeof(drp)));
        CUDA_SYNC_CHECK_ERROR();
    }
    {
        static DeviceTexture d_tex;
        d_tex = PrepareDeviceTexture();

        CHECK_CUDA_ERROR(cudaMemcpyToSymbolAsync(d_texture, &d_tex, sizeof(d_tex)));
        CUDA_SYNC_CHECK_ERROR();
    }
}

void PointRendererCache::UploadCollectionBuffersBackwardsTiled()
{
    static DeviceBilinearAlphaParams dbap;

    for (int i = 0; i < layers_cuda.size(); ++i)
    {
        dbap.bw_sorted_maxed[i] = layers_cuda[i].bw_sorted_maxed;
    }
    CHECK_CUDA_ERROR(cudaMemcpyToSymbolAsync(d_bilinear_alpha_params_bw, &dbap, sizeof(dbap)));
}

__device__ inline float _softplus_backwards(float y, float beta = 1.f, float threshold = 20.f)
{
    if (y > threshold) return 1.f;

    float z = expf(y * beta);


    return z / (z + 1.f);

    // return 1 / (1 + exp(-y));
    // return logf(expf(y * beta) + 1.f) / beta;
    // float z = expf(x * beta) return x * beta > threshold ? x : logf(1 + expf(beta * x));
}

#define MAX_THREADS_X 16
#define MAX_THREADS_Y 16
#define THREADS_PER_BLOCK (MAX_THREADS_X * MAX_THREADS_Y)


template <int num_descriptors, int ELEMENTS_PER_PIXEL>
__global__ __launch_bounds__(THREADS_PER_BLOCK) void BlendBackwardsBilinearTiled(
    DevicePointCloud point_cloud, float* background_color, float* out_background_gradient, int batch,
    ReducedImageInfo cam, bool need_point_gradients, StaticDeviceTensor<int, 1> glob_atomic_c, int max_elements_total,
    int num_layers)
{
    //  int gx = blockIdx.x * blockDim.x + threadIdx.x;
    //  int gy = blockIdx.y * blockDim.y + threadIdx.y;
    //
    //  if (gx >= d_backward_params.in_gradient_image[layer].size(3) ||
    //      gy >= d_backward_params.in_gradient_image[layer].size(2))
    //      return;
    constexpr int WARP_SIZE = 32;

    // auto get_lane_id      = []() -> int { return  };
    int lane_id = (threadIdx.y * blockDim.x + threadIdx.x) % WARP_SIZE;
    int warp_id = (threadIdx.y * blockDim.x + threadIdx.x) / WARP_SIZE;

    uint32_t layer = 0;
    uint32_t gx = 0, gy = 0;
    int ticket = 0;

    auto get_xyz_from_ticket = [&layer, &gx, &gy, &batch, &num_layers](int t)
    {
        int w = d_backward_params.in_gradient_image[0].size(3);
        int h = d_backward_params.in_gradient_image[0].size(2);

        for (int i = 0; i < num_layers; ++i)
        {
            if (t < w * h)
            {
                layer = i;
                gx    = t % w;
                gy    = t / w;
                break;
            }
            t -= w * h;
            w /= 2;
            h /= 2;
        }
    };

    auto get_next_list = [&ticket, &glob_atomic_c, &get_xyz_from_ticket, &max_elements_total]
    {
        ticket = atomicAdd(&glob_atomic_c(0), 1);
        get_xyz_from_ticket(ticket);
    };


    auto get_tex_idx = [&](int index_in_list) -> int
    {
        int texture_index = reinterpret_cast<int*>(
            &d_bilinear_alpha_params_bw.bw_sorted_maxed[layer](batch, gy, gx, index_in_list, 0))[0];
        return texture_index;
    };
    auto get_alpha_val = [&](int index_in_list) -> float
    { return d_bilinear_alpha_params_bw.bw_sorted_maxed[layer](batch, gy, gx, index_in_list, 2); };



    {
        get_next_list();
        __syncwarp();
    }

    constexpr int K_mat_size                = 5;
    constexpr int distortion_polym_max_size = 8;
    constexpr int pose_size                 = 6;


    // lessen warp write to the same memory address (it is serialized), instead write to shared mem and later reduce add
    __shared__ double intrinsics_K_grad[K_mat_size][WARP_SIZE];
    __shared__ double intrinsics_dis_ocam_grad[distortion_polym_max_size][WARP_SIZE];
    __shared__ double pose_grad[pose_size][WARP_SIZE];
    __shared__ double background_grads[num_descriptors][WARP_SIZE];


    // if (lane_id < K_mat_size && warp_id == 0)
    if (warp_id == 0)
    {
        // for (int i = 0; i < WARP_SIZE; ++i) intrinsics_K_grad[lane_id][i] = 0.0;
        if (d_backward_params.out_gradient_intrinsics.data)
            for (int i = 0; i < K_mat_size; ++i) intrinsics_K_grad[i][lane_id] = 0.0;
    }
    else if (warp_id == 1)
    {
        if (d_backward_params.out_gradient_intrinsics.data)
            for (int i = 0; i < distortion_polym_max_size; ++i) intrinsics_dis_ocam_grad[i][lane_id] = 0.0;
    }
    else if (warp_id == 2)
    {
        if (d_backward_params.out_gradient_pose)
            for (int i = 0; i < pose_size; ++i) pose_grad[i][lane_id] = 0.0;
    }
    else if (warp_id == 3)
    {
        for (int i = 0; i < num_descriptors; ++i) background_grads[i][lane_id] = 0.0;
    }
    __syncthreads();
    while (ticket < max_elements_total)
    {
        int activemask    = __activemask();
        int index_in_list = 0;
        float conf_gradients[ELEMENTS_PER_PIXEL];
        {
            for (int i = 0; i < ELEMENTS_PER_PIXEL; ++i)
            {
                conf_gradients[i] = 0.f;
            }
        }
        {
            float alpha_dest = 1.f;
            do
            {
                bool is_foreground = true;
                float alpha_val    = (index_in_list >= ELEMENTS_PER_PIXEL) ? -1.f : get_alpha_val(index_in_list);
                if (alpha_val == -1.f)
                {
                    is_foreground = false;
                    alpha_val     = 1.f;
                }


                int texture_index = 0;
                if (is_foreground)
                {
                    texture_index = get_tex_idx(index_in_list);
                    CUDA_KERNEL_ASSERT(texture_index >= 0);
                    CUDA_KERNEL_ASSERT(texture_index < d_texture.in_texture.sizes[1]);
                }
                float colors[num_descriptors];
                for (int ci = 0; ci < num_descriptors; ++ci)
                    colors[ci] = is_foreground ? d_texture.in_texture(ci, texture_index) : background_color[ci];

                float grad_in[num_descriptors];
                float g_alpha = 0;

                if (d_render_params.saturated_alpha_accumulation)
                {
                    float alpha_TA = ((alpha_dest - alpha_val) > 0.f) ? alpha_val : __saturatef(alpha_dest);
                    for (int ci = 0; ci < num_descriptors; ++ci)
                    {
                        grad_in[ci] = d_backward_params.in_gradient_image[layer](batch, ci, gy, gx);
                        float g     = alpha_TA * grad_in[ci];
                        if (is_foreground)
                            atomicAdd(&d_backward_params.out_gradient_texture(ci, texture_index), g);
                        else
                            atomicAdd(&background_grads[ci][lane_id], (double)g);


                        g_alpha += colors[ci] * grad_in[ci];
                    }

                    if (is_foreground) conf_gradients[index_in_list] += g_alpha;

                    if ((alpha_dest - alpha_val) < 0.f && alpha_dest > 0.f)
                    {
                        // fill case
                        conf_gradients[index_in_list] = 0.f;

                        for (int j = 0; j < index_in_list; ++j)
                        {
                            float g_iter = 0.f;
                            for (int ci = 0; ci < num_descriptors; ++ci) g_iter += colors[ci];
                            conf_gradients[j] -= g_iter;
                        }
                    }

                    alpha_dest = __saturatef(alpha_dest - alpha_val);
                }
                else
                {
                    for (int ci = 0; ci < num_descriptors; ++ci)
                    {
                        grad_in[ci] = d_backward_params.in_gradient_image[layer](batch, ci, gy, gx);
                        float g     = alpha_dest * alpha_val * grad_in[ci];
                        if (is_foreground)
                            atomicAdd(&d_backward_params.out_gradient_texture(ci, texture_index), g);
                        else
                            atomicAdd(&background_grads[ci][lane_id], (double)g);

                        // float* grad_write_address = is_foreground
                        //                                 ? &d_backward_params.out_gradient_texture(ci, texture_index)
                        //                                 : &out_background_gradient[ci];
                        // atomicAdd(grad_write_address, g);

                        g_alpha += colors[ci] * grad_in[ci];
                    }
                    g_alpha *= alpha_dest;
                    //  atomicAdd(&d_backward_params.out_gradient_confidence(0, texture_index), g_alpha);

                    if (is_foreground) conf_gradients[index_in_list] += g_alpha;


                    for (int j = 0; j < index_in_list; ++j)
                    {
                        int texture_index_iter = get_tex_idx(j);
                        // float confidence_val_iter = d_texture.points_confidence_value(0, texture_index_iter);
                        float alpha_val_iter = get_alpha_val(j);
                        float g_iter         = 0;
                        for (int ci = 0; ci < num_descriptors; ++ci)
                        {
                            const float epsilon = 1e-9;
                            float dem           = 1.f / (1.f - alpha_val_iter + epsilon);
                            float g_alpha_iter  = (grad_in[ci] * colors[ci] * alpha_dest * alpha_val * dem);
                            g_iter -= g_alpha_iter;
                            // g += -grad_in[ci] * color_iter * alpha_dest * confidence_val * dem;
                        }
                        // float* grad_address_iter = &d_backward_params.out_gradient_confidence(0, texture_index_iter);
                        // atomicAdd(grad_address_iter, g_iter);
                        conf_gradients[j] += g_iter;
                    }
                    alpha_dest = compute_new_alphadest(alpha_dest, alpha_val);
                }
                if (!is_foreground) break;

                ++index_in_list;
                // texture_index = (index_in_list < ELEMENTS_PER_PIXEL) ? get_tex_idx(index_in_list) : -1;
            } while (index_in_list < ELEMENTS_PER_PIXEL + 1 && alpha_dest >= 0.001);
        }

        auto get_subpixel_position = [&](int index_in_list) -> vec2
        {
            return vec2(d_bilinear_alpha_params_bw.bw_sorted_maxed[layer](batch, gy, gx, index_in_list, 3),
                        d_bilinear_alpha_params_bw.bw_sorted_maxed[layer](batch, gy, gx, index_in_list, 4));
        };
        auto get_subpixel_blend_index = [&](int index_in_list) -> int
        {
            int blend_idx = reinterpret_cast<int*>(
                &d_bilinear_alpha_params_bw.bw_sorted_maxed[layer](batch, gy, gx, index_in_list, 5))[0];
            return blend_idx;
        };
        auto get_point_id = [&](int index_in_list) -> int
        {
            int point_id = reinterpret_cast<int*>(
                &d_bilinear_alpha_params_bw.bw_sorted_maxed[layer](batch, gy, gx, index_in_list, 1))[0];
            return point_id;
        };

        auto get_point_size_opt = [&](int index_in_list) -> float
        { return d_bilinear_alpha_params_bw.bw_sorted_maxed[layer](batch, gy, gx, index_in_list, 6); };

        // return;

        ivec2 p_imgi = ivec2(gx, gy);
        for (int i = 0; i < index_in_list; ++i)
        {
            /*
             * Gradients from each pixel are accumulated independently to one point:
             * The gradient from color w.r.t. the point alpha is computed before this step
             * Here the gradient w.r.t the confidence and the gradient w.r.t. the blend_factors (thus the spatial
             * derivatives) are computed
             */

            int texture_index = get_tex_idx(i);

            vec2 uv         = get_subpixel_position(i);
            int blend_index = get_subpixel_blend_index(i);

            float grad_alpha = conf_gradients[i];

            Matrix<float, 4, 2> J_uv_b;
            vec4 blend_factors = compute_blending_fac(uv, &J_uv_b);

            // if pixel too far away: no gradient
            if (abs(uv.x() - gx) > 1 || abs(uv.y() - gy) > 1) blend_factors *= 0.f;

            float confidence_val = d_texture.points_confidence_value(0, texture_index);

            //  gradient w.r.t. layer:
            //  for dA and dC, this factor is a constant included in alpha
            //           (forward: alpha = blend_fac * conf * layer_fac)
            //               dA_dc = blend_fac * layer_fac
            //               dB_dc = conf * layer_fac
            //               dL_dc = conf * blend_fac (if in layer)

            float layer_factor       = 1.f;
            float J_layerfactor_proj = 0.f;
            if (d_backward_params.out_gradient_layer.data)
            {
                float point_size_opt = get_point_size_opt(i);
                layer_factor         = compute_point_size_fac(point_size_opt, layer, num_layers, &J_layerfactor_proj);
            }
            // write point confidence gradient, dA_dc = blend_fac
            float* point_confidence_grad_address = &d_backward_params.out_gradient_confidence(0, texture_index);
            float grad_point_confidence          = blend_factors[blend_index] * layer_factor * grad_alpha;
            atomicAdd(point_confidence_grad_address, grad_point_confidence);

            // if (!need_point_gradients) continue;

            // if (d_backward_params.in_gradient_image[layer].Image().distanceFromEdge(p_imgi.y(), p_imgi.x()) > 2)
            {
                // if (i > 0) continue;

                // compute dR_dp by singling out the contribution by this pixel, dA_db = confidence
                float grad_blend_single = confidence_val * layer_factor * grad_alpha;


                // compute dP_dA : J_uv_b^T * blend_fac[index]
                vec4 grad_blending;
                grad_blending.setZero();
                grad_blending[blend_index] = grad_blend_single;

                vec2 dR_dp   = J_uv_b.transpose() * grad_blending;
                int point_id = get_point_id(i);


                // adapt for multiresolution rendering
                float scale = 1.f;  // * powf(0.5f, float(layer));

                float grad_scale    = 1.f;
                auto cam2           = cam;
                cam2.crop_transform = cam.crop_transform.scale(scale * grad_scale);


                Sophus::SE3f V = d_render_params.Pose(cam.image_index);
                vec3 position;
                vec3 normal;
                float drop_out_radius;

                thrust::tie(position, normal, drop_out_radius) = point_cloud.GetPoint(point_id);

                if (cam.camera_model_type == CameraModel::PINHOLE_DISTORTION)
                {
                    auto [K, distortion] = d_render_params.PinholeIntrinsics(cam.camera_index);
                    auto [g_point, g_pose, g_k, g_dis, J_softmax_proj] = ProjectPointPinholeBackward(
                        position, normal, dR_dp, V, K, cam2.crop_transform, distortion, d_render_params.check_normal,
                        d_render_params.dist_cutoff, cam2.crop_rotation);

                    if (d_backward_params.out_gradient_layer.data)
                    {
                        // forward:  compute_point_size_fac( ".. selection_of_layers_implicitly .."
                        // proj(softmax(pointsize)) backwards: J_layerfactor_proj * J_proj_softmax * J_softmax_pointsize

                        float J_softmax_ps = _softplus_backwards(point_cloud.GetPointSize(point_id));

                        float grad_layer_ps = 0.f;
                        // J_saturate
                        // if (layer_buf > 0 && layer_buf < 1)
                        {
                            // float grad_a_point = blend_factors[blend_index] * confidence_val * grad_alpha;

                            grad_layer_ps    = J_softmax_ps * J_softmax_proj * J_layerfactor_proj * grad_alpha;
                            float grad_layer = blend_factors[blend_index] * confidence_val * grad_layer_ps;
                            // if (grad_layer != 0) printf("%f ", grad_layer);
                            float* point_layer_buf_grad_address = &d_backward_params.out_gradient_layer(point_id, 0);
                            atomicAdd(point_layer_buf_grad_address, grad_layer);
                            //  printf("%f ", grad_layer);
                        }
                    }


                    if (d_backward_params.out_gradient_points)
                    {
                        for (int k = 0; k < g_point.rows(); ++k)
                            atomicAdd(&d_backward_params.out_gradient_points[point_id][k], g_point(k));
                    }

                    if (d_backward_params.out_gradient_dynamic_points.data)
                    {
                        for (int k = 0; k < g_point.rows(); ++k)
                            atomicAdd(&d_backward_params.out_gradient_dynamic_points(batch, point_id, k), g_point(k));
                    }
                    if (d_backward_params.out_gradient_pose)
                    {
                        //  Extrinsics
                        for (int k = 0; k < g_pose.rows(); ++k) atomicAdd(&pose_grad[k][lane_id], (double)g_pose(k));
                    }
                    if (d_backward_params.out_gradient_intrinsics.data)
                    {
                        // Intrinsics: K
                        for (int k = 0; k < 5; ++k) atomicAdd(&intrinsics_K_grad[k][lane_id], g_k(k));

                        // Instrinsics: Distortion
                        for (int k = 0; k < 8; ++k) atomicAdd(&intrinsics_dis_ocam_grad[k][lane_id], g_dis(k));
                    }
                }
                else if (cam.camera_model_type == CameraModel::OCAM)
                {
                    auto [aff, poly] = d_render_params.OcamIntrinsics(cam.camera_index);

                    vec3 g_point;
                    vec6 g_pose;
                    vec5 g_affine;
                    if (d_backward_params.out_gradient_layer.data)
                    {
                        float point_size = point_cloud.GetPointSize(point_id);

                        BackwardOutputOcamPointsize bocam;
                        bocam = ProjectPointOcamBackwardPointSize(
                            position, normal, point_size, dR_dp, V, cam2.crop_transform, aff, poly,
                            d_render_params.check_normal, d_render_params.dist_cutoff, cam2.crop_rotation);
                        g_point            = bocam.g_point;
                        g_pose             = bocam.g_pose;
                        g_affine           = bocam.g_affine;
                        float g_point_size = bocam.g_pointsize;

                        float J_softmax_ps = _softplus_backwards(point_size);
                        {
                            float grad_layer_ps = J_softmax_ps * g_point_size * J_layerfactor_proj * grad_alpha;
                            float grad_layer    = blend_factors[blend_index] * confidence_val * grad_layer_ps;
                            // if (grad_layer != 0.f) printf("%f ", grad_layer);
                            float* point_layer_buf_grad_address = &d_backward_params.out_gradient_layer(point_id, 0);
                            atomicAdd(point_layer_buf_grad_address, grad_layer);
                            //  printf("%f ", grad_layer);
                        }
                    }
                    else
                    {
                        BackwardOutputOcam bocam;
                        bocam    = ProjectPointOcamBackward(position, normal, dR_dp, V, cam2.crop_transform, aff, poly,
                                                            d_render_params.check_normal, d_render_params.dist_cutoff,
                                                            cam2.crop_rotation);
                        g_point  = bocam.g_point;
                        g_pose   = bocam.g_pose;
                        g_affine = bocam.g_affine;
                    }

                    if (d_backward_params.out_gradient_points)
                    {
                        // Points
                        for (int k = 0; k < g_point.rows(); ++k)
                            atomicAdd(&d_backward_params.out_gradient_points[point_id][k], g_point(k));
                    }
                    if (d_backward_params.out_gradient_dynamic_points.data)
                    {
                        for (int k = 0; k < g_point.rows(); ++k)
                            atomicAdd(&d_backward_params.out_gradient_dynamic_points(batch, point_id, k), g_point(k));
                    }
                    if (d_backward_params.out_gradient_pose)
                    {
                        // Extrinsics
                        for (int k = 0; k < g_pose.rows(); ++k)
                            atomicAdd(&d_backward_params.out_gradient_pose[cam.image_index][k], (double)g_pose(k));
                    }
                    if (d_backward_params.out_gradient_intrinsics.data)
                    {
                        // Intrinsics: Affine
                        for (int k = 0; k < 5; ++k)
                            atomicAdd(&d_backward_params.out_gradient_intrinsics(cam.camera_index, k), g_affine(k));
                    }
                    CUDA_KERNEL_ASSERT("TODO IMPLEMENT POINT SIZE BACK")
                }
                else if (cam.camera_model_type == CameraModel::SPHERICAL)
                {
                    CUDA_KERNEL_ASSERT(false);
                }
            }
        }

        {
            get_next_list();
        }
        __syncwarp(activemask);
        // return;
    }
    __syncthreads();
    auto reduce_and_writeback = [&lane_id](double* sm_field)
    {
        double v = sm_field[lane_id];
        v        = Saiga::CUDA::warpReduceSum<double>(v);
        if (lane_id == 0) sm_field[0] = v;
    };
    if (warp_id == 0)
    {
        for (int i = 0; i < 5; ++i)
        {
            reduce_and_writeback(intrinsics_K_grad[i]);
        }
    }
    if (warp_id == 1)
    {
        if (cam.camera_model_type == CameraModel::PINHOLE_DISTORTION)
        {
            for (int i = 0; i < 8; ++i)
            {
                reduce_and_writeback(intrinsics_dis_ocam_grad[i]);
            }
        }
    }
    if (warp_id == 2)
    {
        for (int i = 0; i < 6; ++i)
        {
            reduce_and_writeback(pose_grad[i]);
        }
    }
    if (warp_id == 3)
    {
        for (int i = 0; i < num_descriptors; ++i)
        {
            reduce_and_writeback(background_grads[i]);
        }
    }

    __syncthreads();

    if (lane_id < distortion_polym_max_size + K_mat_size && warp_id == 0)
    {
        if (d_backward_params.out_gradient_intrinsics.data)
        {
            int fetch_index = lane_id;
            float val       = float(lane_id < K_mat_size ? (intrinsics_K_grad[lane_id][0])
                                                         : (intrinsics_dis_ocam_grad[lane_id - K_mat_size][0]));

            CUDA_KERNEL_ASSERT(fetch_index >= 0 && fetch_index < distortion_polym_max_size + K_mat_size);

            // currently not supporting ocam polynomial gradients
            if (cam.camera_model_type == CameraModel::PINHOLE_DISTORTION || lane_id < K_mat_size)
                atomicAdd(&d_backward_params.out_gradient_intrinsics(cam.camera_index, lane_id), val);
        }
    }
    else if (lane_id < K_mat_size + distortion_polym_max_size + pose_size && warp_id == 0)
    {
        if (d_backward_params.out_gradient_pose)
        {
            int fetch_index = lane_id - (K_mat_size + distortion_polym_max_size);
            CUDA_KERNEL_ASSERT(fetch_index >= 0 && fetch_index < pose_size);

            atomicAdd(&d_backward_params.out_gradient_pose[cam.image_index][fetch_index], pose_grad[fetch_index][0]);
        }
    }
    else if (lane_id < K_mat_size + distortion_polym_max_size + pose_size + num_descriptors && warp_id == 0)
    {
        int fetch_index = lane_id - (K_mat_size + distortion_polym_max_size + pose_size);
        CUDA_KERNEL_ASSERT(fetch_index >= 0 && fetch_index < num_descriptors);
        atomicAdd(&out_background_gradient[fetch_index], (float)background_grads[fetch_index][0]);
    }
}


void PointRendererCache::BlendBackwardsBilinearFast(int batch, NeuralPointCloudCuda point_cloud,
                                                    torch::Tensor background_color, bool use_environment_map)
{
    float* background = background_color.data_ptr<float>();
    SAIGA_ASSERT(background);
    SAIGA_ASSERT(background_color.is_cuda());

    auto out_gradient_background = output_gradient_background.data_ptr<float>();
    SAIGA_ASSERT(output_forward.size() == info->num_layers);


    int max_n;
    torch::Tensor atomic;
    {
        atomic = torch::zeros({1}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32));
        max_n  = 0;
        for (int i = 0; i < info->num_layers; ++i)
        {
            max_n += layers_cuda[i].size.x() * layers_cuda[i].size.y();
        }
    }

    static int bx = 32;  // iDivUp(layers_cuda[1].size.x(), 16);
    static int by = 32;  // iDivUp(layers_cuda[1].size.y(), 16);

    static int threaddim_x = MAX_THREADS_X;
    static int threaddim_y = MAX_THREADS_Y;


    int image_batch_id = batch;
    auto cam           = info->images[image_batch_id];
    switch (info->params.num_texture_channels)
    {
        case 3:
        {
            ::BlendBackwardsBilinearTiled<3, 16><<<dim3(bx, by, 1), dim3(16, 16, 1)>>>(
                point_cloud, background, out_gradient_background, batch, cam, info->params.need_point_gradient, atomic,
                max_n, info->num_layers);

            break;
        }
        case 4:
        {
            ::BlendBackwardsBilinearTiled<4, 16><<<dim3(bx, by, 1), dim3(16, 16, 1)>>>(
                point_cloud, background, out_gradient_background, batch, cam, info->params.need_point_gradient, atomic,
                max_n, info->num_layers);
            break;
        }
        case 6:
        {
            ::BlendBackwardsBilinearTiled<6, 16><<<dim3(bx, by, 1), dim3(16, 16, 1)>>>(
                point_cloud, background, out_gradient_background, batch, cam, info->params.need_point_gradient, atomic,
                max_n, info->num_layers);
            break;
        }
        case 8:
        {
            ::BlendBackwardsBilinearTiled<8, 16><<<dim3(bx, by, 1), dim3(16, 16, 1)>>>(
                point_cloud, background, out_gradient_background, batch, cam, info->params.need_point_gradient, atomic,
                max_n, info->num_layers);
            break;
        }
        case 16:
        {
            ::BlendBackwardsBilinearTiled<16, 16><<<dim3(bx, by, 1), dim3(16, 16, 1)>>>(
                point_cloud, background, out_gradient_background, batch, cam, info->params.need_point_gradient, atomic,
                max_n, info->num_layers);
            break;
        }
    }

    CUDA_SYNC_CHECK_ERROR();
}