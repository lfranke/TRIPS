/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
// functions for forward rendering
// #include "saiga/colorize.h"
#include "saiga/cuda/bitonicSort.h"
#include "saiga/cuda/random.h"
#include "saiga/cuda/reduce.h"
#include "saiga/vision/torch/CudaHelper.h"

#include "AlphaListSort.h"
#include "PointBlending.h"
#include "PointRenderer.h"
#include "PointRendererHelper.h"
#include "RenderConstants.h"

#include <cub/cub.cuh>
#include <cuda/barrier>
#include <cuda/pipeline>

#include "cooperative_groups.h"
#include "helper_math.h"
#include <curand_kernel.h>
__device__ __constant__ DeviceRenderParams d_render_params;
__device__ __constant__ DeviceTexture d_texture;
__device__ __constant__ DeviceForwardParams d_forward_params;
__device__ __constant__ DeviceBackwardParams d_backward_params;

__device__ __constant__ DeviceAlphaCompositionParams d_alpha_comp_params;
__device__ __constant__ DeviceBilinearAlphaParams d_bilinear_alpha_params;


void PointRendererCache::PushParametersForward()
{
    SAIGA_OPTIONAL_TIME_MEASURE("Param Upload", info->timer_system);
    {
        static DeviceForwardParams dfp;
        for (int i = 0; i < info->num_layers; ++i)
        {
            dfp.neural_out[i] = output_forward[i];
            // dfp.blend_out[i] = output_forward_blend[i];
        }
        CHECK_CUDA_ERROR(cudaMemcpyToSymbolAsync(d_forward_params, &dfp, sizeof(dfp)));
    }

    {
        static DeviceRenderParams drp;
        drp = PrepareDeviceRenderParams();

        CHECK_CUDA_ERROR(cudaMemcpyToSymbolAsync(d_render_params, &drp, sizeof(drp)));
        CUDA_SYNC_CHECK_ERROR();
    }
    if (info->scene)
    {
        static DeviceTexture d_tex;
        d_tex = PrepareDeviceTexture();

        CHECK_CUDA_ERROR(cudaMemcpyToSymbolAsync(d_texture, &d_tex, sizeof(d_tex)));
        CUDA_SYNC_CHECK_ERROR();
    }
}





__device__ inline vec2 rotateCropAroundCenter(vec2 point, vec2 center, const ReducedImageInfo& cam)
{
    // center (and cam.wh/2) are in crop size, i.e. 512/256
    point -= center;
    point = cam.crop_rotation * point;
    point += center;
    return point;
}

__inline__ __device__ thrust::tuple<vec2, float, float> ProjPoint(vec3 position, vec3 normal, float drop_out_radius,
                                                                  const ReducedImageInfo& cam, bool check_normal)
{
    vec2 image_p_a;
    vec2 ip;
    float z;
    float radius_pixels;
    Sophus::SE3f V = d_render_params.Pose(cam.image_index);

    if (cam.camera_model_type == CameraModel::PINHOLE_DISTORTION)
    {
        CUDA_KERNEL_ASSERT(cam.camera_model_type == CameraModel::PINHOLE_DISTORTION);
        auto [K, distortion] = d_render_params.PinholeIntrinsics(cam.camera_index);
        thrust::tie(image_p_a, z) =
            ProjectPointPinhole(position, normal, V, K, distortion, check_normal, d_render_params.dist_cutoff);
        radius_pixels = K.fx * cam.crop_transform.fx * drop_out_radius / z;
    }
    else if (cam.camera_model_type == CameraModel::OCAM)
    {
        auto [aff, poly] = d_render_params.OcamIntrinsics(cam.camera_index);
        thrust::tie(image_p_a, z) =
            ProjectPointOcam(position, normal, V, aff, poly, check_normal, d_render_params.dist_cutoff);
        radius_pixels = d_forward_params.neural_out[0].Image().w * cam.crop_transform.fx * drop_out_radius / z;
    }
    else if (cam.camera_model_type == CameraModel::SPHERICAL)
    {
        thrust::tie(image_p_a, z) = ProjectPointSpherical(
            position, normal, V,
            vec2(d_forward_params.neural_out[0].Image().w, d_forward_params.neural_out[0].Image().h), check_normal,
            d_render_params.dist_cutoff);
        radius_pixels = d_forward_params.neural_out[0].Image().w * cam.crop_transform.fx * drop_out_radius / z;
        ip            = image_p_a;
        return {ip, z, radius_pixels};
    }
    else
    {
        CUDA_KERNEL_ASSERT(cam.camera_model_type == CameraModel::PINHOLE_DISTORTION);
    }
    ip = cam.crop_transform.normalizedToImage(image_p_a);
    ip = rotateCropAroundCenter(ip, vec2(cam.w / 2, cam.h / 2), cam);
    return {ip, z, radius_pixels};
}


__device__ inline bool discard_point_for_confidence(int texture_index)
{
    float confidence = d_texture.points_confidence_value(0, texture_index);
    if (confidence < d_render_params.stability_cutoff_value && d_render_params.viewer_only)
    {
        return true;
    }
    return false;
}

__inline__ __device__ thrust::tuple<vec2, float, float> GetProjectedPoint(vec3 position, vec3 normal,
                                                                          float drop_out_radius, int point_id,
                                                                          ReducedImageInfo& cam)
{
    return ProjPoint(position, normal, drop_out_radius, cam, d_render_params.check_normal);
}

void PointRendererCache::UploadLinkedListBuffers()
{
    static DeviceBilinearAlphaParams dbap;

    for (int i = 0; i < info->num_layers; ++i)
    {
        dbap.per_pixel_list_heads[i]   = layers_cuda[i].per_pixel_list_heads;
        dbap.bw_sorted_maxed[i]        = layers_cuda[i].bw_sorted_maxed;
        dbap.per_pixel_list_lengths[i] = layers_cuda[i].per_pixel_list_lengths;
        dbap.scanned_countings[i]      = layers_cuda[i].scanned_counts;
    }
    CHECK_CUDA_ERROR(cudaMemcpyToSymbolAsync(d_bilinear_alpha_params, &dbap, sizeof(dbap)));
}


__device__ inline float _softplus(float x, float beta = 1.f, float threshold = 20.f)
{
    //  return x * beta > threshold ? x : logf(1 + expf(beta * x));
    if (x > threshold) return x;
    return logf(1.f + expf(x * beta)) / beta;
}






#define DEFAULT_BLOCK_SIZE_FAST_COLLECT 256

template <int num_layers>
__global__ __launch_bounds__(DEFAULT_BLOCK_SIZE_FAST_COLLECT) void CountAndCollectTiled(
    __grid_constant__ const DevicePointCloud point_cloud, float* dropout_p,
    __grid_constant__ const ReducedImageInfo cam, int batch, bool train, float* __restrict__ depth_buffer,
    int* __restrict__ max_layer_buffer,
    ConstReadOnlyStaticDeviceTensor<float, 3> full_list_buffer_data /*(batch,numelems,5)*/,
    float* __restrict__ full_list_buffer_data_ptr)

{
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();

    const int threadsPerBlock = blockDim.x * blockDim.y;
    const int threadNumInBlock =
        threadIdx.x + blockDim.x * threadIdx.y;  // (alternatively: threadIdx.y + blockDim.y * threadIdx.x)
    const int blockNumInGrid =
        blockIdx.x + gridDim.x * blockIdx.y;  //  (alternatively: blockIdx.y  + gridDim.y  * blockIdx.x)

    // Load all camera params to Shared Memory
    __shared__ Sophus::SE3f V;
    __shared__ IntrinsicsPinholef K;
    __shared__ Distortionf distortion;
    __shared__ Vector<float, 5> ocam_aff;
    __shared__ ArrayView<const float> ocam_poly;
    if (threadNumInBlock < 32)
    {
        V = d_render_params.Pose(cam.image_index);
    }
    else if (threadNumInBlock < 64)
    {
        if (cam.camera_model_type == CameraModel::PINHOLE_DISTORTION)
        {
            auto [K_l, distortion_l] = d_render_params.PinholeIntrinsics(cam.camera_index);
            K                        = K_l;
            distortion               = distortion_l;
        }
        else if (cam.camera_model_type == CameraModel::OCAM)
        {
            auto [aff_l, poly_l] = d_render_params.OcamIntrinsics(cam.camera_index);
            ocam_aff             = aff_l;
            ocam_poly            = poly_l;
        }
    }
    __syncthreads();

    auto valid_point = [](vec2 ip, float z, int layer, int radius_pixels)
    {
        return !(
            z <= 0 || ip(0) < 0 || ip(0) >= d_bilinear_alpha_params.per_pixel_list_lengths[layer].size(2) - 1 ||
            ip(1) < 0 || ip(1) >= d_bilinear_alpha_params.per_pixel_list_lengths[layer].size(1) - 1 ||
            (d_render_params.drop_out_points_by_radius && radius_pixels < d_render_params.drop_out_radius_threshold));
    };

    for (int point_id = grid.thread_rank(); point_id < point_cloud.Size(); point_id += grid.size())
    {
        // init as invalid
        depth_buffer[point_id] = -1.f;
        // if (point_id >= point_cloud.Size()) continue;
#ifndef FAST_MINIMAL_IMPL
        if (d_render_params.viewer_only)
        {
            int conf_id = point_cloud.GetIndex(point_id);
            if (discard_point_for_confidence(conf_id)) continue;
        }
        else
        {
            if (train && dropout_p)
            {
                bool drop_out = dropout_p[point_id] == 1;
                if (drop_out) continue;
            }
        }
#endif
        if (train && dropout_p)
        {
            bool drop_out = dropout_p[point_id] == 1;
            if (drop_out) continue;
        }

        //    int texture_index    = point_cloud.GetIndex(point_id);
        float point_size_opt = 1.f;
        if (d_render_params.use_layer_point_size) point_size_opt = _softplus(point_cloud.GetPointSize(point_id));
        // if (layer_buf < 0.0001) continue;
        vec2 ip;
        float z;
        float radius_pixels;


        {
            vec3 position;

            vec2 image_p_a;
            float drop_out_radius;
            thrust::tie(position, drop_out_radius) = point_cloud.GetPointWoNormal(point_id);

            if (cam.camera_model_type == CameraModel::PINHOLE_DISTORTION)
            {
                CUDA_KERNEL_ASSERT(cam.camera_model_type == CameraModel::PINHOLE_DISTORTION);
                //  auto [K, distortion] = d_render_params.PinholeIntrinsics(cam.camera_index);
                thrust::tie(image_p_a, z) =
                    ProjectPointPinholeWoNormal(position, V, K, distortion, d_render_params.dist_cutoff);
                radius_pixels = K.fx * cam.crop_transform.fx * drop_out_radius / z;

                point_size_opt = K.fx * cam.crop_transform.fx * point_size_opt / z;
            }
            else if (cam.camera_model_type == CameraModel::OCAM)
            {
                //  auto [aff, poly] = d_render_params.OcamIntrinsics(cam.camera_index);
                float pointsize_ocam;
                thrust::tie(image_p_a, z, pointsize_ocam) =
                    ProjectPointOcamWoNormalWPointsize(position, V, ocam_aff, ocam_poly, 0.15, point_size_opt);
                radius_pixels  = d_forward_params.neural_out[0].Image().w * cam.crop_transform.fx * drop_out_radius / z;
                point_size_opt = pointsize_ocam;
            }
            else if (cam.camera_model_type == CameraModel::SPHERICAL)
            {
                thrust::tie(image_p_a, z) = ProjectPointSphericalWoNormal(
                    position, V,
                    vec2(d_forward_params.neural_out[0].Image().w, d_forward_params.neural_out[0].Image().h),
                    d_render_params.dist_cutoff);
                radius_pixels = d_forward_params.neural_out[0].Image().w * cam.crop_transform.fx * drop_out_radius / z;
                ip            = image_p_a;
            }
            else if (cam.camera_model_type == CameraModel::ORTHO)
            {
                thrust::tie(image_p_a, z) =
                    ProjectPointToOrthographic(position, V, d_forward_params.neural_out[0].Image().w,
                                               d_forward_params.neural_out[0].Image().h, d_render_params.dist_cutoff);
                radius_pixels = d_forward_params.neural_out[0].Image().w * cam.crop_transform.fx * drop_out_radius / z;
            }
            else
            {
                CUDA_KERNEL_ASSERT(cam.camera_model_type == CameraModel::PINHOLE_DISTORTION);
            }
            // if (z <= 0) continue;
            ip = cam.crop_transform.normalizedToImage(image_p_a);
            ip = rotateCropAroundCenter(ip, vec2(cam.w / 2, cam.h / 2), cam);

            // z discard
            // border pixel will be ignored
            // radius discard
            if (!valid_point(ip, z, 0, radius_pixels)) continue;
        }
        // write depth
        depth_buffer[point_id] = z;

        if (!(d_render_params.use_layer_point_size && d_render_params.render_points_in_all_lower_resolutions))
        {
            CUDA_KERNEL_ASSERT(false);
        }
        else
        {
            int data_written_to = point_id;
            {
                // write point data
                int texture_index      = point_cloud.GetIndex(point_id);
                const int offset_batch = 0;  // batch * full_list_buffer_data.strides[0];
                if (train)
                {
                    (full_list_buffer_data_ptr + offset_batch + data_written_to * 5 + 0)[0] = ip.x();
                    (full_list_buffer_data_ptr + offset_batch + data_written_to * 5 + 1)[0] = ip.y();
                    (full_list_buffer_data_ptr + offset_batch + data_written_to * 5 + 2)[0] =
                        reinterpret_cast<float*>(&texture_index)[0];
                    (full_list_buffer_data_ptr + offset_batch + data_written_to * 5 + 3)[0] = point_size_opt;
                    (full_list_buffer_data_ptr + offset_batch + data_written_to * 5 + 4)[0] =
                        reinterpret_cast<float*>(&point_id)[0];
                }
                else
                {
                    float4* addr_data = (float4*)(full_list_buffer_data_ptr + offset_batch + data_written_to * 4);
                    float4 data =
                        make_float4(ip.x(), ip.y(), reinterpret_cast<float*>(&texture_index)[0], point_size_opt);
                    (addr_data)[0] = data;
                }
            }
            int layer_higher = 0;
            if (point_size_opt > 1)
            {
                layer_higher = min(int(ceil(log2f(point_size_opt))), num_layers - 1);
            }
            int layer = 0;
            for (; layer < layer_higher + 1; ++layer, ip *= 0.5f, radius_pixels *= 0.5f)
            {
                ivec2 p_rd = ivec2(__float2int_rd(ip(0)), __float2int_rd(ip(1)));

                // discard upper bounded pixel
                if (!valid_point(p_rd, z, layer, radius_pixels)) break;
#pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    // count list up
                    atomicAdd(&d_bilinear_alpha_params.per_pixel_list_lengths[layer](batch, p_rd.y() + i / 2,
                                                                                     p_rd.x() + i % 2),
                              1);
                }
            }
            max_layer_buffer[point_id] = layer;
        }
    }
}

__global__ __launch_bounds__(DEFAULT_BLOCK_SIZE_FAST_COLLECT) void SplattingPass(
    int batch, StaticDeviceTensor<double, 3> full_list_buffer /* (1,numelems,1)*/,
    double* __restrict__ full_list_buffer_ptr, StaticDeviceTensor<float, 3> full_list_buffer_data /*(1,numelems,5)*/,
    float* __restrict__ depth_buffer, int* __restrict__ max_layer_buffer,
    StaticDeviceTensor<int32_t, 2> layer_lengths /*(1,num_layers)*/)
{
    for (int point_id = cooperative_groups::this_grid().thread_rank() / 4; point_id < full_list_buffer_data.size(1);
         point_id += cooperative_groups::this_grid().size() / 4)
    {
        const int sub_id = cooperative_groups::this_grid().thread_rank() % 4;

        if (point_id >= full_list_buffer_data.size(1)) return;

        float z = depth_buffer[point_id];
        if (z > 0)
        {
            // get max layer
            int max_layer = max_layer_buffer[point_id];

            double data_l = __hiloint2double(reinterpret_cast<int*>(&z)[0], point_id);

            // get uv and z
            float ip_x = full_list_buffer_data(0, point_id, 0);
            float ip_y = full_list_buffer_data(0, point_id, 1);
            // for loop dup
            for (int layer = 0; layer < max_layer; ++layer)
            // if (layer < max_layer)
            {
                ivec2 p_rd = ivec2(__float2int_rd(ip_x), __float2int_rd(ip_y));

                // #pragma unroll
                //             for (int sub_id = 0; sub_id < 4; ++sub_id)
                {
                    // x: i%2; y: i/2
                    const ivec2 splat_point = ivec2(p_rd.x() + sub_id % 2, p_rd.y() + sub_id / 2);

                    const int offset = atomicAdd(
                        &d_bilinear_alpha_params.per_pixel_list_lengths[layer](batch, splat_point.y(), splat_point.x()),
                        1);

                    const int scanned_c =
                        d_bilinear_alpha_params.scanned_countings[layer](batch, splat_point.y(), splat_point.x()) +
                        offset + layer_lengths(0, layer);
                    // layer_lengths_sm[layer];
                    //  int scanned_c                                        = 0;
                    (full_list_buffer_ptr + scanned_c)[0] = data_l;
                    // full_list_buffer.At({0, scanned_c, 0}) = data_l;
                    //  unsigned long long int* addr =
                    //      ((unsigned long long int*)full_list_buffer_ptr) + offset_batch + scanned_c;
                    //  unsigned long long int val = reinterpret_cast<unsigned long long int*>(&data_l)[0];
                    //
                    //  atomicExch(addr, val);

                    // cooperative_groups::memcpy_async(grid, full_list_buffer_ptr + offset_batch +
                    // scanned_c,
                    //                                  &data_l, sizeof(double));
                    // cuda::memcpy_async(full_list_buffer_ptr + offset_batch + scanned_c, &data_l,
                    // sizeof(double),
                    //                    barrier);
                    //  cuda::mem
                }

                //   layer_offset += layer_lengths(batch, layer);
                ip_x *= 0.5f;
                ip_y *= 0.5f;
            }
        }
    }
}

struct SortData3
{
    float depth;
    int index;
    __device__ SortData3() {}
    __device__ SortData3(const SortData3& other)
    {
        depth = other.depth;
        index = other.index;
    }
    __device__ SortData3& operator=(const SortData3& other)
    {
        if (this == &other) return *this;

        depth = other.depth;
        index = other.index;
        return *this;
    }
    __device__ SortData3(const float d, const int i) : depth(d), index(i) {}
    __device__ int& get_index() { return index; }
    // __device__ int get_layer() { return (index_layer & 0xF); }
    // __device__ SortData(const float d, const float i, bool reinterpret) : depth(d)
    // {
    //     index = reinterpret_cast<const int*>(&i)[0];
    // }
};
__device__ __forceinline__ bool operator<(SortData3& a, SortData3& b)
{
    return a.depth < b.depth;
}

#define THREAD_DIMX_PER_BLOCK_MULTITHREADLOAD2 16
#define THREAD_DIMY_PER_BLOCK_MULTITHREADLOAD2 16
#define THREADS_PER_BLOCK_MULTITHREADLOAD2 \
    (THREAD_DIMX_PER_BLOCK_MULTITHREADLOAD2 * THREAD_DIMY_PER_BLOCK_MULTITHREADLOAD2)

template <int NUM_DESCRIPTORS, int ELEMENTS_PER_PIXEL, bool train>
__global__ void __launch_bounds__(THREADS_PER_BLOCK_MULTITHREADLOAD2)
    FastSortFusedNew(int batch,
                     ConstReadOnlyStaticDeviceTensor<double, 3> full_list_buffer /* (batch,pointcloudsize,1)*/,
                     const double* __restrict__ full_list_buffer_data_ptr,
                     ConstReadOnlyStaticDeviceTensor<float, 3> full_list_buffer_data /*(batch,pointcloudsize,4)*/,
                     const float* __restrict__ full_list_buffer_data_data_ptr,
                     const float* __restrict__ background_color, StaticDeviceTensor<int, 1> glob_atomic_c,
                     StaticDeviceTensor<int32_t, 2, int> layer_lengths,
                     StaticDeviceTensor<int32_t, 2> non_zero_indices /*(listlength,1)*/, int num_layers_input)
{
    constexpr float max_depth = 1e25;
    const int thread_idx      = threadIdx.x + threadIdx.y * blockDim.x;
    constexpr int WARP_SIZE   = 32;
    int own_index_in_warp     = thread_idx % WARP_SIZE;
    // const int warp_id_in_block  = thread_idx / WARP_SIZE;

    CUDA_KERNEL_ASSERT(ELEMENTS_PER_PIXEL <= WARP_SIZE / 2);

    constexpr int LISTS_PER_WARP     = WARP_SIZE / ELEMENTS_PER_PIXEL;
    constexpr int WARP_LEADER_STRIDE = ELEMENTS_PER_PIXEL;

    CUDA_KERNEL_ASSERT(WARP_SIZE % ELEMENTS_PER_PIXEL == 0);


    full_list_buffer.setDataPointer(full_list_buffer_data_ptr);
    full_list_buffer_data.setDataPointer(full_list_buffer_data_data_ptr);
    uint32_t ticket = 0;

    uint32_t layer = 0, gx = 0, gy = 0;
    uint32_t start_offset = 0;
    uint32_t length = 1, leader_length = 1;


    SortData3 local_mem_warp = {max_depth, int(-1)};

    uint32_t max_elements_realigned = non_zero_indices.size(0);
    // uint32_t max_elements_realigned = max_elements_total;

    bool loop_run = true;

    auto get_next_list = [&ticket, &glob_atomic_c, &layer, &gx, &gy, &num_layers_input, &length, &local_mem_warp,
                          &max_elements_realigned, &own_index_in_warp, &leader_length, &loop_run, &non_zero_indices]
    {
        bool continue_loop = true;
        // warp leaders fetches next list, computes indices
        unsigned leader_mask = 0;
        leader_mask          = __ballot_sync(0xffffffff, own_index_in_warp % WARP_LEADER_STRIDE == 0);
        if (own_index_in_warp % WARP_LEADER_STRIDE == 0)
        {
            ticket = atomicAdd(&glob_atomic_c(0), 1);

            leader_length = 0;
            if (ticket < max_elements_realigned)
            {
                uint32_t tick = non_zero_indices(ticket, 0);
                // uint32_t tick = ticket;
                for (int i = 0; i < num_layers_input; i++)
                {
                    int w, h;
                    if (train)
                    {
                        w = d_forward_params.neural_out[i].size(3);
                        h = d_forward_params.neural_out[i].size(2);
                    }
                    else
                    {
                        w = d_forward_params.neural_out[i].size(2);
                        h = d_forward_params.neural_out[i].size(1);
                    }

                    if (tick < w * h)
                    {
                        layer = i;
                        gx    = tick % w;
                        gy    = tick / w;
                        break;
                    }
                    tick -= w * h;
                }
                //  length       = d_bilinear_alpha_params.per_pixel_list_lengths[layer](batch, gy, gx);
                //  start_offset = d_bilinear_alpha_params.scanned_countings[layer](batch, gy, gx) + layer_offset;
            }
            else
            {
                length = 0;
            }
            bool all_failed = __all_sync(leader_mask, ticket >= max_elements_realigned);
            continue_loop   = !all_failed;
        }
        __syncwarp();
        loop_run = __shfl_sync(0xffffffff, continue_loop, 0);
        //  clear local mem
        local_mem_warp = {max_depth, int(-1)};
    };

    get_next_list();
    __syncwarp();
    //  while (length > 16 && ticket < max_elements_realigned) get_next_list();

    while (loop_run)  //(ticket < max_elements_realigned)
    {
        const int offset_batch = 0;  // batch * full_list_buffer.strides[0];

        int threads_in_use = 0;
        /*
         * every half warp (or however many ELEMENTS are present) stores the sorted list in its register (sorted by
         * threadnum) then all thread load 32 new elements, bitonic sort them and the locally stored item
         *
         *
         */

        for (; threads_in_use < WARP_SIZE; threads_in_use += WARP_LEADER_STRIDE)
        // for (; leader_threads < WARP_SIZE / 2; leader_threads += WARP_LEADER_STRIDE)
        {
            //      if (leader_threads > 0) break;

            auto get_element_addresses = [&own_index_in_warp, &threads_in_use, &ticket, &length,
                                          &max_elements_realigned, &batch, &leader_length, &start_offset,
                                          &layer_lengths, &num_layers_input](int layer_id, int gy_id, int gx_id)
            {
                if (own_index_in_warp == threads_in_use)
                {
                    if (ticket >= max_elements_realigned || layer_id >= num_layers_input ||
                        gx_id >= d_bilinear_alpha_params.per_pixel_list_lengths[layer_id].size(2) ||
                        gy_id >= d_bilinear_alpha_params.per_pixel_list_lengths[layer_id].size(1))
                    {
                        length = 0;
                    }
                    else
                    {
                        length = d_bilinear_alpha_params.per_pixel_list_lengths[layer_id](batch, gy_id, gx_id);

                        leader_length += length;

                        // int layer_offset = 0;
                        // for (int i = 0; i < layer_id; i++) layer_offset += layer_lengths(batch, i);
                        int layer_offset = layer_lengths(0, layer_id);
                        start_offset =
                            d_bilinear_alpha_params.scanned_countings[layer_id](batch, gy_id, gx_id) + layer_offset;
                    }
                }
                length       = __shfl_sync(0xffffffff, length, threads_in_use);
                start_offset = __shfl_sync(0xffffffff, start_offset, threads_in_use);
            };

            int start_layer = __shfl_sync(0xffffffff, layer, threads_in_use);
            int x_id        = gx;
            int y_id        = gy;

            int l_id = start_layer;

            get_element_addresses(l_id, y_id, x_id);

            {
                for (int elem_i = 0; elem_i < length; elem_i += WARP_SIZE)
                {
#ifdef DEBUG_PRINT
                    __syncwarp();
                    printf("Load Addr and len (%i,%i) - \n", start_offset + elem_i + own_index_in_warp, length);
#endif
                    const int local_memory_start_thread = threads_in_use;
                    const int local_memory_end_thread   = local_memory_start_thread + ELEMENTS_PER_PIXEL;

                    int remaining_elements = max(0, min(int(length) - elem_i, 32));

                    SortData3 local_loaded_element;

                    // each thread loads one element
                    if (own_index_in_warp < remaining_elements)
                    {
                        int fetch_index      = start_offset + elem_i + own_index_in_warp;
                        int2 data_comb       = ((int2*)(full_list_buffer_data_ptr + offset_batch + fetch_index))[0];
                        local_loaded_element = SortData3(reinterpret_cast<const float*>(&data_comb.y)[0], data_comb.x);

#ifdef DEBUG_PRINT
                        printf("Loaded (%f,%i) - \n", local_loaded_element.depth, local_loaded_element.index);
#endif
                    }
                    else
                    {
                        local_loaded_element = {max_depth, int(-1)};
                    }

                    // early continue if all elements are further away then current list
                    {
                        const int last_elem_warp_index = threads_in_use + (WARP_LEADER_STRIDE - 1);
                        const float last_depth_in_local_list =
                            __shfl_sync(0xffffffff, local_mem_warp.depth, last_elem_warp_index);
                        if (elem_i != 0 &&
                            __all_sync(0xffffffff, local_loaded_element.depth > last_depth_in_local_list))
                            continue;
                    }
#ifdef DEBUG_PRINT
                    __syncwarp();
                    CUDA_KERNEL_ASSERT(local_loaded_element.depth > 0);
#endif

                    bool first = elem_i == 0;  // && start_layer == l_id);

                    // first sorting: thread 0-15: new elements, 16-31 list stored, sort descending
                    SortData3 local_sort_storage;
                    if (first)
                    {
                        local_sort_storage = local_loaded_element;
                    }
                    else
                    {
                        // fetch local memory to latter part of list, then fill with lower part of list
                        local_sort_storage = local_mem_warp;
                        if (threads_in_use == 0)
                        {
                            local_sort_storage.depth = __shfl_xor_sync(0xffffffff, local_sort_storage.depth, 16);
                            local_sort_storage.index = __shfl_xor_sync(0xffffffff, local_sort_storage.index, 16);
                        }
                        if (own_index_in_warp < ELEMENTS_PER_PIXEL)
                        {
                            local_sort_storage = local_loaded_element;
                        }
                    }

#ifdef DEBUG_PRINT
                    __syncwarp();
                    printf("Init Half (%f,%i) - \n", local_sort_storage.depth, local_sort_storage.index);

                    CUDA_KERNEL_ASSERT(local_sort_storage.depth > 0);
                    __syncwarp();
#endif

                    // first sort: elem 0-15 new, elem 16-31 local mem
                    {
                        float2 data_f2;
                        ((SortData3*)&data_f2)[0] = local_sort_storage;
                        data_f2                   = Saiga::CUDA::bitonicWarpSort(data_f2, own_index_in_warp);
                        local_sort_storage        = ((SortData3*)&data_f2)[0];
                    }
#ifdef DEBUG_PRINT
                    printf("Sorted1 (%f,%i) - \n", local_sort_storage.depth, local_sort_storage.index);

                    CUDA_KERNEL_ASSERT(local_sort_storage.depth > 0);

                    // local_sort_storage = reinterpret_cast<SortData*>(&sorted)[0];

                    __syncwarp();
#endif
                    // second sort only if enought elements remaining and not first iter
                    if (!first && remaining_elements > ELEMENTS_PER_PIXEL)
                    // if (remaining_elements > ELEMENTS_PER_PIXEL)
                    {
#ifdef DEBUG_PRINT
                        CUDA_KERNEL_ASSERT(local_sort_storage.depth > 0);

                        __syncwarp();
                        printf("XORed (%f,%i) - \n", local_sort_storage.depth, local_sort_storage.index);
#endif

                        // remaining elements
                        if (own_index_in_warp >= ELEMENTS_PER_PIXEL) local_sort_storage = local_loaded_element;
#ifdef DEBUG_PRINT
                        __syncwarp();
                        printf("OtherHalf (%f,%i) - \n", local_sort_storage.depth, local_sort_storage.index);


                        CUDA_KERNEL_ASSERT(local_sort_storage.depth > 0);
#endif
                        // second sort: elem 0-15 local mem, elem 16-31 new
                        {
                            float2 data2_f2;
                            ((SortData3*)&data2_f2)[0] = local_sort_storage;
                            data2_f2                   = Saiga::CUDA::bitonicWarpSort(data2_f2, own_index_in_warp);
                            local_sort_storage         = ((SortData3*)&data2_f2)[0];
                        }
                    }
#ifdef DEBUG_PRINT
                    __syncwarp();
                    CUDA_KERNEL_ASSERT(local_sort_storage.depth > 0);
                    printf("Sorted2 (%f,%i) - \n", local_sort_storage.depth, local_sort_storage.index);
#endif

                    if (threads_in_use > 0)
                    {
                        // move result to correct
                        local_sort_storage.depth =
                            __shfl_xor_sync(0xffffffff, local_sort_storage.depth, threads_in_use);
                        local_sort_storage.index =
                            __shfl_xor_sync(0xffffffff, local_sort_storage.index, threads_in_use);
#ifdef DEBUG_PRINT
                        __syncwarp();
                        CUDA_KERNEL_ASSERT(local_sort_storage.depth > 0);
                        printf("Leaderthread xor (%f,%i) - \n", local_sort_storage.depth, local_sort_storage.index);
#endif
                    }

                    // write nearest 16 elements back
                    if (own_index_in_warp >= local_memory_start_thread && own_index_in_warp < local_memory_end_thread)
                        local_mem_warp = local_sort_storage;
#ifdef DEBUG_PRINT
                    __syncwarp();
                    printf("LocalMem (%f,%i) - \n", local_mem_warp.depth, local_mem_warp.index);


                    //    break;
#endif
                    // first = false;
                }
                // l_id += 1;
                // x_id /= 2;
                // y_id /= 2;
                //   if (l_id < max_layer) get_element_addresses(l_id, y_id, x_id);
                //   if (l_id < 1) get_element_addresses(l_id, y_id, x_id);
            }
        }
        __syncwarp();
        // debug test:
#if 0
        {
            float run_elem = local_mem_warp.depth;
            if (own_index_in_warp == 0)
            {
                //   printf("(%f,%i) - ", local_mem_warp.depth, local_mem_warp.index);
                CUDA_KERNEL_ASSERT(run_elem != max_depth);
            }

            for (int i = 1; i < 16; ++i)
            {
                float other_depth = __shfl_sync(0xffffffff, local_mem_warp.depth, i);
                int other_index   = __shfl_sync(0xffffffff, local_mem_warp.index, i);
                if (own_index_in_warp == 0)
                {
                    // printf("(%f,%i) - ", other_depth, other_index);
                    if (!(run_elem <= other_depth))
                    {
                        printf("%f - %f\n", run_elem, other_depth);
                    }
                    //  CUDA_KERNEL_ASSERT(other_depth != max_depth);
                    CUDA_KERNEL_ASSERT(run_elem <= other_depth);
                }
                run_elem = other_depth;
            }
        }
#    ifdef DEBUG_PRINT
        __syncwarp();
        printf("-------- \n");
#    endif
#endif

        {
            const bool is_leader_thread =
                own_index_in_warp % WARP_LEADER_STRIDE == 0 && own_index_in_warp < threads_in_use;

            // accumulate
            float alpha_dest = 1.f;
            // float color_out[4] = {0.f, 0.f, 0.f, 0.f};
            // float4 color_out = make_float4(0.f, 0.f, 0.f, 0.f);
            float color_out[NUM_DESCRIPTORS];
            for (int i = 0; i < NUM_DESCRIPTORS; ++i) color_out[i] = 0.f;

            if (own_index_in_warp < threads_in_use)  // disregard half threads with last element
            {
                const int local_leader_thread = (own_index_in_warp / WARP_LEADER_STRIDE) * WARP_LEADER_STRIDE;
                unsigned mask_sync            = 0xffffffff;
                if (threads_in_use < WARP_SIZE) mask_sync = (1 << threads_in_use) - 1;

                // get relevent values from leader
                gx     = __shfl_sync(mask_sync, gx, local_leader_thread);
                gy     = __shfl_sync(mask_sync, gy, local_leader_thread);
                layer  = __shfl_sync(mask_sync, layer, local_leader_thread);
                length = __shfl_sync(mask_sync, leader_length, local_leader_thread);

                unsigned ballot_sync_mask = 0;
                // if ((own_index_in_warp - local_leader_thread) < length)
                ballot_sync_mask = __ballot_sync(mask_sync, (own_index_in_warp - local_leader_thread) < length);

                // float4 color = make_float4(0, 0, 0, 0);

                if (own_index_in_warp - local_leader_thread < length)
                {
                    mask_sync = ballot_sync_mask;

                    __syncwarp(mask_sync);

                    int fetch_idx = local_mem_warp.get_index();
                    //     printf("LocalMem (%f,%i) fid(%i) - \n", local_mem_warp.depth,
                    //     local_mem_warp.index.fetch_idx);

                    CUDA_KERNEL_ASSERT(fetch_idx != -1);

                    float ipx, ipy;
                    int texture_index;
                    float point_size_opt = 1.f;

                    if (!train)
                    {
                        const float4 data_fetch = ((float4*)&full_list_buffer_data(0, fetch_idx, 0))[0];
                        ipx                     = data_fetch.x;
                        ipy                     = data_fetch.y;
                        texture_index           = reinterpret_cast<const int*>(&data_fetch.z)[0];
                        if (d_render_params.use_layer_point_size) point_size_opt = data_fetch.w;
                    }
                    else
                    {
                        ipx           = full_list_buffer_data(0, fetch_idx, 0);
                        ipy           = full_list_buffer_data(0, fetch_idx, 1);
                        texture_index = reinterpret_cast<const int*>(&full_list_buffer_data(0, fetch_idx, 2))[0];
                        if (d_render_params.use_layer_point_size)
                            point_size_opt = full_list_buffer_data(0, fetch_idx, 3);
                    }

                    const float layer_mult_fac = 1.f / float(1 << layer);
                    vec2 ip                    = vec2(ipx * layer_mult_fac, ipy * layer_mult_fac);
                    // vec4 blend_vec             = compute_blending_fac(ip);
                    // int blend_index            = blend_fac_index(ip, vec2(gx, gy));
                    // float bilinear_fac         = blend_vec[blend_index];

                    float bilinear_fac = compute_blending_fac_wo_index(ip, ivec2(gx, gy));

                    // for lower layers: if the point is too far away discard the contribution by setting alpha to
                    // zero
                    if (abs(ip.x() - gx) > 1 || abs(ip.y() - gy) > 1) bilinear_fac = 0.f;

                    float alpha_bilin = bilinear_fac * d_texture.points_confidence_value(0, texture_index);

                    /// if (d_render_params.use_layer_point_size)
                    ///{
                    float layer_factor = compute_point_size_fac(point_size_opt, layer, num_layers_input);
                    alpha_bilin *= layer_factor;
                    // }

                    // float color0 = d_texture.in_texture(0, texture_index);
                    // float color1 = d_texture.in_texture(1, texture_index);
                    // float color2 = d_texture.in_texture(2, texture_index);
                    // float color3 = d_texture.in_texture(3, texture_index);

#define ALPHA_DEST_CUTOFF 0.001f
#if 0
                    if (d_render_params.saturated_alpha_accumulation)
                    {
                        // color = make_float4(color0, color1, color2, color3) * alpha_bilin;


                        float alpha_dest_precomputed = 1.f;
                        alpha_dest                   = alpha_dest_precomputed;
                        for (int index_in_list = 1; index_in_list < ELEMENTS_PER_PIXEL; ++index_in_list)
                        {
                            alpha_dest_precomputed =
                                __shfl_up_sync(mask_sync, alpha_dest_precomputed - alpha_bilin, 1, 16);
                            if (own_index_in_warp % ELEMENTS_PER_PIXEL == index_in_list)
                                alpha_dest = (alpha_dest_precomputed >= ALPHA_DEST_CUTOFF)
                                                 ? (((alpha_dest_precomputed - alpha_bilin) > 0.f)
                                                        ? 1.f
                                                        : __saturatef(alpha_dest) / (alpha_bilin + 1e-9))
                                                 : 0.f;
                        }

                        // color *= alpha_dest;
                        for (int i = 0; i < NUM_DESCRIPTORS; ++i)
                        {
                            color_out[i] = alpha_dest * alpha_bilin * d_texture.in_texture(i, texture_index);
                        }
                    }
                    else
#endif
                    {
                        //  color = make_float4(color0, color1, color2, color3) * alpha_bilin;

                        // normal blend
                        float alpha_dest_precomputed = 1.f;
                        alpha_dest                   = alpha_dest_precomputed;
                        for (int index_in_list = 1; index_in_list < ELEMENTS_PER_PIXEL; ++index_in_list)
                        {
                            alpha_dest_precomputed =
                                __shfl_up_sync(mask_sync, alpha_dest_precomputed * (1 - alpha_bilin), 1, 16);
                            if (own_index_in_warp % ELEMENTS_PER_PIXEL == index_in_list)
                                alpha_dest =
                                    (alpha_dest_precomputed >= ALPHA_DEST_CUTOFF) ? alpha_dest_precomputed : 0.f;
                        }
                        for (int i = 0; i < NUM_DESCRIPTORS; ++i)
                        {
                            color_out[i] = alpha_dest * alpha_bilin * d_texture.in_texture(i, texture_index);
                        }
                        // color *= alpha_dest;
                    }
#if 1
                    if (train)
                    {
                        if (alpha_dest >= ALPHA_DEST_CUTOFF)
                        {
                            // write out intermediates for backwards
                            // texture id, point id, alpha_val, vec2 subpixel_pos, int blendindex, point_id
                            int index_in_list = own_index_in_warp % ELEMENTS_PER_PIXEL;
                            d_bilinear_alpha_params.bw_sorted_maxed[layer](batch, gy, gx, index_in_list, 0) =
                                reinterpret_cast<const float*>(&texture_index)[0];
                            d_bilinear_alpha_params.bw_sorted_maxed[layer](batch, gy, gx, index_in_list, 1) =
                                full_list_buffer_data(0, fetch_idx, 4);
                            d_bilinear_alpha_params.bw_sorted_maxed[layer](batch, gy, gx, index_in_list, 2) =
                                alpha_bilin;
                            d_bilinear_alpha_params.bw_sorted_maxed[layer](batch, gy, gx, index_in_list, 3) = ip.x();
                            d_bilinear_alpha_params.bw_sorted_maxed[layer](batch, gy, gx, index_in_list, 4) = ip.y();
                            int blend_index = blend_fac_index(ip, vec2(gx, gy));
                            d_bilinear_alpha_params.bw_sorted_maxed[layer](batch, gy, gx, index_in_list, 5) =
                                reinterpret_cast<const float*>(&blend_index)[0];
                            if (d_render_params.use_layer_point_size)
                                d_bilinear_alpha_params.bw_sorted_maxed[layer](batch, gy, gx, index_in_list, 6) =
                                    point_size_opt;
                        }
                    }
#endif
                }

                for (int offset = 1; offset < ELEMENTS_PER_PIXEL; offset *= 2)
                {
                    for (int i = 0; i < NUM_DESCRIPTORS; ++i)
                    {
                        // color_out[i] = alpha_dest * alpha_bilin * d_texture.in_texture(i, texture_index);
                        color_out[i] += __shfl_xor_sync(mask_sync, color_out[i], offset, 16);
                    }
                    // color.x += __shfl_xor_sync(mask_sync, color.x, offset, 16);
                    // color.y += __shfl_xor_sync(mask_sync, color.y, offset, 16);
                    // color.z += __shfl_xor_sync(mask_sync, color.z, offset, 16);
                    // color.w += __shfl_xor_sync(mask_sync, color.w, offset, 16);
                }

                // if (is_leader_thread)
                //{
                //     // alpha_dest = alpha_dest_precomputed;
                //     color_out = color;
                // }

                alpha_dest =
                    __shfl_sync(mask_sync, alpha_dest, local_leader_thread + min(length, ELEMENTS_PER_PIXEL) - 1);
            }

            if (is_leader_thread)
            {
                if (alpha_dest >= ALPHA_DEST_CUTOFF)
                {
                    //    if (!environment_map)
                    {
                        // float4 bg_col = make_float4(background_color[0], background_color[1],
                        // background_color[2],
                        //                             background_color[3]);
                        //// color_out[ci] = compute_blend(alpha_dest, 1.f, background_color[ci], color_out[ci]);
                        // color_out += bg_col * (alpha_dest * 1.f);
                        for (int i = 0; i < NUM_DESCRIPTORS; ++i)
                        {
                            color_out[i] += background_color[i] * (alpha_dest * 1.f);
                        }
                    }
                }
                if (train)
                {
                    // d_forward_params.neural_out[layer](batch, 0, gy, gx) = color_out.x;
                    // d_forward_params.neural_out[layer](batch, 1, gy, gx) = color_out.y;
                    // d_forward_params.neural_out[layer](batch, 2, gy, gx) = color_out.z;
                    // d_forward_params.neural_out[layer](batch, 3, gy, gx) = color_out.w;
                    for (int i = 0; i < NUM_DESCRIPTORS; ++i)
                    {
                        d_forward_params.neural_out[layer](batch, i, gy, gx) = color_out[i];
                    }
                }
                else
                {
                    //((float4*)&d_forward_params.neural_out[layer](batch, gy, gx, 0))[0] = color_out;
                    //// d_forward_params.neural_out[layer](batch, gy, gx, 0) = color_out.x;
                    //// d_forward_params.neural_out[layer](batch, gy, gx, 1) = color_out.y;
                    //// d_forward_params.neural_out[layer](batch, gy, gx, 2) = color_out.z;
                    //// d_forward_params.neural_out[layer](batch, gy, gx, 3) = color_out.w;
                    for (int i = 0; i < NUM_DESCRIPTORS; ++i)
                    {
                        // d_forward_params.neural_out[layer](batch, i, gy, gx) = color_out[i];
                        d_forward_params.neural_out[layer](batch, gy, gx, i) = color_out[i];
                    }
                }
            }
        }
        get_next_list();

#ifdef DEBUG_PRINT
        static int counter_l = 0;
        counter_l++;
        if (counter_l > 5)
            while (1)
                ;
#endif
    }
}



/// NEW IMPL!!!!!!!!!!!!!!!!!!!!!!!!!!!

void PointRendererCache::RenderFast16(int batch, NeuralPointCloudCuda point_cloud, bool train,
                                      torch::Tensor background_color, CUDA::CudaTimerSystem* timer_system)
{
    // run once per batch
    float* dropout = (info->train && dropout_points.defined())
                         ? dropout_points.data_ptr<float>() + dropout_points.stride(0) * batch
                         : nullptr;

    auto cam = info->images[batch];
    SAIGA_ASSERT(cam.camera_index >= 0 && cam.image_index >= 0);

    // temp storage1
    torch::Tensor temp_full_buffer = torch::empty({1, point_cloud->Size(), train ? 5 : 4},
                                                  torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
    torch::Tensor temp_depth_buffer =
        torch::empty({point_cloud->Size()}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));

    torch::Tensor temp_max_layer_buffer =
        torch::empty({point_cloud->Size()}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32));
    // temp_depth_buffer.zero_();
    // temp_max_layer_buffer.zero_();
    //  CountAndCollectTiled
    //   atomicAdd on per_pixel_list_lengths, thus length per list
    //   full_buffer_data: all collected buffer
    {
        SAIGA_OPTIONAL_TIME_MEASURE("Count Render", info->timer_system);

        const int points_per_thread_collection_pass = 8;

        int c = iDivUp(point_cloud->Size(), DEFAULT_BLOCK_SIZE_FAST_COLLECT * points_per_thread_collection_pass);
        if (info->num_layers == 1)
        {
            ::CountAndCollectTiled<1><<<c, DEFAULT_BLOCK_SIZE_FAST_COLLECT>>>(
                point_cloud, dropout, cam, batch, info->train, temp_depth_buffer.data_ptr<float>(),
                temp_max_layer_buffer.data_ptr<int>(), temp_full_buffer, temp_full_buffer.data_ptr<float>());
        }
        else if (info->num_layers == 2)
        {
            ::CountAndCollectTiled<2><<<c, DEFAULT_BLOCK_SIZE_FAST_COLLECT>>>(
                point_cloud, dropout, cam, batch, info->train, temp_depth_buffer.data_ptr<float>(),
                temp_max_layer_buffer.data_ptr<int>(), temp_full_buffer, temp_full_buffer.data_ptr<float>());
        }
        else if (info->num_layers == 3)
        {
            ::CountAndCollectTiled<3><<<c, DEFAULT_BLOCK_SIZE_FAST_COLLECT>>>(
                point_cloud, dropout, cam, batch, info->train, temp_depth_buffer.data_ptr<float>(),
                temp_max_layer_buffer.data_ptr<int>(), temp_full_buffer, temp_full_buffer.data_ptr<float>());
        }
        else if (info->num_layers == 4)
        {
            ::CountAndCollectTiled<4><<<c, DEFAULT_BLOCK_SIZE_FAST_COLLECT>>>(
                point_cloud, dropout, cam, batch, info->train, temp_depth_buffer.data_ptr<float>(),
                temp_max_layer_buffer.data_ptr<int>(), temp_full_buffer, temp_full_buffer.data_ptr<float>());
        }
        else if (info->num_layers == 5)
        {
            ::CountAndCollectTiled<5><<<c, DEFAULT_BLOCK_SIZE_FAST_COLLECT>>>(
                point_cloud, dropout, cam, batch, info->train, temp_depth_buffer.data_ptr<float>(),
                temp_max_layer_buffer.data_ptr<int>(), temp_full_buffer, temp_full_buffer.data_ptr<float>());
        }
        else if (info->num_layers == 6)
        {
            ::CountAndCollectTiled<6><<<c, DEFAULT_BLOCK_SIZE_FAST_COLLECT>>>(
                point_cloud, dropout, cam, batch, info->train, temp_depth_buffer.data_ptr<float>(),
                temp_max_layer_buffer.data_ptr<int>(), temp_full_buffer, temp_full_buffer.data_ptr<float>());
        }
        else if (info->num_layers == 7)
        {
            ::CountAndCollectTiled<7><<<c, DEFAULT_BLOCK_SIZE_FAST_COLLECT>>>(
                point_cloud, dropout, cam, batch, info->train, temp_depth_buffer.data_ptr<float>(),
                temp_max_layer_buffer.data_ptr<int>(), temp_full_buffer, temp_full_buffer.data_ptr<float>());
        }
        else if (info->num_layers == 8)
        {
            ::CountAndCollectTiled<8><<<c, DEFAULT_BLOCK_SIZE_FAST_COLLECT>>>(
                point_cloud, dropout, cam, batch, info->train, temp_depth_buffer.data_ptr<float>(),
                temp_max_layer_buffer.data_ptr<int>(), temp_full_buffer, temp_full_buffer.data_ptr<float>());
        }
        else
        {
            SAIGA_EXIT_ERROR("invalid number of layers");
        }
        CUDA_SYNC_CHECK_ERROR();
    }

    //  std::cout << "temp_max_layer_buffer: " << TensorInfo(temp_max_layer_buffer) << std::endl;
    //  std::cout << "temp_depth_buffer: " << TensorInfo(temp_depth_buffer) << std::endl;
    //  std::cout << "temp_full_buffer: " << TensorInfo(temp_full_buffer) << std::endl;

    // scan lengths buffer, thus offset buffer for full_data
    torch::Tensor layer_lengths;
    torch::Tensor full_list_buffer;

    torch::Tensor non_zero_indices_t;
    {
        std::vector<torch::Tensor> indices_more_than_X;
        int num_elements = 0;
        int num_lists    = 0;
        std::vector<int> list_lengths;
        list_lengths.push_back(0);
        {
            SAIGA_OPTIONAL_TIME_MEASURE("Scan Counts and division", info->timer_system);
            void* d_temp_storage                = NULL;
            size_t temp_storage_bytes_allocated = 0;
            torch::Tensor tmp_tensor;
            for (int i = 0; i < info->num_layers; ++i)
            {
                auto& l = layers_cuda[i];
                //     PrintTensorInfo(l.per_pixel_list_lengths);
                int num_items = l.size.x() * l.size.y();
                int* d_in     = l.per_pixel_list_lengths.data_ptr<int>() + batch * l.per_pixel_list_lengths.stride(0);
                int* d_out    = l.scanned_counts.data_ptr<int>() + batch * l.scanned_counts.stride(0);

                size_t temp_storage_bytes = 0;
                cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes, d_in, d_out, num_items);
                if (temp_storage_bytes_allocated < temp_storage_bytes)
                {
                    temp_storage_bytes_allocated = temp_storage_bytes;
                    tmp_tensor =
                        torch::empty({iDivUp((long)temp_storage_bytes, 4L)}, torch::TensorOptions(torch::kCUDA));
                    d_temp_storage = tmp_tensor.data_ptr();
                }
                cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);

                int min_value                     = 0;
                auto lengths_this_layer           = l.per_pixel_list_lengths.slice(0, batch, batch + 1).view({-1});
                auto more_or_equal_than_X_        = lengths_this_layer.ge(min_value).squeeze();
                auto more_or_equal_than_X_indices = more_or_equal_than_X_.nonzero().to(torch::kInt32) + num_lists;
                // if (!indices_more_than_X.defined())
                //     indices_more_than_X = more_or_equal_than_X_indices;
                // else
                //     indices_more_than_X =
                //         torch::cat({indices_more_than_X, more_or_equal_than_X_indices}, 0);
                indices_more_than_X.push_back(more_or_equal_than_X_indices);
                num_lists += l.size.x() * l.size.y();

                int max_scanned = l.scanned_counts.slice(0, batch, batch + 1)
                                      .slice(1, l.size.y() - 1, l.size.y())
                                      .slice(2, l.size.x() - 1, l.size.x())
                                      .item<int>();
                int last_elem = l.per_pixel_list_lengths.slice(0, batch, batch + 1)
                                    .slice(1, l.size.y() - 1, l.size.y())
                                    .slice(2, l.size.x() - 1, l.size.x())
                                    .item<int>();
                num_elements += max_scanned + last_elem;

                int prev_len = list_lengths.back();
                list_lengths.push_back(max_scanned + last_elem + prev_len);
            }
        }
        {
            SAIGA_OPTIONAL_TIME_MEASURE("Alloc", info->timer_system);
            layer_lengths = torch::from_blob(list_lengths.data(), {1, info->num_layers},
                                             torch::TensorOptions().dtype(torch::kInt32))
                                .contiguous()
                                .clone()
                                .cuda();
            for (int i = 0; i < info->num_layers; ++i)
            {
                auto& l = layers_cuda[i];
                l.per_pixel_list_lengths.zero_();
            }
            full_list_buffer =
                torch::empty({1, num_elements, 1}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat64));

            non_zero_indices_t = torch::cat({indices_more_than_X}, 0);
        }
    }

    // SplattingPass
    //  read z, layer, ip
    // splat to relevant points in buffer
    {
        SAIGA_OPTIONAL_TIME_MEASURE("Splat", timer_system);

        // static int c = 256;

        static int BS = DEFAULT_BLOCK_SIZE_FAST_COLLECT;
        //  ImGui::SliderInt("BS", &BS, 32, 1024);
        // static int c = 256;

        static int ppt = 1;
        // ImGui::SliderInt("ppt", &ppt, 1, 128);

        const int points_per_thread_splatting_pass = 1;
        torch::Tensor atomic1 = torch::zeros({1}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32));

        int c = iDivUp(point_cloud->Size() * 4, BS * ppt);

        // ImGui::InputInt("c", &c);

        SplattingPass<<<c, BS>>>(batch, full_list_buffer, full_list_buffer.data_ptr<double>(), temp_full_buffer,
                                 temp_depth_buffer.data_ptr<float>(), temp_max_layer_buffer.data_ptr<int>(),
                                 layer_lengths);
        //, atomic1);
        //  ,
        //  non_zero_indices_t, atomic1);
        CUDA_SYNC_CHECK_ERROR();
    }
    // std::cout << "full_list_buffer: " << TensorInfo(full_list_buffer) << std::endl;

    // sortaccum

    {
        static int bx = 128;  // iDivUp(layers_cuda[1].size.x(), 16);
        static int by = 32;   // iDivUp(layers_cuda[1].size.y(), 16);
        SAIGA_OPTIONAL_TIME_MEASURE("FastSortAndBlendMore", timer_system);

        float* background = background_color.data_ptr<float>();

        torch::Tensor atomic2 = torch::zeros({1}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32));

        //  std::cout << "non_zero_t: " << TensorInfo(non_zero_indices_t) << std::endl;

        if (train)
        {
            switch (info->params.num_texture_channels)
            {
                case 3:
                {
                    FastSortFusedNew<3, 16, true><<<dim3(bx, by, 1), dim3(THREAD_DIMX_PER_BLOCK_MULTITHREADLOAD2,
                                                                          THREAD_DIMY_PER_BLOCK_MULTITHREADLOAD2, 1)>>>(
                        batch, full_list_buffer, full_list_buffer.data_ptr<double>(), temp_full_buffer,
                        temp_full_buffer.data_ptr<float>(), background, atomic2, layer_lengths, non_zero_indices_t,
                        info->num_layers);
                    break;
                }
                case 4:
                {
                    FastSortFusedNew<4, 16, true><<<dim3(bx, by, 1), dim3(THREAD_DIMX_PER_BLOCK_MULTITHREADLOAD2,
                                                                          THREAD_DIMY_PER_BLOCK_MULTITHREADLOAD2, 1)>>>(
                        batch, full_list_buffer, full_list_buffer.data_ptr<double>(), temp_full_buffer,
                        temp_full_buffer.data_ptr<float>(), background, atomic2, layer_lengths, non_zero_indices_t,
                        info->num_layers);
                    break;
                }
                case 6:
                {
                    FastSortFusedNew<6, 16, true><<<dim3(bx, by, 1), dim3(THREAD_DIMX_PER_BLOCK_MULTITHREADLOAD2,
                                                                          THREAD_DIMY_PER_BLOCK_MULTITHREADLOAD2, 1)>>>(
                        batch, full_list_buffer, full_list_buffer.data_ptr<double>(), temp_full_buffer,
                        temp_full_buffer.data_ptr<float>(), background, atomic2, layer_lengths, non_zero_indices_t,
                        info->num_layers);
                    break;
                }
                case 8:
                {
                    FastSortFusedNew<8, 16, true><<<dim3(bx, by, 1), dim3(THREAD_DIMX_PER_BLOCK_MULTITHREADLOAD2,
                                                                          THREAD_DIMY_PER_BLOCK_MULTITHREADLOAD2, 1)>>>(
                        batch, full_list_buffer, full_list_buffer.data_ptr<double>(), temp_full_buffer,
                        temp_full_buffer.data_ptr<float>(), background, atomic2, layer_lengths, non_zero_indices_t,
                        info->num_layers);
                    break;
                }
                case 16:
                {
                    FastSortFusedNew<16, 16, true>
                        <<<dim3(bx, by, 1),
                           dim3(THREAD_DIMX_PER_BLOCK_MULTITHREADLOAD2, THREAD_DIMY_PER_BLOCK_MULTITHREADLOAD2, 1)>>>(
                            batch, full_list_buffer, full_list_buffer.data_ptr<double>(), temp_full_buffer,
                            temp_full_buffer.data_ptr<float>(), background, atomic2, layer_lengths, non_zero_indices_t,
                            info->num_layers);
                    break;
                }
                default:
                    SAIGA_ASSERT(false, "NOT IMPLEMENTED");
            };
        }
        else
        {
            switch (info->params.num_texture_channels)
            {
                case 3:
                {
                    FastSortFusedNew<3, 16, false>
                        <<<dim3(bx, by, 1),
                           dim3(THREAD_DIMX_PER_BLOCK_MULTITHREADLOAD2, THREAD_DIMY_PER_BLOCK_MULTITHREADLOAD2, 1)>>>(
                            batch, full_list_buffer, full_list_buffer.data_ptr<double>(), temp_full_buffer,
                            temp_full_buffer.data_ptr<float>(), background, atomic2, layer_lengths, non_zero_indices_t,
                            info->num_layers);
                    break;
                }
                case 4:
                {
                    FastSortFusedNew<4, 16, false>
                        <<<dim3(bx, by, 1),
                           dim3(THREAD_DIMX_PER_BLOCK_MULTITHREADLOAD2, THREAD_DIMY_PER_BLOCK_MULTITHREADLOAD2, 1)>>>(
                            batch, full_list_buffer, full_list_buffer.data_ptr<double>(), temp_full_buffer,
                            temp_full_buffer.data_ptr<float>(), background, atomic2, layer_lengths, non_zero_indices_t,
                            info->num_layers);
                    break;
                }
                case 6:
                {
                    FastSortFusedNew<6, 16, false>
                        <<<dim3(bx, by, 1),
                           dim3(THREAD_DIMX_PER_BLOCK_MULTITHREADLOAD2, THREAD_DIMY_PER_BLOCK_MULTITHREADLOAD2, 1)>>>(
                            batch, full_list_buffer, full_list_buffer.data_ptr<double>(), temp_full_buffer,
                            temp_full_buffer.data_ptr<float>(), background, atomic2, layer_lengths, non_zero_indices_t,
                            info->num_layers);
                    break;
                }
                case 8:
                {
                    FastSortFusedNew<8, 16, false>
                        <<<dim3(bx, by, 1),
                           dim3(THREAD_DIMX_PER_BLOCK_MULTITHREADLOAD2, THREAD_DIMY_PER_BLOCK_MULTITHREADLOAD2, 1)>>>(
                            batch, full_list_buffer, full_list_buffer.data_ptr<double>(), temp_full_buffer,
                            temp_full_buffer.data_ptr<float>(), background, atomic2, layer_lengths, non_zero_indices_t,
                            info->num_layers);
                    break;
                }
                case 16:
                {
                    FastSortFusedNew<16, 16, false>
                        <<<dim3(bx, by, 1),
                           dim3(THREAD_DIMX_PER_BLOCK_MULTITHREADLOAD2, THREAD_DIMY_PER_BLOCK_MULTITHREADLOAD2, 1)>>>(
                            batch, full_list_buffer, full_list_buffer.data_ptr<double>(), temp_full_buffer,
                            temp_full_buffer.data_ptr<float>(), background, atomic2, layer_lengths, non_zero_indices_t,
                            info->num_layers);
                    break;
                }
            }
        }
        CUDA_SYNC_CHECK_ERROR();
    }


    // output: colors array, (save for backwards)
}




#define FAST_MINIMAL_IMPL

template <int num_layers>
__global__ __launch_bounds__(DEFAULT_BLOCK_SIZE_FAST_COLLECT) void CountTiled(
    __grid_constant__ const DevicePointCloud point_cloud, float* dropout_p,
    __grid_constant__ const ReducedImageInfo cam, int batch, bool train)
{
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();

    const int threadsPerBlock = blockDim.x * blockDim.y;
    const int threadNumInBlock =
        threadIdx.x + blockDim.x * threadIdx.y;  // (alternatively: threadIdx.y + blockDim.y * threadIdx.x)
    const int blockNumInGrid =
        blockIdx.x + gridDim.x * blockIdx.y;  //  (alternatively: blockIdx.y  + gridDim.y  * blockIdx.x)

    // Load all camera params to Shared Memory
    __shared__ Sophus::SE3f V;
    __shared__ IntrinsicsPinholef K;
    __shared__ Distortionf distortion;
    __shared__ Vector<float, 5> ocam_aff;
    __shared__ ArrayView<const float> ocam_poly;
    if (threadNumInBlock < 32)
    {
        V = d_render_params.Pose(cam.image_index);
    }
    else if (threadNumInBlock < 64)
    {
        if (cam.camera_model_type == CameraModel::PINHOLE_DISTORTION)
        {
            auto [K_l, distortion_l] = d_render_params.PinholeIntrinsics(cam.camera_index);
            K                        = K_l;
            distortion               = distortion_l;
        }
        else if (cam.camera_model_type == CameraModel::OCAM)
        {
            auto [aff_l, poly_l] = d_render_params.OcamIntrinsics(cam.camera_index);
            ocam_aff             = aff_l;
            ocam_poly            = poly_l;
        }
    }
    __syncthreads();

    for (int point_id = grid.thread_rank(); point_id < point_cloud.Size(); point_id += grid.size())
    {
        // if (point_id >= point_cloud.Size()) continue;

#ifndef FAST_MINIMAL_IMPL
        if (d_render_params.viewer_only)
        {
            int conf_id = point_cloud.GetIndex(point_id);
            if (discard_point_for_confidence(conf_id)) continue;
        }
        else
        {
            if (train && dropout_p)
            {
                bool drop_out = dropout_p[point_id] == 1;
                if (drop_out) continue;
            }
        }
#endif
        if (train && dropout_p)
        {
            bool drop_out = dropout_p[point_id] == 1;
            if (drop_out) continue;
        }

        int texture_index    = point_cloud.GetIndex(point_id);
        float point_size_opt = 1.f;
        if (d_render_params.use_layer_point_size) point_size_opt = _softplus(point_cloud.GetPointSize(point_id));
        // if (layer_buf < 0.0001) continue;
        vec2 ip;
        float z;
        float radius_pixels;

        {
            vec3 position;

            vec2 image_p_a;
            float drop_out_radius;
            thrust::tie(position, drop_out_radius) = point_cloud.GetPointWoNormal(point_id);

            if (cam.camera_model_type == CameraModel::PINHOLE_DISTORTION)
            {
                CUDA_KERNEL_ASSERT(cam.camera_model_type == CameraModel::PINHOLE_DISTORTION);
                //  auto [K, distortion] = d_render_params.PinholeIntrinsics(cam.camera_index);
                thrust::tie(image_p_a, z) =
                    ProjectPointPinholeWoNormal(position, V, K, distortion, d_render_params.dist_cutoff);
                radius_pixels = K.fx * cam.crop_transform.fx * drop_out_radius / z;

                point_size_opt = K.fx * cam.crop_transform.fx * point_size_opt / z;
            }
            else if (cam.camera_model_type == CameraModel::OCAM)
            {
                //  auto [aff, poly] = d_render_params.OcamIntrinsics(cam.camera_index);
                thrust::tie(image_p_a, z) = ProjectPointOcamWoNormal(position, V, ocam_aff, ocam_poly, 0.15);
                radius_pixels = d_forward_params.neural_out[0].Image().w * cam.crop_transform.fx * drop_out_radius / z;
                // if (d_render_params.use_layer_point_size) CUDA_KERNEL_ASSERT(false);
                point_size_opt = d_forward_params.neural_out[0].Image().w * cam.crop_transform.fx * point_size_opt / z;
            }
            else if (cam.camera_model_type == CameraModel::SPHERICAL)
            {
                thrust::tie(image_p_a, z) = ProjectPointSphericalWoNormal(
                    position, V,
                    vec2(d_forward_params.neural_out[0].Image().w, d_forward_params.neural_out[0].Image().h),
                    d_render_params.dist_cutoff);
                radius_pixels = d_forward_params.neural_out[0].Image().w * cam.crop_transform.fx * drop_out_radius / z;
                ip            = image_p_a;
            }
            else if (cam.camera_model_type == CameraModel::ORTHO)
            {
                thrust::tie(image_p_a, z) =
                    ProjectPointToOrthographic(position, V, d_forward_params.neural_out[0].Image().w,
                                               d_forward_params.neural_out[0].Image().h, d_render_params.dist_cutoff);
                radius_pixels = d_forward_params.neural_out[0].Image().w * cam.crop_transform.fx * drop_out_radius / z;
            }
            else
            {
                CUDA_KERNEL_ASSERT(cam.camera_model_type == CameraModel::PINHOLE_DISTORTION);
            }
            // if (z <= 0) continue;
            ip = cam.crop_transform.normalizedToImage(image_p_a);
            ip = rotateCropAroundCenter(ip, vec2(cam.w / 2, cam.h / 2), cam);

            // z discard
            // border pixel will be ignored
            // radius discard
            if (z <= 0 || ip(0) < 0 || ip(0) >= d_bilinear_alpha_params.per_pixel_list_lengths[0].size(2) - 1 ||
                ip(1) < 0 || ip(1) >= d_bilinear_alpha_params.per_pixel_list_lengths[0].size(1) - 1 ||
                (d_render_params.drop_out_points_by_radius &&
                 radius_pixels < d_render_params.drop_out_radius_threshold))
                continue;

            // if (d_render_params.drop_out_points_by_radius && radius_pixels <
            // d_render_params.drop_out_radius_threshold)
            //{
            //     continue;
            // }
        }



        if (d_render_params.use_layer_point_size)
        {
            // log2(<1) is neg; num_layer is max possible layers
            int layer_lower  = 0;
            int layer_higher = 0;
            if (point_size_opt > 1)
            {
                layer_lower  = min(int(floor(log2f(point_size_opt))), num_layers - 1);
                layer_higher = min(int(ceil(log2f(point_size_opt))), num_layers - 1);
            }
            if (!d_render_params.render_points_in_all_lower_resolutions)
            {
                for (int layer = 0; layer < layer_lower; ++layer)
                {
                    radius_pixels *= 0.5f;
                    ip *= 0.5f;
                }
                for (int layer = layer_lower; layer < layer_higher + 1; ++layer)
                {
                    ivec2 p_rd = ivec2(__float2int_rd(ip(0)), __float2int_rd(ip(1)));

                    // discard upper bounded pixel
                    if (p_rd(0) < 0 || p_rd(0) >= d_bilinear_alpha_params.per_pixel_list_lengths[layer].size(2) - 1 ||
                        p_rd(1) < 0 || p_rd(1) >= d_bilinear_alpha_params.per_pixel_list_lengths[layer].size(1) - 1 ||
                        (d_render_params.drop_out_points_by_radius &&
                         radius_pixels < d_render_params.drop_out_radius_threshold))
                        break;
#pragma unroll
                    for (int i = 0; i < 4; ++i)
                    {
                        // count list up
                        atomicAdd(&d_bilinear_alpha_params.per_pixel_list_lengths[layer](batch, p_rd.y() + i / 2,
                                                                                         p_rd.x() + i % 2),
                                  1);
                    }

                    radius_pixels *= 0.5f;
                    ip *= 0.5f;
                }
            }
            else
            {
                for (int layer = 0; layer < layer_higher + 1; ++layer)
                {
                    ivec2 p_rd = ivec2(__float2int_rd(ip(0)), __float2int_rd(ip(1)));

                    // discard upper bounded pixel
                    if (p_rd(0) < 0 || p_rd(0) >= d_bilinear_alpha_params.per_pixel_list_lengths[layer].size(2) - 1 ||
                        p_rd(1) < 0 || p_rd(1) >= d_bilinear_alpha_params.per_pixel_list_lengths[layer].size(1) - 1 ||
                        (d_render_params.drop_out_points_by_radius &&
                         radius_pixels < d_render_params.drop_out_radius_threshold))
                        break;
#pragma unroll
                    for (int i = 0; i < 4; ++i)
                    {
                        // count list up
                        atomicAdd(&d_bilinear_alpha_params.per_pixel_list_lengths[layer](batch, p_rd.y() + i / 2,
                                                                                         p_rd.x() + i % 2),
                                  1);
                    }

                    radius_pixels *= 0.5f;
                    ip *= 0.5f;
                }
            }
        }
        else
        {
#pragma unroll
            for (int layer = 0; layer < num_layers; ++layer, radius_pixels *= 0.5f, ip *= 0.5f)
            {
                ivec2 p_rd = ivec2(__float2int_rd(ip(0)), __float2int_rd(ip(1)));

                // discard upper bounded pixel
                if (p_rd(0) < 0 || p_rd(0) >= d_bilinear_alpha_params.per_pixel_list_lengths[layer].size(2) - 1 ||
                    p_rd(1) < 0 || p_rd(1) >= d_bilinear_alpha_params.per_pixel_list_lengths[layer].size(1) - 1 ||
                    (d_render_params.drop_out_points_by_radius &&
                     radius_pixels < d_render_params.drop_out_radius_threshold))
                    break;
#pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    // count list up
                    atomicAdd(&d_bilinear_alpha_params.per_pixel_list_lengths[layer](batch, p_rd.y() + i / 2,
                                                                                     p_rd.x() + i % 2),
                              1);
                }
            }
        }

    }
}


void PointRendererCache::CountTiled(int batch, NeuralPointCloudCuda point_cloud, bool train)
{
    SAIGA_ASSERT(point_cloud);

    static int points_per_thread_collection_pass = 8;

    static int num_per_t = 1024;
    // ImGui::Begin("test");

    // ImGui::SliderInt("points_per_thread_collection_pass", &points_per_thread_collection_pass, 1, 256);

    // ImGui::End();
    {
        int image_batch_id = batch;

        float* dropout = (info->train && dropout_points.defined())
                             ? dropout_points.data_ptr<float>() + dropout_points.stride(0) * image_batch_id
                             : nullptr;

        auto cam = info->images[image_batch_id];
        SAIGA_ASSERT(cam.camera_index >= 0 && cam.image_index >= 0);

        int c = iDivUp(point_cloud->Size(), DEFAULT_BLOCK_SIZE_FAST_COLLECT * points_per_thread_collection_pass);



        if (info->num_layers == 1)
        {
            ::CountTiled<1><<<c, DEFAULT_BLOCK_SIZE_FAST_COLLECT>>>(point_cloud, dropout, cam, batch, info->train);
        }
        else if (info->num_layers == 2)
        {
            ::CountTiled<2><<<c, DEFAULT_BLOCK_SIZE_FAST_COLLECT>>>(point_cloud, dropout, cam, batch, info->train);
        }
        else if (info->num_layers == 3)
        {
            ::CountTiled<3><<<c, DEFAULT_BLOCK_SIZE_FAST_COLLECT>>>(point_cloud, dropout, cam, batch, info->train);
        }
        else if (info->num_layers == 4)
        {
            // cudaLaunchCooperativeKernel((void*)(::CollectTiled<4>), c, DEFAULT_BLOCK_SIZE_FAST_COLLECT,
            // kernelArgs);
            ::CountTiled<4><<<c, DEFAULT_BLOCK_SIZE_FAST_COLLECT>>>(point_cloud, dropout, cam, batch, info->train);
        }
        else if (info->num_layers == 5)
        {
            ::CountTiled<5><<<c, DEFAULT_BLOCK_SIZE_FAST_COLLECT>>>(point_cloud, dropout, cam, batch, info->train);
        }
        else if (info->num_layers == 6)
        {
            ::CountTiled<6><<<c, DEFAULT_BLOCK_SIZE_FAST_COLLECT>>>(point_cloud, dropout, cam, batch, info->train);
        }
        else if (info->num_layers == 7)
        {
            ::CountTiled<7><<<c, DEFAULT_BLOCK_SIZE_FAST_COLLECT>>>(point_cloud, dropout, cam, batch, info->train);
        }
        else if (info->num_layers == 8)
        {
            ::CountTiled<8><<<c, DEFAULT_BLOCK_SIZE_FAST_COLLECT>>>(point_cloud, dropout, cam, batch, info->train);
        }
        else
        {
            SAIGA_EXIT_ERROR("invalid number of layers");
        }
    }
    CUDA_SYNC_CHECK_ERROR();
}



///collect

template <int num_layers>
__global__ __launch_bounds__(DEFAULT_BLOCK_SIZE_FAST_COLLECT) void CollectTiled2(
    DevicePointCloud point_cloud, float* dropout_p, ReducedImageInfo cam, int batch,
    StaticDeviceTensor<double, 3> full_list_buffer /* (batch,numelems,1)*/,
    StaticDeviceTensor<float, 3> full_list_buffer_data /*(batch,numelems,5)*/,
    StaticDeviceTensor<int32_t, 2> layer_lengths /*(batch,num_layers)*/, bool train)
{
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();


    // Load all camera params to Shared Memory
    __shared__ Sophus::SE3f V;
    __shared__ IntrinsicsPinholef K;
    __shared__ Distortionf distortion;
    __shared__ Vector<float, 5> ocam_aff;
    __shared__ ArrayView<const float> ocam_poly;
    {
        const int threadNumInBlock = threadIdx.x + blockDim.x * threadIdx.y;
        if (threadNumInBlock < 32)
        {
            V = d_render_params.Pose(cam.image_index);
        }
        else if (threadNumInBlock < 64)
        {
            if (cam.camera_model_type == CameraModel::PINHOLE_DISTORTION)
            {
                auto [K_l, distortion_l] = d_render_params.PinholeIntrinsics(cam.camera_index);
                K                        = K_l;
                distortion               = distortion_l;
            }
            else if (cam.camera_model_type == CameraModel::OCAM)
            {
                auto [aff_l, poly_l] = d_render_params.OcamIntrinsics(cam.camera_index);
                ocam_aff             = aff_l;
                ocam_poly            = poly_l;
            }
        }
    }
    __syncthreads();

    for (int point_id = grid.thread_rank(); point_id < point_cloud.Size(); point_id += grid.size())
    {
        // if (point_id >= point_cloud.Size()) continue;

#ifndef FAST_MINIMAL_IMPL
        if (d_render_params.viewer_only)
        {
            int conf_id = point_cloud.GetIndex(point_id);
            if (discard_point_for_confidence(conf_id)) continue;
        }
        else
        {
            if (train && dropout_p)
            {
                bool drop_out = dropout_p[point_id] == 1;
                if (drop_out) continue;
            }
        }
#endif
        if (train && dropout_p)
        {
            bool drop_out = dropout_p[point_id] == 1;
            if (drop_out) continue;
        }

        float point_size_opt = 1.f;
        if (d_render_params.use_layer_point_size) point_size_opt = _softplus(point_cloud.GetPointSize(point_id));
        // if (layer_buf < 0.0001) continue;
        vec2 ip;
        float z;
        float radius_pixels;

        {
            vec3 position;

            vec2 image_p_a;
            float drop_out_radius;
            thrust::tie(position, drop_out_radius) = point_cloud.GetPointWoNormal(point_id);

            if (cam.camera_model_type == CameraModel::PINHOLE_DISTORTION)
            {
                CUDA_KERNEL_ASSERT(cam.camera_model_type == CameraModel::PINHOLE_DISTORTION);
                //  auto [K, distortion] = d_render_params.PinholeIntrinsics(cam.camera_index);
                thrust::tie(image_p_a, z) =
                    ProjectPointPinholeWoNormal(position, V, K, distortion, d_render_params.dist_cutoff);
                radius_pixels = K.fx * cam.crop_transform.fx * drop_out_radius / z;

                if (d_render_params.use_layer_point_size)
                    point_size_opt = K.fx * cam.crop_transform.fx * point_size_opt / z;
            }
            else if (cam.camera_model_type == CameraModel::OCAM)
            {
                //  auto [aff, poly] = d_render_params.OcamIntrinsics(cam.camera_index);
                thrust::tie(image_p_a, z) = ProjectPointOcamWoNormal(position, V, ocam_aff, ocam_poly, 0.15);
                radius_pixels = d_forward_params.neural_out[0].Image().w * cam.crop_transform.fx * drop_out_radius / z;
                if (d_render_params.use_layer_point_size) CUDA_KERNEL_ASSERT(false);
            }
            else if (cam.camera_model_type == CameraModel::SPHERICAL)
            {
                thrust::tie(image_p_a, z) = ProjectPointSphericalWoNormal(
                    position, V,
                    vec2(d_forward_params.neural_out[0].Image().w, d_forward_params.neural_out[0].Image().h),
                    d_render_params.dist_cutoff);
                radius_pixels = d_forward_params.neural_out[0].Image().w * cam.crop_transform.fx * drop_out_radius / z;
                ip            = image_p_a;
            }
            else if (cam.camera_model_type == CameraModel::ORTHO)
            {
                thrust::tie(image_p_a, z) =
                    ProjectPointToOrthographic(position, V, d_forward_params.neural_out[0].Image().w,
                                               d_forward_params.neural_out[0].Image().h, d_render_params.dist_cutoff);
                radius_pixels = d_forward_params.neural_out[0].Image().w * cam.crop_transform.fx * drop_out_radius / z;
            }
            else
            {
                CUDA_KERNEL_ASSERT(cam.camera_model_type == CameraModel::PINHOLE_DISTORTION);
            }
            // if (z <= 0) continue;
            ip = cam.crop_transform.normalizedToImage(image_p_a);
            ip = rotateCropAroundCenter(ip, vec2(cam.w / 2, cam.h / 2), cam);

            // z discard
            // border pixel will be ignored
            // radius discard
            if (z <= 0 || ip(0) < 0 || ip(0) >= cam.w - 1 || ip(1) < 0 || ip(1) >= cam.h - 1 ||
                (d_render_params.drop_out_points_by_radius &&
                 radius_pixels < d_render_params.drop_out_radius_threshold))
                continue;

            // if (d_render_params.drop_out_points_by_radius && radius_pixels <
            // d_render_params.drop_out_radius_threshold)
            //{
            //     continue;
            // }
        }
        vec2 ip_org      = ip;
        int layer_offset = 0;
        if (d_render_params.use_layer_point_size)
        {
            // log2(<1) is neg; num_layer is max possible layers
            int layer_lower  = 0;
            int layer_higher = 0;
            if (point_size_opt > 1)
            {
                layer_lower  = min(int(floor(log2f(point_size_opt))), num_layers - 1);
                layer_higher = min(int(ceil(log2f(point_size_opt))), num_layers - 1);
            }
            // get layer fac: (point_size_opt - layer_lower) -  (layer_higher - layer_lower)
            // done in accum

            for (int layer = 0; layer < layer_lower; ++layer)
            {
                layer_offset += layer_lengths(batch, layer);
                radius_pixels *= 0.5f;
                ip *= 0.5f;
            }
            __syncwarp();

            bool data_written   = false;
            int data_written_to = 0;
            for (int layer = layer_lower; layer < layer_higher + 1; ++layer)
            {
                ivec2 p_rd = ivec2(__float2int_rd(ip(0)), __float2int_rd(ip(1)));

                // discard upper bounded pixel
                if (p_rd(0) < 0 || p_rd(0) >= d_bilinear_alpha_params.per_pixel_list_lengths[layer].size(2) - 1 ||
                    p_rd(1) < 0 || p_rd(1) >= d_bilinear_alpha_params.per_pixel_list_lengths[layer].size(1) - 1 ||
                    (d_render_params.drop_out_points_by_radius &&
                     radius_pixels < d_render_params.drop_out_radius_threshold))
                    break;



#pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    ivec2 splat_point = ivec2(p_rd.x() + i % 2, p_rd.y() + i / 2);

                    int offset = atomicAdd(
                        &d_bilinear_alpha_params.per_pixel_list_lengths[layer](batch, splat_point.y(), splat_point.x()),
                        1);
                    int scanned_c =
                        d_bilinear_alpha_params.scanned_countings[layer](batch, splat_point.y(), splat_point.x()) +
                        offset + layer_offset;

                    // write data to buffers
                    if (!data_written)
                    {
                        data_written_to = scanned_c;
                        // write point data
                        // float4 data_buf = make_float4(ip.x(), ip.y(),
                        // reinterpret_cast<float*>(&texture_index)[0],
                        //                              reinterpret_cast<float*>(&point_id)[0]);
                        //
                        //((float4*)&full_list_buffer_data(batch, data_ll_index, 0))[0] = data_buf;
                        int texture_index = point_cloud.GetIndex(point_id);

                        full_list_buffer_data.At({batch, scanned_c, 0}) = ip_org.x();
                        full_list_buffer_data.At({batch, scanned_c, 1}) = ip_org.y();
                        full_list_buffer_data.At({batch, scanned_c, 2}) = reinterpret_cast<float*>(&texture_index)[0];
                        if (d_render_params.use_layer_point_size)
                            full_list_buffer_data(batch, scanned_c, 3) = point_size_opt;
                        if (train)
                            full_list_buffer_data.At({batch, scanned_c, 4}) = reinterpret_cast<float*>(&point_id)[0];

                        data_written = true;
                    }
                    double data_l = __hiloint2double(reinterpret_cast<int*>(&z)[0], data_written_to);
                    ((double*)&full_list_buffer.At({batch, scanned_c, 0}))[0] = data_l;
                }
                layer_offset += layer_lengths(batch, layer);
                radius_pixels *= 0.5f;
                ip *= 0.5f;
            }
        }
        else
        {
#pragma unroll
            for (int layer = 0; layer < num_layers; ++layer)
            {
                ivec2 p_rd = ivec2(__float2int_rd(ip(0)), __float2int_rd(ip(1)));

                // discard upper bounded pixel
                if (p_rd(0) < 0 || p_rd(0) >= d_bilinear_alpha_params.per_pixel_list_lengths[layer].size(2) - 1 ||
                    p_rd(1) < 0 || p_rd(1) >= d_bilinear_alpha_params.per_pixel_list_lengths[layer].size(1) - 1 ||
                    (d_render_params.drop_out_points_by_radius &&
                     radius_pixels < d_render_params.drop_out_radius_threshold))
                    break;
#pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    ivec2 splat_point = ivec2(p_rd.x() + i % 2, p_rd.y() + i / 2);

                    int offset = atomicAdd(
                        &d_bilinear_alpha_params.per_pixel_list_lengths[layer](batch, splat_point.y(), splat_point.x()),
                        1);
                    int scanned_c =
                        d_bilinear_alpha_params.scanned_countings[layer](batch, splat_point.y(), splat_point.x()) +
                        offset + layer_offset;

                    // write data to buffers
                    {
                        // write point data
                        // float4 data_buf = make_float4(ip.x(), ip.y(),
                        // reinterpret_cast<float*>(&texture_index)[0],
                        //                              reinterpret_cast<float*>(&point_id)[0]);
                        //
                        //((float4*)&full_list_buffer_data(batch, data_ll_index, 0))[0] = data_buf;
                        int texture_index = point_cloud.GetIndex(point_id);

                        full_list_buffer_data.At({batch, scanned_c, 0}) = ip_org.x();
                        full_list_buffer_data.At({batch, scanned_c, 1}) = ip_org.y();
                        full_list_buffer_data.At({batch, scanned_c, 2}) = reinterpret_cast<float*>(&texture_index)[0];
                        if (d_render_params.use_layer_point_size)
                            full_list_buffer_data(batch, scanned_c, 3) = point_size_opt;
                        if (train)
                            full_list_buffer_data.At({batch, scanned_c, 4}) = reinterpret_cast<float*>(&point_id)[0];


                        double data_l = __hiloint2double(reinterpret_cast<int*>(&z)[0], scanned_c);
                        ((double*)&full_list_buffer.At({batch, scanned_c, 0}))[0] = data_l;
                    }
                }
                layer_offset += layer_lengths(batch, layer);
                radius_pixels *= 0.5f;
                ip *= 0.5f;
            }
            /*

            // if (d_render_params.use_layer_point_size)
            {
                // log2(<1) is neg; num_layer is max possible layers
                int layer_lower  = 0;
                int layer_higher = num_layers - 1;
                if (d_render_params.use_layer_point_size)
                {
                    if (point_size_opt > 1)
                    {
                        layer_lower  = min(int(floor(log2f(point_size_opt))), num_layers - 1);
                        layer_higher = min(int(ceil(log2f(point_size_opt))), num_layers - 1);
                    }
                    else
                        layer_higher = 0;
                }


                // get layer fac: (point_size_opt - layer_lower) -  (layer_higher - layer_lower)
                // done in accum

                int layer_offset         = 0;
                auto add_to_layer_offset = [&layer_offset, &batch, &layer_lengths](int layer)
                { layer_offset += layer_lengths(batch, layer); };

                for (int layer = 0; layer < layer_lower;
                     add_to_layer_offset(layer), ++layer, radius_pixels *= 0.5f, ip *= 0.5f)
                {
                }
                for (int layer = layer_lower; layer < layer_higher + 1;
                     add_to_layer_offset(layer), ++layer, radius_pixels *= 0.5f, ip *= 0.5f)
                {
                    ivec2 p_rd = ivec2(__float2int_rd(ip(0)), __float2int_rd(ip(1)));

                    // discard upper bounded pixel
                    if (p_rd(0) < 0 || p_rd(0) >= d_bilinear_alpha_params.per_pixel_list_heads[layer].size(2) - 1 ||
                        p_rd(1) < 0 || p_rd(1) >= d_bilinear_alpha_params.per_pixel_list_heads[layer].size(1) - 1 ||
                        (d_render_params.drop_out_points_by_radius &&
                         radius_pixels < d_render_params.drop_out_radius_threshold))
                        break;


                    for (int i = 0; i < 4; ++i)
                    {
                        ivec2 splat_point = ivec2(p_rd.x() + i % 2, p_rd.y() + i / 2);

                        int offset = atomicAdd(
                            &d_bilinear_alpha_params.per_pixel_list_lengths[layer](batch, splat_point.y(),
            splat_point.x()), 1); int scanned_c = d_bilinear_alpha_params.scanned_countings[layer](batch,
            splat_point.y(), splat_point.x()) + offset + layer_offset;

                        // write data to buffers
                        {
                            // write point data
                            // float4 data_buf = make_float4(ip.x(), ip.y(),
                            // reinterpret_cast<float*>(&texture_index)[0],
                            //                              reinterpret_cast<float*>(&point_id)[0]);
                            //
                            //((float4*)&full_list_buffer_data(batch, data_ll_index, 0))[0] = data_buf;
                            full_list_buffer_data.At({batch, scanned_c, 0}) = ip.x();
                            full_list_buffer_data.At({batch, scanned_c, 1}) = ip.y();
                            full_list_buffer_data.At({batch, scanned_c, 2}) =
            reinterpret_cast<float*>(&texture_index)[0]; full_list_buffer_data.At({batch, scanned_c, 3}) =
            reinterpret_cast<float*>(&point_id)[0]; if (d_render_params.use_layer_point_size)
                                full_list_buffer_data(batch, scanned_c, 4) = point_size_opt;


                            double data_l = __hiloint2double(reinterpret_cast<int*>(&z)[0], scanned_c);
                            ((double*)&full_list_buffer.At({batch, scanned_c, 0}))[0] = data_l;
                        }
                    }
                }
            }*/
        }
    }
}


template <int num_layers, int points_per_thread>
__global__ __launch_bounds__(DEFAULT_BLOCK_SIZE_FAST_COLLECT) void CollectTiled2Pointsize(
    __grid_constant__ const DevicePointCloud point_cloud, float* dropout_p,
    __grid_constant__ const ReducedImageInfo cam, int batch,
    ConstReadOnlyStaticDeviceTensor<double, 3> full_list_buffer /* (batch,numelems,1)*/,
    double* __restrict__ full_list_buffer_ptr,
    ConstReadOnlyStaticDeviceTensor<float, 3> full_list_buffer_data /*(batch,numelems,5)*/,
    float* __restrict__ full_list_buffer_data_ptr,
    StaticDeviceTensor<int32_t, 2, int> layer_lengths /*(batch,num_layers)*/,
    StaticDeviceTensor<int32_t, 1> atomic_c, /*(1)*/
    bool train)
{
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();

#    ifdef SM
    // Load all camera params to Shared Memory
    __shared__ Sophus::SE3f V;
    __shared__ IntrinsicsPinholef K;
    __shared__ Distortionf distortion;
    __shared__ Vector<float, 5> ocam_aff;
    __shared__ ArrayView<const float> ocam_poly;
    __shared__ int layer_lengths_sm[num_layers];
    const int threadNumInBlock = threadIdx.x + blockDim.x * threadIdx.y;
    {
        if (threadNumInBlock < 32)
        {
            V = d_render_params.Pose(cam.image_index);
        }
        else if (threadNumInBlock < 64)
        {
            if (cam.camera_model_type == CameraModel::PINHOLE_DISTORTION)
            {
                auto [K_l, distortion_l] = d_render_params.PinholeIntrinsics(cam.camera_index);
                K                        = K_l;
                distortion               = distortion_l;
            }
            else if (cam.camera_model_type == CameraModel::OCAM)
            {
                auto [aff_l, poly_l] = d_render_params.OcamIntrinsics(cam.camera_index);
                ocam_aff             = aff_l;
                ocam_poly            = poly_l;
            }
        }
        else if (threadNumInBlock == 64)
        {
            int layer_offset = 0;
            for (int i = 0; i < num_layers; ++i)
            {
                layer_lengths_sm[i] = layer_offset;
                layer_offset += layer_lengths(batch, i);
            }
        }
    }
    __syncthreads();
#    else
    Sophus::SE3f V;
    IntrinsicsPinholef K;
    Distortionf distortion;
    Vector<float, 5> ocam_aff;
    ArrayView<const float> ocam_poly;
    int layer_lengths_sm[num_layers];

    V = d_render_params.Pose(cam.image_index);

    if (cam.camera_model_type == CameraModel::PINHOLE_DISTORTION)
    {
        auto [K_l, distortion_l] = d_render_params.PinholeIntrinsics(cam.camera_index);
        K                        = K_l;
        distortion               = distortion_l;
    }
    else if (cam.camera_model_type == CameraModel::OCAM)
    {
        auto [aff_l, poly_l] = d_render_params.OcamIntrinsics(cam.camera_index);
        ocam_aff             = aff_l;
        ocam_poly            = poly_l;
    }

    int layer_offset = 0;
    for (int i = 0; i < num_layers; ++i)
    {
        layer_lengths_sm[i] = layer_offset;
        layer_offset += layer_lengths(batch, i);
    }

#    endif
    for (int point_id = grid.thread_rank(); point_id < point_cloud.Size(); point_id += grid.size())
    {
        {
#    ifndef FAST_MINIMAL_IMPL
            if (d_render_params.viewer_only)
            {
                int conf_id = point_cloud.GetIndex(point_id);
                if (discard_point_for_confidence(conf_id)) continue;
            }
            else
            {
                if (train && dropout_p)
                {
                    bool drop_out = dropout_p[point_id] == 1;
                    if (drop_out) continue;
                }
            }
#    endif
            if (train && dropout_p)
            {
                bool drop_out = dropout_p[point_id] == 1;
                if (drop_out) continue;
            }

            float point_size_opt = _softplus(point_cloud.GetPointSize(point_id));
            // if (layer_buf < 0.0001) continue;
            vec2 ip;
            float z;
            float radius_pixels;


            {
                vec3 position;

                vec2 image_p_a;
                float drop_out_radius;
                thrust::tie(position, drop_out_radius) = point_cloud.GetPointWoNormal(point_id);

                if (cam.camera_model_type == CameraModel::PINHOLE_DISTORTION)
                {
                    CUDA_KERNEL_ASSERT(cam.camera_model_type == CameraModel::PINHOLE_DISTORTION);
                    //  auto [K, distortion] = d_render_params.PinholeIntrinsics(cam.camera_index);
                    thrust::tie(image_p_a, z) =
                        ProjectPointPinholeWoNormal(position, V, K, distortion, d_render_params.dist_cutoff);
                    radius_pixels = K.fx * cam.crop_transform.fx * drop_out_radius / z;

                    point_size_opt = K.fx * cam.crop_transform.fx * point_size_opt / z;
                }
                else if (cam.camera_model_type == CameraModel::OCAM)
                {
                    //  auto [aff, poly] = d_render_params.OcamIntrinsics(cam.camera_index);
                    thrust::tie(image_p_a, z) = ProjectPointOcamWoNormal(position, V, ocam_aff, ocam_poly, 0.15);
                    radius_pixels =
                        d_forward_params.neural_out[0].Image().w * cam.crop_transform.fx * drop_out_radius / z;
                    // if (d_render_params.use_layer_point_size) CUDA_KERNEL_ASSERT(false);
                    point_size_opt =
                        d_forward_params.neural_out[0].Image().w * cam.crop_transform.fx * point_size_opt / z;
                }
                else if (cam.camera_model_type == CameraModel::SPHERICAL)
                {
                    thrust::tie(image_p_a, z) = ProjectPointSphericalWoNormal(
                        position, V,
                        vec2(d_forward_params.neural_out[0].Image().w, d_forward_params.neural_out[0].Image().h),
                        d_render_params.dist_cutoff);
                    radius_pixels =
                        d_forward_params.neural_out[0].Image().w * cam.crop_transform.fx * drop_out_radius / z;
                    ip = image_p_a;
                }
                else if (cam.camera_model_type == CameraModel::ORTHO)
                {
                    thrust::tie(image_p_a, z) = ProjectPointToOrthographic(
                        position, V, d_forward_params.neural_out[0].Image().w, d_forward_params.neural_out[0].Image().h,
                        d_render_params.dist_cutoff);
                    radius_pixels =
                        d_forward_params.neural_out[0].Image().w * cam.crop_transform.fx * drop_out_radius / z;
                }
                else
                {
                    //   CUDA_KERNEL_ASSERT(cam.camera_model_type == CameraModel::PINHOLE_DISTORTION);
                }
                // if (z <= 0) continue;
                ip = cam.crop_transform.normalizedToImage(image_p_a);
                ip = rotateCropAroundCenter(ip, vec2(cam.w / 2, cam.h / 2), cam);

                // z discard
                // border pixel will be ignored
                // radius discard
                //   if (z <= 0 || ip(0) < 0 || ip(0) >= cam.w - 1 || ip(1) < 0 || ip(1) >= cam.h - 1 ||
                //       (d_render_params.drop_out_points_by_radius &&
                //        radius_pixels < d_render_params.drop_out_radius_threshold))
                //       continue;

                // if (d_render_params.drop_out_points_by_radius && radius_pixels <
                // d_render_params.drop_out_radius_threshold)
                //{
                //     continue;
                // }
            }
            //__syncwarp();

            if (!d_render_params.render_points_in_all_lower_resolutions)
            {
                if (!(z <= 0 || ip(0) < 0 || ip(0) >= cam.w - 1 || ip(1) < 0 || ip(1) >= cam.h - 1 ||
                      (d_render_params.drop_out_points_by_radius &&
                       radius_pixels < d_render_params.drop_out_radius_threshold)))
                {
                    int data_written_to = atomicAdd(atomic_c.data, 1);
                    {
                        // write data to buffers
                        {
                            // write point data
                            // float4 data_buf = make_float4(ip.x(), ip.y(),
                            // reinterpret_cast<float*>(&texture_index)[0],
                            //                              reinterpret_cast<float*>(&point_id)[0]);
                            //
                            //((float4*)&full_list_buffer_data(batch, data_ll_index, 0))[0] = data_buf;
                            int texture_index = point_cloud.GetIndex(point_id);

                            const int offset_batch = batch * full_list_buffer_data.strides[0];


                            if (train)
                            {
                                // full_list_buffer_data(batch, data_written_to, 0) = ip.x();
                                // full_list_buffer_data(batch, data_written_to, 1) = ip.y();
                                // full_list_buffer_data(batch, data_written_to, 2) =
                                //     reinterpret_cast<float*>(&texture_index)[0];
                                // full_list_buffer_data(batch, data_written_to, 3) = point_size_opt;
                                // full_list_buffer_data(batch, data_written_to, 4) =
                                // reinterpret_cast<float*>(&point_id)[0];

                                (full_list_buffer_data_ptr + offset_batch + data_written_to * 5 + 0)[0] = ip.x();
                                (full_list_buffer_data_ptr + offset_batch + data_written_to * 5 + 1)[0] = ip.y();
                                (full_list_buffer_data_ptr + offset_batch + data_written_to * 5 + 2)[0] =
                                    reinterpret_cast<float*>(&texture_index)[0];
                                (full_list_buffer_data_ptr + offset_batch + data_written_to * 5 + 3)[0] =
                                    point_size_opt;
                                (full_list_buffer_data_ptr + offset_batch + data_written_to * 5 + 4)[0] =
                                    reinterpret_cast<float*>(&point_id)[0];
                            }
                            else
                            {
                                float4* addr_data =
                                    (float4*)(full_list_buffer_data_ptr + offset_batch + data_written_to * 4);
                                float4 data = make_float4(ip.x(), ip.y(), reinterpret_cast<float*>(&texture_index)[0],
                                                          point_size_opt);
                                (addr_data)[0] = data;
                            }
                        }
                    }
                    double data_l = __hiloint2double(reinterpret_cast<int*>(&z)[0], data_written_to);

                    {
                        // log2(<1) is neg; num_layer is max possible layers
                        int layer_lower  = 0;
                        int layer_higher = 0;

                        if (point_size_opt > 1)
                        {
                            float log_ps = log2f(point_size_opt);
                            layer_lower  = min(int(floor(log_ps)), num_layers - 1);
                            layer_higher = min(int(ceil(log_ps)), num_layers - 1);

                            ip *= powf(0.5f, layer_lower);
                            radius_pixels *= powf(0.5f, layer_lower);
                        }

                        {
                            int layer  = layer_lower;
                            ivec2 p_rd = ivec2(__float2int_rd(ip(0)), __float2int_rd(ip(1)));


                            // discard upper bounded pixel
                            if (!(p_rd(0) < 0 ||
                                  p_rd(0) >= d_bilinear_alpha_params.per_pixel_list_lengths[layer].size(2) - 1 ||
                                  p_rd(1) < 0 ||
                                  p_rd(1) >= d_bilinear_alpha_params.per_pixel_list_lengths[layer].size(1) - 1 ||
                                  (d_render_params.drop_out_points_by_radius &&
                                   radius_pixels < d_render_params.drop_out_radius_threshold)))
                            {
#    ifdef ACCUM_IN_SM
                                int index_sm = atomicAdd(&atomic_local, 1);
#    endif
                                // #    pragma unroll
                                for (int i = 0; i < 4; ++i)
                                {
                                    // x: i%2; y: i/2
                                    const ivec2 splat_point = ivec2(p_rd.x() + i % 2, p_rd.y() + i / 2);

                                    int offset    = atomicAdd(&d_bilinear_alpha_params.per_pixel_list_lengths[layer](
                                                               batch, splat_point.y(), splat_point.x()),
                                                              1);
                                    int scanned_c = d_bilinear_alpha_params.scanned_countings[layer](
                                                        batch, splat_point.y(), splat_point.x()) +
                                                    offset + layer_lengths_sm[layer];
#    ifdef ACCUM_IN_SM
                                    write_indices_shared[index_sm * 4 + i] = scanned_c;
                                    write_data_shared[index_sm * 4 + i]    = data_l;
#    else

                                    const int offset_batch = batch * full_list_buffer.strides[0];
                                    (full_list_buffer_ptr + offset_batch + scanned_c)[0] = data_l;
                                    // full_list_buffer(batch, scanned_c, 0) = data_l;
#    endif
                                }

                                //   layer_offset += layer_lengths(batch, layer);
                                radius_pixels *= 0.5f;
                                ip *= 0.5f;

                                // continue;
                                if (layer_higher != layer_lower)
                                {
                                    int layer  = layer_higher;
                                    ivec2 p_rd = ivec2(__float2int_rd(ip(0)), __float2int_rd(ip(1)));

                                    // discard upper bounded pixel
                                    if (!(p_rd(0) < 0 ||
                                          p_rd(0) >=
                                              d_bilinear_alpha_params.per_pixel_list_lengths[layer].size(2) - 1 ||
                                          p_rd(1) < 0 ||
                                          p_rd(1) >=
                                              d_bilinear_alpha_params.per_pixel_list_lengths[layer].size(1) - 1 ||
                                          (d_render_params.drop_out_points_by_radius &&
                                           radius_pixels < d_render_params.drop_out_radius_threshold)))
                                    {
#    ifdef ACCUM_IN_SM
                                        index_sm = atomicAdd(&atomic_local, 1);
#    endif
                                        // #    pragma unroll
                                        for (int i = 0; i < 4; ++i)
                                        {
                                            ivec2 splat_point = ivec2(p_rd.x() + i % 2, p_rd.y() + i / 2);

                                            int offset =
                                                atomicAdd(&d_bilinear_alpha_params.per_pixel_list_lengths[layer](
                                                              batch, splat_point.y(), splat_point.x()),
                                                          1);
                                            int scanned_c = d_bilinear_alpha_params.scanned_countings[layer](
                                                                batch, splat_point.y(), splat_point.x()) +
                                                            offset + layer_lengths_sm[layer];

                                            //  double data_l = __hiloint2double(reinterpret_cast<int*>(&z)[0],
                                            //  data_written_to);
#    ifdef ACCUM_IN_SM
                                            write_indices_shared[index_sm * 4 + i] = scanned_c;
                                            write_data_shared[index_sm * 4 + i]    = data_l;
#    else
                                            // full_list_buffer(batch, scanned_c, 0) = data_l;
                                            const int offset_batch = batch * full_list_buffer.strides[0];
                                            (full_list_buffer_ptr + offset_batch + scanned_c)[0] = data_l;

#    endif
                                        }
                                    }
                                }
                            }
                            // __syncwarp();
                        }
                    }
                }
            }

#    ifdef ACCUM_IN_SM

            __syncthreads();

            for (int i = 0; i < 8; ++i)
            {
                int index       = threadNumInBlock + i * DEFAULT_BLOCK_SIZE_FAST_COLLECT;
                int write_index = write_indices_shared[index];
                if (write_index != -1) full_list_buffer(batch, write_index, 0) = write_data_shared[index];
            }

            __syncthreads();

            // reset data
            for (int i = 0; i < 8; ++i)
                write_indices_shared[threadNumInBlock + i * DEFAULT_BLOCK_SIZE_FAST_COLLECT] = -1;
            if (threadIdx.x == 0) atomic_local = 0;

            __syncthreads();
#    endif

            else
            {
                if (!(z <= 0 || ip(0) < 0 || ip(0) >= d_bilinear_alpha_params.per_pixel_list_lengths[0].size(2) - 1 ||
                      ip(1) < 0 || ip(1) >= d_bilinear_alpha_params.per_pixel_list_lengths[0].size(2) - 1) ||
                    (d_render_params.drop_out_points_by_radius &&
                     radius_pixels < d_render_params.drop_out_radius_threshold))
                {
                    // write data:
                    int data_written_to = atomicAdd(atomic_c.data, 1);
                    {
                        // write point data
                        int texture_index      = point_cloud.GetIndex(point_id);
                        const int offset_batch = batch * full_list_buffer_data.strides[0];
                        if (train)
                        {
                            (full_list_buffer_data_ptr + offset_batch + data_written_to * 5 + 0)[0] = ip.x();
                            (full_list_buffer_data_ptr + offset_batch + data_written_to * 5 + 1)[0] = ip.y();
                            (full_list_buffer_data_ptr + offset_batch + data_written_to * 5 + 2)[0] =
                                reinterpret_cast<float*>(&texture_index)[0];
                            (full_list_buffer_data_ptr + offset_batch + data_written_to * 5 + 3)[0] = point_size_opt;
                            (full_list_buffer_data_ptr + offset_batch + data_written_to * 5 + 4)[0] =
                                reinterpret_cast<float*>(&point_id)[0];
                        }
                        else
                        {
                            float4* addr_data =
                                (float4*)(full_list_buffer_data_ptr + offset_batch + data_written_to * 4);
                            float4 data    = make_float4(ip.x(), ip.y(), reinterpret_cast<float*>(&texture_index)[0],
                                                         point_size_opt);
                            (addr_data)[0] = data;
                        }
                    }
                    double data_l = __hiloint2double(reinterpret_cast<int*>(&z)[0], data_written_to);


                    int layer_lower  = 0;
                    int layer_higher = 0;
                    if (point_size_opt > 1)
                    {
                        layer_lower  = min(int(floor(log2f(point_size_opt))), num_layers - 1);
                        layer_higher = min(int(ceil(log2f(point_size_opt))), num_layers - 1);
                    }

                    for (int layer = 0; layer < layer_higher + 1; ++layer)
                    {
                        ivec2 p_rd = ivec2(__float2int_rd(ip(0)), __float2int_rd(ip(1)));

                        // discard upper bounded pixel
                        if (p_rd(0) < 0 ||
                            p_rd(0) >= d_bilinear_alpha_params.per_pixel_list_lengths[layer].size(2) - 1 ||
                            p_rd(1) < 0 ||
                            p_rd(1) >= d_bilinear_alpha_params.per_pixel_list_lengths[layer].size(1) - 1 ||
                            (d_render_params.drop_out_points_by_radius &&
                             radius_pixels < d_render_params.drop_out_radius_threshold))
                            break;
                            //     continue;
#    pragma unroll
                        for (int i = 0; i < 4; ++i)
                        {
                            // x: i%2; y: i/2
                            const ivec2 splat_point = ivec2(p_rd.x() + i % 2, p_rd.y() + i / 2);

                            int offset = atomicAdd(&d_bilinear_alpha_params.per_pixel_list_lengths[layer](
                                                       batch, splat_point.y(), splat_point.x()),
                                                   1);
                            // int offset = d_bilinear_alpha_params.per_pixel_list_lengths[layer](batch,
                            // splat_point.y(),
                            //                                                                    splat_point.x())++;
                            int scanned_c = d_bilinear_alpha_params.scanned_countings[layer](batch, splat_point.y(),
                                                                                             splat_point.x()) +
                                            offset + layer_lengths_sm[layer];
                            // int scanned_c                                        = 0;
                            const int offset_batch                               = batch * full_list_buffer.strides[0];
                            (full_list_buffer_ptr + offset_batch + scanned_c)[0] = data_l;
                            // unsigned long long int* addr =
                            //     ((unsigned long long int*)full_list_buffer_ptr) + offset_batch + scanned_c;
                            // unsigned long long int val = reinterpret_cast<unsigned long long int*>(&data_l)[0];
                            //
                            // atomicExch(addr, val);

                            // cooperative_groups::memcpy_async(grid, full_list_buffer_ptr + offset_batch +
                            // scanned_c,
                            //                                  &data_l, sizeof(double));
                            //  full_list_buffer(batch, scanned_c, 0) = data_l;
                            // cuda::memcpy_async(full_list_buffer_ptr + offset_batch + scanned_c, &data_l,
                            // sizeof(double),
                            //                    barrier);
                            //  cuda::mem
                        }

                        //   layer_offset += layer_lengths(batch, layer);
                        radius_pixels *= 0.5f;
                        ip *= 0.5f;
                    }
                }
            }
        }
    }
}



void PointRendererCache::CollectTiled2(int batch, NeuralPointCloudCuda point_cloud, torch::Tensor full_list_buffer,
                                       torch::Tensor full_list_buffer_data, torch::Tensor layer_lengths, bool train)
{
    SAIGA_ASSERT(point_cloud);

    const int points_per_thread_collection_pass = 8;

    static int num_per_t = 1024;
    // ImGui::Begin("test");

    // ImGui::SliderInt("points_per_thread_collection_pass", &points_per_thread_collection_pass, 1, 256);

    // ImGui::End();
    {
        torch::Tensor atomic = torch::zeros({1}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32));

        int image_batch_id = batch;

        float* dropout = (info->train && dropout_points.defined())
                             ? dropout_points.data_ptr<float>() + dropout_points.stride(0) * image_batch_id
                             : nullptr;

        auto cam = info->images[image_batch_id];
        SAIGA_ASSERT(cam.camera_index >= 0 && cam.image_index >= 0);

        int c = iDivUp(point_cloud->Size(), DEFAULT_BLOCK_SIZE_FAST_COLLECT * points_per_thread_collection_pass);


        if (info->params.use_layer_point_size)
        {
            // std::cout << "TT" << std::endl;
            if (info->num_layers == 1)
            {
                ::CollectTiled2Pointsize<1, points_per_thread_collection_pass><<<c, DEFAULT_BLOCK_SIZE_FAST_COLLECT>>>(
                    point_cloud, dropout, cam, batch, full_list_buffer, full_list_buffer.data_ptr<double>(),
                    full_list_buffer_data, full_list_buffer_data.data_ptr<float>(), layer_lengths, atomic, info->train);
                CUDA_SYNC_CHECK_ERROR();
            }
            else if (info->num_layers == 2)
            {
                ::CollectTiled2Pointsize<2, points_per_thread_collection_pass><<<c, DEFAULT_BLOCK_SIZE_FAST_COLLECT>>>(
                    point_cloud, dropout, cam, batch, full_list_buffer, full_list_buffer.data_ptr<double>(),
                    full_list_buffer_data, full_list_buffer_data.data_ptr<float>(), layer_lengths, atomic, info->train);
                CUDA_SYNC_CHECK_ERROR();
            }
            else if (info->num_layers == 3)
            {
                ::CollectTiled2Pointsize<3, points_per_thread_collection_pass><<<c, DEFAULT_BLOCK_SIZE_FAST_COLLECT>>>(
                    point_cloud, dropout, cam, batch, full_list_buffer, full_list_buffer.data_ptr<double>(),
                    full_list_buffer_data, full_list_buffer_data.data_ptr<float>(), layer_lengths, atomic, info->train);
                CUDA_SYNC_CHECK_ERROR();
            }
            else if (info->num_layers == 4)
            {
                ::CollectTiled2Pointsize<4, points_per_thread_collection_pass><<<c, DEFAULT_BLOCK_SIZE_FAST_COLLECT>>>(
                    point_cloud, dropout, cam, batch, full_list_buffer, full_list_buffer.data_ptr<double>(),
                    full_list_buffer_data, full_list_buffer_data.data_ptr<float>(), layer_lengths, atomic, info->train);
                CUDA_SYNC_CHECK_ERROR();
            }
            else if (info->num_layers == 5)
            {
                ::CollectTiled2Pointsize<5, points_per_thread_collection_pass><<<c, DEFAULT_BLOCK_SIZE_FAST_COLLECT>>>(
                    point_cloud, dropout, cam, batch, full_list_buffer, full_list_buffer.data_ptr<double>(),
                    full_list_buffer_data, full_list_buffer_data.data_ptr<float>(), layer_lengths, atomic, info->train);
                CUDA_SYNC_CHECK_ERROR();
            }
            else if (info->num_layers == 6)
            {
                ::CollectTiled2Pointsize<6, points_per_thread_collection_pass><<<c, DEFAULT_BLOCK_SIZE_FAST_COLLECT>>>(
                    point_cloud, dropout, cam, batch, full_list_buffer, full_list_buffer.data_ptr<double>(),
                    full_list_buffer_data, full_list_buffer_data.data_ptr<float>(), layer_lengths, atomic, info->train);
                CUDA_SYNC_CHECK_ERROR();
            }
            else if (info->num_layers == 7)
            {
                ::CollectTiled2Pointsize<7, points_per_thread_collection_pass><<<c, DEFAULT_BLOCK_SIZE_FAST_COLLECT>>>(
                    point_cloud, dropout, cam, batch, full_list_buffer, full_list_buffer.data_ptr<double>(),
                    full_list_buffer_data, full_list_buffer_data.data_ptr<float>(), layer_lengths, atomic, info->train);
                CUDA_SYNC_CHECK_ERROR();
            }
            else if (info->num_layers == 8)
            {
                ::CollectTiled2Pointsize<8, points_per_thread_collection_pass><<<c, DEFAULT_BLOCK_SIZE_FAST_COLLECT>>>(
                    point_cloud, dropout, cam, batch, full_list_buffer, full_list_buffer.data_ptr<double>(),
                    full_list_buffer_data, full_list_buffer_data.data_ptr<float>(), layer_lengths, atomic, info->train);
                CUDA_SYNC_CHECK_ERROR();
            }
            else
            {
                SAIGA_EXIT_ERROR("invalid number of layers");
            }
        }
        else
        {
            if (info->num_layers == 1)
            {
                ::CollectTiled2<1><<<c, DEFAULT_BLOCK_SIZE_FAST_COLLECT>>>(point_cloud, dropout, cam, batch,
                                                                           full_list_buffer, full_list_buffer_data,
                                                                           layer_lengths, info->train);
                CUDA_SYNC_CHECK_ERROR();
            }
            else if (info->num_layers == 2)
            {
                ::CollectTiled2<2><<<c, DEFAULT_BLOCK_SIZE_FAST_COLLECT>>>(point_cloud, dropout, cam, batch,
                                                                           full_list_buffer, full_list_buffer_data,
                                                                           layer_lengths, info->train);
                CUDA_SYNC_CHECK_ERROR();
            }
            else if (info->num_layers == 3)
            {
                ::CollectTiled2<3><<<c, DEFAULT_BLOCK_SIZE_FAST_COLLECT>>>(point_cloud, dropout, cam, batch,
                                                                           full_list_buffer, full_list_buffer_data,
                                                                           layer_lengths, info->train);
                CUDA_SYNC_CHECK_ERROR();
            }
            else if (info->num_layers == 4)
            {
                ::CollectTiled2<4><<<c, DEFAULT_BLOCK_SIZE_FAST_COLLECT>>>(point_cloud, dropout, cam, batch,
                                                                           full_list_buffer, full_list_buffer_data,
                                                                           layer_lengths, info->train);
                CUDA_SYNC_CHECK_ERROR();
            }
            else if (info->num_layers == 5)
            {
                ::CollectTiled2<5><<<c, DEFAULT_BLOCK_SIZE_FAST_COLLECT>>>(point_cloud, dropout, cam, batch,
                                                                           full_list_buffer, full_list_buffer_data,
                                                                           layer_lengths, info->train);
                CUDA_SYNC_CHECK_ERROR();
            }
            else if (info->num_layers == 6)
            {
                ::CollectTiled2<6><<<c, DEFAULT_BLOCK_SIZE_FAST_COLLECT>>>(point_cloud, dropout, cam, batch,
                                                                           full_list_buffer, full_list_buffer_data,
                                                                           layer_lengths, info->train);
                CUDA_SYNC_CHECK_ERROR();
            }
            else if (info->num_layers == 7)
            {
                ::CollectTiled2<7><<<c, DEFAULT_BLOCK_SIZE_FAST_COLLECT>>>(point_cloud, dropout, cam, batch,
                                                                           full_list_buffer, full_list_buffer_data,
                                                                           layer_lengths, info->train);
                CUDA_SYNC_CHECK_ERROR();
            }
            else if (info->num_layers == 8)
            {
                ::CollectTiled2<8><<<c, DEFAULT_BLOCK_SIZE_FAST_COLLECT>>>(point_cloud, dropout, cam, batch,
                                                                           full_list_buffer, full_list_buffer_data,
                                                                           layer_lengths, info->train);
                CUDA_SYNC_CHECK_ERROR();
            }
            else
            {
                SAIGA_EXIT_ERROR("invalid number of layers");
            }
        }
    }
    CUDA_SYNC_CHECK_ERROR();
}

///Sort and Blend


#define MAX_THREADS_X 16
#define MAX_THREADS_Y 16
#define THREADS_PER_BLOCK (MAX_THREADS_X * MAX_THREADS_Y)
struct SortData
{
    float depth;
    int index;
    __device__ SortData() {}
    __device__ SortData(const SortData& other)
    {
        depth = other.depth;
        index = other.index;
    }
    __device__ SortData& operator=(const SortData& other)
    {
        if (this == &other) return *this;

        depth = other.depth;
        index = other.index;
        return *this;
    }
    __device__ SortData(const float d, const int i) : depth(d), index(i) {}
    __device__ SortData(const float d, const float i, bool reinterpret) : depth(d)
    {
        index = reinterpret_cast<const int*>(&i)[0];
    }
};
__device__ __forceinline__ bool operator<(SortData& a, SortData& b)
{
    return a.depth < b.depth;
}

template <int TILE_SIZE, int ELEMENTS_PER_PIXEL, int max_layers, bool train>
__global__ void __launch_bounds__(THREADS_PER_BLOCK)  //__launch_bounds__(THREADS_PER_BLOCK,2048 / THREADS_PER_BLOCK)
    FastSortAndBlendWPrep2(int batch,
                           ConstReadOnlyStaticDeviceTensor<double, 3> full_list_buffer /* (batch,pointcloudsize,1)*/,
                           const double* __restrict__ full_list_buffer_data_ptr,
                           ConstReadOnlyStaticDeviceTensor<float, 3> full_list_buffer_data /*(batch,pointcloudsize,4)*/,
                           const float* __restrict__ full_list_buffer_data_data_ptr,
                           const float* __restrict__ background_color, StaticDeviceTensor<int, 1> glob_atomic_c,
                           StaticDeviceTensor<int32_t, 2, int> layer_lengths, bool environment_map,
                           int max_elements_total, StaticDeviceTensor<int32_t, 2> non_zero_indices /*(listlength,1)*/)
{
    constexpr float max_depth = 1e25;
    // constexpr int BILINEAR_MULTIPLICATION_FACTOR = 4;

    full_list_buffer.setDataPointer(full_list_buffer_data_ptr);
    full_list_buffer_data.setDataPointer(full_list_buffer_data_data_ptr);
    uint32_t ticket = 0;

    uint32_t layer = 0, gx = 0, gy = 0;
    uint32_t start_offset = 0;
    uint32_t length       = 0;



    SortData local_mem[ELEMENTS_PER_PIXEL];
    //  auto get_elem = [&local_mem](int index) -> SortData& { return local_mem[index]; };

    /*
    auto get_next_list = [&ticket, &glob_atomic_c, &layer, &gx, &gy, &batch, &start_offset, &layer_lengths, &length,
                          &get_elem, &max_elements_total]
    {
        int layer_offset = 0;

        // ticket = atomicAdd(&glob_atomic_c(0), 1);
        length = 0;
        while (length == 0)
        {
            ticket = atomicAdd(&glob_atomic_c(0), 1);
            if (ticket >= max_elements_total) return;
            layer_offset = 0;
            int tick     = ticket;
            for (int i = 0; i < max_layers; i++)
            {
                int w = d_forward_params.neural_out[i].size(3);
                int h = d_forward_params.neural_out[i].size(2);

                if (tick < w * h)
                {
                    layer  = i;
                    gx     = tick % w;
                    gy     = tick / w;
                    length = d_bilinear_alpha_params.per_pixel_list_lengths[layer](batch, gy, gx);
                    break;
                }
                layer_offset += layer_lengths(batch, i);
                tick -= w * h;
            }
        }
        start_offset = d_bilinear_alpha_params.scanned_countings[layer](batch, gy, gx) + layer_offset;
        // clear local mem
        for (int i = 0; i < ELEMENTS_PER_PIXEL; ++i) get_elem(i) = {max_depth, int(-1)};
    };*/
    uint32_t layer_offset           = 0;
    uint32_t max_elements_realigned = non_zero_indices.size(0);

    auto get_next_list = [&ticket, &glob_atomic_c, &layer, &gx, &gy, &batch, &start_offset, &layer_lengths, &length,
                          &local_mem, &layer_offset, &non_zero_indices, &max_elements_realigned]
    {
        ticket = atomicAdd(&glob_atomic_c(0), 1);
        if (ticket >= max_elements_realigned) return;
        layer_offset  = 0;
        uint32_t tick = non_zero_indices(ticket, 0);
        for (int i = 0; i < max_layers; i++)
        {
            int w, h;
            if (train)
            {
                w = d_forward_params.neural_out[i].size(3);
                h = d_forward_params.neural_out[i].size(2);
            }
            else
            {
                w = d_forward_params.neural_out[i].size(2);
                h = d_forward_params.neural_out[i].size(1);
            }

            if (tick < w * h)
            {
                layer = i;
                gx    = tick % w;
                gy    = tick / w;
                break;
            }
            layer_offset += layer_lengths(batch, i);
            tick -= w * h;
        }
        length       = d_bilinear_alpha_params.per_pixel_list_lengths[layer](batch, gy, gx);
        start_offset = d_bilinear_alpha_params.scanned_countings[layer](batch, gy, gx) + layer_offset;
        // clear local mem
        for (int i = 0; i < ELEMENTS_PER_PIXEL; ++i) local_mem[i] = {max_depth, int(-1)};
    };

    get_next_list();
    //  while (length > 16 && ticket < max_elements_realigned) get_next_list();

    while (ticket < max_elements_realigned)
    {
        auto process_sortdata = [&local_mem](SortData& elem)
        {
            float depth_elem = elem.depth;
            // early discard
            if (depth_elem >= local_mem[ELEMENTS_PER_PIXEL - 1].depth)
            {
                return;
            }

            // #        pragma unroll
            for (int pos_to_insert = 0; pos_to_insert < ELEMENTS_PER_PIXEL; ++pos_to_insert)
            {
                float d_in_list = local_mem[pos_to_insert].depth;
                if (depth_elem < d_in_list)
                {
                    if (d_in_list != max_depth)
                    {
                        for (int i = ELEMENTS_PER_PIXEL - 1; i > pos_to_insert; --i)
                        {
                            local_mem[i] = local_mem[i - 1];
                        }
                    }
                    // insert elements
                    local_mem[pos_to_insert] = elem;
                    return;
                }
            }
        };
        int elem_i             = 0;
        const int offset_batch = batch * full_list_buffer.strides[0];

        // 4-element access should be memory aligned, thus if starting at non-alignment read first element
        if (start_offset % 2 == 1)
        {
            const int run_pos = start_offset + elem_i;
            int2 data_comb    = ((int2*)(full_list_buffer_data_ptr + offset_batch + run_pos))[0];

            SortData elem(reinterpret_cast<const float*>(&data_comb.y)[0], data_comb.x);
            process_sortdata(elem);
            ++elem_i;
        }
        for (; elem_i + 1 < length; elem_i += 2)
        {
            const int run_pos         = start_offset + elem_i;
            const int4 data_combined2 = ((int4*)(full_list_buffer_data_ptr + offset_batch + run_pos))[0];

            SortData elem(reinterpret_cast<const float*>(&data_combined2.y)[0], data_combined2.x);
            process_sortdata(elem);
            SortData elem2(reinterpret_cast<const float*>(&data_combined2.w)[0], data_combined2.z);
            process_sortdata(elem2);
            //   __syncwarp();
        }
        // read final element (non-aligned
        if (elem_i < length)
        {
            const int run_pos    = start_offset + elem_i;
            const int2 data_comb = ((int2*)(full_list_buffer_data_ptr + offset_batch + run_pos))[0];

            SortData elem(reinterpret_cast<const float*>(&data_comb.y)[0], data_comb.x);
            process_sortdata(elem);
            ++elem_i;
        }
        //   __syncwarp();

        // accumulate
        float alpha_dest = 1.f;
        // float color_out[4] = {0.f, 0.f, 0.f, 0.f};
        float4 color_out = make_float4(0.f, 0.f, 0.f, 0.f);

        // for (int index_in_list = 0, fetch_idx = get_elem(0).index;
        //      fetch_idx >= 0 && index_in_list < ELEMENTS_PER_PIXEL && alpha_dest >= 0.001;
        //      ++index_in_list, fetch_idx = (index_in_list < ELEMENTS_PER_PIXEL) ? get_elem(index_in_list).index :
        //      -1)
        for (int index_in_list = 0; index_in_list < min(length, ELEMENTS_PER_PIXEL); ++index_in_list)
        {
            int fetch_idx = local_mem[index_in_list].index;
            CUDA_KERNEL_ASSERT(fetch_idx != -1);

            float ipx, ipy;
            int texture_index;
            float point_size_opt = 1.f;

            if (!train)
            {
                const float4 data_fetch = ((float4*)&full_list_buffer_data(batch, fetch_idx, 0))[0];
                ipx                     = data_fetch.x;
                ipy                     = data_fetch.y;
                texture_index           = reinterpret_cast<const int*>(&data_fetch.z)[0];
                if (d_render_params.use_layer_point_size) point_size_opt = data_fetch.w;
            }
            else
            {
                ipx           = full_list_buffer_data(batch, fetch_idx, 0);
                ipy           = full_list_buffer_data(batch, fetch_idx, 1);
                texture_index = reinterpret_cast<const int*>(&full_list_buffer_data(batch, fetch_idx, 2))[0];
                if (d_render_params.use_layer_point_size) point_size_opt = full_list_buffer_data(batch, fetch_idx, 3);
            }

            const float layer_mult_fac = 1.f / float(1 << layer);
            vec2 ip                    = vec2(ipx * layer_mult_fac, ipy * layer_mult_fac);
            vec4 blend_vec             = compute_blending_fac(ip);
            int blend_index            = blend_fac_index(ip, vec2(gx, gy));
            float bilinear_fac         = blend_vec[blend_index];

            float alpha_bilin = bilinear_fac * d_texture.points_confidence_value(0, texture_index);

            if (d_render_params.use_layer_point_size)
            {
                float layer_factor = compute_point_size_fac(point_size_opt, layer, max_layers);
                alpha_bilin *= layer_factor;
            }
            //  if (train)
            {
                // for (int ci = 0; ci < d_render_params.num_texture_channels; ++ci)
                // {
                float color0 = d_texture.in_texture(0, texture_index);
                float color1 = d_texture.in_texture(1, texture_index);
                float color2 = d_texture.in_texture(2, texture_index);
                float color3 = d_texture.in_texture(3, texture_index);
                float4 color = make_float4(color0, color1, color2, color3);
                // compute_blend_fused<float4>(make_float4(alpha_dest, alpha_dest, alpha_dest, alpha_dest),
                //                             make_float4(alpha_bilin, alpha_bilin, alpha_bilin, alpha_bilin),
                //                             color, color_out);
                color_out += color * (alpha_dest * alpha_bilin);
            }
            // else
            // {
            //     float4 color = reinterpret_cast<float4*>(&d_texture.in_texture(0, texture_index))[0];
            //     color_out += color * (alpha_dest * alpha_bilin);
            // }
            alpha_dest = compute_new_alphadest(alpha_dest, alpha_bilin);

            if (train)
            {
                // write out intermediates for backwards
                // texture id, point id, alpha_val, vec2 subpixel_pos, int blendindex, point_id
                d_bilinear_alpha_params.bw_sorted_maxed[layer](batch, gy, gx, index_in_list, 0) =
                    reinterpret_cast<const float*>(&texture_index)[0];
                d_bilinear_alpha_params.bw_sorted_maxed[layer](batch, gy, gx, index_in_list, 1) =
                    full_list_buffer_data(batch, fetch_idx, 4);
                d_bilinear_alpha_params.bw_sorted_maxed[layer](batch, gy, gx, index_in_list, 2) = alpha_bilin;
                d_bilinear_alpha_params.bw_sorted_maxed[layer](batch, gy, gx, index_in_list, 3) = ip.x();
                d_bilinear_alpha_params.bw_sorted_maxed[layer](batch, gy, gx, index_in_list, 4) = ip.y();
                d_bilinear_alpha_params.bw_sorted_maxed[layer](batch, gy, gx, index_in_list, 5) =
                    reinterpret_cast<const float*>(&blend_index)[0];
                if (d_render_params.use_layer_point_size)
                    d_bilinear_alpha_params.bw_sorted_maxed[layer](batch, gy, gx, index_in_list, 6) = point_size_opt;
            }
            if (alpha_dest < 0.001) break;
        }

        //  __syncwarp();
        //  color_out[0] /= 16.f;
        // blend background (opacity 1) and write out
        // for (int ci = 0; ci < d_render_params.num_texture_channels; ++ci)
        // {
        if (alpha_dest >= 0.001)
        {
            if (!environment_map)
            {
                float4 bg_col =
                    make_float4(background_color[0], background_color[1], background_color[2], background_color[3]);
                // color_out[ci] = compute_blend(alpha_dest, 1.f, background_color[ci], color_out[ci]);
                color_out += bg_col * (alpha_dest * 1.f);
            }
        }
        if (train)
        {
            d_forward_params.neural_out[layer](batch, 0, gy, gx) = color_out.x;
            d_forward_params.neural_out[layer](batch, 1, gy, gx) = color_out.y;
            d_forward_params.neural_out[layer](batch, 2, gy, gx) = color_out.z;
            d_forward_params.neural_out[layer](batch, 3, gy, gx) = color_out.w;
        }
        else
        {
            ((float4*)&d_forward_params.neural_out[layer](batch, gy, gx, 0))[0] = color_out;
            // d_forward_params.neural_out[layer](batch, gy, gx, 0) = color_out.x;
            // d_forward_params.neural_out[layer](batch, gy, gx, 1) = color_out.y;
            // d_forward_params.neural_out[layer](batch, gy, gx, 2) = color_out.z;
            // d_forward_params.neural_out[layer](batch, gy, gx, 3) = color_out.w;
        }
        //    __syncwarp();
        get_next_list();
        // while (length <= 16 && ticket < max_elements_realigned) get_next_list();

        //    __syncwarp();
    }
}
#define THREAD_DIMX_PER_BLOCK_MULTITHREADLOAD 16
#define THREAD_DIMY_PER_BLOCK_MULTITHREADLOAD 16

// #define DEBUG_PRINT
#ifdef DEBUG_PRINT
#    define THREAD_DIMX_PER_BLOCK_MULTITHREADLOAD 32
#    define THREAD_DIMY_PER_BLOCK_MULTITHREADLOAD 1
#endif
#define THREADS_PER_BLOCK_MULTITHREADLOAD \
    (THREAD_DIMX_PER_BLOCK_MULTITHREADLOAD * THREAD_DIMY_PER_BLOCK_MULTITHREADLOAD)
//__launch_bounds__(THREADS_PER_BLOCK,2048 / THREADS_PER_BLOCK)

template <unsigned int SIZE = 32>
inline __device__ float2 shuffleSwapCompare(float2 x, int mask, int direction)
{
    auto y = Saiga::CUDA::shfl_xor(x, mask, SIZE);
    return x.x < y.x == direction ? y : x;
}

__device__ void set_bit(int& val, int bit_num)
{
    int set_mask = 1 << bit_num;
    val |= set_mask;
}
__device__ void unset_bit(int& val, int bit_num)
{
    int set_mask = ~(1 << bit_num);
    val &= set_mask;
}


struct SortData2
{
    float depth;
    int index_layer;
    __device__ SortData2() {}
    __device__ SortData2(const SortData2& other)
    {
        depth       = other.depth;
        index_layer = other.index_layer;
    }
    __device__ SortData2& operator=(const SortData2& other)
    {
        if (this == &other) return *this;

        depth       = other.depth;
        index_layer = other.index_layer;
        return *this;
    }
    __device__ SortData2(const float d, const int i, const int layer) : depth(d), index_layer(i)
    {
        int ind     = i << 4;
        index_layer = ind | layer;
    }
    __device__ int get_index() { return (index_layer >> 4); }
    __device__ int get_layer() { return (index_layer & 0xF); }
    // __device__ SortData(const float d, const float i, bool reinterpret) : depth(d)
    // {
    //     index = reinterpret_cast<const int*>(&i)[0];
    // }
};
__device__ __forceinline__ bool operator<(SortData2& a, SortData2& b)
{
    return a.depth < b.depth;
}


template <int NUM_DESCRIPTORS, int ELEMENTS_PER_PIXEL, bool train>
__global__ void __launch_bounds__(THREADS_PER_BLOCK_MULTITHREADLOAD) FastSortAndBlendWPrep2WarpSharedWork(
    int batch, ConstReadOnlyStaticDeviceTensor<double, 3> full_list_buffer /* (batch,pointcloudsize,1)*/,
    const double* __restrict__ full_list_buffer_data_ptr,
    ConstReadOnlyStaticDeviceTensor<float, 3> full_list_buffer_data /*(batch,pointcloudsize,4)*/,
    const float* __restrict__ full_list_buffer_data_data_ptr, const float* __restrict__ background_color,
    StaticDeviceTensor<int, 1> glob_atomic_c, StaticDeviceTensor<int32_t, 2, int> layer_lengths, bool environment_map,
    int max_elements_total, StaticDeviceTensor<int32_t, 2> non_zero_indices /*(listlength,1)*/, int num_layers_input)
{
    constexpr float max_depth = 1e25;
    const int thread_idx      = threadIdx.x + threadIdx.y * blockDim.x;
    constexpr int WARP_SIZE   = 32;
    int own_index_in_warp     = thread_idx % WARP_SIZE;
    // const int warp_id_in_block  = thread_idx / WARP_SIZE;

    const int FAILURE_VAL_LIST = int(-1) << 4;

    CUDA_KERNEL_ASSERT(ELEMENTS_PER_PIXEL <= WARP_SIZE / 2);

    constexpr int LISTS_PER_WARP     = WARP_SIZE / ELEMENTS_PER_PIXEL;
    constexpr int WARP_LEADER_STRIDE = ELEMENTS_PER_PIXEL;

    CUDA_KERNEL_ASSERT(WARP_SIZE % ELEMENTS_PER_PIXEL == 0);


    full_list_buffer.setDataPointer(full_list_buffer_data_ptr);
    full_list_buffer_data.setDataPointer(full_list_buffer_data_data_ptr);
    uint32_t ticket = 0;

    uint32_t layer = 0, gx = 0, gy = 0;
    uint32_t start_offset = 0;
    uint32_t length = 1, leader_length = 1;


    SortData2 local_mem_warp = {max_depth, int(FAILURE_VAL_LIST), int(0)};

    uint32_t max_elements_realigned = non_zero_indices.size(0);
    // uint32_t max_elements_realigned = max_elements_total;

    bool loop_run = true;

    auto get_next_list = [&ticket, &glob_atomic_c, &layer, &gx, &gy, &num_layers_input, &length, &local_mem_warp,
                          &max_elements_realigned, &own_index_in_warp, &leader_length, &loop_run, &non_zero_indices,
                          &FAILURE_VAL_LIST]
    {
        bool continue_loop = true;
        // warp leaders fetches next list, computes indices
        unsigned leader_mask = 0;
        leader_mask          = __ballot_sync(0xffffffff, own_index_in_warp % WARP_LEADER_STRIDE == 0);
        if (own_index_in_warp % WARP_LEADER_STRIDE == 0)
        {
            ticket = atomicAdd(&glob_atomic_c(0), 1);

            leader_length = 0;
            if (ticket < max_elements_realigned)
            {
                uint32_t tick = non_zero_indices(ticket, 0);
                // uint32_t tick = ticket;
                for (int i = 0; i < num_layers_input; i++)
                {
                    int w, h;
                    if (train)
                    {
                        w = d_forward_params.neural_out[i].size(3);
                        h = d_forward_params.neural_out[i].size(2);
                    }
                    else
                    {
                        w = d_forward_params.neural_out[i].size(2);
                        h = d_forward_params.neural_out[i].size(1);
                    }

                    if (tick < w * h)
                    {
                        layer = i;
                        gx    = tick % w;
                        gy    = tick / w;
                        break;
                    }
                    tick -= w * h;
                }
                //  length       = d_bilinear_alpha_params.per_pixel_list_lengths[layer](batch, gy, gx);
                //  start_offset = d_bilinear_alpha_params.scanned_countings[layer](batch, gy, gx) + layer_offset;
            }
            else
            {
                length = 0;
            }
            bool all_failed = __all_sync(leader_mask, ticket >= max_elements_realigned);
            continue_loop   = !all_failed;
        }
        __syncwarp();
        loop_run = __shfl_sync(0xffffffff, continue_loop, 0);
        //  clear local mem
        local_mem_warp = {max_depth, int(FAILURE_VAL_LIST), int(0)};
    };

    get_next_list();
    __syncwarp();
    //  while (length > 16 && ticket < max_elements_realigned) get_next_list();

    while (loop_run)  //(ticket < max_elements_realigned)
    {
        const int offset_batch = batch * full_list_buffer.strides[0];

        int threads_in_use = 0;
        /*
         * every half warp (or however many ELEMENTS are present) stores the sorted list in its register (sorted by
         * threadnum) then all thread load 32 new elements, bitonic sort them and the locally stored item
         *
         *
         */

        for (; threads_in_use < WARP_SIZE; threads_in_use += WARP_LEADER_STRIDE)
        // for (; leader_threads < WARP_SIZE / 2; leader_threads += WARP_LEADER_STRIDE)
        {
            //      if (leader_threads > 0) break;

            auto get_element_addresses = [&own_index_in_warp, &threads_in_use, &ticket, &length,
                                          &max_elements_realigned, &batch, &leader_length, &start_offset,
                                          &layer_lengths, &num_layers_input](int layer_id, int gy_id, int gx_id)
            {
                if (own_index_in_warp == threads_in_use)
                {
                    if (ticket >= max_elements_realigned || layer_id >= num_layers_input ||
                        gx_id >= d_bilinear_alpha_params.per_pixel_list_lengths[layer_id].size(2) ||
                        gy_id >= d_bilinear_alpha_params.per_pixel_list_lengths[layer_id].size(1))
                    {
                        length = 0;
                    }
                    else
                    {
                        length = d_bilinear_alpha_params.per_pixel_list_lengths[layer_id](batch, gy_id, gx_id);

                        leader_length += length;

                        int layer_offset = 0;
                        for (int i = 0; i < layer_id; i++) layer_offset += layer_lengths(batch, i);

                        start_offset =
                            d_bilinear_alpha_params.scanned_countings[layer_id](batch, gy_id, gx_id) + layer_offset;
                    }
                }
                length       = __shfl_sync(0xffffffff, length, threads_in_use);
                start_offset = __shfl_sync(0xffffffff, start_offset, threads_in_use);
            };

            int start_layer = __shfl_sync(0xffffffff, layer, threads_in_use);
            int x_id        = gx;
            int y_id        = gy;

            int l_id        = start_layer;
            const int max_l = d_render_params.combine_lists ? num_layers_input : l_id + 1;
            // const int max_l = l_id + 1;

            get_element_addresses(l_id, y_id, x_id);

            //    if (length == 0) continue;

            for (; l_id < max_l; l_id += 1, x_id /= 2, y_id /= 2)
            {
                // not first iteration (done before loop begin)
                if (l_id != start_layer) get_element_addresses(l_id, y_id, x_id);

                for (int elem_i = 0; elem_i < length; elem_i += WARP_SIZE)
                {
#    ifdef DEBUG_PRINT
                    __syncwarp();
                    printf("Load Addr and len (%i,%i) - \n", start_offset + elem_i + own_index_in_warp, length);
#    endif
                    const int local_memory_start_thread = threads_in_use;
                    const int local_memory_end_thread   = local_memory_start_thread + ELEMENTS_PER_PIXEL;

                    int remaining_elements = max(0, min(int(length) - elem_i, 32));

                    SortData2 local_loaded_element;

                    // each thread loads one element
                    if (own_index_in_warp < remaining_elements)
                    {
                        int fetch_index = start_offset + elem_i + own_index_in_warp;
                        int2 data_comb  = ((int2*)(full_list_buffer_data_ptr + offset_batch + fetch_index))[0];
                        local_loaded_element =
                            SortData2(reinterpret_cast<const float*>(&data_comb.y)[0], data_comb.x, l_id);

#    ifdef DEBUG_PRINT
                        printf("Loaded (%f,%i) - \n", local_loaded_element.depth, local_loaded_element.index);
#    endif
                    }
                    else
                    {
                        local_loaded_element = {max_depth, int(FAILURE_VAL_LIST), int(0)};
                    }

                    // early continue if all elements are further away then current list
                    {
                        const int last_elem_warp_index = threads_in_use + (WARP_LEADER_STRIDE - 1);
                        const float last_depth_in_local_list =
                            __shfl_sync(0xffffffff, local_mem_warp.depth, last_elem_warp_index);
                        if (elem_i != 0 &&
                            __all_sync(0xffffffff, local_loaded_element.depth > last_depth_in_local_list))
                            continue;
                    }
#    ifdef DEBUG_PRINT
                    __syncwarp();
                    CUDA_KERNEL_ASSERT(local_loaded_element.depth > 0);
#    endif

                    bool first = false;  //(elem_i == 0 && start_layer == l_id);

                    // first sorting: thread 0-15: new elements, 16-31 list stored, sort descending
                    SortData2 local_sort_storage;
                    if (first)
                    {
                        local_sort_storage = local_loaded_element;
                    }
                    else
                    {
                        // fetch local memory to latter part of list, then fill with lower part of list
                        local_sort_storage = local_mem_warp;
                        if (threads_in_use == 0)
                        {
                            local_sort_storage.depth = __shfl_xor_sync(0xffffffff, local_sort_storage.depth, 16);
                            local_sort_storage.index_layer =
                                __shfl_xor_sync(0xffffffff, local_sort_storage.index_layer, 16);
                        }
                        if (own_index_in_warp < ELEMENTS_PER_PIXEL)
                        {
                            local_sort_storage = local_loaded_element;
                        }
                    }

#    ifdef DEBUG_PRINT
                    __syncwarp();
                    printf("Init Half (%f,%i) - \n", local_sort_storage.depth, local_sort_storage.index);

                    CUDA_KERNEL_ASSERT(local_sort_storage.depth > 0);
                    __syncwarp();
#    endif

                    // first sort: elem 0-15 new, elem 16-31 local mem
                    {
                        float2 data_f2;
                        ((SortData2*)&data_f2)[0] = local_sort_storage;
                        data_f2                   = Saiga::CUDA::bitonicWarpSort(data_f2, own_index_in_warp);
                        local_sort_storage        = ((SortData2*)&data_f2)[0];
                    }
#    ifdef DEBUG_PRINT
                    printf("Sorted1 (%f,%i) - \n", local_sort_storage.depth, local_sort_storage.index);

                    CUDA_KERNEL_ASSERT(local_sort_storage.depth > 0);

                    // local_sort_storage = reinterpret_cast<SortData*>(&sorted)[0];

                    __syncwarp();
#    endif
                    // second sort only if enought elements remaining and not first iter
                    if (!first && remaining_elements > ELEMENTS_PER_PIXEL)
                    // if (remaining_elements > ELEMENTS_PER_PIXEL)
                    {
#    ifdef DEBUG_PRINT
                        CUDA_KERNEL_ASSERT(local_sort_storage.depth > 0);

                        __syncwarp();
                        printf("XORed (%f,%i) - \n", local_sort_storage.depth, local_sort_storage.index);
#    endif

                        // remaining elements
                        if (own_index_in_warp >= ELEMENTS_PER_PIXEL) local_sort_storage = local_loaded_element;
#    ifdef DEBUG_PRINT
                        __syncwarp();
                        printf("OtherHalf (%f,%i) - \n", local_sort_storage.depth, local_sort_storage.index);


                        CUDA_KERNEL_ASSERT(local_sort_storage.depth > 0);
#    endif
                        // second sort: elem 0-15 local mem, elem 16-31 new
                        {
                            float2 data2_f2;
                            ((SortData2*)&data2_f2)[0] = local_sort_storage;
                            data2_f2                   = Saiga::CUDA::bitonicWarpSort(data2_f2, own_index_in_warp);
                            local_sort_storage         = ((SortData2*)&data2_f2)[0];
                        }
                    }
#    ifdef DEBUG_PRINT
                    __syncwarp();
                    CUDA_KERNEL_ASSERT(local_sort_storage.depth > 0);
                    printf("Sorted2 (%f,%i) - \n", local_sort_storage.depth, local_sort_storage.index);
#    endif

                    if (threads_in_use > 0)
                    {
                        // move result to correct
                        local_sort_storage.depth =
                            __shfl_xor_sync(0xffffffff, local_sort_storage.depth, threads_in_use);
                        local_sort_storage.index_layer =
                            __shfl_xor_sync(0xffffffff, local_sort_storage.index_layer, threads_in_use);
#    ifdef DEBUG_PRINT
                        __syncwarp();
                        CUDA_KERNEL_ASSERT(local_sort_storage.depth > 0);
                        printf("Leaderthread xor (%f,%i) - \n", local_sort_storage.depth, local_sort_storage.index);
#    endif
                    }

                    // write nearest 16 elements back
                    if (own_index_in_warp >= local_memory_start_thread && own_index_in_warp < local_memory_end_thread)
                        local_mem_warp = local_sort_storage;
#    ifdef DEBUG_PRINT
                    __syncwarp();
                    printf("LocalMem (%f,%i) - \n", local_mem_warp.depth, local_mem_warp.index);


                    //    break;
#    endif
                    // first = false;
                }
                // l_id += 1;
                // x_id /= 2;
                // y_id /= 2;
                //   if (l_id < max_layer) get_element_addresses(l_id, y_id, x_id);
                //   if (l_id < 1) get_element_addresses(l_id, y_id, x_id);
            }
        }
        __syncwarp();
        // debug test:
#    if 0
        {
            float run_elem = local_mem_warp.depth;
            if (own_index_in_warp == 0)
            {
                //   printf("(%f,%i) - ", local_mem_warp.depth, local_mem_warp.index);
                CUDA_KERNEL_ASSERT(run_elem != max_depth);
            }

            for (int i = 1; i < 16; ++i)
            {
                float other_depth = __shfl_sync(0xffffffff, local_mem_warp.depth, i);
                int other_index   = __shfl_sync(0xffffffff, local_mem_warp.index, i);
                if (own_index_in_warp == 0)
                {
                    // printf("(%f,%i) - ", other_depth, other_index);
                    if (!(run_elem <= other_depth))
                    {
                        printf("%f - %f\n", run_elem, other_depth);
                    }
                    //  CUDA_KERNEL_ASSERT(other_depth != max_depth);
                    CUDA_KERNEL_ASSERT(run_elem <= other_depth);
                }
                run_elem = other_depth;
            }
        }
#        ifdef DEBUG_PRINT
        __syncwarp();
        printf("-------- \n");
#        endif
#    endif

        {
            const bool is_leader_thread =
                own_index_in_warp % WARP_LEADER_STRIDE == 0 && own_index_in_warp < threads_in_use;

            // accumulate
            float alpha_dest = 1.f;
            // float color_out[4] = {0.f, 0.f, 0.f, 0.f};
            // float4 color_out = make_float4(0.f, 0.f, 0.f, 0.f);
            float color_out[NUM_DESCRIPTORS];
            for (int i = 0; i < NUM_DESCRIPTORS; ++i) color_out[i] = 0.f;

            if (own_index_in_warp < threads_in_use)  // disregard half threads with last element
            {
                const int local_leader_thread = (own_index_in_warp / WARP_LEADER_STRIDE) * WARP_LEADER_STRIDE;
                unsigned mask_sync            = 0xffffffff;
                if (threads_in_use < WARP_SIZE) mask_sync = (1 << threads_in_use) - 1;

                // get relevent values from leader
                gx     = __shfl_sync(mask_sync, gx, local_leader_thread);
                gy     = __shfl_sync(mask_sync, gy, local_leader_thread);
                layer  = __shfl_sync(mask_sync, layer, local_leader_thread);
                length = __shfl_sync(mask_sync, leader_length, local_leader_thread);

                unsigned ballot_sync_mask = 0;
                // if ((own_index_in_warp - local_leader_thread) < length)
                ballot_sync_mask = __ballot_sync(mask_sync, (own_index_in_warp - local_leader_thread) < length);

                // float4 color = make_float4(0, 0, 0, 0);

                if (own_index_in_warp - local_leader_thread < length)
                {
                    mask_sync = ballot_sync_mask;

                    __syncwarp(mask_sync);

                    int fetch_idx = local_mem_warp.get_index();
                    //     printf("LocalMem (%f,%i) fid(%i) - \n", local_mem_warp.depth,
                    //     local_mem_warp.index.fetch_idx);

                    CUDA_KERNEL_ASSERT(fetch_idx != FAILURE_VAL_LIST);

                    float ipx, ipy;
                    int texture_index;
                    float point_size_opt = 1.f;

                    if (!train)
                    {
                        const float4 data_fetch = ((float4*)&full_list_buffer_data(batch, fetch_idx, 0))[0];
                        ipx                     = data_fetch.x;
                        ipy                     = data_fetch.y;
                        texture_index           = reinterpret_cast<const int*>(&data_fetch.z)[0];
                        if (d_render_params.use_layer_point_size) point_size_opt = data_fetch.w;
                    }
                    else
                    {
                        ipx           = full_list_buffer_data(batch, fetch_idx, 0);
                        ipy           = full_list_buffer_data(batch, fetch_idx, 1);
                        texture_index = reinterpret_cast<const int*>(&full_list_buffer_data(batch, fetch_idx, 2))[0];
                        if (d_render_params.use_layer_point_size)
                            point_size_opt = full_list_buffer_data(batch, fetch_idx, 3);
                    }

                    const float layer_mult_fac = 1.f / float(1 << layer);
                    vec2 ip                    = vec2(ipx * layer_mult_fac, ipy * layer_mult_fac);
                    vec4 blend_vec             = compute_blending_fac(ip);
                    int blend_index            = blend_fac_index(ip, vec2(gx, gy));
                    float bilinear_fac         = blend_vec[blend_index];

                    // for lower layers: if the point is too far away discard the contribution by setting alpha to
                    // zero
                    if (abs(ip.x() - gx) > 1 || abs(ip.y() - gy) > 1) bilinear_fac = 0.f;

                    float alpha_bilin = bilinear_fac * d_texture.points_confidence_value(0, texture_index);

                    if (d_render_params.use_layer_point_size)
                    {
                        float layer_factor = compute_point_size_fac(point_size_opt, layer, num_layers_input);
                        alpha_bilin *= layer_factor;
                    }

                    // float color0 = d_texture.in_texture(0, texture_index);
                    // float color1 = d_texture.in_texture(1, texture_index);
                    // float color2 = d_texture.in_texture(2, texture_index);
                    // float color3 = d_texture.in_texture(3, texture_index);

#    define ALPHA_DEST_CUTOFF 0.001f

                    if (d_render_params.saturated_alpha_accumulation)
                    {
                        // color = make_float4(color0, color1, color2, color3) * alpha_bilin;


                        float alpha_dest_precomputed = 1.f;
                        alpha_dest                   = alpha_dest_precomputed;
                        for (int index_in_list = 1; index_in_list < ELEMENTS_PER_PIXEL; ++index_in_list)
                        {
                            alpha_dest_precomputed =
                                __shfl_up_sync(mask_sync, alpha_dest_precomputed - alpha_bilin, 1, 16);
                            if (own_index_in_warp % ELEMENTS_PER_PIXEL == index_in_list)
                                alpha_dest = (alpha_dest_precomputed >= ALPHA_DEST_CUTOFF)
                                                 ? (((alpha_dest_precomputed - alpha_bilin) > 0.f)
                                                        ? 1.f
                                                        : __saturatef(alpha_dest) / (alpha_bilin + 1e-9))
                                                 : 0.f;
                        }

                        // color *= alpha_dest;
                        for (int i = 0; i < NUM_DESCRIPTORS; ++i)
                        {
                            color_out[i] = alpha_dest * alpha_bilin * d_texture.in_texture(i, texture_index);
                        }
                    }
                    else
                    {
                        //  color = make_float4(color0, color1, color2, color3) * alpha_bilin;

                        // normal blend
                        float alpha_dest_precomputed = 1.f;
                        alpha_dest                   = alpha_dest_precomputed;
                        for (int index_in_list = 1; index_in_list < ELEMENTS_PER_PIXEL; ++index_in_list)
                        {
                            alpha_dest_precomputed =
                                __shfl_up_sync(mask_sync, alpha_dest_precomputed * (1 - alpha_bilin), 1, 16);
                            if (own_index_in_warp % ELEMENTS_PER_PIXEL == index_in_list)
                                alpha_dest =
                                    (alpha_dest_precomputed >= ALPHA_DEST_CUTOFF) ? alpha_dest_precomputed : 0.f;
                        }
                        for (int i = 0; i < NUM_DESCRIPTORS; ++i)
                        {
                            color_out[i] = alpha_dest * alpha_bilin * d_texture.in_texture(i, texture_index);
                        }
                        // color *= alpha_dest;
                    }
#    if 1
                    if (train)
                    {
                        if (alpha_dest >= ALPHA_DEST_CUTOFF)
                        {
                            // write out intermediates for backwards
                            // texture id, point id, alpha_val, vec2 subpixel_pos, int blendindex, point_id
                            int index_in_list = own_index_in_warp % ELEMENTS_PER_PIXEL;
                            d_bilinear_alpha_params.bw_sorted_maxed[layer](batch, gy, gx, index_in_list, 0) =
                                reinterpret_cast<const float*>(&texture_index)[0];
                            d_bilinear_alpha_params.bw_sorted_maxed[layer](batch, gy, gx, index_in_list, 1) =
                                full_list_buffer_data(batch, fetch_idx, 4);
                            d_bilinear_alpha_params.bw_sorted_maxed[layer](batch, gy, gx, index_in_list, 2) =
                                alpha_bilin;
                            d_bilinear_alpha_params.bw_sorted_maxed[layer](batch, gy, gx, index_in_list, 3) = ip.x();
                            d_bilinear_alpha_params.bw_sorted_maxed[layer](batch, gy, gx, index_in_list, 4) = ip.y();
                            d_bilinear_alpha_params.bw_sorted_maxed[layer](batch, gy, gx, index_in_list, 5) =
                                reinterpret_cast<const float*>(&blend_index)[0];
                            if (d_render_params.use_layer_point_size)
                                d_bilinear_alpha_params.bw_sorted_maxed[layer](batch, gy, gx, index_in_list, 6) =
                                    point_size_opt;
                        }
                    }
#    endif
                }

                for (int offset = 1; offset < ELEMENTS_PER_PIXEL; offset *= 2)
                {
                    for (int i = 0; i < NUM_DESCRIPTORS; ++i)
                    {
                        // color_out[i] = alpha_dest * alpha_bilin * d_texture.in_texture(i, texture_index);
                        color_out[i] += __shfl_xor_sync(mask_sync, color_out[i], offset, 16);
                    }
                    // color.x += __shfl_xor_sync(mask_sync, color.x, offset, 16);
                    // color.y += __shfl_xor_sync(mask_sync, color.y, offset, 16);
                    // color.z += __shfl_xor_sync(mask_sync, color.z, offset, 16);
                    // color.w += __shfl_xor_sync(mask_sync, color.w, offset, 16);
                }

                // if (is_leader_thread)
                //{
                //     // alpha_dest = alpha_dest_precomputed;
                //     color_out = color;
                // }

                alpha_dest =
                    __shfl_sync(mask_sync, alpha_dest, local_leader_thread + min(length, ELEMENTS_PER_PIXEL) - 1);
            }

            if (is_leader_thread)
            {
                if (alpha_dest >= ALPHA_DEST_CUTOFF)
                {
                    if (!environment_map)
                    {
                        // float4 bg_col = make_float4(background_color[0], background_color[1],
                        // background_color[2],
                        //                             background_color[3]);
                        //// color_out[ci] = compute_blend(alpha_dest, 1.f, background_color[ci], color_out[ci]);
                        // color_out += bg_col * (alpha_dest * 1.f);
                        for (int i = 0; i < NUM_DESCRIPTORS; ++i)
                        {
                            color_out[i] += background_color[i] * (alpha_dest * 1.f);
                        }
                    }
                }
                if (train)
                {
                    // d_forward_params.neural_out[layer](batch, 0, gy, gx) = color_out.x;
                    // d_forward_params.neural_out[layer](batch, 1, gy, gx) = color_out.y;
                    // d_forward_params.neural_out[layer](batch, 2, gy, gx) = color_out.z;
                    // d_forward_params.neural_out[layer](batch, 3, gy, gx) = color_out.w;
                    for (int i = 0; i < NUM_DESCRIPTORS; ++i)
                    {
                        d_forward_params.neural_out[layer](batch, i, gy, gx) = color_out[i];
                    }
                }
                else
                {
                    //((float4*)&d_forward_params.neural_out[layer](batch, gy, gx, 0))[0] = color_out;
                    //// d_forward_params.neural_out[layer](batch, gy, gx, 0) = color_out.x;
                    //// d_forward_params.neural_out[layer](batch, gy, gx, 1) = color_out.y;
                    //// d_forward_params.neural_out[layer](batch, gy, gx, 2) = color_out.z;
                    //// d_forward_params.neural_out[layer](batch, gy, gx, 3) = color_out.w;
                    for (int i = 0; i < NUM_DESCRIPTORS; ++i)
                    {
                        // d_forward_params.neural_out[layer](batch, i, gy, gx) = color_out[i];
                        d_forward_params.neural_out[layer](batch, gy, gx, i) = color_out[i];
                    }
                }
            }
        }
        get_next_list();

#    ifdef DEBUG_PRINT
        static int counter_l = 0;
        counter_l++;
        if (counter_l > 5)
            while (1)
                ;
#    endif
    }
}



void PointRendererCache::FusedSortAndBlend2(int batch, torch::Tensor full_list_buffer,
                                            torch::Tensor full_list_buffer_data, torch::Tensor background_color,
                                            bool train, bool use_environment_map, int length_of_list,
                                            torch::Tensor layer_lengths, torch::Tensor indices_more_than_X,
                                            torch::Tensor indices_less_than_X, CUDA::CudaTimerSystem* timer_system)
{
    {
        float* background = background_color.data_ptr<float>();

        static int bx          = 128;  // iDivUp(layers_cuda[1].size.x(), 16);
        static int by          = 32;   // iDivUp(layers_cuda[1].size.y(), 16);
        static int bx_lower    = 16;   // iDivUp(layers_cuda[1].size.x(), 16);
        static int by_lower    = 8;    // iDivUp(layers_cuda[1].size.y(), 16);
        static int threaddim_x = MAX_THREADS_X;
        static int threaddim_y = MAX_THREADS_Y;
        //  for (int i = 0; i < info->num_layers; ++i)
        //      std::cout << batch << "(before): " << TensorInfo(output_forward[i].slice(0, batch, batch +
        //      1)) << std::endl;

        torch::Tensor atomic1;
        torch::Tensor atomic2;
        {
            SAIGA_OPTIONAL_TIME_MEASURE("Alloc ", timer_system);
            atomic1 = torch::zeros({1}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32));
            atomic2 = torch::zeros({1}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32));


            //    ImGui::InputInt("bx ", &bx);
            //    ImGui::SameLine();
            //    ImGui::InputInt("by ", &by);
            //    ImGui::InputInt("bx_lower ", &bx_lower);
            //    ImGui::SameLine();
            //    ImGui::InputInt("by_lower", &by_lower);
            //    ImGui::InputInt("tx ", &threaddim_x);
            //    ImGui::SameLine();
            //    ImGui::InputInt("ty ", &threaddim_y);
        }
        int max_n = 0;
        for (int i = 0; i < info->num_layers; ++i)
        {
            max_n += layers_cuda[i].size.x() * layers_cuda[i].size.y();
        }
        max_n = length_of_list;

        static bool compare_org = false;
        //   ImGui::Checkbox("compare_org", &compare_org);
        {
            // int batch, StaticDeviceTensor<int, 1> max_elements /*(1)*/,
            // StaticDeviceTensor<int32_t, 2> padded_lists /*(batch, w*h*w/2*h/2 ... , 2)*/,
            // ConstReadOnlyStaticDeviceTensor<double, 3> full_list_buffer /* (batch,pointcloudsize,1)*/,
            // const double* __restrict__ full_list_buffer_data_ptr,
            // ConstReadOnlyStaticDeviceTensor<float, 3> full_list_buffer_data /*(batch,pointcloudsize,4)*/,
            // const float* __restrict__ full_list_buffer_data_data_ptr,
            // const float* __restrict__ background_color, StaticDeviceTensor<int, 1> glob_atomic_c2,
            // bool environment_map
#if 0
            {
                SAIGA_OPTIONAL_TIME_MEASURE("FastSortAndBlendLess", timer_system);

                if (train)
                {
                    // cudaFuncSetAttribute(&FastSortAndBlendWPrep2<16, 16, max_layers, true>,
                    //                      cudaFuncAttributePreferredSharedMemoryCarveout,
                    //                      cudaSharedmemCarveoutMaxL1);
                    FastSortAndBlendWPrep2<16, 16, max_layers, true>
                        <<<dim3(bx_lower, by_lower, 1), dim3(MAX_THREADS_X, MAX_THREADS_Y, 1)>>>(
                            batch, full_list_buffer, full_list_buffer.data_ptr<double>(), full_list_buffer_data,
                            full_list_buffer_data.data_ptr<float>(), background, atomic1, layer_lengths,
                            use_environment_map, max_n, indices_less_than_X);
                }
                else
                {
                    // cudaFuncSetAttribute(&FastSortAndBlendWPrep2<16, 16, max_layers, false>,
                    //                      cudaFuncAttributePreferredSharedMemoryCarveout,
                    //                      cudaSharedmemCarveoutMaxL1);
                    FastSortAndBlendWPrep2<16, 16, max_layers, false>
                        <<<dim3(bx_lower, by_lower, 1), dim3(MAX_THREADS_X, MAX_THREADS_Y, 1)>>>(
                            batch, full_list_buffer, full_list_buffer.data_ptr<double>(), full_list_buffer_data,
                            full_list_buffer_data.data_ptr<float>(), background, atomic1, layer_lengths,
                            use_environment_map, max_n, indices_less_than_X);
                }
                CUDA_SYNC_CHECK_ERROR();
            }
#endif
            //   for (int i = 0; i < info->num_layers; ++i)
            if (compare_org)
            {
                SAIGA_OPTIONAL_TIME_MEASURE("FastSortAndBlendOLD", timer_system);

                if (train)
                {
                    // cudaFuncSetAttribute(&FastSortAndBlendWPrep2<16, 16, max_layers, true>,
                    //                      cudaFuncAttributePreferredSharedMemoryCarveout,
                    //                      cudaSharedmemCarveoutMaxL1);
                    FastSortAndBlendWPrep2<16, 16, max_layers, true>
                        <<<dim3(bx, by, 1), dim3(MAX_THREADS_X, MAX_THREADS_Y, 1)>>>(
                            batch, full_list_buffer, full_list_buffer.data_ptr<double>(), full_list_buffer_data,
                            full_list_buffer_data.data_ptr<float>(), background, atomic2, layer_lengths,
                            use_environment_map, max_n, indices_more_than_X);
                }
                else
                {
                    // cudaFuncSetAttribute(&FastSortAndBlendWPrep2<16, 16, max_layers, false>,
                    //                      cudaFuncAttributePreferredSharedMemoryCarveout,
                    //                      cudaSharedmemCarveoutMaxL1);
                    FastSortAndBlendWPrep2<16, 16, max_layers, false>
                        <<<dim3(bx, by, 1), dim3(MAX_THREADS_X, MAX_THREADS_Y, 1)>>>(
                            batch, full_list_buffer, full_list_buffer.data_ptr<double>(), full_list_buffer_data,
                            full_list_buffer_data.data_ptr<float>(), background, atomic2, layer_lengths,
                            use_environment_map, max_n, indices_more_than_X);
                }
                CUDA_SYNC_CHECK_ERROR();
            }
            else
            {
                SAIGA_OPTIONAL_TIME_MEASURE("FastSortAndBlendMore", timer_system);

                if (train)
                {
                    switch (info->params.num_texture_channels)
                    {
                        case 3:
                        {
                            FastSortAndBlendWPrep2WarpSharedWork<3, 16, true>
                                <<<dim3(bx, by, 1), dim3(THREAD_DIMX_PER_BLOCK_MULTITHREADLOAD,
                                                         THREAD_DIMY_PER_BLOCK_MULTITHREADLOAD, 1)>>>(
                                    batch, full_list_buffer, full_list_buffer.data_ptr<double>(), full_list_buffer_data,
                                    full_list_buffer_data.data_ptr<float>(), background, atomic2, layer_lengths,
                                    use_environment_map, max_n, indices_more_than_X, info->num_layers);
                            break;
                        }
                        case 4:
                        {
                            FastSortAndBlendWPrep2WarpSharedWork<4, 16, true>
                                <<<dim3(bx, by, 1), dim3(THREAD_DIMX_PER_BLOCK_MULTITHREADLOAD,
                                                         THREAD_DIMY_PER_BLOCK_MULTITHREADLOAD, 1)>>>(
                                    batch, full_list_buffer, full_list_buffer.data_ptr<double>(), full_list_buffer_data,
                                    full_list_buffer_data.data_ptr<float>(), background, atomic2, layer_lengths,
                                    use_environment_map, max_n, indices_more_than_X, info->num_layers);
                            break;
                        }
                        case 8:
                        {
                            FastSortAndBlendWPrep2WarpSharedWork<8, 16, true>
                                <<<dim3(bx, by, 1), dim3(THREAD_DIMX_PER_BLOCK_MULTITHREADLOAD,
                                                         THREAD_DIMY_PER_BLOCK_MULTITHREADLOAD, 1)>>>(
                                    batch, full_list_buffer, full_list_buffer.data_ptr<double>(), full_list_buffer_data,
                                    full_list_buffer_data.data_ptr<float>(), background, atomic2, layer_lengths,
                                    use_environment_map, max_n, indices_more_than_X, info->num_layers);
                            break;
                        }
                        case 16:
                        {
                            FastSortAndBlendWPrep2WarpSharedWork<16, 16, true>
                                <<<dim3(bx, by, 1), dim3(THREAD_DIMX_PER_BLOCK_MULTITHREADLOAD,
                                                         THREAD_DIMY_PER_BLOCK_MULTITHREADLOAD, 1)>>>(
                                    batch, full_list_buffer, full_list_buffer.data_ptr<double>(), full_list_buffer_data,
                                    full_list_buffer_data.data_ptr<float>(), background, atomic2, layer_lengths,
                                    use_environment_map, max_n, indices_more_than_X, info->num_layers);
                            break;
                        }
                        default:
                            SAIGA_ASSERT(false, "NOT IMPLEMENTED");
                    };
                }
                else
                {
                    switch (info->params.num_texture_channels)
                    {
                        case 3:
                        {
                            FastSortAndBlendWPrep2WarpSharedWork<3, 16, false>
                                <<<dim3(bx, by, 1), dim3(THREAD_DIMX_PER_BLOCK_MULTITHREADLOAD,
                                                         THREAD_DIMY_PER_BLOCK_MULTITHREADLOAD, 1)>>>(
                                    batch, full_list_buffer, full_list_buffer.data_ptr<double>(), full_list_buffer_data,
                                    full_list_buffer_data.data_ptr<float>(), background, atomic2, layer_lengths,
                                    use_environment_map, max_n, indices_more_than_X, info->num_layers);
                            break;
                        }
                        case 4:
                        {
                            FastSortAndBlendWPrep2WarpSharedWork<4, 16, false>
                                <<<dim3(bx, by, 1), dim3(THREAD_DIMX_PER_BLOCK_MULTITHREADLOAD,
                                                         THREAD_DIMY_PER_BLOCK_MULTITHREADLOAD, 1)>>>(
                                    batch, full_list_buffer, full_list_buffer.data_ptr<double>(), full_list_buffer_data,
                                    full_list_buffer_data.data_ptr<float>(), background, atomic2, layer_lengths,
                                    use_environment_map, max_n, indices_more_than_X, info->num_layers);
                            break;
                        }
                        case 8:
                        {
                            FastSortAndBlendWPrep2WarpSharedWork<8, 16, false>
                                <<<dim3(bx, by, 1), dim3(THREAD_DIMX_PER_BLOCK_MULTITHREADLOAD,
                                                         THREAD_DIMY_PER_BLOCK_MULTITHREADLOAD, 1)>>>(
                                    batch, full_list_buffer, full_list_buffer.data_ptr<double>(), full_list_buffer_data,
                                    full_list_buffer_data.data_ptr<float>(), background, atomic2, layer_lengths,
                                    use_environment_map, max_n, indices_more_than_X, info->num_layers);
                            break;
                        }
                        case 16:
                        {
                            FastSortAndBlendWPrep2WarpSharedWork<16, 16, false>
                                <<<dim3(bx, by, 1), dim3(THREAD_DIMX_PER_BLOCK_MULTITHREADLOAD,
                                                         THREAD_DIMY_PER_BLOCK_MULTITHREADLOAD, 1)>>>(
                                    batch, full_list_buffer, full_list_buffer.data_ptr<double>(), full_list_buffer_data,
                                    full_list_buffer_data.data_ptr<float>(), background, atomic2, layer_lengths,
                                    use_environment_map, max_n, indices_more_than_X, info->num_layers);
                            break;
                        }
                    }
                }
                CUDA_SYNC_CHECK_ERROR();
            }
        }

        CUDA_SYNC_CHECK_ERROR();
        //  for (int i = 0; i < info->num_layers; ++i)
        //      std::cout << batch << "(after): " << TensorInfo(output_forward[i].slice(0, batch, batch +
        //      1)) << std::endl;
    }
}
