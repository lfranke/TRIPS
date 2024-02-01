/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/cuda/random.h"
#include "saiga/cuda/reduce.h"
#include "saiga/vision/cameraModel/OCam.h"
#include "saiga/vision/kernels/BA.h"
#include "saiga/vision/torch/CudaHelper.h"

#include "PointRenderer.h"
#include "PointRendererHelper.h"
#include "RenderConstants.h"
#include "config.h"
#include "data/Dataset.h"
//
// #include <ATen/ATen.h>
// #include <ATen/AccumulateType.h>
// #include <ATen/ExpandUtils.h>
// #include <ATen/cuda/detail/IndexUtils.cuh>

#include "cooperative_groups.h"
#include <curand_kernel.h>

#ifdef CUDA_DEBUG
#    define CUDA_DEBUG_ASSERT(_x) CUDA_KERNEL_ASSERT(_x)
#else
#    define CUDA_DEBUG_ASSERT(_x)
#endif



/////////////////////////////////////////////////////////////////////////////////////////
// Warp specific functions
#ifdef _CG_HAS_MATCH_COLLECTIVE

template <int GROUP_SIZE = 32>
__device__ cooperative_groups::coalesced_group subgroupPartitionNV(ivec2 p)
{
    using namespace cooperative_groups;
    thread_block block                   = this_thread_block();
    thread_block_tile<GROUP_SIZE> tile32 = tiled_partition<GROUP_SIZE>(block);

    coalesced_group g1 = labeled_partition(tile32, p(0));
    coalesced_group g2 = labeled_partition(tile32, p(1));

    details::_coalesced_group_data_access acc;
    return acc.construct_from_mask<coalesced_group>(acc.get_mask(g1) & acc.get_mask(g2));
}


template <typename T, int GROUP_SIZE = 32>
__device__ T subgroupPartitionedAddNV(T value, cooperative_groups::coalesced_group group)
{
    int s = group.size();
    int r = group.thread_rank();

    for (int offset = GROUP_SIZE / 2; offset > 0; offset /= 2)
    {
        auto v = group.template shfl_down(value, offset);
        if (r + offset < s) value += v;
    }
    return value;
}

template <typename T, int GROUP_SIZE = 32>
__device__ T subgroupPartitionedMinNV(T value, cooperative_groups::coalesced_group group)
{
    int s = group.size();
    int r = group.thread_rank();

    for (int offset = GROUP_SIZE / 2; offset > 0; offset /= 2)
    {
        auto v = group.template shfl_down(value, offset);
        if (r + offset < s) value = min(value, v);
    }
    return value;
}

#endif

__device__ inline int getGlobalThreadID()
{
    // https://forums.developer.nvidia.com/t/calculate-global-thread-id/23541
    int threadsPerBlock = blockDim.x * blockDim.y;
    int threadNumInBlock =
        threadIdx.x + blockDim.x * threadIdx.y;  // (alternatively: threadIdx.y + blockDim.y * threadIdx.x)
    int blockNumInGrid =
        blockIdx.x + gridDim.x * blockIdx.y;  //  (alternatively: blockIdx.y  + gridDim.y  * blockIdx.x)
    int globalThreadNum = blockNumInGrid * threadsPerBlock + threadNumInBlock;
    return globalThreadNum;
}



/////////////////////////////////////////////////////////////////////////////////////
// octahedron environment map span
HD inline Saiga::vec6 get_oct_factors(vec3 dir)
{
    vec6 oct_factors;
    // l1 norm on appearence vector
    float l1 = abs(dir(0)) + abs(dir(1)) + abs(dir(2));
    dir      = dir / l1;  // dir.lpNorm<1>();
    // project on octahedron
    oct_factors[0] = fmax(0.f, dir(0));
    oct_factors[1] = fmax(0.f, -dir(0));
    oct_factors[2] = fmax(0.f, dir(1));
    oct_factors[3] = fmax(0.f, -dir(1));
    oct_factors[4] = fmax(0.f, dir(2));
    oct_factors[5] = fmax(0.f, -dir(2));

    return oct_factors;
}

HD inline Saiga::vec3 get_reflection_dir(vec3 dir, vec3 normal)
{
    vec3 d = dir.normalized();
    vec3 n = normal.normalized();
    vec3 r = d - 2 * (d.dot(n)) * n;
    return r;
}

HD inline Saiga::mat3 getTBN(vec3 normal)
{
    vec3 n            = normal.normalized();
    vec3 bi_tangent_f = vec3(0, 1, 0);
    if (bi_tangent_f == n) bi_tangent_f = vec3(1, 0, 0);
    vec3 tangent   = (n.cross(bi_tangent_f)).normalized();
    vec3 bitangent = (n.cross(tangent)).normalized();

    mat3 TBN;  //= mat3(tangent, bitangent, normal);
    TBN.col(0) = tangent;
    TBN.col(1) = bitangent;
    TBN.col(2) = normal;
    //   TBN << tangent, bitangent, normal;
    return TBN;
}

HD inline Saiga::vec3 get_tangent_room_dir(vec3 dir, vec3 normal)
{
    mat3 tbn = getTBN(normal);
    vec3 res = tbn.inverse() * dir;
    return res;
}


/////////////////////////////////////////////////////////////////////////////////////
// pseudo random functions

// loosely following park-miller
HD inline int intrnd(int seed)  // 1<=seed<=m
{
    int const a = 16807;       // ie 7**5
    int const m = 2147483647;  // ie 2**31-1
    seed        = (long(seed * a)) % m;
    return seed;
}

HD inline float getRandForPointAndEpoch(int point_id, int epoch)
{
    int a1      = intrnd(point_id + 1);
    int a2      = intrnd(epoch + 1);
    int const m = 2147483647;  // ie 2**31-1
    int r       = intrnd((a1 * a2 * (point_id + 1) * (epoch + 1))) % m;
    return (float(r) / float(m)) * 0.5 + 0.5;
}

HD inline bool check_randomly_discard_point(int point_id, int epoch, float cutoff = 0.5)
{
    float r = getRandForPointAndEpoch(point_id, epoch);
    return (r > cutoff);
}



/////////////////////////////////////////////////////////////////////////////////////
// Projection functions

HD inline thrust::pair<vec2, float> ProjectPointPinhole(vec3 p, vec3 n, Sophus::SE3f V, IntrinsicsPinholef K,
                                                        Distortionf distortion, bool check_normal, float dist_cutoff)
{
    vec3 world_p = vec3(p(0), p(1), p(2));


    vec3 view_p = TransformPoint<float>(V, world_p);
    float z     = view_p.z();
    z           = fmax(z, 0.f);

    if (check_normal)
    {
        CUDA_KERNEL_ASSERT(isfinite(n(0)) & isfinite(n(1)) & isfinite(n(2)));
        vec3 world_n = vec3(n(0), n(1), n(2));
        vec3 view_n  = V.so3() * world_n;
        if (dot(view_p, view_n) > 0)
        {
            z = 0;
        }
    }

    vec2 norm_p = DivideByZ<float>(view_p);

    vec2 dist_p = distortNormalizedPoint<float>(norm_p, distortion, nullptr, nullptr, dist_cutoff);

    if (dist_p(0) == 100000)
    {
        z = 0;
    }

    vec2 image_p = K.normalizedToImage(dist_p, nullptr, nullptr);


    return {image_p, z};
}


HD inline thrust::pair<vec2, float> ProjectPointToOrthographic(vec3 p, Sophus::SE3f V, int w, int h, float dist_cutoff)
{
    vec3 world_p = vec3(p(0), p(1), p(2));
    vec3 view_p  = TransformPoint<float>(V, world_p);

    float z      = view_p.z();
    z            = fmax(z, 0.f);
    vec2 image_p = vec2(view_p.x() * w, view_p.y() * h);

    return {image_p, z};
}

HD inline thrust::pair<vec2, float> ProjectPointPinholeWoNormal(vec3 p, Sophus::SE3f V, IntrinsicsPinholef K,
                                                                Distortionf distortion, float dist_cutoff)
{
    vec3 world_p = vec3(p(0), p(1), p(2));


    vec3 view_p = TransformPoint<float>(V, world_p);
    float z     = view_p.z();
    z           = fmax(z, 0.f);

    vec2 norm_p = DivideByZ<float>(view_p);

    vec2 dist_p = distortNormalizedPoint<float>(norm_p, distortion, nullptr, nullptr, dist_cutoff);

    if (dist_p(0) == 100000)
    {
        z = 0;
    }

    vec2 image_p = K.normalizedToImage(dist_p, nullptr, nullptr);
    return {image_p, z};
}

HD inline vec3 ProjWorldToOCAM(vec3 p, Sophus::SE3f V, Vector<float, 5> a, ArrayView<const float> poly,
                               float dist_cutoff)
{
    vec3 world_p = vec3(p(0), p(1), p(2));

    vec3 view_p = TransformPoint<float>(V, world_p);
    vec3 ip_z   = ProjectOCam(view_p, a, poly, dist_cutoff);

    return ip_z;
}

HD inline thrust::pair<vec2, float> ProjectPointOcam(vec3 p, vec3 n, Sophus::SE3f V, Vector<float, 5> a,
                                                     ArrayView<const float> poly, bool check_normal, float dist_cutoff)
{
    CUDA_KERNEL_ASSERT(isfinite(n(0)) & isfinite(n(1)) & isfinite(n(2)));

    vec3 world_p = vec3(p(0), p(1), p(2));
    vec3 world_n = vec3(n(0), n(1), n(2));

    vec3 view_p = TransformPoint<float>(V, world_p);

    vec3 ip_z    = ProjectOCam(view_p, a, poly, dist_cutoff);
    vec2 image_p = ip_z.head<2>();
    float z      = ip_z(2);

    vec3 view_n = V.so3() * world_n;
    if (check_normal && (dot(view_p, view_n) > 0))
    {
        z = 0;
    }
    //    printf("%f, %f", image_p.x(), image_p.y());

    return {image_p, z};
}

HD inline thrust::pair<vec2, float> ProjectPointOcamWoNormal(vec3 p, Sophus::SE3f V, Vector<float, 5> a,
                                                             ArrayView<const float> poly, float dist_cutoff)
{
    vec3 world_p = vec3(p(0), p(1), p(2));

    vec3 view_p = TransformPoint<float>(V, world_p);

    vec3 ip_z    = ProjectOCam(view_p, a, poly, dist_cutoff);
    vec2 image_p = ip_z.head<2>();
    float z      = ip_z(2);

    return {image_p, z};
}


HD inline thrust::tuple<vec2, float, float> ProjectPointOcamWoNormalWPointsize(vec3 p, Sophus::SE3f V,
                                                                               Vector<float, 5> a,
                                                                               ArrayView<const float> poly,
                                                                               float dist_cutoff, float pointsize)
{
    vec3 world_p = vec3(p(0), p(1), p(2));

    vec3 view_p = TransformPoint<float>(V, world_p);

    vec4 ip_z_ps = ProjectOCamAndApproxSize(view_p, pointsize, a, poly, dist_cutoff);
    vec2 image_p = ip_z_ps.head<2>();
    float z      = ip_z_ps(2);


    return {image_p, z, ip_z_ps(3)};
}

// Equirectangular projection
HD inline vec3 ProjectSpherical(vec3 p)
{
    // z and y flipped from wiki

    float dist = p.norm();
    float phi  = atan2f(normalize(p)(2), normalize(p)(0)) / (M_PI);
    float chi  = -(acosf(normalize(p)(1)) - M_PI / 2);

    return vec3(1 - (phi * 0.5 + .5), chi * 0.5 + .5, dist);


    //  float r     = p.norm();
    //  float theta = acos(p(1) / r);
    //  // float phi   = acos(p(0) / sqrt(p(0) * p(0) + p(2) * p(2)));
    //  float phi = atan2f(normalize(p)(2), normalize(p)(0));
    //  // if (p(2) < 0) phi *= -1;
    //
    //  return vec3(phi / M_PI, theta / M_PI, r);

    // float dist    = p.norm();
    // float phi     = atan2f(normalize(p)(2), normalize(p)(0)) / (M_PI);
    // float chi     = -(acosf(normalize(p)(1)) - M_PI / 2) / 2;



    // float dist = p.norm();
    // return vec3(atan(p(1) / p(0)) / M_PI, acos(p(2) / dist) / M_PI, dist);

    // float dist  = p.norm();
    // float theta = acos(p(2) / dist) / M_PI;
    // float phi   = atan(p(1) / p(0)) / M_PI;
    //// float chi   = -(acosf(normalize(p)(1)) - M_PI / 2) / 2;
    //
    // return vec3(phi, theta, dist);
    //  return vec3(1 - (phi * 0.5 + .5), theta * 0.5 + .5, dist);


    // float theta = acos(p(2) / dist);
    // float phi   = atan(p(1) / p(0));
    // float y = 0.5 * log((1 + sinf(phi)) / (1 - sinf(phi)));
    // return vec3(theta, y, dist);



    // vec3 p_n = normalize(p);
    // float v1 = atanf(p_n.y() / p_n.x()) / M_PI;
    // float v2 = acosf(p_n.z()) - (M_PI / 2);
    // return vec3(v1 * 0.5 + 0.5, v2 * 0.5 + 0.5, dist);
}

HD inline thrust::pair<vec2, float> ProjectPointSpherical(vec3 p, vec3 n, Sophus::SE3f V, vec2 wh, bool check_normal,
                                                          float dist_cutoff)
{
    CUDA_KERNEL_ASSERT(isfinite(n(0)) & isfinite(n(1)) & isfinite(n(2)));

    vec3 world_p = vec3(p(0), p(1), p(2));
    vec3 world_n = vec3(n(0), n(1), n(2));

    vec3 view_p = TransformPoint<float>(V, world_p);

    vec3 ip_z    = ProjectSpherical(view_p);
    vec2 image_p = ip_z.head<2>();

    image_p.x() *= wh.x();
    image_p.y() *= wh.y();

    // image_p *= vec2(K.cx, K.cy);
    float z = ip_z(2);

    vec3 view_n = V.so3() * world_n;
    //   if (check_normal & dot(view_p, view_n) > 0)
    //   {
    //       z = 0;
    //   }


    return {image_p, z};
}
HD inline thrust::pair<vec2, float> ProjectPointSphericalWoNormal(vec3 p, Sophus::SE3f V, vec2 wh, float dist_cutoff)
{
    vec3 world_p = vec3(p(0), p(1), p(2));

    vec3 view_p = TransformPoint<float>(V, world_p);

    vec3 ip_z    = ProjectSpherical(view_p);
    vec2 image_p = ip_z.head<2>();

    image_p.x() *= wh.x();
    image_p.y() *= wh.y();

    // image_p *= vec2(K.cx, K.cy);
    float z = ip_z(2);


    return {image_p, z};
}



struct BackwardOutputPinhole
{
    vec3 g_point  = vec3::Zero();
    vec6 g_pose   = vec6::Zero();
    vec5 g_k      = vec5::Zero();
    vec8 g_dis    = vec8::Zero();
    float g_layer = 0;
};

// Backpropagates the image space gradient to the point position
//
// Return [gradient_point, gradient_pose]
HD inline BackwardOutputPinhole ProjectPointPinholeBackward(vec3 p, vec3 n, vec2 grad, Sophus::SE3f V,
                                                            IntrinsicsPinholef K, IntrinsicsPinholef crop_transform,
                                                            Distortionf distortion, bool check_normal,
                                                            float dist_cutoff, Matrix<float, 2, 2> crop_rot

)
{
    using T = float;


    vec3 world_p = vec3(p(0), p(1), p(2));

    Matrix<T, 3, 3> J_point;
    Matrix<T, 3, 6> J_pose;
    vec3 view_p = TransformPoint<float>(V, world_p, &J_pose, &J_point);
    float z     = view_p.z();
    CUDA_KERNEL_ASSERT(z > 0);
    if (z <= 0) return {};

    if (check_normal)
    {
        CUDA_KERNEL_ASSERT(isfinite(n(0)) & isfinite(n(1)) & isfinite(n(2)));
        vec3 world_n = vec3(n(0), n(1), n(2));
        vec3 view_n  = V.so3() * world_n;
        CUDA_KERNEL_ASSERT(dot(view_p, view_n) <= 0);
        if (dot(view_p, view_n) > 0) return {};
    }
    // forward: layer_factor = K.fx * cam.crop_transform.fx * layer_buf / z;
    // backward: dLdlayer_factor = K.fx * cam.crop_transform.fx / z;
    float g_layer = K.fx * crop_transform.fx / z;

    Matrix<T, 2, 3> J_p_div;
    vec2 norm_p = DivideByZ<float>(view_p, &J_p_div);

    Matrix<T, 2, 2> J_p_dis;
    Matrix<T, 2, 8> J_dis_dis;
    vec2 dist_p = distortNormalizedPoint<float>(norm_p, distortion, &J_p_dis, &J_dis_dis, dist_cutoff);

    Matrix<T, 2, 2> J_p_K1, J_p_K2;
    Matrix<T, 2, 5> J_k_K1;
    vec2 image_p = K.normalizedToImage(dist_p, &J_p_K1, &J_k_K1);
    image_p      = crop_transform.normalizedToImage(image_p, &J_p_K2, nullptr);
    //(void)image_p;
    // crop rotation gradient is forward rotation
    Matrix<T, 2, 2> J_rot = crop_rot;

    vec2 grad_p_rot = J_rot.transpose() * grad;

    vec2 grad_p_k2 = J_p_K2.transpose() * grad_p_rot;

    vec3 g_point =
        J_point.transpose() * (J_p_div.transpose() * (J_p_dis.transpose() * (J_p_K1.transpose() * grad_p_k2)));

    vec6 g_pose = J_pose.transpose() * (J_p_div.transpose() * (J_p_dis.transpose() * (J_p_K1.transpose() * grad_p_k2)));


    vec5 g_k   = J_k_K1.transpose() * grad_p_k2;
    vec8 g_dis = J_dis_dis.transpose() * (J_p_K1.transpose() * grad_p_k2);

    return {g_point, g_pose, g_k, g_dis, g_layer};
}



struct BackwardOutputOcam
{
    vec3 g_point  = vec3::Zero();
    vec6 g_pose   = vec6::Zero();
    vec5 g_affine = vec5::Zero();
};

// Backpropagates the image space gradient to the point position
//
// Return [gradient_point, gradient_pose]
HD inline BackwardOutputOcam ProjectPointOcamBackward(vec3 p, vec3 n, vec2 grad, Sophus::SE3f V,
                                                      IntrinsicsPinholef crop_transform, Vector<float, 5> a,
                                                      ArrayView<const float> poly, bool check_normal, float dist_cutoff,
                                                      Matrix<float, 2, 2> crop_rot)
{
    using T = float;


    vec3 world_p = vec3(p(0), p(1), p(2));
    vec3 world_n = vec3(n(0), n(1), n(2));

    Matrix<T, 3, 3> J_point;
    Matrix<T, 3, 6> J_pose;
    vec3 view_p = TransformPoint<float>(V, world_p, &J_pose, &J_point);


    if (check_normal)
    {
        vec3 view_n = V.so3() * world_n;
        if (!isfinite(n(0)) || (check_normal && dot(view_p, view_n) > 0))
        {
            printf("invalid normal %f %f %f \n", n(0), n(1), n(2));
        }
        if (dot(view_p, view_n) > 0)
        {
            CUDA_KERNEL_ASSERT(!check_normal || dot(view_p, view_n) <= 0);
            CUDA_KERNEL_ASSERT(isfinite(n(0)) & isfinite(n(1)) & isfinite(n(2)));
            return {};
        }
    }

    Matrix<T, 2, 3> J_p_ocam;
    Matrix<T, 2, 5> J_affine_ocam;
    vec3 ip_z    = ProjectOCam<T>(view_p, a, poly, dist_cutoff, &J_p_ocam, &J_affine_ocam);
    vec2 image_p = ip_z.head<2>();
    float z      = ip_z(2);

    // crop rotation gradient is forward rotation
    Matrix<T, 2, 2> J_rot = crop_rot;
    vec2 grad_p_rot       = J_rot.transpose() * grad;

    Matrix<T, 2, 2> J_p_crop;
    image_p = crop_transform.normalizedToImage(image_p, &J_p_crop, nullptr);


    vec2 grad_p_k2 = J_p_crop.transpose() * grad_p_rot;

    vec3 g_point  = J_point.transpose() * (J_p_ocam.transpose() * grad_p_k2);
    vec6 g_pose   = J_pose.transpose() * (J_p_ocam.transpose() * grad_p_k2);
    vec5 g_affine = J_affine_ocam.transpose() * grad_p_k2;

    return {g_point, g_pose, g_affine};
}



struct BackwardOutputOcamPointsize
{
    vec3 g_point      = vec3::Zero();
    vec6 g_pose       = vec6::Zero();
    vec5 g_affine     = vec5::Zero();
    float g_pointsize = 0;
};

// Backpropagates the image space gradient to the point position
//
// Return [gradient_point, gradient_pose, g_affine, g_ps]
HD inline BackwardOutputOcamPointsize ProjectPointOcamBackwardPointSize(
    vec3 p, vec3 n, float pointsize, vec2 grad, Sophus::SE3f V, IntrinsicsPinholef crop_transform, Vector<float, 5> a,
    ArrayView<const float> poly, bool check_normal, float dist_cutoff, Matrix<float, 2, 2> crop_rot)
{
    using T = float;


    vec3 world_p = vec3(p(0), p(1), p(2));
    vec3 world_n = vec3(n(0), n(1), n(2));

    Matrix<T, 3, 3> J_point;
    Matrix<T, 3, 6> J_pose;
    vec3 view_p = TransformPoint<float>(V, world_p, &J_pose, &J_point);


    if (check_normal)
    {
        vec3 view_n = V.so3() * world_n;
        if (!isfinite(n(0)) || (check_normal && dot(view_p, view_n) > 0))
        {
            printf("invalid normal %f %f %f \n", n(0), n(1), n(2));
        }
        if (dot(view_p, view_n) > 0)
        {
            CUDA_KERNEL_ASSERT(!check_normal || dot(view_p, view_n) <= 0);
            CUDA_KERNEL_ASSERT(isfinite(n(0)) & isfinite(n(1)) & isfinite(n(2)));
            return {};
        }
    }

    Matrix<T, 2, 3> J_p_ocam;
    Matrix<T, 2, 5> J_affine_ocam;
    Matrix<T, 1, 1> J_point_size;
    vec4 ip_z    = ProjectOCamAndApproxSize<T>(view_p, pointsize, a, poly, dist_cutoff, &J_p_ocam, &J_affine_ocam,
                                               &J_point_size);  //(view_p, pointsize, a, poly, dist_cutoff);
    vec2 image_p = ip_z.head<2>();
    float z      = ip_z(2);

    float grad_point_size = J_point_size(0, 0);

    // crop rotation gradient is forward rotation
    Matrix<T, 2, 2> J_rot = crop_rot;
    vec2 grad_p_rot       = J_rot.transpose() * grad;

    Matrix<T, 2, 2> J_p_crop;
    image_p = crop_transform.normalizedToImage(image_p, &J_p_crop, nullptr);


    vec2 grad_p_k2 = J_p_crop.transpose() * grad_p_rot;

    vec3 g_point  = J_point.transpose() * (J_p_ocam.transpose() * grad_p_k2);
    vec6 g_pose   = J_pose.transpose() * (J_p_ocam.transpose() * grad_p_k2);
    vec5 g_affine = J_affine_ocam.transpose() * grad_p_k2;

    return {g_point, g_pose, g_affine, grad_point_size};
}


/////////////////////////////////////////////////////////////////////////////////////
// various


// Blog: https://ai.googleblog.com/2019/08/turbo-improved-rainbow-colormap-for.html
// Code: https://gist.github.com/mikhailov-work/0d177465a8151eb6ede1768d51d476c7
__device__ inline vec3 colorizeTurbo(float x)
{
    const vec4 kRedVec4   = vec4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
    const vec4 kGreenVec4 = vec4(0.09140261, 2.19418839, 4.84296658, -14.18503333);
    const vec4 kBlueVec4  = vec4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
    const vec2 kRedVec2   = vec2(-152.94239396, 59.28637943);
    const vec2 kGreenVec2 = vec2(4.27729857, 2.82956604);
    const vec2 kBlueVec2  = vec2(-89.90310912, 27.34824973);

    x       = __saturatef(x);
    vec4 v4 = vec4(1.0, x, x * x, x * x * x);
    // vec2 v2 = v4.zw * v4.z;
    vec2 v2 = vec2(v4[2], v4[3]) * v4[2];
    return vec3(dot(v4, kRedVec4) + dot(v2, kRedVec2), dot(v4, kGreenVec4) + dot(v2, kGreenVec2),
                dot(v4, kBlueVec4) + dot(v2, kBlueVec2));
}



HD inline thrust::pair<float, uint32_t> ExtractIndex(Packtype i)
{
    uint32_t depthi = i >> 32;
    float depth     = reinterpret_cast<float*>(&depthi)[0];
    uint32_t index  = i & 0xFFFFFFFFUL;
    return {depth, index};
}


HD inline Packtype PackIndex(float depth, uint32_t index)
{
    uint32_t depthi = reinterpret_cast<uint32_t*>(&depth)[0];
    return (Packtype(depthi) << 32) | Packtype(index);
}


struct RenderImages
{
    ImageView<Packtype> depth_index[max_layers];
};


struct OutputImages
{
    ImageView<long> output[max_layers];
};

HD inline vec3 ProjWorldToPinhole(vec3 p, Sophus::SE3f V, IntrinsicsPinholef K, Distortionf distortion,
                                  float dist_cutoff)
{
    vec3 world_p = vec3(p(0), p(1), p(2));


    vec3 view_p  = TransformPoint<float>(V, world_p);
    float z      = view_p.z();
    z            = fmax(z, 0.f);
    vec2 image_p = vec2(0, 0);
    if (z != 0)
    {
        vec2 norm_p = DivideByZ<float>(view_p);

        vec2 dist_p = distortNormalizedPoint<float>(norm_p, distortion, nullptr, nullptr, dist_cutoff);
        if (dist_p(0) == 100000)
        {
            z = 0;
        }
        image_p = K.normalizedToImage(dist_p, nullptr, nullptr);
    }


    return vec3(image_p.x(), image_p.y(), z);
}
