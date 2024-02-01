/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


// #define CUDA_NDEBUG

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/util/easylogging++.h"
#include "saiga/cuda/cudaHelper.h"
#include "saiga/cuda/reduce_global.h"
#include "saiga/cuda/tests/test.h"
#include "saiga/cuda/tests/test_helper.h"
#include "saiga/cuda/thread_info.h"
#include "saiga/vision/torch/CudaHelper.h"

#include "EnvironmentMap.h"
#include "models/MyAdam.h"

std::ostream& operator<<(std::ostream& stream, const RayList& rays)
{
    stream << "RayList: lxbxhxw [" << rays.num_layers << "x" << rays.num_batches << "x" << rays.height << "x"
           << rays.width << "]; tensors (origin, direction, alpha_accum) " << TensorInfo(rays.origin) << " "
           << TensorInfo(rays.direction) << " " << TensorInfo(rays.alpha_dest_accumulated) << std::endl;
    return stream;
}

torch::Tensor rgbwMap(int channels, int num_images, int resolution)
{
    std::vector<torch::Tensor> col;
    for (int i = 0; i < channels; ++i)
    {
        auto t1 = torch::ones({1, channels, 1, resolution, resolution}, torch::TensorOptions(torch::kCUDA));
        if (i > 0)
        {
            t1.slice(1, 0, i) *= 0;
        }
        t1.slice(1, i, 4) *= 0;
        col.push_back(t1);
    }
    return torch::cat(col, 2).contiguous();
}

torch::Tensor arangeMap(int channels, int num_images, int resolution)
{
    std::vector<torch::Tensor> col;
    for (int i = 0; i < num_images; ++i)
    {
        auto t1 = torch::ones({1, channels, 1, resolution, resolution}, torch::TensorOptions(torch::kCUDA));
        std::vector<torch::Tensor> t;

        for (int j = 0; j < channels; ++j)
        {
            auto m1    = torch::linspace(0, 1, resolution);
            auto m2    = torch::linspace(0, 1, resolution);
            auto meshx = torch::matmul(m1.unsqueeze(1), m2.unsqueeze(0)).unsqueeze(0).unsqueeze(0).unsqueeze(0);
            t.push_back(meshx);
            PrintTensorInfo(meshx);
        }
        t1 = torch::cat(t, 1);

        if (i > 0)
        {
            t1.slice(1, 0, i) *= 0;
        }
        t1.slice(1, i, 4) *= 0;
        col.push_back(t1);
    }
    return torch::cat(col, 2).contiguous();
}

torch::Tensor rgbwRandMap(int channels, int num_images, int resolution)
{
    std::vector<torch::Tensor> col;
    for (int i = 0; i < num_images; ++i)
    {
        auto t1 = torch::rand({1, channels, 1, resolution, resolution}, torch::TensorOptions(torch::kCUDA));
        if (i > 0)
        {
            t1.slice(1, 0, i).set_(torch::zeros_like(t1.slice(1, 0, i)));
        }
        t1.slice(1, i, 4).set_(torch::zeros_like(t1.slice(1, i, 4)));
        t1 = t1.contiguous();
        col.push_back(t1);
    }
    return torch::cat(col, 2).contiguous();
}

#define MAX_NUM_SPHERES 8
__constant__ float d_radii[MAX_NUM_SPHERES];

EnvironmentMapImpl::EnvironmentMapImpl(int channels, int h, int w, bool log_texture, int up_axis, int num_images,
                                       float inner_radius, float radius_factor, bool non_subzero_texture)
    : channels(channels), up_axis(up_axis), non_subzero_texture(non_subzero_texture)
{
    SAIGA_ASSERT(num_images <= MAX_NUM_SPHERES, "currently not more spheres are supported");
    int resolution = std::max(h, w);
    density        = torch::full({1, 1, num_images, resolution, resolution}, 1.f, torch::TensorOptions(torch::kCUDA));
    color = torch::full({1, channels, num_images, resolution, resolution}, 1.f, torch::TensorOptions(torch::kCUDA));


    register_parameter("density", density);
    register_parameter("color", color);

    for (int i = 0; i < num_images; ++i)
    {
        radii.push_back(inner_radius * pow(radius_factor, i));
        std::cout << "Radius " << i << " " << radii.back() << std::endl;
    }
    CHECK_CUDA_ERROR(cudaMemcpyToSymbolAsync(d_radii, radii.data(), sizeof(float) * radii.size()));
}
void EnvironmentMapImpl::CreateEnvMapOptimizer(float env_col_learning_rate, float env_density_learning_rate)
{
    std::vector<torch::optim::OptimizerParamGroup> g;
    {
        auto opts = std::make_unique<torch::optim::AdamOptions>(env_col_learning_rate);
        std::vector<torch::Tensor> ts;
        ts.push_back(color);
        g.emplace_back(ts, std::move(opts));
    }
    {
        auto opts = std::make_unique<torch::optim::AdamOptions>(env_density_learning_rate);
        std::vector<torch::Tensor> ts;
        ts.push_back(density);
        g.emplace_back(ts, std::move(opts));
    }
    optimizer_adam = std::make_shared<torch::optim::Adam>(g, torch::optim::AdamOptions(1));
}



HD inline vec2 SphericalCoordinates(vec3 r)
{
    const float pi = 3.14159265358979323846;

    vec2 lookup_coord = vec2(r[0], r[1]);
    // Note, coordinate system is (mathematical [x,y,z]) => (here: [x,-z,y])
    // also Note, r is required to be normalized for theta.
    // spheric
    float phi = atan2(-r[2], r[0]);
    float x   = phi / (2.0 * pi);  // in [-.5,.5]
    x         = fract(x);  // uv-coord in [0,1]    // is not needed. just for convenience (but it changes seam-position)

    float theta = acos(r[1]);  // [1,-1] ->  [0,Pi]    // acos(r.y/length(r)), if r not normalized
    float y     = theta / pi;  // uv in [0,1]
    y           = 1 - y;       // texture-coordinate-y is flipped in opengl

    lookup_coord = vec2(x, y);

    return lookup_coord;
}

__global__ void BuildSphericalUV(StaticDeviceTensor<double, 2> poses, StaticDeviceTensor<float, 2> intrinsics,
                                 StaticDeviceTensor<float, 3> uvs_out, ReducedImageInfo cam)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    if (gx >= uvs_out.sizes[1] || gy >= uvs_out.sizes[0]) return;


    vec2 ip(gx, gy);



    Sophus::SE3f V = (((Sophus::SE3d*)(&poses(cam.image_index, 0)))[0]).cast<float>();

    float* ptr             = &intrinsics(cam.camera_index, 0);
    IntrinsicsPinholef K   = ((vec5*)ptr)[0];
    Distortionf distortion = ((vec8*)(ptr + 5))[0];


    vec2 dist_p = K.unproject2(cam.crop_transform.unproject2(ip));
    vec2 np     = undistortNormalizedPointSimple<float>(dist_p, distortion);

    vec3 inp(np(0), np(1), 1);

    vec3 wp = V.inverse() * inp;

    vec3 dir = (wp - V.inverse().translation()).normalized();

    // CV -> opengl
    // dir(1) *= -1;
    // dir(2) *= -1;

    vec2 uv = SphericalCoordinates(dir);

    uv = uv * 2 - vec2(1, 1);

    uvs_out(gy, gx, 0) = uv[0];
    uvs_out(gy, gx, 1) = uv[1];


    // uvs_out.At({gy, gx, 0}) = dir(0);
    // uvs_out.At({gy, gx, 1}) = dir(1);
}

__global__ void BuildSphericalRayList(StaticDeviceTensor<double, 2> poses, StaticDeviceTensor<float, 2> intrinsics,
                                      StaticDeviceTensor<float, 2> rays_origin,
                                      StaticDeviceTensor<float, 2> rays_direction, ReducedImageInfo cam,
                                      int layer_width, int layer_height, StaticDeviceTensor<float, 1> ocam_cam2world)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    if (gx >= layer_width || gy >= layer_height) return;
    int ray_index = gy * layer_width + gx;

    vec2 ip(gx, gy);

    Sophus::SE3f V = (((Sophus::SE3d*)(&poses(cam.image_index, 0)))[0]).cast<float>();

    auto inv_rot_p = [&](vec2 point, vec2 center)
    {
        point -= center;
        point = cam.crop_rotation.inverse() * point;
        point += center;
        return point;
    };
    vec3 wp    = vec3(0, 0, 0);
    vec3 dir   = vec3(0, 1, 0);
    float* ptr = &intrinsics(cam.camera_index, 0);
    if (cam.camera_model_type == CameraModel::PINHOLE_DISTORTION)
    {
        IntrinsicsPinholef K   = ((vec5*)ptr)[0];
        Distortionf distortion = ((vec8*)(ptr + 5))[0];

        ip = inv_rot_p(ip, vec2(layer_width / 2, layer_height / 2));

        vec2 dist_p = K.unproject2(cam.crop_transform.unproject2(ip));
        vec2 np     = undistortNormalizedPointSimple<float>(dist_p, distortion);

        vec3 inp(np(0), np(1), 1);
        wp  = V.inverse() * inp;
        dir = (wp - V.inverse().translation()).normalized();
    }
    else if (cam.camera_model_type == CameraModel::OCAM)
    {
        vec5 affine_params   = ((vec5*)ptr)[0];
        float* cam2world_ptr = ocam_cam2world.data;
        ArrayView<const float> poly(cam2world_ptr, ocam_cam2world.size(0));
        // CUDA_KERNEL_ASSERT(poly.size() == 7);
        ip = inv_rot_p(ip, vec2(layer_width / 2, layer_height / 2));

        vec2 ip_crop = cam.crop_transform.unproject2(ip);

        ArrayView av    = ocam_cam2world;
        vec3 ocam_undis = UnprojectOCam<float>(vec2(ip_crop.x(), ip_crop.y()), 1.f, affine_params, poly);

        if (ocam_undis.z() < 0)
        {
            wp  = vec3(0, 0, 0);
            dir = vec3(0, 0, 0);
        }
        else
        {
            wp  = V.inverse() * ocam_undis;
            dir = (wp - V.inverse().translation()).normalized();
        }
    }
    else
    {
        float phi = ((ip.x() / float(layer_width) - 0.5f) * 2.f) * M_PI;
        float chi = (((ip.y() / float(layer_height) - 0.5f) * 2.f) * -1.f + M_PI / 2);

        float x = cosf(phi);
        float z = sinf(phi);
        float y = cosf(chi);

        vec3 cartesian = vec3(x, y, z);  // vec3(cosf(phi) * sinf(chi), sinf(phi) * sinf(chi), cosf(chi));
        wp             = V.inverse() * cartesian;
        dir            = (wp - V.inverse().translation()).normalized();
    }
    for (int i = 0; i < 3; ++i)
    {
        rays_origin(ray_index, i)    = wp(i);
        rays_direction(ray_index, i) = dir(i);
    }
}


std::vector<torch::Tensor> EnvironmentMapImpl::Sample(torch::Tensor poses, torch::Tensor intrinsics,
                                                      ArrayView<ReducedImageInfo> info_batch, int num_layers,
                                                      std::shared_ptr<SceneData> scene,
                                                      std::vector<std::vector<torch::Tensor>> layers_cuda,
                                                      CUDA::CudaTimerSystem* timer_system)
{
    int num_batches = info_batch.size();
    std::vector<std::vector<RayList>> ray_lists(num_layers);
    {
        SAIGA_OPTIONAL_TIME_MEASURE("Allocate Ray List", timer_system);

        int h = info_batch.front().h;
        int w = info_batch.front().w;

        for (int i = 0; i < num_layers; ++i)
        {
            for (int b = 0; b < num_batches; ++b)
            {
                RayList rl = RayList();
                rl.Allocate(w * h, 3);
                ray_lists[i].push_back(rl);
            }
            h /= 2;
            w /= 2;
        }
    }
    {
        SAIGA_OPTIONAL_TIME_MEASURE("Create Rays", timer_system);

        int h       = info_batch.front().h;
        int w       = info_batch.front().w;
        float scale = 1;
        for (int i = 0; i < num_layers; ++i)
        {
            int bx = iDivUp(w, 16);
            int by = iDivUp(h, 16);

            for (int b = 0; b < num_batches; ++b)
            {
                auto cam                            = info_batch[b];
                torch::Tensor tensor_ocam_cam2world = torch::ones({1}).cuda();
                if (cam.camera_model_type == CameraModel::OCAM)
                {
                    int cam_index         = cam.camera_index;
                    auto ocam_cam2world   = scene->scene_cameras[cam_index].ocam.cast<float>().poly_cam2world;
                    tensor_ocam_cam2world = torch::from_blob(ocam_cam2world.data(), {(long)ocam_cam2world.size()},
                                                             torch::TensorOptions().dtype(torch::kFloat32))
                                                .clone()
                                                .cuda();
                }

                cam.crop_transform = cam.crop_transform.scale(scale);
                BuildSphericalRayList<<<dim3(bx, by, 1), dim3(16, 16, 1)>>>(poses, intrinsics, ray_lists[i][b].origin,
                                                                            ray_lists[i][b].direction, cam, w, h,
                                                                            tensor_ocam_cam2world);
            }

            scale /= 2;
            h /= 2;
            w /= 2;
        }
        CUDA_SYNC_CHECK_ERROR();
    }

    std::vector<torch::Tensor> result(num_layers);
    {
        SAIGA_OPTIONAL_TIME_MEASURE("Sample Map", timer_system);

        int h       = info_batch.front().h;
        int w       = info_batch.front().w;
        float scale = 1;
        for (int i = 0; i < num_layers; ++i)
        {
            //     SAIGA_OPTIONAL_TIME_MEASURE("Layer "+ std::to_string(i), timer_system);

            std::vector<torch::Tensor> batch_of_envs;
            for (int b = 0; b < num_batches; ++b)
            {
                //       SAIGA_OPTIONAL_TIME_MEASURE("Batch "+ std::to_string(b), timer_system);

                auto alpha_dest_layer = layers_cuda[i][b];

                auto col_env = forward_mps(ray_lists[i][b], alpha_dest_layer.reshape({-1}));
                col_env      = col_env.reshape({channels, h, w});
                batch_of_envs.push_back(col_env.unsqueeze(0));
            }
            h /= 2;
            w /= 2;
            result[i] = torch::cat(batch_of_envs, 0);
        }
    }

    return result;
}

std::vector<torch::Tensor> EnvironmentMapImpl::Sample2(torch::Tensor poses, torch::Tensor intrinsics,
                                                       ArrayView<ReducedImageInfo> info_batch, int num_layers,
                                                       std::shared_ptr<SceneData> scene,
                                                       std::vector<std::vector<torch::Tensor>> layers_cuda,
                                                       CUDA::CudaTimerSystem* timer_system)
{
    int num_batches  = info_batch.size();
    RayList ray_list = RayList();

    {
        SAIGA_OPTIONAL_TIME_MEASURE("Allocate Ray List", timer_system);

        int h = info_batch.front().h;
        int w = info_batch.front().w;
        ray_list.Allocate(w, h, num_batches, num_layers, 3);
    }
    {
        SAIGA_OPTIONAL_TIME_MEASURE("Create Rays", timer_system);

        int h       = info_batch.front().h;
        int w       = info_batch.front().w;
        float scale = 1;
        for (int i = 0; i < num_layers; ++i)
        {
            int bx = iDivUp(w, 16);
            int by = iDivUp(h, 16);

            for (int b = 0; b < num_batches; ++b)
            {
                auto cam = info_batch[b];

                torch::Tensor tensor_ocam_cam2world = torch::ones({1}).cuda();
                if (cam.camera_model_type == CameraModel::OCAM)
                {
                    int cam_index       = cam.camera_index;
                    auto ocam_cam2world = scene->scene_cameras[cam_index].ocam.cast<float>().poly_cam2world;

                    tensor_ocam_cam2world = torch::from_blob(ocam_cam2world.data(), {(long)ocam_cam2world.size()},
                                                             torch::TensorOptions().dtype(torch::kFloat32))
                                                .clone()
                                                .cuda();
                }

                cam.crop_transform = cam.crop_transform.scale(scale);

                auto alpha_dest_layer = layers_cuda[i][b];
                ray_list.getSubViewAlphaDest(b, i).copy_(alpha_dest_layer.reshape({-1, 1}));

                // ray_list.getSubViewAlphaDest(b, i) = (alpha_dest_layer.reshape({-1, 1}));
                BuildSphericalRayList<<<dim3(bx, by, 1), dim3(16, 16, 1)>>>(
                    poses, intrinsics, ray_list.getSubViewOrigin(b, i), ray_list.getSubViewDirection(b, i), cam, w, h,
                    tensor_ocam_cam2world);
                //  BuildSphericalRayList<<<dim3(bx, by, 1), dim3(16, 16, 1)>>>(poses, intrinsics,
                //  ray_lists[i][b].origin, ray_lists[i][b].direction, cam, w, h, tensor_ocam_cam2world);
            }

            scale /= 2;
            h /= 2;
            w /= 2;
        }
        CUDA_SYNC_CHECK_ERROR();
    }
    ray_list.alpha_dest_accumulated = ray_list.alpha_dest_accumulated.contiguous();



    //      std::vector<torch::Tensor> result(num_layers);
    std::vector<torch::Tensor> result(num_layers);
    {
        torch::Tensor col_env;
        {
            SAIGA_OPTIONAL_TIME_MEASURE("Sample Map", timer_system);
            // static bool new_env = true;
            // ImGui::Checkbox("new_env", &new_env);
            // if (new_env)
            col_env = forward_mps5(ray_list, timer_system);
            // col_env = forward_mps2(ray_list, timer_system);
            //  else
            //      col_env = forward_mps4(ray_list, timer_system);

            CUDA_SYNC_CHECK_ERROR();
        }
        {
            SAIGA_OPTIONAL_TIME_MEASURE("Split Tensor", timer_system);

            int h           = info_batch.front().h;
            int w           = info_batch.front().w;
            int start_index = 0;
            for (int i = 0; i < num_layers; ++i)
            {
                int stride_this_layer = h * w;
                std::vector<torch::Tensor> batch_of_envs;

                for (int b = 0; b < num_batches; ++b)
                {
                    torch::Tensor blob_layer = col_env.slice(1, start_index, start_index + stride_this_layer);
                    batch_of_envs.push_back(blob_layer.reshape({channels, h, w}).unsqueeze(0));
                    start_index += stride_this_layer;
                }
                result[i] = torch::cat(batch_of_envs, 0);
                h /= 2;
                w /= 2;
            }
        }
    }

    return result;
}



// origin, direction: [Rays, 3] ; result: [Rays,3] ; mask [Rays,1]
__global__ void CreateSampleUVMap(StaticDeviceTensor<float, 2> rays_origin, StaticDeviceTensor<float, 2> rays_direction,
                                  int up_axis, int num_spheres, int sphere_index, float radius_sphere,
                                  StaticDeviceTensor<float, 2> out_result, StaticDeviceTensor<float, 2> out_mask)
{
    int ray_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (ray_index >= rays_origin.size(0)) return;

    vec3 origin, direction;
    for (int i = 0; i < 3; ++i)
    {
        origin(i)    = rays_origin(ray_index, i);
        direction(i) = rays_direction(ray_index, i);
    }

    // invalid ray (should be ocam only)
    if (origin == vec3(0, 0, 0) && direction == vec3(0, 0, 0))
    {
        out_mask(ray_index, 0) = 0;
        for (int i = 0; i < 3; ++i)
        {
            out_result(ray_index, i) = direction(i);
        }
        return;
    }

    CUDA_KERNEL_ASSERT(num_spheres > sphere_index);

    vec3 sphere_vec;
    bool success_mask;
    // the last sphere is placed at r=inf which means we directly use the ray direction as coordinates
    if (sphere_index == num_spheres - 1)
    {
        sphere_vec   = direction;
        success_mask = true;
    }
    else
    {
        float q2a = (2 * (direction.array() * direction.array())).sum();
        float qb  = (2 * (origin.array() * direction.array())).sum();
        float f   = qb * qb - 2 * q2a * (origin.array() * origin.array()).sum();

        auto det = [&](float r) { return f + 2 * q2a * (r * r); };

        auto d       = det(radius_sphere);
        success_mask = d >= 0.f;

        float result = (-qb + sqrtf(d)) / q2a;

        auto position_on_sphere = origin + result * direction;
        sphere_vec              = position_on_sphere / length(position_on_sphere);
    }
    out_mask(ray_index, 0) = success_mask ? 1 : 0;

    auto xyz2equirect = [&](vec3 xyz, float resolution)
    {
        float lat, lon;
        if (up_axis == 0)
        {
            lat = asinf(xyz.x());
            lon = atan2f(xyz.y(), xyz.z());
        }
        else if (up_axis == 1)
        {
            lat = asinf(xyz.y());
            lon = atan2f(xyz.x(), xyz.z());
        }
        else if (up_axis == 2)
        {
            lat = asinf(xyz.z());
            lon = atan2f(xyz.x(), xyz.y());
        }
        else
        {
            CUDA_KERNEL_ASSERT(false);
        }
        auto u = 2 * resolution * (lon / 2 / pi<float>());
        auto v = 2 * resolution * (lat / pi<float>());
        return vec2(u, v);
    };

    auto uv = xyz2equirect(sphere_vec, 1);
    float z = (float)sphere_index / (num_spheres - 1) * 2 - 1;
    if (num_spheres == 1)
    {
        z = 0;
    }
    auto uvz = vec3(uv.x(), uv.y(), z);
    for (int i = 0; i < 3; ++i)
    {
        out_result(ray_index, i) = uvz(i);
    }
}


// origin, direction: [Rays, 3] ; result: [Sphere_num, Rays,3] ; mask [Sphere_num, Rays,1]
__global__ void CreateSampleUVMapAllSpheres(StaticDeviceTensor<float, 2> rays_origin,
                                            StaticDeviceTensor<float, 2> rays_direction,
                                            StaticDeviceTensor<float, 2> ray_alpha_dest, int up_axis, int num_spheres,
                                            StaticDeviceTensor<float, 3> out_result,
                                            StaticDeviceTensor<float, 3> out_mask)
{
    int ray_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (ray_index >= rays_origin.size(0)) return;

    vec3 origin, direction;
    for (int i = 0; i < 3; ++i)
    {
        origin(i)    = rays_origin(ray_index, i);
        direction(i) = rays_direction(ray_index, i);
    }

    // invalid ray (should be ocam only)
    if ((origin == vec3(0, 0, 0) && direction == vec3(0, 0, 0)))  //|| ray_alpha_dest(ray_index) < 0.001)
    {
        for (int sphere_index = 0; sphere_index < num_spheres; ++sphere_index)
        {
            out_mask(sphere_index, ray_index, 0) = 0;
            for (int i = 0; i < 3; ++i)
            {
                out_result(sphere_index, ray_index, i) = -2;
            }
        }
        return;
    }
    float q2a = (2 * (direction.array() * direction.array())).sum();
    float qb  = (2 * (origin.array() * direction.array())).sum();
    float f   = qb * qb - 2 * q2a * (origin.array() * origin.array()).sum();

    for (int sphere_index = 0; sphere_index < num_spheres; ++sphere_index)
    {
        vec3 sphere_vec;
        bool success_mask;
        // the last sphere is placed at r=inf which means we directly use the ray direction as coordinates
        if (sphere_index == num_spheres - 1)
        {
            sphere_vec   = direction;
            success_mask = true;
        }
        else
        {
            auto det            = [&](float r) { return f + 2 * q2a * (r * r); };
            float radius_sphere = d_radii[sphere_index];
            auto d              = det(radius_sphere);
            success_mask        = d >= 0.f;

            float result = (-qb + sqrtf(d)) / q2a;

            auto position_on_sphere = origin + result * direction;
            sphere_vec              = position_on_sphere / length(position_on_sphere);
        }
        out_mask(sphere_index, ray_index, 0) = success_mask ? 1 : 0;

        auto xyz2equirect = [&](vec3 xyz, float resolution)
        {
            float lat, lon;
            if (up_axis == 0)
            {
                lat = asinf(xyz.x());
                lon = atan2f(xyz.y(), xyz.z());
            }
            else if (up_axis == 1)
            {
                lat = asinf(xyz.y());
                lon = atan2f(xyz.x(), xyz.z());
            }
            else if (up_axis == 2)
            {
                lat = asinf(xyz.z());
                lon = atan2f(xyz.x(), xyz.y());
            }
            else
            {
                CUDA_KERNEL_ASSERT(false);
            }
            auto u = 2 * resolution * (lon / 2 / pi<float>());
            auto v = 2 * resolution * (lat / pi<float>());
            return vec2(u, v);
        };

        auto uv = xyz2equirect(sphere_vec, 1);
        float z = (float)sphere_index / (num_spheres - 1) * 2 - 1;
        if (num_spheres == 1)
        {
            z = 0;
        }
        auto uvz = vec3(uv.x(), uv.y(), z);
        for (int i = 0; i < 3; ++i)
        {
            out_result(sphere_index, ray_index, i) = uvz(i);
        }
    }
}


// #define DEBUG_ENVIRONMENT_SPHERES
#ifdef DEBUG_ENVIRONMENT_SPHERES
static int start = 0;
static int end   = 4;
#endif

torch::Tensor EnvironmentMapImpl::forward_mps5(RayList rays, CUDA::CudaTimerSystem* timer_system)
{
    torch::Tensor final_color = torch::zeros({channels, rays.size()}, rays.direction.options());
    int num_elements          = rays.size();


    auto origin            = rays.origin;
    auto direction         = rays.direction;
    auto uvz_all           = torch::empty({(long)radii.size(), num_elements, 3}, origin.options());
    auto success_masks_all = torch::empty({(long)radii.size(), num_elements, 1}, origin.options());
    {
        SAIGA_OPTIONAL_TIME_MEASURE("Compute UV", timer_system);
        CreateSampleUVMapAllSpheres<<<iDivUp(num_elements, 256), 256>>>(
            origin, direction, rays.alpha_dest_accumulated, up_axis, radii.size(), uvz_all, success_masks_all);
        CUDA_SYNC_CHECK_ERROR();
    }
    auto opt = torch::nn::functional::GridSampleFuncOptions();
    opt      = opt.padding_mode(torch::kZeros).mode(torch::kBilinear).align_corners(true);

    torch::Tensor alpha_all;
    torch::Tensor color_all;

    {
        SAIGA_OPTIONAL_TIME_MEASURE("Grid_sample", timer_system);

        auto uvz_fetch = uvz_all.reshape({1, (long)radii.size() * num_elements, 1, 1, 3});
        // uvz      = uvz.reshape({1, uvz.size(0), 1, 1, 3});

        // [1, 1, num_rays, 1, 1]
        auto density_raw = torch::nn::functional::grid_sample(density, uvz_fetch, opt);
        density_raw      = density_raw.reshape({1, (long)radii.size(), num_elements});
        alpha_all        = torch::sigmoid(density_raw.permute({1, 0, 2}));

        color_all = torch::nn::functional::grid_sample(color, uvz_fetch, opt);
        color_all = color_all.reshape({channels, (long)radii.size(), num_elements});
        color_all = color_all.permute({1, 0, 2});
        if (non_subzero_texture) color_all = torch::abs(color_all);
    }
#ifdef DEBUG_ENVIRONMENT_SPHERES

    ImGui::SliderInt("start sphere", &start, 0, 4);
    ImGui::SliderInt("end sphere", &end, 1, 4);
    for (int sphere_index = start; sphere_index < end; ++sphere_index)
#else
    for (int sphere_index = 0; sphere_index < radii.size(); ++sphere_index)
#endif
    {
        SAIGA_OPTIONAL_TIME_MEASURE("Sphere " + std::to_string(sphere_index), timer_system);

        auto success_mask = success_masks_all.slice(0, sphere_index, sphere_index + 1).squeeze(0);

        auto alpha = alpha_all.slice(0, sphere_index, sphere_index + 1).squeeze(0);
        if (sphere_index == radii.size() - 1)
        {
            // set the last sphere (at r=inf) to have an alpha of 1 (fully opaque)
            alpha = torch::ones_like(alpha);
        }
        alpha = alpha * success_mask.squeeze().unsqueeze(0);

        auto color = color_all.slice(0, sphere_index, sphere_index + 1).squeeze(0);

        final_color = rays.alpha_dest_accumulated.squeeze().unsqueeze(0) * alpha * color + final_color;

        rays.alpha_dest_accumulated = ((1 - alpha.squeeze()) * rays.alpha_dest_accumulated.squeeze()).unsqueeze(1);
    }
    return final_color;
}


torch::Tensor EnvironmentMapImpl::forward_mps4(RayList rays, CUDA::CudaTimerSystem* timer_system)
{
    torch::Tensor final_color = torch::zeros({channels, rays.size()}, rays.direction.options());
    int num_elements          = rays.size();

    auto origin            = rays.origin;
    auto direction         = rays.direction;
    auto uvz_all           = torch::empty({(long)radii.size(), num_elements, 3}, origin.options());
    auto success_masks_all = torch::empty({(long)radii.size(), num_elements, 1}, origin.options());
    {
        SAIGA_OPTIONAL_TIME_MEASURE("Compute UV", timer_system);
        CreateSampleUVMapAllSpheres<<<iDivUp(num_elements, 256), 256>>>(
            origin, direction, rays.alpha_dest_accumulated, up_axis, radii.size(), uvz_all, success_masks_all);
        CUDA_SYNC_CHECK_ERROR();
    }


#ifdef DEBUG_ENVIRONMENT_SPHERES

    ImGui::SliderInt("start sphere", &start, 0, 4);
    ImGui::SliderInt("end sphere", &end, 1, 4);
    for (int sphere_index = start; sphere_index < end; ++sphere_index)
#else
    for (int sphere_index = 0; sphere_index < radii.size(); ++sphere_index)
#endif
    {
        SAIGA_OPTIONAL_TIME_MEASURE("Sphere " + std::to_string(sphere_index), timer_system);


        auto opt = torch::nn::functional::GridSampleFuncOptions();
        opt      = opt.padding_mode(torch::kBorder).mode(torch::kBilinear).align_corners(true);

        auto uvz = uvz_all.slice(0, sphere_index, sphere_index + 1).squeeze(0);
        uvz      = uvz.reshape({1, uvz.size(0), 1, 1, 3});

        auto success_mask = success_masks_all.slice(0, sphere_index, sphere_index + 1).squeeze(0);

        torch::Tensor density_raw;
        torch::Tensor color_raw;

        {
            SAIGA_OPTIONAL_TIME_MEASURE("Grid_sample " + std::to_string(sphere_index), timer_system);

            // [1, 1, num_rays, 1, 1]
            density_raw = torch::nn::functional::grid_sample(density, uvz, opt);
            density_raw = density_raw.reshape({1, num_elements});

            color_raw = torch::nn::functional::grid_sample(color, uvz, opt);
            color_raw = color_raw.reshape({channels, num_elements});
        }
        auto alpha = torch::sigmoid(density_raw);

        if (sphere_index == radii.size() - 1)
        {
            // set the last sphere (at r=inf) to have an alpha of 1 (fully opaque)
            alpha = torch::ones_like(alpha);
        }
        alpha = alpha * success_mask.squeeze().unsqueeze(0);

        auto color = color_raw;
        if (non_subzero_texture) color = torch::abs(color);

        final_color = rays.alpha_dest_accumulated.squeeze().unsqueeze(0) * alpha * color + final_color;

        rays.alpha_dest_accumulated = ((1 - alpha.squeeze()) * rays.alpha_dest_accumulated.squeeze()).unsqueeze(1);
    }
    return final_color;
}


torch::Tensor EnvironmentMapImpl::forward_mps3(RayList rays, CUDA::CudaTimerSystem* timer_system)
{
    torch::Tensor final_color = torch::zeros({channels, rays.size()}, rays.direction.options());
    int num_elements          = rays.size();

#ifdef DEBUG_ENVIRONMENT_SPHERES

    ImGui::SliderInt("start sphere", &start, 0, 4);
    ImGui::SliderInt("end sphere", &end, 1, 4);
    for (int sphere_index = start; sphere_index < end; ++sphere_index)
#else
    for (int sphere_index = 0; sphere_index < radii.size(); ++sphere_index)
#endif
    {
        float r = radii[sphere_index];
        SAIGA_OPTIONAL_TIME_MEASURE("Sphere " + std::to_string(sphere_index), timer_system);
        auto origin       = rays.origin;
        auto direction    = rays.direction;
        auto uvz          = torch::empty_like(origin);
        auto success_mask = torch::empty({num_elements, 1}, origin.options());

        CreateSampleUVMap<<<iDivUp(num_elements, 256), 256>>>(origin, direction, up_axis, radii.size(), sphere_index, r,
                                                              uvz, success_mask);
        CUDA_SYNC_CHECK_ERROR();

        auto opt = torch::nn::functional::GridSampleFuncOptions();
        opt      = opt.padding_mode(torch::kBorder).mode(torch::kBilinear).align_corners(true);

        uvz = uvz.reshape({1, uvz.size(0), 1, 1, 3});

        torch::Tensor density_raw;
        torch::Tensor color_raw;

        {
            SAIGA_OPTIONAL_TIME_MEASURE("Grid_sample " + std::to_string(sphere_index), timer_system);

            // [1, 1, num_rays, 1, 1]
            density_raw = torch::nn::functional::grid_sample(density, uvz, opt);
            density_raw = density_raw.reshape({1, num_elements});

            color_raw = torch::nn::functional::grid_sample(color, uvz, opt);
            color_raw = color_raw.reshape({channels, num_elements});
        }

        auto alpha = torch::sigmoid(density_raw);

        if (sphere_index == radii.size() - 1)
        {
            // set the last sphere (at r=inf) to have an alpha of 1 (fully opaque)
            alpha = torch::ones_like(alpha);
        }
        alpha = alpha * success_mask.squeeze().unsqueeze(0);


        auto color = color_raw;
        if (non_subzero_texture) color = torch::abs(color);

        final_color = rays.alpha_dest_accumulated.squeeze().unsqueeze(0) * alpha * color + final_color;

        rays.alpha_dest_accumulated = ((1 - alpha.squeeze()) * rays.alpha_dest_accumulated.squeeze()).unsqueeze(1);
    }
    return final_color;
}



torch::Tensor EnvironmentMapImpl::forward_mps2(RayList rays, CUDA::CudaTimerSystem* timer_system)
{
    torch::Tensor final_color = torch::zeros({channels, rays.size()}, rays.direction.options());
#ifdef DEBUG_ENVIRONMENT_SPHERES
    ImGui::SliderInt("start sphere", &start, 0, 4);
    ImGui::SliderInt("end sphere", &end, 1, 4);
    for (int sphere_index = start; sphere_index < end; ++sphere_index)
#else
    for (int sphere_index = 0; sphere_index < radii.size(); ++sphere_index)
#endif
    {
        float r = radii[sphere_index];
        SAIGA_OPTIONAL_TIME_MEASURE("Sphere " + std::to_string(sphere_index), timer_system);


        //    std::cout << std::endl << "MultiSphereImage::forward" << std::endl;
        // std::cout << TensorInfo(rays.origin) << " _" << rays.origin.size(1) <<  std::endl;
        // rays.origin.unsqueeze_(0);
        // rays.direction.unsqueeze_(0);
        // CHECK_EQ(rays.origin.dim(), 3);
        // CHECK_EQ(rays.origin.size(1), 1);

        auto origin = rays.origin;        // .view({-1, 3});
                                          //  PrintTensorInfo(origin);
        auto direction = rays.direction;  //.view({-1, 3});

        // self.q2a : torch.Tensor = 2 * (dirs * dirs).sum(-1)
        // self.qb : torch.Tensor = 2 * (origins * dirs).sum(-1)
        // self.f = self.qb.square() - 2 * self.q2a * (origins * origins).sum(-1)

        auto q2a = 2 * (direction * direction).sum(-1);
        //   PrintTensorInfo(q2a);

        auto qb = 2 * (origin * direction).sum(-1);
        auto f  = qb.square() - 2 * q2a * (origin * origin).sum(-1);

        // def _det(self, r : float) -> torch.Tensor:
        //     return self.f + 2 * self.q2a * (r * r)
        auto det = [&](float r) { return f + 2 * q2a * (r * r); };

        auto d            = det(r);
        auto success_mask = d >= 0.f;
        auto result       = torch::zeros_like(q2a);
        result            = (-qb + torch::sqrt(d)) / q2a;

        // overwrite all failed intersection with 1 so we don't have nan's in our calculation
        result = result.masked_fill(true ^ success_mask, 1);
        //   CHECK(result.isfinite().all().item().toBool());



        auto position_on_sphere = origin + result.unsqueeze(-1) * direction;
        auto sphere_vec         = position_on_sphere / torch::norm(position_on_sphere, 2, 1, true);

        auto xyz2equirect = [&](torch::Tensor xyz, float resolution)
        {
            auto x = xyz.slice(xyz.dim() - 1, 0, 1);
            auto y = xyz.slice(xyz.dim() - 1, 1, 2);
            auto z = xyz.slice(xyz.dim() - 1, 2, 3);

            torch::Tensor lat, lon;

            if (up_axis == 0)
            {
                lat = torch::asin(x);
                lon = torch::atan2(y, z);
            }
            else if (up_axis == 1)
            {
                lat = torch::asin(y);
                lon = torch::atan2(x, z);
            }
            else if (up_axis == 2)
            {
                lat = torch::asin(z);
                lon = torch::atan2(x, y);
            }
            else
            {
                CHECK(false);
            }


            auto u = 2 * resolution * (lon / 2 / pi<float>());
            auto v = 2 * resolution * (lat / pi<float>());
            return torch::cat({u, v}, -1);
        };

        if (sphere_index == radii.size() - 1)
        {
            // the last sphere is placed at r=inf which means we directly use the ray direction as
            // coordinates
            sphere_vec   = direction;
            success_mask = torch::ones_like(success_mask);
        }

        // [num_rays, 2] in range [-1, 1]
        auto uv = xyz2equirect(sphere_vec, 1);

        float z = (float)sphere_index / (NumImages() - 1) * 2 - 1;
        if (NumImages() == 1)
        {
            z = 0;
        }
        auto uvz = torch::cat({uv, torch::full({uv.size(0), 1}, z, uv.options())}, 1);


        auto opt = torch::nn::functional::GridSampleFuncOptions();
        opt      = opt.padding_mode(torch::kBorder).mode(torch::kBilinear).align_corners(true);

        uvz = uvz.reshape({1, uvz.size(0), 1, 1, 3});

        torch::Tensor density_raw;
        torch::Tensor color_raw;

        {
            SAIGA_OPTIONAL_TIME_MEASURE("Grid_sample " + std::to_string(sphere_index), timer_system);

            // [1, 1, num_rays, 1, 1]
            density_raw = torch::nn::functional::grid_sample(density, uvz, opt);
            density_raw = density_raw.reshape({1, uv.size(0)});

            color_raw = torch::nn::functional::grid_sample(color, uvz, opt);
            color_raw = color_raw.reshape({channels, uv.size(0)});
        }

        // auto density     = torch::softplus(density_raw) * success_mask.unsqueeze(0);
        // auto neg_density = -density * 20;
        // auto alpha       = 1. - torch::exp(neg_density);

        auto alpha = torch::sigmoid(density_raw) * success_mask.unsqueeze(0);

        if (sphere_index == radii.size() - 1)
        {
            // set the last sphere (at r=inf) to have an alpha of 1 (fully opaque)
            alpha = torch::ones_like(alpha);
        }

        // auto weight = torch::exp(neg_density_integral_without_last) * alpha;

        auto color = color_raw;
        if (non_subzero_texture) color = torch::abs(color);

        //  std::cout << rays << std::endl;
        //  std::cout << TensorInfo(rays.alpha_dest_accumulated.squeeze().unsqueeze(0)) <<
        //  TensorInfo(alpha) << TensorInfo(color) << TensorInfo(final_color) << std::endl;
        final_color = rays.alpha_dest_accumulated.squeeze().unsqueeze(0) * alpha * color + final_color;

        rays.alpha_dest_accumulated = ((1 - alpha.squeeze()) * rays.alpha_dest_accumulated.squeeze()).unsqueeze(1);
        // final_color += color * weight;
        // neg_density_integral_without_last += neg_density;
    }
    return final_color;
}


torch::Tensor EnvironmentMapImpl::forward_mps(RayList rays, torch::Tensor alpha_dest_weights,
                                              CUDA::CudaTimerSystem* timer_system)
{
    torch::Tensor neg_density_integral_without_last = torch::zeros({1, rays.size()}, rays.direction.options());
    torch::Tensor final_color                       = torch::zeros({channels, rays.size()}, rays.direction.options());

    // static int start = 0;
    // ImGui::SliderInt("start sphere", &start, 0, 4);
    // static int end = 4;
    // ImGui::SliderInt("end sphere", &end, 1, 4);
    torch::Tensor alpha_dest_accum = alpha_dest_weights;

    for (int sphere_index = 0; sphere_index < radii.size(); ++sphere_index)
    // for (int sphere_index = start; sphere_index < end; ++sphere_index)
    {
        float r            = radii[sphere_index];
        static int counter = 0;
        //  SAIGA_OPTIONAL_TIME_MEASURE("Sphere "+ std::to_string(sphere_index), timer_system);


        //    std::cout << std::endl << "MultiSphereImage::forward" << std::endl;
        // std::cout << TensorInfo(rays.origin) << " _" << rays.origin.size(1) <<  std::endl;
        // rays.origin.unsqueeze_(0);
        // rays.direction.unsqueeze_(0);
        // CHECK_EQ(rays.origin.dim(), 3);
        // CHECK_EQ(rays.origin.size(1), 1);

        auto origin    = rays.origin;     // .view({-1, 3});
        auto direction = rays.direction;  //.view({-1, 3});

        // self.q2a : torch.Tensor = 2 * (dirs * dirs).sum(-1)
        // self.qb : torch.Tensor = 2 * (origins * dirs).sum(-1)
        // self.f = self.qb.square() - 2 * self.q2a * (origins * origins).sum(-1)

        auto q2a = 2 * (direction * direction).sum(-1);
        auto qb  = 2 * (origin * direction).sum(-1);
        auto f   = qb.square() - 2 * q2a * (origin * origin).sum(-1);

        // def _det(self, r : float) -> torch.Tensor:
        //     return self.f + 2 * self.q2a * (r * r)
        auto det = [&](float r) { return f + 2 * q2a * (r * r); };

        auto d            = det(r);
        auto success_mask = d >= 0.f;
        auto result       = torch::zeros_like(q2a);
        result            = (-qb + torch::sqrt(d)) / q2a;

        // overwrite all failed intersection with 1 so we don't have nan's in our calculation
        result = result.masked_fill(true ^ success_mask, 1);
        CHECK(result.isfinite().all().item().toBool());



        auto position_on_sphere = origin + result.unsqueeze(-1) * direction;
        auto sphere_vec         = position_on_sphere / torch::norm(position_on_sphere, 2, 1, true);

        auto xyz2equirect = [&](torch::Tensor xyz, float resolutin)
        {
            auto x = xyz.slice(xyz.dim() - 1, 0, 1);
            auto y = xyz.slice(xyz.dim() - 1, 1, 2);
            auto z = xyz.slice(xyz.dim() - 1, 2, 3);

            torch::Tensor lat, lon;

            if (up_axis == 0)
            {
                lat = torch::asin(x);
                lon = torch::atan2(y, z);
            }
            else if (up_axis == 1)
            {
                lat = torch::asin(y);
                lon = torch::atan2(x, z);
            }
            else if (up_axis == 2)
            {
                lat = torch::asin(z);
                lon = torch::atan2(x, y);
            }
            else
            {
                CHECK(false);
            }


            auto u = 2 * resolutin * (lon / 2 / pi<float>());
            auto v = 2 * resolutin * (lat / pi<float>());
            return torch::cat({u, v}, -1);
        };

        if (sphere_index == radii.size() - 1)
        {
            // the last sphere is placed at r=inf which means we directly use the ray direction as
            // coordinates
            sphere_vec   = direction;
            success_mask = torch::ones_like(success_mask);
        }

        // [num_rays, 2] in range [-1, 1]
        auto uv = xyz2equirect(sphere_vec, 1);

        float z = (float)sphere_index / (NumImages() - 1) * 2 - 1;
        if (NumImages() == 1)
        {
            z = 0;
        }
        auto uvz = torch::cat({uv, torch::full({uv.size(0), 1}, z, uv.options())}, 1);


        auto opt = torch::nn::functional::GridSampleFuncOptions();
        opt      = opt.padding_mode(torch::kBorder).mode(torch::kBilinear).align_corners(true);

        uvz = uvz.reshape({1, uvz.size(0), 1, 1, 3});

        // [1, 1, num_rays, 1, 1]
        auto density_raw = torch::nn::functional::grid_sample(density, uvz, opt);
        density_raw      = density_raw.reshape({1, uv.size(0)});

        auto color_raw = torch::nn::functional::grid_sample(color, uvz, opt);
        color_raw      = color_raw.reshape({channels, uv.size(0)});


        // auto density     = torch::softplus(density_raw) * success_mask.unsqueeze(0);
        // auto neg_density = -density * 20;
        // auto alpha       = 1. - torch::exp(neg_density);

        auto alpha = torch::sigmoid(density_raw) * success_mask.unsqueeze(0);

        if (sphere_index == radii.size() - 1)
        {
            // set the last sphere (at r=inf) to have an alpha of 1 (fully opaque)
            alpha = torch::ones_like(alpha);
        }

        // auto weight = torch::exp(neg_density_integral_without_last) * alpha;

        auto color = color_raw;
        if (non_subzero_texture) color = torch::abs(color);

        // std::cout << TensorInfo(alpha_dest_accum) << TensorInfo(alpha) << TensorInfo(color) <<
        // TensorInfo(final_color) << std::endl;
        final_color = alpha_dest_accum * alpha * color + final_color;

        alpha_dest_accum = (1 - alpha) * alpha_dest_accum;
        // final_color += color * weight;
        // neg_density_integral_without_last += neg_density;
    }
    return final_color;
}
