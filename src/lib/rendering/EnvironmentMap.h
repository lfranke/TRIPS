/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/cuda/imgui_cuda.h"
#include "saiga/vision/torch/TorchHelper.h"

#include "config.h"
#include "data/NeuralStructure.h"
#include "data/SceneData.h"
#include "data/Settings.h"
// #define LOW_ENV_MAP


// This struct contains a list of rays as tensors.
struct RayList
{
    // float [num_rays, D]
    torch::Tensor origin;

    // float [num_rays, D]
    torch::Tensor direction;

    torch::Tensor alpha_dest_accumulated;

    int width       = 0;
    int height      = 0;
    int num_layers  = 0;
    int num_batches = 0;
    RayList() {}

    // Stacks all rays into a single list
    RayList(const std::vector<RayList>& list)
    {
        std::vector<torch::Tensor> origin_list;
        std::vector<torch::Tensor> direction_list;
        std::vector<torch::Tensor> alpha_list;
        for (auto& l : list)
        {
            origin_list.push_back(l.origin);
            direction_list.push_back(l.direction);
            alpha_list.push_back(l.alpha_dest_accumulated);
        }
        origin                 = torch::cat(origin_list, 0);
        direction              = torch::cat(direction_list, 0);
        alpha_dest_accumulated = torch::cat(alpha_list, 0);
    }

    void Allocate(int num_rays, int D)
    {
        origin                 = torch::empty({num_rays, D}, torch::TensorOptions().device(torch::kCUDA));
        direction              = torch::empty({num_rays, D}, torch::TensorOptions().device(torch::kCUDA));
        alpha_dest_accumulated = torch::empty({num_rays, 1}, torch::TensorOptions().device(torch::kCUDA));
    }

    void Allocate(int w, int h, int num_b, int num_l, int D)
    {
        width       = w;
        height      = h;
        num_batches = num_b;
        num_layers  = num_l;

        int num_rays = 0;
        for (int i = 0; i < num_layers; ++i)
        {
            num_rays += w * h * num_batches;
            h /= 2;
            w /= 2;
        }
        origin                 = torch::empty({num_rays, D}, torch::TensorOptions().device(torch::kCUDA));
        direction              = torch::empty({num_rays, D}, torch::TensorOptions().device(torch::kCUDA));
        alpha_dest_accumulated = torch::empty({num_rays, 1}, torch::TensorOptions().device(torch::kCUDA));
    }



    /* ordering is: [num_batches + layer0][num_batches x layer1] ...
     *  ____________________________________________________
     *  |      ||      ||      ||      |    |    |    |    | ....
     *  |      ||      ||      ||      |    |    |    |    |
     *  |      ||      ||      ||      |--------------------
     *  |      ||      ||      ||      |
     *  --------------------------------
     */
    std::pair<int, int> GetSubViewIndices(int batch_id, int layer_id)
    {
        std::vector<int> sizes_of_subtensors;
        int w = width;
        int h = height;
        for (int i = 0; i < num_layers; ++i)
        {
            int size_wh = w * h;
            sizes_of_subtensors.push_back(size_wh);
            w /= 2;
            h /= 2;
        }
        int start_num_rays = 0;
        // get offset in layers
        for (int i = 0; i < layer_id; ++i)
        {
            start_num_rays += sizes_of_subtensors[i] * num_batches;
        }
        // add offset in batch num
        start_num_rays += batch_id * sizes_of_subtensors[layer_id];
        return {start_num_rays, sizes_of_subtensors[layer_id]};
    }

    /*
    int size_of_subtensor = (width / std::pow(2, layer_id)) * (height / std::pow(2, layer_id));
    int start_num_rays = 0;
    int w              = width;
    int h              = height;
    for (int i = 0; i < layer_id + 1; ++i)
    {
        size_of_subtensor = w * h;
        w /= 2;
        h /= 2;
    }

    w = width;
    h = height;
    for (int i = 0; i < layer_id; ++i)
    {
        start_num_rays += num_batches * w * h;
        w /= 2;
        h /= 2;
    }
    start_num_rays += batch_id * w * h;


            int size_of_one_batch = 0;
            int size_of_index_to_layer = 0;

            for (int i = 0; i < num_layers; ++i)
            {
                if(i<=layer_id)
                    size_of_index_to_layer+=(w*h);
                size_of_one_batch+=(w*h);
                h /= 2;
                w /= 2;
            }
            start_num_rays = batch_id*size_of_bulk + size_of_index_to_layer;

            for (int i = 0; i < layer_num; ++i)
            {
                for (int b = 0; b < batch_id; ++b){
                    start_num_rays+=(w*h)*batch_id;
                }
                h /= 2;
                w /= 2;
            }
            w = width;
            h = height;
            for (int i = 0; i < layer_num; ++i)
            {
                start_num_rays+=(w*h);
                h /= 2;
                w /= 2;
            }
            */

    torch::Tensor getSubViewDirection(int batch_id, int layer_num)
    {
        int start, length;
        std::tie(start, length) = GetSubViewIndices(batch_id, layer_num);
        return direction.slice(0, start, start + length);
    }
    torch::Tensor getSubViewOrigin(int batch_id, int layer_num)
    {
        int start, length;
        std::tie(start, length) = GetSubViewIndices(batch_id, layer_num);
        return origin.slice(0, start, start + length);
    }
    torch::Tensor getSubViewAlphaDest(int batch_id, int layer_num)
    {
        int start, length;
        std::tie(start, length) = GetSubViewIndices(batch_id, layer_num);
        return alpha_dest_accumulated.slice(0, start, start + length);
    }

    RayList getSubRayListView(int batch_id, int layer_num)
    {
        RayList rl                = RayList();
        rl.origin                 = getSubViewOrigin(batch_id, layer_num);
        rl.direction              = getSubViewDirection(batch_id, layer_num);
        rl.alpha_dest_accumulated = getSubViewAlphaDest(batch_id, layer_num);
        rl.width                  = width;
        rl.height                 = height;
        rl.num_layers             = num_layers;
        rl.num_batches            = num_batches;

        return rl;
    }
    void to(torch::Device device)
    {
        // if(linear_pixel_location.defined()) linear_pixel_location = linear_pixel_location.to(device);
        origin                 = origin.to(device);
        direction              = direction.to(device);
        alpha_dest_accumulated = alpha_dest_accumulated.to(device);

        // if (pixel_uv.defined()) pixel_uv = pixel_uv.to(device);
    }

    RayList SubSample(torch::Tensor index)
    {
        RayList result;
        // result.linear_pixel_location = torch::index_select(linear_pixel_location, 0, index);
        result.origin                 = torch::index_select(origin, 0, index);
        result.direction              = torch::index_select(direction, 0, index);
        result.alpha_dest_accumulated = torch::index_select(alpha_dest_accumulated, 0, index);
        return result;
    }

    size_t Memory()
    {
        return direction.numel() * sizeof(float) + origin.numel() * sizeof(float);
        // +linear_pixel_location.numel() * sizeof(long);
    }

    int size() const { return origin.size(0); }

    int Dim() const { return origin.size(1); }

    template <int D>
    std::pair<Eigen::Vector<float, D>, Eigen::Vector<float, D>> GetRay(int i) const
    {
        SAIGA_ASSERT(i < size());
        SAIGA_ASSERT(D == Dim());
        Eigen::Vector<float, D> o;
        Eigen::Vector<float, D> d;
        for (int k = 0; k < D; ++k)
        {
            o(k) = origin.template data_ptr<float>()[i * origin.stride(0) + k * origin.stride(1)];
            d(k) = direction.template data_ptr<float>()[i * direction.stride(0) + k * direction.stride(1)];
        }

        return {o, d};
    }
    friend std::ostream& operator<<(std::ostream& stream, const RayList& rays);
};


class EnvironmentMapImpl : public torch::nn::Module
{
   public:
    // up_axis: int: 0,1,2 for the axis used as up.
    EnvironmentMapImpl(int channels, int h, int w, bool log_texture, int axis = 0, int num_images = 4,
                       float inner_radius = 20.f, float radius_factor = 5.f, bool non_subzero_texture = false);

    void CreateEnvMapOptimizer(float env_col_learning_rate = 0.02f, float env_density_learning_rate = 0.001f);


    // Samples the env. map in all layers and all images of the batch.
    // The result is an array of tensor where each element resebles one layer of the stack.
    std::vector<torch::Tensor> Sample(torch::Tensor poses, torch::Tensor intrinsics,
                                      ArrayView<ReducedImageInfo> info_batch, int num_layers,
                                      std::shared_ptr<SceneData> scene,
                                      std::vector<std::vector<torch::Tensor>> layers_cuda,
                                      CUDA::CudaTimerSystem* timer_system = nullptr);

    std::vector<torch::Tensor> Sample2(torch::Tensor poses, torch::Tensor intrinsics,
                                       ArrayView<ReducedImageInfo> info_batch, int num_layers,
                                       std::shared_ptr<SceneData> scene,
                                       std::vector<std::vector<torch::Tensor>> layers_cuda,
                                       CUDA::CudaTimerSystem* timer_system = nullptr);


    // returns for a give ray the color
    // return:      [3, num_rays]
    torch::Tensor forward_mps(RayList rays, torch::Tensor alpha_dest_weights,
                              CUDA::CudaTimerSystem* timer_system = nullptr);
    torch::Tensor forward_mps2(RayList rays, CUDA::CudaTimerSystem* timer_system = nullptr);
    torch::Tensor forward_mps3(RayList rays, CUDA::CudaTimerSystem* timer_system = nullptr);
    torch::Tensor forward_mps4(RayList rays, CUDA::CudaTimerSystem* timer_system = nullptr);
    torch::Tensor forward_mps5(RayList rays, CUDA::CudaTimerSystem* timer_system = nullptr);

    std::shared_ptr<torch::optim::Adam> optimizer_adam;

    // [1, 1, num_images, resolution, resolution]
    torch::Tensor density;
    // [1, num_desc, num_images, resolution, resolution]
    torch::Tensor color;

    int NumImages() { return density.size(2); }

   private:
    // the radii of the spheres
    // length == num_images (from constructor)
    std::vector<float> radii;
    int up_axis;
    int channels;
    bool non_subzero_texture;
};
TORCH_MODULE(EnvironmentMap);
