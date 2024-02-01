/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#pragma once
#include "saiga/core/Core.h"
#include "saiga/core/util/FileSystem.h"
#include "saiga/core/util/directory.h"
#include "saiga/core/util/tostring.h"
#include "saiga/cuda/imgui_cuda.h"
#include "saiga/vision/torch/ImageTensor.h"

#include "SceneData.h"
#include "config.h"
#include "rendering/RenderInfo.h"

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <torch/torch.h>
#include <vector>

TemplatedImage<vec2> InitialUVImage(int h, int w, ivec2 border_for_crops = vec2(0, 0));
TemplatedImage<vec3> InitialDirectionImage(int w, int h, CameraModel camera_model_type, IntrinsicsPinholef K,
                                           Distortionf distortion, OCam<double> ocam,
                                           ivec2 border_for_crops = vec2(0, 0));



using NeuralTrainData = std::shared_ptr<TorchFrameData>;

class SceneDataTrainSampler
{
   public:
    SceneDataTrainSampler() {}
    SceneDataTrainSampler(std::shared_ptr<SceneData> dataset, std::vector<int> indices, bool down_scale,
                          ivec2 crop_size, int inner_batch_size, bool use_image_mask, bool crop_rotate,
                          int max_distance_from_image_center, int warmup_epochs = -1, bool timer = false);

    std::vector<NeuralTrainData> Get(int index);

    int Size() const { return indices.size(); }

    void Start(int epoch);
    void Finish() const
    {
        if (timer_system)
        {
            //   timer_system->EndFrame();
            timer_system->PrintTable(std::cout);
        }
    }

   public:
    int inner_batch_size = 1;

    // different for each camera
    std::vector<ivec2> image_size_crop;
    std::vector<ivec2> image_size_input;
    std::vector<TemplatedImage<vec2>> uv_target;
    std::vector<TemplatedImage<vec3>> direction_target;

    std::shared_ptr<SceneData> scene;
    std::vector<int> indices;

    int num_classes = -1;

    bool down_scale                    = false;
    bool crop_rotate                   = false;
    bool random_translation            = true;
    bool sample_gaussian               = false;
    bool random_zoom                   = true;
    bool prefere_border                = true;
    bool use_image_mask                = false;
    int inner_sample_size              = 1;
    int max_distance_from_image_center = -1;

    int warmup_epochs = -1;

    vec2 min_max_zoom = vec2(0.75, 1.5);

    std::shared_ptr<Saiga::CUDA::CudaTimerSystem> timer_system = nullptr;
};

namespace torch
{
class MultiDatasetSampler : public torch::data::samplers::Sampler<>
{
   public:
    MultiDatasetSampler(std::vector<uint64_t> sizes, int batch_size, bool shuffle);

    void reset(torch::optional<size_t> new_size = torch::nullopt) override
    {
        current_index = 0;
        allocateOrReset();
    }

    /// Returns the next batch of indices.
    optional<std::vector<size_t>> next(size_t batch_size) override;

    /// Serializes the `RandomSampler` to the `archive`.
    void save(serialize::OutputArchive& archive) const override {}

    /// Deserializes the `RandomSampler` from the `archive`.
    void load(serialize::InputArchive& archive) override {}

    /// Returns the current index of the `RandomSampler`.
    size_t index() const noexcept { return current_index; }


    int Size() { return batch_offset_size.size(); }
    int NumImages() { return combined_indices.size(); }

    void allocateOrReset();

    // dataset sampler state
    size_t current_index = 0;
    std::vector<std::pair<int, int>> combined_indices;
    // a pointer into the array above
    std::vector<std::pair<int, int>> batch_offset_size;

    // members, initially set
    std::vector<uint64_t> sizes;
    int batch_size;
    bool shuffle;
};
}  // namespace torch


class TorchSingleSceneDataset : public torch::data::Dataset<TorchSingleSceneDataset, NeuralTrainData>
{
   public:
    TorchSingleSceneDataset(std::vector<SceneDataTrainSampler> sampler);
    virtual torch::optional<size_t> size() const override { return sampler.front().Size(); }
    virtual std::vector<NeuralTrainData> get2(size_t index);


    virtual NeuralTrainData get(size_t index) override { return {}; }

    std::vector<NeuralTrainData> get_batch(torch::ArrayRef<size_t> indices) override;

   private:
    std::vector<SceneDataTrainSampler> sampler;
};
