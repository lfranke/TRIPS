/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#pragma once

#include "saiga/core/Core.h"

#include "SceneData.h"
#include "Settings.h"
#include "config.h"
#include "data/NeuralStructure.h"
#include "models/NeuralCamera.h"
#include "models/NeuralTexture.h"
#include "rendering/EnvironmentMap.h"
#include "rendering/NeuralPointCloudCuda.h"


using namespace Saiga;

class NeuralScene
{
   public:
    NeuralScene(std::shared_ptr<SceneData> scene, std::shared_ptr<CombinedParams> params, bool eval_only = false);

    void BuildOutlierCloud(int n);

    void Train(int epoch_id, bool train);

    void to(torch::Device device)
    {
        if (environment_map)
        {
            environment_map->to(device);
        }
        texture->to(device);

        camera->to(device);
        intrinsics->to(device);
        poses->to(device);
        point_cloud_cuda->to(device);
        if (outlier_point_cloud_cuda)
        {
            outlier_point_cloud_cuda->to(device);
        }
    }

    void SaveCheckpoint(const std::string& dir, bool reduced);

    void LoadCheckpoint(const std::string& dir);

    void CreateTextureOptimizer();

    void CreateStructureOptimizer();

    void ShrinkTextureOptimizer(torch::Tensor indices_to_keep);

    void AppendToTextureOptimizer(int new_size);

    void Log(const std::string& log_dir);

    void OptimizerStep(int epoch_id, bool structure_only);

    void OptimizerClear(int epoch_id, bool structure_only);

    void UpdateLearningRate(int epoch_id, double factor);

    // Download + Save in 'scene'
    void DownloadIntrinsics();

    void DownloadPoses();

    void AddPointsViaPointGrowing(int factor = 2, float distance = 1.f, bool update_optimizer = true);

    void AddNewPoints(std::vector<vec3> positions, std::vector<vec3> normal);

    void AddNewPoints(torch::Tensor random_values, bool update_optimizer = true);
    void AddNewPoints(std::vector<vec3>& positions, bool update_optimizer = true);

    void AddNewRandomForEnvSphere(int num_spheres, float inner_radius, float env_radius_factor, int num_points,
                                  bool update_optimizer = true);

    void AddNewRandomPoints(float factor);

    void AddNewRandomPointsInValuefilledBB(int num_points_to_add, float percent_of_boxes = 0.05);

    void AddNewRandomPointsFromCTStack(int num_points_to_add, std::string path, float ct_volume_scale = 5.f,
                                       vec3 ct_volume_translation = vec3(0, 0, 1));

    void AddNewRandomPointsFromCTHdr(torch::Tensor hdr_img_stack, int max_num_points_to_add, float ct_volume_scale,
                                     vec3 ct_volume_translation, AABB aabb);

    void RemovePoints(torch::Tensor indices, bool update_optimizer = true);

   public:
    friend class NeuralPipeline;

    std::shared_ptr<SceneData> scene;

    NeuralPointCloudCuda point_cloud_cuda         = nullptr;
    NeuralPointCloudCuda outlier_point_cloud_cuda = nullptr;

    NeuralPointTexture texture = nullptr;

    EnvironmentMap environment_map = nullptr;
    NeuralCamera camera            = nullptr;
    PoseModule poses               = nullptr;
    IntrinsicsModule intrinsics    = nullptr;


    //[batch, n_points, 3]
    torch::Tensor dynamic_refinement_t;

    std::shared_ptr<torch::optim::Optimizer> camera_adam_optimizer, camera_sgd_optimizer;
    std::shared_ptr<torch::optim::Optimizer> texture_optimizer;
    std::shared_ptr<torch::optim::Optimizer> structure_optimizer;

    torch::DeviceType device = torch::kCUDA;
    std::shared_ptr<CombinedParams> params;
};
