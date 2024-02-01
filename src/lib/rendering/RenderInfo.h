/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/cuda/imageProcessing/image.h"

#include "NeuralPointCloudCuda.h"
#include "config.h"
#include "data/SceneData.h"

#include <torch/torch.h>



struct TorchFrameData
{
    ReducedImageInfo img;

    torch::Tensor target;

    // binary float tensor, where the loss should only be taken if target_mask == 1
    torch::Tensor target_mask;

    torch::Tensor uv, uv_local;

    torch::Tensor direction;

    // long index for the camera
    // used for training camera specific parameters
    torch::Tensor camera_index;
    torch::Tensor scale;
    torch::Tensor timestep;

    int scene_id;
    void to(torch::Device device)
    {
        if (target.defined()) target = target.to(device);
        if (target_mask.defined()) target_mask = target_mask.to(device);
        if (uv.defined()) uv = uv.to(device);
        if (uv_local.defined()) uv_local = uv_local.to(device);
        if (direction.defined()) direction = direction.to(device);
        if (camera_index.defined()) camera_index = camera_index.to(device);
        if (timestep.defined()) timestep = timestep.to(device);
        if (scale.defined()) scale = scale.to(device);
    }

    void print()
    {
        std::cout << "TorchFrameData: " << std::endl;
        img.print();
        std::cout << "\t target: " << TensorInfo(target) << "\t target_mask: " << TensorInfo(target_mask)
                  << "\t uv: " << TensorInfo(uv) << "\t camera_index: " << TensorInfo(camera_index)
                  << "\t scale: " << TensorInfo(scale) << "\t scene_id: " << scene_id << std::endl;
    }
};
