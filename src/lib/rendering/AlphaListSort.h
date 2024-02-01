/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/cuda/imgui_cuda.h"

#include "NeuralPointCloudCuda.h"

void SegmentedSortBitonicHelper(torch::Tensor counts, torch::Tensor scanned_counts, torch::Tensor data,
                                CUDA::CudaTimerSystem* timer);
void SegmentedSortBitonicHelper2(torch::Tensor counts, torch::Tensor scanned_counts, torch::Tensor data,
                                 CUDA::CudaTimerSystem* timer = nullptr);

void SegmentedSortCubHelper(torch::Tensor counts, torch::Tensor scanned_counts, torch::Tensor& data,
                            CUDA::CudaTimerSystem* timer);