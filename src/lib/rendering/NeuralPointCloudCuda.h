/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/cuda/cuda.h"
#include "saiga/cuda/thrust_helper.h"
#include "saiga/normal_packing.h"
#include "saiga/vision/torch/TorchHelper.h"

#include "NeuralPointCloud.h"

#include <torch/torch.h>

#include "cuda_fp16.h"



class NeuralPointCloudCudaImpl : public NeuralPointCloud, public torch::nn::Module
{
   public:
    NeuralPointCloudCudaImpl(const Saiga::UnifiedMesh& model, bool use_grid_loss = false, float cell_ws_size = 1.f,
                             AABB custom_point_manip_aabb = AABB(), bool use_pointsize = true);

    void MakeOutlier(int max_index);

    std::vector<int> Indices();
    void SetIndices(std::vector<int>& indices);

    /* Following [Schütz et al. 22], 80*128 conseq points */
    void UpdateCellStructureForRendering(size_t conseq_points = 10240);

    void UpdateCellStructureForPointOptim(float size_of_box_in_ws, AABB custom_aabb = AABB());

    torch::Tensor GetPerPointBBIndex();
    torch::Tensor GetPerPointBBValue();
    torch::Tensor DebugBBIndexToCol();

    std::vector<vec3> DebugColorsPerBoxCPU();
    void ResetCellValues()
    {
        t_cell_value.set_(torch::zeros_like(t_cell_value));
        t_cell_access_count.set_(torch::zeros_like(t_cell_access_count));
    }
    void NormalizeBBCellValue()
    {
        t_cell_access_count =
            torch::where(t_cell_access_count != 0, t_cell_access_count, torch::ones_like(t_cell_access_count));
        t_cell_value /= t_cell_access_count;
    }

    void RemoveSelected(torch::Tensor to_keep)
    {
        torch::NoGradGuard ngg;
        std::cout << TensorInfo(to_keep) << std::endl;
        std::cout << TensorInfo(t_position) << std::endl;
        t_position = t_position.index({to_keep});
        std::cout << TensorInfo(t_position) << std::endl;
        std::cout << TensorInfo(t_normal) << std::endl;
        t_normal = t_normal.index({to_keep});
        std::cout << TensorInfo(t_normal) << std::endl;
        std::cout << TensorInfo(t_original_color) << std::endl;
        t_original_color = t_original_color.index({to_keep});
        std::cout << TensorInfo(t_original_color) << std::endl;
        std::cout << TensorInfo(t_index) << std::endl;
        t_index = t_index.index({to_keep});
        std::cout << TensorInfo(t_index) << std::endl;
        // std::cout << TensorInfo(t_original_index) << std::endl;
        // t_original_index = t_original_index.index({to_keep});
        // std::cout << TensorInfo(t_original_index) << std::endl;
    }

    torch::Tensor DebugColorsPerBox();
    // void Reorder(torch::Tensor indices);

    int Size();
    Saiga::UnifiedMesh Mesh();

    // [n, 4]
    torch::Tensor t_position;

    //[n,1]
    torch::Tensor t_point_size;
    // [n, 4]
    torch::Tensor t_position_displacement;

    // [n, 4]
    torch::Tensor t_normal;
    // torch::Tensor t_normal_test;

    // [n, 4]
    torch::Tensor t_original_color;

    // [n, 1]
    torch::Tensor t_index;
    // [n, 1]
    //  torch::Tensor t_original_index;

    // [cell_n,3] float
    torch::Tensor t_cell_bb_min;
    // [cell_n,3] float
    torch::Tensor t_cell_bb_length;
    // [cell_n,1] float
    torch::Tensor t_cell_value;
    // [cell_n,1] int
    torch::Tensor t_cell_access_count;

    using PointType  = vec4;
    using NormalType = vec4;
};


TORCH_MODULE(NeuralPointCloudCuda);


// A simple helper class to make the kernels more compact.
struct DevicePointCloud
{
    float4* __restrict__ position;
    float4* __restrict__ position_displacement;
    // int4* normal_test;
    // half2* normal;
    int* __restrict__ normal;
    int* __restrict__ index;
    float* __restrict__ point_size;

    int3* cell_bb_min;
    int3* cell_bb_length;
    float* cell_value;
    int* cell_access_count;

    int n;
    int n_cells;

    DevicePointCloud() = default;

    DevicePointCloud(NeuralPointCloudCuda pc)
    {
        // SAIGA_ASSERT(pc->t_position.size(0) == pc->t_index.size(0));
        SAIGA_ASSERT(pc->t_position.size(0) == pc->Size());

        position = (float4*)pc->t_position.data_ptr<float>();
        if (pc->t_normal.defined())
        {
            SAIGA_ASSERT(pc->t_position.size(0) == pc->t_normal.size(0));
            normal = pc->t_normal.data_ptr<int>();
        }
        else
        {
            normal = nullptr;
        }

        if (pc->t_position_displacement.defined() && pc->t_position_displacement.sizes().size() > 1)
        {
            SAIGA_ASSERT(pc->t_position.size(0) == pc->t_position_displacement.size(0));
            position_displacement = (float4*)pc->t_position_displacement.data_ptr<float>();
        }
        else
        {
            position_displacement = nullptr;
        }

        index = (int*)pc->t_index.data_ptr();

        if (pc->t_point_size.defined())
        {
            point_size = pc->t_point_size.data_ptr<float>();
        }
        else
        {
            point_size = nullptr;
        }

        n = pc->Size();

        if (pc->t_cell_bb_min.defined())
        {
            n_cells = pc->t_cell_bb_min.sizes()[0];

            cell_bb_min       = (int3*)pc->t_cell_bb_min.data_ptr<float>();
            cell_bb_length    = (int3*)pc->t_cell_bb_length.data_ptr<float>();
            cell_value        = pc->t_cell_value.data_ptr<float>();
            cell_access_count = pc->t_cell_access_count.data_ptr<int>();
        }

        n_cells = pc->t_cell_bb_min.size(0);
    }
    HD inline thrust::tuple<vec3, vec3, float> GetCellBB(int cell_id)
    {
        vec3 bb_min;
        vec3 bb_len;
        float val;
        reinterpret_cast<int3*>(&bb_min)[0] = cell_bb_min[cell_id];
        reinterpret_cast<int3*>(&bb_len)[0] = cell_bb_length[cell_id];
        reinterpret_cast<float*>(&val)[0]   = cell_value[cell_id];

        return {bb_min, bb_len, val};
    }
    HD inline void SetValueForCell(int cell_id, float val) { cell_value[cell_id] = val; }
    HD inline float* GetPointerForValueForCell(int cell_id) { return &cell_value[cell_id]; }
    HD inline int* GetPointerForAccessCountForCell(int cell_id) { return &cell_access_count[cell_id]; }

    HD inline thrust::tuple<vec3, vec3, float> GetPoint(int point_index)
    {
        vec4 p;
        vec4 p_displacement = vec4(0, 0, 0, 0);
        vec4 n_test;

        // float4 global memory loads are vectorized!
        // reinterpret_cast<int4*>(&p)[0] = position[point_index];
        vec4* p_p = reinterpret_cast<vec4*>(&position[point_index]);
        p         = p_p[0];

        if (position_displacement)
        {
            reinterpret_cast<float4*>(&p_displacement)[0] = position_displacement[point_index];
        }

        vec3 n;

        if (normal)
        {
            auto enc = normal[point_index];
            n        = UnpackNormal10Bit(enc);
        }

        float drop_out_radius = p(3);

        vec3 pos = p.head<3>();
        if (position_displacement)
        {
            pos += p_displacement.head<3>();
        }

        return {pos, n.head<3>(), drop_out_radius};
    }

    HD inline thrust::tuple<vec3, float> GetPointWoNormal(int point_index) const
    {
        vec4 p;
        vec4 p_displacement = vec4(0, 0, 0, 0);

        // float4 global memory loads are vectorized!
        // reinterpret_cast<int4*>(&p)[0] = position[point_index];
        // vec4* p_p = reinterpret_cast<vec4*>(&position[point_index]);
        // p         = p_p[0];


        float4 p_f4 = position[point_index];

        p = reinterpret_cast<vec4*>(&p_f4)[0];



        // reinterpret_cast<int4*>(&p)[0] = position[point_index];
        // reinterpret_cast<int4*>(&p)[0] = position[point_index];

        if (position_displacement)
        {
            reinterpret_cast<float4*>(&p_displacement)[0] = position_displacement[point_index];
        }
        float drop_out_radius = p(3);

        vec3 pos = p.head<3>();
        if (position_displacement)
        {
            pos += p_displacement.head<3>();
        }

        return {pos, drop_out_radius};
    }

    HD inline int GetIndex(int tid) const { return index[tid]; }
    HD inline float GetPointSize(int tid) const { return point_size[tid]; }

    HD inline void SetIndex(int tid, int value) const { index[tid] = value; }

    HD inline int Size() const { return n; }
};
