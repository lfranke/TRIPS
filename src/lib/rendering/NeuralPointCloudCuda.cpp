/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "NeuralPointCloudCuda.h"

#include "saiga/normal_packing.h"

inline float _softplus(float x, float beta = 1.f, float threshold = 20.f)
{
    //  return x * beta > threshold ? x : logf(1 + expf(beta * x));
    if (x > threshold) return x;
    return logf(1.f + expf(x * beta)) / beta;
}


inline float inverse_softplus(float x, float beta = 1.f, float threshold = 20.f)
{
    if (x > threshold) return x;

    return log(exp(x * beta) - 1) / beta;
}

//
// #define USE_LAYER_EXPL


NeuralPointCloudCudaImpl::NeuralPointCloudCudaImpl(const UnifiedMesh& model, bool use_grid_loss, float cell_ws_size,
                                                   AABB custom_point_manip_aabb, bool use_pointsize)
    : NeuralPointCloud(model)
{
    std::vector<vec4> data_position;
    std::vector<int> data_normal_compressed;
    std::vector<vec4> original_colors;

    std::vector<float> data_point_size;

    std::vector<int32_t> indices;
    for (int i = 0; i < points.size(); ++i)
    {
        indices.push_back(points[i].index);

        float drop_out_radius = 0;
        if (data.size() == points.size())
        {
            drop_out_radius = data[i](3);
        }
        data_point_size.push_back(inverse_softplus(data[i](0) * 0.5f));

#if 0
        if (normal.size() == points.size())
        {
            vec3 n = normal[i].head<3>();
            SAIGA_ASSERT(n.allFinite());
            auto n_enc = PackNormal10Bit(n);
            data_normal_compressed.push_back(n_enc);
        }
#endif
        original_colors.push_back(color[i]);
        data_position.push_back(make_vec4(points[i].position, drop_out_radius));
    }
    t_position = torch::from_blob(data_position.data(), {(long)data_position.size(), 4},
                                  torch::TensorOptions().dtype(torch::kFloat32))
                     .contiguous()
                     .cuda()
                     .clone();

    if (!data_normal_compressed.empty())
    {
        t_normal = torch::from_blob(data_normal_compressed.data(), {(long)data_normal_compressed.size(), 1},
                                    torch::TensorOptions().dtype(torch::kInt32))
                       .contiguous()
                       .cuda()
                       .clone();
        register_buffer("t_normal", t_normal);
    }

    t_original_color = torch::from_blob(original_colors.data(), {(long)original_colors.size(), 4},
                                        torch::TensorOptions().dtype(torch::kFloat32))
                           .contiguous()
                           .cuda()
                           .clone();

    t_index = torch::from_blob(indices.data(), {(long)indices.size(), 1}, torch::TensorOptions().dtype(torch::kInt32))
                  .contiguous()
                  .cuda()
                  .clone();



    //   t_original_index = t_index.clone();

    t_point_size = torch::from_blob(data_point_size.data(), {(long)data_point_size.size(), 1},
                                    torch::TensorOptions().dtype(torch::kFloat32))
                       .contiguous()
                       .cuda()
                       .clone();

    // UpdateCellStructure();

    register_parameter("t_position", t_position);
    register_buffer("t_index", t_index);
    register_buffer("t_original_color", t_original_color);

    // #ifdef USE_LAYER_EXPL
    if (use_pointsize) register_parameter("t_point_size", t_point_size);
    // #endif

    SAIGA_ASSERT(t_position.isfinite().all().item().toBool());

    size_t total_mem = t_position.nbytes() + t_index.nbytes() + t_original_color.nbytes();
    if (t_normal.defined()) total_mem += t_normal.nbytes();

    if (use_grid_loss)
    {
        UpdateCellStructureForPointOptim(cell_ws_size, custom_point_manip_aabb);

        register_buffer("t_cell_bb_min", t_cell_bb_min);
        register_buffer("t_cell_bb_length", t_cell_bb_length);
        register_buffer("t_cell_value", t_cell_value);
        register_buffer("t_cell_access_count", t_cell_access_count);
        total_mem +=
            t_cell_bb_min.nbytes() + t_cell_bb_length.nbytes() + t_cell_value.nbytes() + t_cell_access_count.nbytes();
    }

    std::cout << "GPU memory - Point Cloud: " << total_mem / 1000000.0 << "MB" << std::endl;
}

void NeuralPointCloudCudaImpl::UpdateCellStructureForPointOptim(float size_of_box_in_ws, AABB custom_aabb)
{
    torch::NoGradGuard ngg;
    AABB aabb = custom_aabb;

    // no custom box given
    if (aabb.maxSize() < 0.01)
    {
        auto mesh = this->Mesh();
        std::cout << mesh.NumVertices() << std::endl;
        aabb = mesh.BoundingBox();
        std::cout << aabb << std::endl;
        float epsilon = 0.01;
        aabb.min -= vec3(epsilon, epsilon, epsilon);
        aabb.max += vec3(epsilon, epsilon, epsilon);
    }
    std::cout << "Custom AABB: " << custom_aabb << std::endl;
    vec3 num_slices =
        ((aabb.max - aabb.min).array() / vec3(size_of_box_in_ws, size_of_box_in_ws, size_of_box_in_ws).array()).ceil();
    std::cout << "Bounding Box:" << num_slices << std::endl;
    std::cout << "Bounding Box: " << aabb << "; slices: " << num_slices.x() << " * " << num_slices.y() << " * "
              << num_slices.z() << " ; "
              << "; full num: " << num_slices.x() * num_slices.y() * num_slices.z() << std::endl;

    size_t l_buf = int(num_slices.x() * num_slices.y() * num_slices.z());
    std::vector<vec3> bb_mins(l_buf);
    std::vector<vec3> bb_lengths(l_buf);

    for (int z_n = 0; z_n < num_slices.z(); ++z_n)
    {
        for (int y_n = 0; y_n < num_slices.y(); ++y_n)
        {
            for (int x_n = 0; x_n < num_slices.x(); ++x_n)
            {
                unsigned int index = z_n * num_slices.y() * num_slices.x() + y_n * num_slices.x() + x_n;
                bb_mins[index]     = aabb.min.array() + vec3(x_n, y_n, z_n).array() * size_of_box_in_ws;
                bb_lengths[index]  = vec3(size_of_box_in_ws, size_of_box_in_ws, size_of_box_in_ws);
                //      values[index]=index;
            }
        }
    }

    t_cell_bb_min = torch::from_blob(bb_mins.data(), {(long)l_buf, 3}, torch::TensorOptions().dtype(torch::kFloat))
                        .clone()
                        .cuda()
                        .contiguous();
    t_cell_bb_length =
        torch::from_blob(bb_lengths.data(), {(long)l_buf, 3}, torch::TensorOptions().dtype(torch::kFloat))
            .clone()
            .cuda()
            .contiguous();
    t_cell_value        = torch::zeros({(long)l_buf, 1}, torch::TensorOptions().dtype(torch::kFloat)).cuda();
    t_cell_access_count = torch::zeros({(long)l_buf, 1}, torch::TensorOptions().dtype(torch::kInt32)).cuda();


    // t_cell_value = torch::arange(float(l_buf), torch::TensorOptions().dtype(torch::kFloat)).unsqueeze(1).cuda();
    // t_cell_value/=t_cell_value.max();
}

/*
static torch::Tensor hsv2rgb(torch::Tensor input)
{
    assert(false);
    std::cout << TensorInfo(input) << std::endl;
    using namespace torch::indexing;
    auto h = input.index({Slice(),Slice(0)});
    auto s = input.index({Slice(),Slice(1)});
    auto v = input.index({Slice(),Slice(2)});
    auto h_ = (h - torch::floor(h / 360) * 360) / 60;
    auto c = s* v;
    auto x = c * (1 - torch::abs(torch::fmod(h_, 2) - 1));

    auto zero = torch::zeros_like(c);
    auto y = torch::stack({
                              torch::stack({c, x, zero},1),
                              torch::stack({x, c, zero},1),
                              torch::stack({zero, c, x}, 1),
                              torch::stack({zero, x, c}, 1),
                              torch::stack({x, zero, c}, 1),
                              torch::stack({c, zero, x}, 1)},0);

     auto index = torch::repeat_interleave(torch::floor(h_).unsqueeze(1), 3, 1).unsqueeze(0).to(torch::kLong);
     auto rgb = (y.gather(0, index) + (v - c)).squeeze(0);
     return rgb;
}
 */


vec3 cols[24] = {vec3(255, 0, 0),     vec3(255, 255, 0),   vec3(0, 234, 255),   vec3(170, 0, 255),
                 vec3(255, 127, 0),   vec3(191, 255, 0),   vec3(0, 149, 255),   vec3(255, 0, 170),
                 vec3(255, 212, 0),   vec3(106, 255, 0),   vec3(0, 64, 255),    vec3(237, 185, 185),
                 vec3(185, 215, 237), vec3(231, 233, 185), vec3(220, 185, 237), vec3(185, 237, 224),
                 vec3(143, 35, 35),   vec3(35, 98, 143),   vec3(143, 106, 35),  vec3(107, 35, 143),
                 vec3(79, 143, 35),   vec3(255, 255, 255), vec3(115, 115, 115), vec3(204, 204, 204)};
/*
torch::Tensor NeuralPointCloudCudaImpl::DebugBBIndexToCol(torch::Tensor in_d){
    float PHI = (1 + sqrt(5))/2;
    auto n = in_d * PHI - torch::floor(in_d * PHI);
    auto hue = floor(n * 256);
    hue = in_d.squeeze()/10;
    hue = hue.unsqueeze(1);
    return
torch::cat({hsv2rgb(torch::cat({hue,torch::full_like(hue,0.5),torch::full_like(hue,0.95)},1).unsqueeze(2).unsqueeze(3)).squeeze(),in_d},1);
}
*/

torch::Tensor NeuralPointCloudCudaImpl::DebugColorsPerBox()
{
    torch::Tensor c = torch::from_blob((float*)&cols[0], {24, 3}, torch::TensorOptions().dtype(torch::kFloat32))
                          .contiguous()
                          .clone() /
                      255.0;
    torch::Tensor res = c.clone();
    for (int i = 1; i <= t_cell_bb_min.size(0) / 24; ++i)
    {
        res = torch::cat({res, c}, 0);
    }
    return res.contiguous();
}
std::vector<vec3> NeuralPointCloudCudaImpl::DebugColorsPerBoxCPU()
{
    std::vector<vec3> res;
    for (int i = 0; i <= t_cell_bb_min.size(0); ++i)
    {
        res.push_back(cols[i % 24]);
    }
    return res;
}

torch::Tensor NeuralPointCloudCudaImpl::DebugBBIndexToCol()
{
    using namespace torch::indexing;
    // torch::Tensor ind = torch::full({t_position.size(0),1},-1).cuda();
    torch::Tensor debug = torch::full_like(t_position, -15).cuda();
    for (int cell_id = 0; cell_id < t_cell_bb_min.size(0); ++cell_id)
    {
        torch::Tensor intersection_bool = torch::full({t_position.size(0), 1}, true).cuda();
        for (int j = 0; j < 3; ++j)
        {
            intersection_bool &= (t_position.index({Slice(), Slice(j, j + 1)}) >=
                                  t_cell_bb_min.index({Slice(cell_id, cell_id + 1), Slice(j, j + 1)}));
            intersection_bool &=
                (t_position.index({Slice(), Slice(j, j + 1)}) <=
                 (t_cell_bb_min + t_cell_bb_length).index({Slice(cell_id, cell_id + 1), Slice(j, j + 1)}));
        }
        //  ind.index_put_({intersection_bool},cell_id);
        debug.index_put_({intersection_bool.squeeze(), 0}, cols[cell_id % 24].x() / 255.f);
        debug.index_put_({intersection_bool.squeeze(), 1}, cols[cell_id % 24].y() / 255.f);
        debug.index_put_({intersection_bool.squeeze(), 2}, cols[cell_id % 24].z() / 255.f);
    }
    return debug;
    // std::cout << ind << std::endl;
    // return ind.to(torch::kFloat).squeeze().unsqueeze(0);
}

torch::Tensor NeuralPointCloudCudaImpl::GetPerPointBBIndex()
{
    using namespace torch::indexing;
    torch::Tensor ind = torch::full({t_position.size(0), 1}, -1).cuda();
    // torch::Tensor ind = torch::full_like(t_position.slice(1,0,1),-15).cuda();

    for (int cell_id = 0; cell_id < t_cell_bb_min.size(0); ++cell_id)
    {
        torch::Tensor intersection_bool = torch::full({t_position.size(0), 1}, true).cuda();
        for (int j = 0; j < 3; ++j)
        {
            intersection_bool &= (t_position.index({Slice(), Slice(j, j + 1)}) >=
                                  t_cell_bb_min.index({Slice(cell_id, cell_id + 1), Slice(j, j + 1)}));
            intersection_bool &=
                (t_position.index({Slice(), Slice(j, j + 1)}) <=
                 (t_cell_bb_min + t_cell_bb_length).index({Slice(cell_id, cell_id + 1), Slice(j, j + 1)}));
        }
        ind.index_put_({intersection_bool}, cell_id);
        // debug.index_put_({intersection_bool.squeeze(),0},cols[cell_id%24].x()/255.f);
        // debug.index_put_({intersection_bool.squeeze(),1},cols[cell_id%24].y()/255.f);
        // debug.index_put_({intersection_bool.squeeze(),2},cols[cell_id%24].z()/255.f);
    }
    return ind.to(torch::kFloat).squeeze().unsqueeze(0);
}

torch::Tensor NeuralPointCloudCudaImpl::GetPerPointBBValue()
{
    using namespace torch::indexing;
    torch::Tensor ind = torch::full({t_position.size(0), 1}, -1.f).cuda();
    // torch::Tensor ind = torch::full_like(t_position.slice(1,0,1),-15).cuda();

    for (int cell_id = 0; cell_id < t_cell_bb_min.size(0); ++cell_id)
    {
        torch::Tensor intersection_bool = torch::full({t_position.size(0), 1}, true).cuda();
        for (int j = 0; j < 3; ++j)
        {
            intersection_bool &= (t_position.index({Slice(), Slice(j, j + 1)}) >=
                                  t_cell_bb_min.index({Slice(cell_id, cell_id + 1), Slice(j, j + 1)}));
            intersection_bool &=
                (t_position.index({Slice(), Slice(j, j + 1)}) <=
                 (t_cell_bb_min + t_cell_bb_length).index({Slice(cell_id, cell_id + 1), Slice(j, j + 1)}));
        }
        ind.index_put_({intersection_bool}, t_cell_value.index({Slice(cell_id, cell_id + 1), 0}));
        // debug.index_put_({intersection_bool.squeeze(),0},cols[cell_id%24].x()/255.f);
        // debug.index_put_({intersection_bool.squeeze(),1},cols[cell_id%24].y()/255.f);
        // debug.index_put_({intersection_bool.squeeze(),2},cols[cell_id%24].z()/255.f);
    }
    std::cout << TensorInfo(ind) << std::endl;
    return ind.to(torch::kFloat).squeeze().unsqueeze(0);
}



void NeuralPointCloudCudaImpl::UpdateCellStructureForRendering(size_t conseq_points)
{
    torch::NoGradGuard ngg;

    size_t full_point_batches = points.size() / conseq_points;
    size_t last_point_batch   = points.size() % conseq_points;

    size_t l_buf = full_point_batches + (last_point_batch > 0 ? 1 : 0);

    std::vector<vec3> bb_mins(l_buf);
    std::vector<vec3> bb_lengths(l_buf);

    {
        size_t i = 0;
        // #pragma omp parallel for
        for (i = 0; i < full_point_batches; ++i)
        {
            AABB point_batch_aabb;
            for (size_t j = 0; j < conseq_points; ++j)
            {
                size_t index = i * conseq_points + j;
                auto pos     = points[index];
                point_batch_aabb.growBox(pos.position);
            }
            bb_mins[i]    = point_batch_aabb.min;
            bb_lengths[i] = point_batch_aabb.max - point_batch_aabb.min;
        }
        AABB point_batch_aabb;
        for (size_t j = 0; j < last_point_batch; ++j)
        {
            size_t index = i * conseq_points + j;
            auto pos     = points[index];
            point_batch_aabb.growBox(pos.position);
        }
    }
    t_cell_bb_min = torch::from_blob(bb_mins.data(), {(long)full_point_batches + 1, 3},
                                     torch::TensorOptions().dtype(torch::kFloat32))
                        .clone()
                        .cuda()
                        .contiguous();
    t_cell_bb_length = torch::from_blob(bb_lengths.data(), {(long)full_point_batches + 1, 3},
                                        torch::TensorOptions().dtype(torch::kFloat32))
                           .clone()
                           .cuda()
                           .contiguous();
}


#if 0
    torch::NoGradGuard ngg;
    using namespace torch::indexing;
    auto bbmin = std::get<0>(torch::min(t_position,0,false)).index({Slice(0,3)});
    auto bbmax = std::get<0>(torch::max(t_position,0,false)).index({Slice(0,3)});

    auto slices_t = (bbmax - bbmin) / gridsize;
    vec3 slices_f = vec3(slices_t[0].item<float>(),slices_t[1].item<float>(),slices_t[2].item<float>());
    for( int i=0; i<3; ++i)
        slices_f[i] = std::ceil(slices_f[i]);
    ivec3 slices = ivec3(int(slices_f.x()),int(slices_f.y()),int(slices_f.z()));

    std::vector<std::vector<int>> indices;
    for (int i = 0; i < slices.x() * slices.y() * slices.z(); ++i)
    {
        indices.push_back(std::vector<int>());
    }
    //helper functions to adresss linearized space back and forth
    auto access_index = [&](ivec3 index)
    {
        return index.z()*slices[1]*slices[0] + index.y()*slices[0] + index.x();
    };
    auto access_3d = [&](int ind)
    {
        int index = ind;
        int z = index / int(slices[1]*slices[0]);
        index -= (z * int(slices[1]*slices[0]));
        int y = index / int(slices[0]);
        int x = index % int(slices[0]);
        return ivec3(x,y,z);
    };

    for(int i = 0; i< t_position.sizes()[0]; ++i){
        auto p = t_position[i].index({Slice(0,3)}) - bbmin;
        ivec3 ind = ivec3(floor(p[0].item<float>() / gridsize),floor(p[1].item<float>() / gridsize),floor(p[2].item<float>()/ gridsize));
        indices[access_index(ind)].push_back(i);
    }
    std::vector<int> full_index_list;
    for(auto list: indices){
        full_index_list.insert(full_index_list.end(), list.begin(),list.end());
    }
    auto indices_tensor = torch::from_blob(full_index_list.data(), {(long)full_index_list.size(), 1}, torch::TensorOptions().dtype(torch::kInt32))
                      .clone().to(torch::kInt64)
                      .cuda()
                      .contiguous();
    SAIGA_ASSERT(indices_tensor.sizes()[0] == t_position.sizes()[0]);

    int sum_of_points = 0;
    torch::Tensor starts  = torch::zeros(indices.size(),torch::TensorOptions().dtype(torch::kInt32));
    torch::Tensor lengths = torch::zeros(indices.size(),torch::TensorOptions().dtype(torch::kInt32));
    torch::Tensor cell_bb_min = torch::zeros({int(indices.size()),3},torch::TensorOptions().dtype(torch::kFloat32));
    torch::Tensor cell_bb_length = torch::zeros({int(indices.size()),3},torch::TensorOptions().dtype(torch::kFloat32));
    for(int i=0; i< indices.size(); ++i){
        auto list = indices[i];
        starts[i] = sum_of_points;
        lengths[i] = int(list.size());
        sum_of_points+=list.size();
        auto index3d = access_3d(i);
        for(int j=0; j<3; ++j)
        {
            cell_bb_min[i][j] = bbmin[j] + index3d[j] * gridsize;
            cell_bb_length[i][j] = gridsize;
        }
    }

    t_cells_start = starts.unsqueeze(1).to(torch::kInt32)
                      .contiguous()
                      .cuda()
                      .clone();

    t_cells_length = lengths.unsqueeze(1).to(torch::kInt32)
                       .contiguous()
                       .cuda()
                       .clone();
    t_cell_bb_min = cell_bb_min
                       .contiguous()
                       .cuda()
                       .clone();
    t_cell_bb_length = cell_bb_length
                       .contiguous()
                       .cuda()
                       .clone();

    Reorder(indices_tensor.squeeze(1));
#endif


// void NeuralPointCloudCudaImpl::Reorder(torch::Tensor indices){
//     torch::NoGradGuard ngg;
//
//     SAIGA_ASSERT(indices.sizes().size() == 1);
//
//     t_position = t_position.index({indices}).contiguous();
//     t_index = t_index.index({indices}).contiguous();
//     t_normal = t_normal.index({indices}).contiguous();
//
//
// }


Saiga::UnifiedMesh NeuralPointCloudCudaImpl::Mesh()
{
    Saiga::UnifiedMesh mesh;

    std::cout << "Extracing Point Cloud from device data" << std::endl;
    std::cout << "Tensors included:" << std::endl;
    // Position
    PrintTensorInfo(t_position);
    std::vector<vec4> data_position(t_position.size(0), vec4(-1, -1, -1, -1));
    torch::Tensor cp_position = t_position.contiguous().cpu();
    memcpy(data_position[0].data(), cp_position.data_ptr(), sizeof(vec4) * data_position.size());

    for (auto p : data_position)
    {
        mesh.position.push_back(p.head<3>());
        mesh.data.push_back(vec4(0, 0, 0, p(3)));
    }
    // Normal
    if (t_normal.defined())
    {
        std::cout << "Normal: " << TensorInfo(t_normal) << std::endl;
        std::vector<int> data_normal(t_normal.size(0));
        torch::Tensor cp_normal = t_normal.contiguous().cpu();
        memcpy(data_normal.data(), cp_normal.data_ptr(), sizeof(int) * data_normal.size());

        for (auto n : data_normal)
        {
            vec3 n_dec = UnpackNormal10Bit(n);
            mesh.normal.push_back(n_dec);
        }
    }

    // Color
    PrintTensorInfo(t_original_color);
    std::vector<vec4> org_col(t_original_color.size(0), vec4(-1, -1, -1, -1));
    torch::Tensor cp_col = t_original_color.contiguous().cpu();
    memcpy(org_col[0].data(), cp_col.data_ptr(), sizeof(vec4) * org_col.size());
    std::cout << "End of tensors included:" << std::endl;

    for (auto c : org_col)
    {
        mesh.color.push_back(c);
    }

    return mesh;
}
void NeuralPointCloudCudaImpl::MakeOutlier(int max_index)
{
    SAIGA_ASSERT(0);
    torch::NoGradGuard ngg;
    t_index.uniform_(0, max_index);
}
std::vector<int> NeuralPointCloudCudaImpl::Indices()
{
    std::vector<int> indices(t_index.size(0));

    torch::Tensor cp_index = t_index.contiguous().cpu();

    memcpy(indices.data(), cp_index.data_ptr(), sizeof(int) * indices.size());

    return indices;
}
void NeuralPointCloudCudaImpl::SetIndices(std::vector<int>& indices)
{
#if 1
    // torch::NoGradGuard  ngg;
    t_index.set_data(
        torch::from_blob(indices.data(), {(long)indices.size(), 1}, torch::TensorOptions().dtype(torch::kFloat32))
            .contiguous()
            .cuda()
            .clone());
#else
    t_index = torch::from_blob(indices.data(), {(long)indices.size(), 1}, torch::TensorOptions().dtype(torch::kFloat32))
                  .contiguous()
                  .cuda()
                  .clone();
#endif
}
int NeuralPointCloudCudaImpl::Size()
{
    SAIGA_ASSERT(t_position.defined());
    return t_position.size(0);
}
