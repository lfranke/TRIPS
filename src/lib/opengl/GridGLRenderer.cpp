/**
* Copyright (c) 2024 Darius Rückert and Linus Franke
* Licensed under the MIT License.
* See LICENSE file for more information.
*/
#include "GridGLRenderer.h"
#ifdef UNIX
struct ocam_model
{
    ivec2 image_size;
    float c;
    float d;
    float e;
    float cx;
    float cy;
    int world2cam_size;
    float world2cam[20];

    int cam2world_size;
    float cam2world[20];
};
template <typename T>
std::vector<T> tensor_to_vector(torch::Tensor t)
{
    auto t_x = t.contiguous().cpu();
    SAIGA_ASSERT(t.sizes().size() == 2);
    SAIGA_ASSERT(t.dtype() == torch::kFloat);
    uint32_t floats_per_elem = sizeof(T) / sizeof(float);
    std::vector<T> vec(t_x.numel() / floats_per_elem);
    // std::cout << t.sizes() << t.numel()<< std::endl;
    std::memcpy((float*)vec.data(), t_x.data_ptr<float>(), t_x.numel() * sizeof(float));
    //   std::cout << t_x.numel() << std::endl;

    // std::vector<T> vec = std::vector<T>(t_x.data_ptr<float>(), t_x.data_ptr<float>()+t_x.numel());
    return vec;
}

GridGLRenderer::GridGLRenderer(NeuralPointCloudCuda neural_pc)
{
    ocam_model_ssbo.createGLBuffer(nullptr, sizeof(ocam_model), GL_DYNAMIC_DRAW);

    std::vector<vec3> grid_mins = tensor_to_vector<vec3>(neural_pc->t_cell_bb_min);
    //    std::cout << neural_pc->t_cell_bb_min.sizes() << ": " << grid_mins.size() << std::endl;
    std::cout << "1" << std::endl;
    std::vector<vec3> grid_lengths = tensor_to_vector<vec3>(neural_pc->t_cell_bb_length);
    std::cout << "1" << std::endl;
    std::vector<float> grid_values = tensor_to_vector<float>(neural_pc->t_cell_value);
    std::cout << "1" << std::endl;
    std::vector<vec3> grid_colors = neural_pc->DebugColorsPerBoxCPU();
    std::cout << "1" << std::endl;
    std::vector<vec3> pos_cube = {
        vec3(-1.0f, -1.0f, -1.0f),                            // triangle 1 : begi
        vec3(-1.0f, -1.0f, 1.0f),  vec3(-1.0f, 1.0f, 1.0f),   // triangle 1 : en)d
        vec3(1.0f, 1.0f, -1.0f),                              // triangle 2 : begi)n
        vec3(-1.0f, -1.0f, -1.0f), vec3(-1.0f, 1.0f, -1.0f),  // triangle 2 : en)d
        vec3(1.0f, -1.0f, 1.0f),   vec3(-1.0f, -1.0f, -1.0f), vec3(1.0f, -1.0f, -1.0f),  vec3(1.0f, 1.0f, -1.0f),
        vec3(1.0f, -1.0f, -1.0f),  vec3(-1.0f, -1.0f, -1.0f), vec3(-1.0f, -1.0f, -1.0f), vec3(-1.0f, 1.0f, 1.0f),
        vec3(-1.0f, 1.0f, -1.0f),  vec3(1.0f, -1.0f, 1.0f),   vec3(-1.0f, -1.0f, 1.0f),  vec3(-1.0f, -1.0f, -1.0f),
        vec3(-1.0f, 1.0f, 1.0f),   vec3(-1.0f, -1.0f, 1.0f),  vec3(1.0f, -1.0f, 1.0f),   vec3(1.0f, 1.0f, 1.0f),
        vec3(1.0f, -1.0f, -1.0f),  vec3(1.0f, 1.0f, -1.0f),   vec3(1.0f, -1.0f, -1.0f),  vec3(1.0f, 1.0f, 1.0f),
        vec3(1.0f, -1.0f, 1.0f),   vec3(1.0f, 1.0f, 1.0f),    vec3(1.0f, 1.0f, -1.0f),   vec3(-1.0f, 1.0f, -1.0f),
        vec3(1.0f, 1.0f, 1.0f),    vec3(-1.0f, 1.0f, -1.0f),  vec3(-1.0f, 1.0f, 1.0f),   vec3(1.0f, 1.0f, 1.0f),
        vec3(-1.0f, 1.0f, 1.0f),   vec3(1.0f, -1.0f, 1.0f)};

    std::vector<PositionIndex> vb_pos;
    std::vector<uint32_t> idx_buf;
    std::vector<vec4> colors_value;
    std::vector<vec4> colors_index;
    sorted_grid_values.clear();

    for (int i = 0; i < grid_mins.size(); ++i)
    {
        for (int p = 0; p < pos_cube.size(); ++p)
        {
            idx_buf.push_back(vb_pos.size());
            PositionIndex pi;
            pi.position =
                grid_mins[i].array() + (pos_cube[p] * 0.5 + vec3(0.5, 0.5, 0.5)).array() * grid_lengths[i].array();
            pi.index = vb_pos.size();
            vb_pos.push_back(pi);
            sorted_grid_values.push_back(grid_values[i]);
            colors_value.push_back(vec4(grid_values[i], grid_values[i], grid_values[i], 1));
            colors_index.push_back(vec4(grid_colors[i].x(), grid_colors[i].y(), grid_colors[i].z(), 1));
        }
    }

    shader = Saiga::shaderLoader.load<Saiga::Shader>("grid_render.glsl");
    gl_grid.setDrawMode(GL_TRIANGLES);
    gl_grid.set(vb_pos, GL_STATIC_DRAW);

    gl_color.create(colors_index, GL_STATIC_DRAW);
    gl_grid.addExternalBuffer(gl_color, 1, 4, GL_FLOAT, GL_FALSE, sizeof(vec4), 0);

    gl_data.create(colors_value, GL_STATIC_DRAW);
    gl_grid.addExternalBuffer(gl_data, 4, 4, GL_FLOAT, GL_FALSE, sizeof(vec4), 0);
    std::cout << "s1" << std::endl;
    std::sort(sorted_grid_values.begin(), sorted_grid_values.end());
    std::cout << "s1" << std::endl;
}

void GridGLRenderer::render(const FrameData& fd, float cutoff_val, int mode, bool cutoff_as_percent)
{
    float cutoff = cutoff_val;
    if (cutoff_as_percent)
    {
        cutoff_val = 1.f - clamp(cutoff_val, 0.f, 1.0f);
        cutoff     = sorted_grid_values[int(sorted_grid_values.size() * cutoff_val)];
    }
    if (shader->bind())
    {
        //  glPointSize(1);
        //  glEnable(GL_PROGRAM_POINT_SIZE);
        auto cam = fd.GLCamera();

        shader->upload(0, mat4(mat4::Identity()));
        shader->upload(1, cam.view);
        shader->upload(2, cam.proj);

        mat4 v = fd.pose.inverse().matrix().cast<float>().eval();
        mat3 k = fd.K.matrix();
        vec2 s(fd.w, fd.h);

        shader->upload(3, v);
        shader->upload(4, k);
        shader->upload(5, s);
        shader->upload(6, scale);

        vec4 dis1 = fd.distortion.Coeffs().head<4>();
        vec4 dis2 = fd.distortion.Coeffs().tail<4>();

        shader->upload(7, dis1);
        shader->upload(8, dis2);
        shader->upload(9, exp2(fd.exposure_value));

        // render settings
        shader->upload(10, mode);

        shader->upload(11, cutoff);
        shader->upload(12, (int)fd.camera_model_type);

        ocam_model ocam_mod;
        ocam_mod.c              = fd.ocam.c;
        ocam_mod.d              = fd.ocam.d;
        ocam_mod.e              = fd.ocam.e;
        ocam_mod.cx             = fd.ocam.cx;
        ocam_mod.cy             = fd.ocam.cy;
        ocam_mod.image_size     = ivec2(fd.w, fd.h);
        ocam_mod.world2cam_size = fd.ocam.poly_world2cam.size();

        for (int i = 0; i < fd.ocam.poly_world2cam.size(); ++i)
        {
            ocam_mod.world2cam[i] = fd.ocam.poly_world2cam[i];
        }
        //  std::cout << fd.ocam.poly_world2cam.size() << " _ " << fd.ocam.poly_cam2world.size() << std::endl;
        ocam_model_ssbo.updateBuffer(&ocam_mod, sizeof(ocam_model), 0);

        ocam_model_ssbo.bind(6);



        gl_grid.bindAndDraw();

        // glDisable(GL_PROGRAM_POINT_SIZE);
        shader->unbind();
        //   ocam_model_ssbo.unbind();
    }
}
#endif