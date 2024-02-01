/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#pragma once

#include "NeuralPointCloudOpenGL.h"
#include "models/Pipeline.h"
#ifdef UNIX


class GridGLRenderer : public Saiga::Object3D
{
   public:
    GridGLRenderer(NeuralPointCloudCuda neural_pc);

    void render(const FrameData& fd, float cutoff_val, int mode, bool cutoff_as_percent = false);

   private:
    std::vector<float> sorted_grid_values;

    UniformBuffer ocam_model_ssbo;

    std::shared_ptr<Saiga::Shader> shader;

    Saiga::VertexBuffer<PositionIndex> gl_grid;

    Saiga::TemplatedBuffer<vec4> gl_normal = {GL_ARRAY_BUFFER};
    Saiga::TemplatedBuffer<vec4> gl_color  = {GL_ARRAY_BUFFER};
    Saiga::TemplatedBuffer<vec4> gl_data   = {GL_ARRAY_BUFFER};
};
#endif