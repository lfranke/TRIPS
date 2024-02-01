/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/cuda/imgui_cuda.h"
#include "saiga/cuda/interop.h"
#include "saiga/opengl/all.h"
#include "saiga/opengl/rendering/deferredRendering/tone_mapper.h"
#include "saiga/opengl/rendering/lighting/bloom.h"

#include "config.h"
#include "data/Dataset.h"
#include "models/Pipeline.h"
#include "opengl/SceneViewer.h"

// #define MINIMAL_GUI


// Helper class which is able to use a pretrained network to render a neural point cloud
// The output is an RGB image of the given viewpoint
class RealTimeRenderer
{
   public:
    RealTimeRenderer(std::shared_ptr<SceneData> scene);

    void Forward(ImageInfo fd, int rotate_result_90deg = 0, vec3 debug_refl_dir = vec3(0, 0, 0));

    torch::Tensor Render(ImageInfo fd, vec3 debug_refl_dir = vec3(0, 0, 0), int border_to_render = 0);

    // flags:
    // 0: color
    //
    void RenderColor(ImageInfo fd, int flags, vec3 debug_refl_dir = vec3(0, 0, 0));

    void SetupRenderedDebugColor();

    void imgui();

    void overrideForCustomCam(ImageInfo& fd, float& old_cutoff);

    std::shared_ptr<Texture> getClosestGTImage(ImageInfo fd);

    struct Experiment
    {
        // full (absolute) directory of the experimenet folder
        std::string dir;

        // only the name
        std::string name;

        struct EP
        {
            // full (absolute) directory of the epxxxx folder
            std::string dir;

            // only the name. for example "ep0005"
            std::string name;

            // for example "kemenate"
            std::string scene_name;

            // ep number
            int ep = 0;
        };
        std::vector<EP> eps;

        Experiment(std::string dir, std::string name, std::string scene_name, bool render_able = true);
    };

    std::string experiments_base = "experiments/";
    std::vector<Experiment> experiments;
    int current_ex      = 0;
    int current_ep      = 0;
    int current_best_gt = -1;
    int best_gt_counter = 0;

    bool mouse_on_view = false;
    void LoadNets();

    TemplatedImage<vec4> output_image;
    TemplatedImage<ucvec4> output_image_ldr;
    std::shared_ptr<Texture> output_texture, output_texture_ldr, output_color, best_gt_texture;
    std::shared_ptr<Saiga::CUDA::Interop> texure_interop, color_interop;
    AABB custom_discard_aabb;
    bool reset_new_ex                      = false;
    bool render_env_map                    = true;
    NeuralPointTexture debug_color_texture = nullptr;
    torch::Tensor debug_color_texture_texture;

    std::shared_ptr<SceneData> scene;
    std::shared_ptr<NeuralScene> ns;
    std::shared_ptr<NeuralPipeline> pipeline;

    // The real-time camera parameters for live viewing
    IntrinsicsModule rt_intrinsics = nullptr;
    PoseModule rt_extrinsics       = nullptr;

    torch::Tensor uv_tensor, uv_tensor_center, direction_tensor;
    bool use_center_tensor = false;


    bool use_gl_tonemapping = false;
    bool use_bloom          = false;
    bool render_color       = true;

    // The custom camera can be controlled by the user and might be a different model
    bool use_custom_camera = true;

    float stability_cutoff_value            = 0.f;
    float stability_cutoff_value_multiplier = 1.f;

    // int current_state_debug_show          = 0;
    // std::vector<std::string> debug_states = {"normal", "confidence_value"};
    // int use_which_layer                   = -1;

    int discard_with_state                       = 0;
    std::vector<std::string> discard_with_option = {"_via_org_confidence_value", "_via_org_confidence_value"};
    int use_which_layer_discard                  = -1;
    bool invert_colors_and_red                   = false;
    // bool mask_out_original_points                = false;

    bool use_discard_in_main_render_window = false;
    bool use_visualize_confidence_as_full  = false;

    int color_layer = 1;
#ifndef MINIMAL_GUI
    int color_flags   = 0;
    float color_scale = 1.f;
#else
    int color_flags   = 1;
    float color_scale = 10.f;
#endif


    int current_eval_epoch = 600;
    // int color_flags   = 1;
    // float color_scale = 16.f;

    float fixed_confidence = 0.f;

    int manuel_timestep = -1;

    vec3 point_spawn_debug_color = vec3(1, 0, 1);
    vec3 prev_debug_col          = vec3(1, 1, 1);

    int render_mode = 0;

    torch::DeviceType device = torch::kCUDA;
    CUDA::CudaTimerSystem timer_system;

    // Default saiga renderer uses 16-bit float for HDR content
    ToneMapper tone_mapper = {GL_RGBA32F};
    Bloom bloom            = {GL_RGBA32F};
    std::shared_ptr<CombinedParams> params;

    TemplatedImage<ucvec4> DownloadRender()
    {
        if (use_gl_tonemapping)
        {
            SAIGA_ASSERT(output_texture_ldr);
            TemplatedImage<ucvec4> tmp(output_texture_ldr->getHeight(), output_texture_ldr->getWidth());

            output_texture_ldr->bind();
            glGetTexImage(output_texture_ldr->getTarget(), 0, GL_RGBA, GL_UNSIGNED_BYTE, tmp.data());
            assert_no_glerror();
            output_texture_ldr->unbind();

            return tmp;
        }
        else
        {
            SAIGA_ASSERT(output_texture);
            TemplatedImage<ucvec4> tmp(output_texture->getHeight(), output_texture->getWidth());

            output_texture->bind();
            glGetTexImage(output_texture->getTarget(), 0, GL_RGBA, GL_UNSIGNED_BYTE, tmp.data());
            assert_no_glerror();
            output_texture->unbind();

            return tmp;
        }
    }

    TemplatedImage<ucvec4> DownloadColor()
    {
        SAIGA_ASSERT(output_color);
        TemplatedImage<ucvec4> tmp(output_color->getHeight(), output_color->getWidth());

        output_color->bind();
        glGetTexImage(output_color->getTarget(), 0, GL_RGBA, GL_UNSIGNED_BYTE, tmp.data());
        assert_no_glerror();
        output_color->unbind();
        return tmp;
    }

    TemplatedImage<ucvec4> DownloadGt()
    {
        SAIGA_ASSERT(best_gt_texture);
        TemplatedImage<ucvec4> tmp(best_gt_texture->getHeight(), best_gt_texture->getWidth());

        best_gt_texture->bind();
        glGetTexImage(best_gt_texture->getTarget(), 0, GL_RGBA, GL_UNSIGNED_BYTE, tmp.data());
        assert_no_glerror();
        best_gt_texture->unbind();
        return tmp;
    }
};