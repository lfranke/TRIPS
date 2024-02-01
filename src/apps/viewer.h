/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/geometry/cameraAnimation.h"
#include "saiga/core/glfw/all.h"
#include "saiga/opengl/assets/AssetRenderSystem.h"
#include "saiga/opengl/assets/all.h"
#include "saiga/opengl/rendering/deferredRendering/deferredRendering.h"
#include "saiga/opengl/rendering/forwardRendering/forwardRendering.h"
#include "saiga/opengl/rendering/renderer.h"
#include "saiga/opengl/window/WindowTemplate.h"
#include "saiga/opengl/window/glfw_window.h"
#include "saiga/opengl/world/LineSoup.h"
#include "saiga/opengl/world/TextureDisplay.h"
#include "saiga/opengl/world/pointCloud.h"

#include "config.h"
#include "opengl/RealTimeRenderer.h"
#include "opengl/SceneViewer.h"

#include "viewer_base.h"
using namespace Saiga;


using WindowType              = glfw_Window;
constexpr WindowManagement wm = WindowManagement::GLFW;


class Viewer : public StandaloneWindow<wm, DeferredRenderer>, public glfw_KeyListener, glfw_MouseListener, ViewerBase
{
   public:
    Viewer(std::string scene_dir, std::unique_ptr<DeferredRenderer> renderer_, std::unique_ptr<WindowType> window_);
    ~Viewer() {}


    void LoadSceneImpl();


    void update(float dt) override
    {
        if (next_scene != current_scene)
        {
            LoadSceneImpl();
        }

        renderer->tone_mapper.params_dirty = true;


        scene->scene_camera.update(dt);
    }

    void interpolate(float dt, float interpolation) override
    {
        if (renderer->use_mouse_input_in_3dview || renderer->use_mouse_input_in_3dview || mouse_in_gt)
        {
            scene->scene_camera.interpolate(dt, interpolation);
        }
        mouse_in_gt = false;
    }

    void render(RenderInfo render_info) override;
    bool mouse_in_gt = false;


    void Recording(ImageInfo& fd);

    virtual void keyPressed(int key, int scancode, int mods) override
    {
        switch (key)
        {
            case GLFW_KEY_Q:
            {
                auto& f                                     = scene->scene->frames[scene->selected_capture];
                renderer->tone_mapper.params.exposure_value = f.exposure_value;
                camera->setModelMatrix(f.OpenglModel());
                camera->updateFromModel();
                break;
            }
            case GLFW_KEY_ESCAPE:
            {
                window->close();
                break;
            }
            case GLFW_KEY_T:
            {
                camera->rotateGlobal(camera->getDirection().head<3>(), 5.f);
                break;
            }
            case GLFW_KEY_G:
            {
                camera->rotateGlobal(camera->getDirection().head<3>(), -5.f);
                break;
            }
            default:
                break;
        };
    }


    void mousePressed(int key, int x, int y) override
    {
        ivec2 global_pixel = ivec2(x, y);
        ivec2 local_pixel  = renderer->WindowCoordinatesToViewport(global_pixel);

        switch (key)
        {
            case GLFW_MOUSE_BUTTON_RIGHT:
            {
                auto ray = ::camera->PixelRay(local_pixel.cast<float>(), renderer->viewport_size.x(),
                                              renderer->viewport_size.y(), true);
                scene->Select(ray);

                break;
            }
            case GLFW_MOUSE_BUTTON_LEFT:
            {
                break;
            }
        }
    }
    float render_scale = 1.f;

    bool debug_directionality = false;
    vec3 debug_refl_dir       = vec3(0, 0, -1);
    // used to rotate all results left -1, not 0, right +1
    int rotate_result_90deg     = 0;
    bool render_grid            = false;
    float grid_cutoff_val       = 0.05f;
    bool grid_cutoff_as_percent = true;
    int grid_mode               = 0;


   private:
    std::shared_ptr<LineVertexColoredAsset> spline_mesh;
    SplinePath camera_spline;

    ViewMode view_mode;
    std::shared_ptr<DirectionalLight> sun;
    TextureDisplay display;

    // std::shared_ptr<GridGLRenderer> grid_renderer;

    std::unique_ptr<Framebuffer> target_framebuffer;

    // video recording variables
    std::string recording_dir;
};
