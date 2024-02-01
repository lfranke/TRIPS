/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/glfw/all.h"

#include "../config.h"
#include "data/SceneData.h"
#include "opengl/NeuralPointCloudOpenGL.h"

using namespace Saiga;

class SceneViewer : public glfw_KeyListener
{
   public:
    SceneViewer(std::shared_ptr<SceneData> scene);

    void OptimizePoints();

    void CreateMasks(bool mult_old_mask, float size_of_mask_in_ws = 0.1f);

    void RemoveInvalidPoints();

    void SetRandomPointColor();

    // Render the shapes for the cameras
    void RenderDebug(Camera* cam);

    void RenderMultiscalarSpheres(Camera* cam, bool recompute, int num_spheres = 4, float inner_radius = 10.f,
                                  float radius_factor = 5.f, vec3 mid_point = vec3(0, 0, 0));

    std::shared_ptr<ColoredAsset> capture_asset;
    std::shared_ptr<LineVertexColoredAsset> frustum_asset;

    void Select(Ray r);

    void UpdatePointCloudBuffer();

    FrameData& Current()
    {
        SAIGA_ASSERT(selected_capture >= 0 && selected_capture < scene->frames.size());
        return scene->frames[selected_capture];
    }

    void DeleteCurrent()
    {
        if (selected_capture != -1)
        {
            scene->frames[selected_capture] = scene->frames.back();
            scene->frames.pop_back();
            selected_capture = -1;
        }
    }

    void imgui();


    std::shared_ptr<NeuralPointCloudOpenGL> gl_points;

    std::shared_ptr<SceneData> scene;

    UnifiedModel model;

    CameraModel free_view_camera_type = CameraModel::PINHOLE_DISTORTION;
    // IntrinsicsPinholef default_pinhole = IntrinsicsPinholef(1000.f, 1000.f, 1920.f / 2.f, 1080.f / 2.f, 0.f);
    /// IntrinsicsPinholef default_pinhole = IntrinsicsPinholef(512.f, 512.f, 512.f, 512.f, 0.f);
    // IntrinsicsPinholef default_pinhole = IntrinsicsPinholef(1500.f, 1500.f, 540.f, 960.f, 0.f);
    float pinhole_cx = 540.f;
    float pinhole_cy = 960.f;
    float pinhole_fx = 1500.f;
    float pinhole_fy = 1500.f;

    std::vector<float> default_cam_to_world =
        std::vector<float>({-1372.79, 0, 0.00036144, -2.81704e-07, 2.64439e-10, -1.07414e-13, 1.76e-17});
    std::vector<float> default_world_to_cam =
        std::vector<float>({2180.44, 1372.7, -196.498, -283.005, 139.021, 500.373, -99.8263, -687.294, 46.9305, 777.54,
                            120.427, -587.192, -227.447, 242.275, 158.027, -26.5849, -39.8955, -8.11842});
    OCam<float> default_ocam = OCam<float>(5472, 3648, {1.00019, -7.17476e-05, 0.000243079, 1824.93, 2725.09},
                                           default_cam_to_world, default_world_to_cam);

    IntrinsicsPinholef crop_debug_intrinsics;
    ivec2 crop_debug_size;
    Matrix<float, 2, 2> crop_debug_rotation;
    float rot_angle             = 0.f;
    bool use_crop_for_rendering = false;


    enum VIEWER_CAMERA_STATE
    {
        DEFAULT    = 0,
        PINHOLE    = 1,
        OCAM       = 2,
        SPHERICAL  = 3,
        ORTHO      = 4,
        EXTRA_EVAL = 5,
    } viewer_camera = VIEWER_CAMERA_STATE::DEFAULT;


    ImageInfo CurrentFrameData() const
    {
        ImageInfo fd;
        fd.crop_rotation.setZero();
        fd.crop_rotation(0, 0) = 1;
        fd.crop_rotation(1, 1) = 1;
        if (viewer_camera == VIEWER_CAMERA_STATE::DEFAULT)
        {
            fd.w                 = scene->scene_cameras[0].w;  // * scene->dataset_params.render_scale;
            fd.h                 = scene->scene_cameras[0].h;  // * scene->dataset_params.render_scale;
            fd.K                 = scene->scene_cameras[0].K;
            fd.camera_model_type = CameraModel::PINHOLE_DISTORTION;
            if (scene->scene_cameras[0].camera_model_type == CameraModel::OCAM)
            {
                // cx and cy are "switched"" in the ocam model, as it is computed with x and y flipped
                // fd.K = IntrinsicsPinholef(scene->scene_cameras[0].w * scene->dataset_params.render_scale,
                //                          scene->scene_cameras[0].h * scene->dataset_params.render_scale,
                //                          scene->scene_cameras[0].ocam.cy, scene->scene_cameras[0].ocam.cx, 0);

                fd.camera_model_type = CameraModel::OCAM;
            }
            fd.distortion = scene->scene_cameras[0].distortion;
            fd.ocam       = scene->scene_cameras[0].ocam.cast<float>();
        }
        else if (viewer_camera == VIEWER_CAMERA_STATE::PINHOLE)
        {
            fd.w = 1080.f;  // default_pinhole.cx * 2;  // * scene->dataset_params.render_scale;
            fd.h = 1920.f;  // default_pinhole.cy * 2;  // * scene->dataset_params.render_scale;
            // fd.w = scene->scene_cameras[0].w;
            // fd.h = scene->scene_cameras[0].h;
            IntrinsicsPinholef default_pinhole =
                IntrinsicsPinholef(pinhole_fx, pinhole_fy, pinhole_cx, pinhole_cy, 0.f);

            //  fd.K = IntrinsicsPinholef(scene->scene_cameras[0].w / 2, scene->scene_cameras[0].w / 2,
            //         scene->scene_cameras[0].w / 2, scene->scene_cameras[0].h / 2, 0);
            fd.K                 = default_pinhole;
            fd.camera_model_type = CameraModel::PINHOLE_DISTORTION;
            fd.distortion        = Distortionf();
        }
        else if (viewer_camera == VIEWER_CAMERA_STATE::OCAM)
        {
            fd.w                               = default_ocam.w * 0.5f;
            fd.h                               = default_ocam.h * 0.5f;
            IntrinsicsPinholef default_pinhole = IntrinsicsPinholef(1500.f, 1500.f, 540.f, 960.f, 0.f);

            fd.K                 = default_pinhole;
            fd.camera_model_type = CameraModel::OCAM;
            // fd.crop_transform    = fd.crop_transform.scale(0.5f);
            fd.ocam = default_ocam;
        }
        else if (viewer_camera == VIEWER_CAMERA_STATE::SPHERICAL)
        {
            fd.camera_model_type = CameraModel::SPHERICAL;
            fd.w                 = 4096;
            fd.h                 = 2048;
        }
        else if (viewer_camera == VIEWER_CAMERA_STATE::ORTHO)
        {
            fd.camera_model_type = CameraModel::ORTHO;
            fd.w                 = 2048;
            fd.h                 = 2048;
        }
        else if (viewer_camera == VIEWER_CAMERA_STATE::EXTRA_EVAL)
        {
            fd.camera_model_type = CameraModel::PINHOLE_DISTORTION;
            fd.w                 = scene->extra_eval_cameras[0].w;
            fd.h                 = scene->extra_eval_cameras[0].h;
            fd.K                 = scene->extra_eval_cameras[0].K;
            fd.distortion        = scene->extra_eval_cameras[0].distortion;
        }
        fd.w *= scene->dataset_params.render_scale;
        fd.h *= scene->dataset_params.render_scale;
        if (use_crop_for_rendering)
        {
            fd.crop_transform = crop_debug_intrinsics;
            fd.crop_rotation  = crop_debug_rotation;
            fd.w              = crop_debug_size.x();
            fd.h              = crop_debug_size.y();
        }
        fd.crop_transform = fd.crop_transform.scale(scene->dataset_params.render_scale);
        fd.pose           = Sophus::SE3f::fitToSE3(scene_camera.model * GL2CVView()).cast<double>();

        return fd;
    }
    Glfw_Camera<PerspectiveCamera> scene_camera;

    // temp values
    int selected_capture = -1;
    bool render_frustums = true;

    bool render_custom_aabb         = false;
    bool use_custom_aabb_to_cut_out = false;
    bool no_env_map                 = false;
    AABB custom_aabb;
    // only for ImGui
    vec2 max_min_custom_aabb    = vec2(-3, 3);
    bool custom_aabb_dirty_flag = true;
    vec4 custom_aabb_color      = vec4(1, 0, 0, 1);


    bool render_experiment_sphere = true;
    bool render_custom_sphere     = false;
    int custom_num_spheres        = 4;
    float custom_inner_radius     = 10.f;
    float custom_radius_factor    = 5.f;
    bool custom_sphere_dirty_flag = true;

    virtual void keyPressed(int key, int scancode, int mods) override
    {
        switch (key)
        {
            case GLFW_KEY_X:
            {
                DeleteCurrent();
                break;
            }
            default:
                break;
        };
    }
};
