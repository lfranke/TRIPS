/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "SceneViewer.h"

#include "saiga/colorize.h"
#include "saiga/core/geometry/AccelerationStructure.h"
#include "saiga/core/imgui/imgui.h"
#include "saiga/core/model/model_from_shape.h"
#include "saiga/core/util/ProgressBar.h"
#include "saiga/vision/torch/RandomCrop.h"

#include "opengl/RealTimeRenderer.h"

float sphere_radius = 0.01;
float frustum_size  = 0.05;

// pose time view
// float sphere_radius = 0.0001;
// float frustum_size =  0.001;

SceneViewer::SceneViewer(std::shared_ptr<SceneData> scene) : scene(scene)
{
    if (std::filesystem::exists(scene->dataset_params.file_model))
    {
        model = UnifiedModel(scene->dataset_params.file_model).AddMissingDummyTextures();
        model.ComputeColor();
        std::cout << "[Scene] Model (Tris) " << model.TotalTriangles() << std::endl;
    }


    UpdatePointCloudBuffer();

    {
        auto obj = IcoSphereMesh(Sphere(vec3(0, 0, 0), sphere_radius), 3);
        obj.SetVertexColor(vec4(1, 1, 1, 1));
        capture_asset = std::make_shared<ColoredAsset>(obj);
    }


    scene_camera.proj                = scene->GLProj();
    scene_camera.zNear               = scene->dataset_params.znear;
    scene_camera.zFar                = scene->dataset_params.zfar;
    scene_camera.global_up           = scene->dataset_params.scene_up_vector;
    scene_camera.recompute_on_resize = false;

    scene_camera.movementSpeed = 5;
    //    scene_camera.mode0 = CameraControlMode::ROTATE_FIRST_PERSON_FIX_UP_VECTOR;
    scene_camera.mode0 = CameraControlMode::ROTATE_FIRST_PERSON;
    scene_camera.mode1 = CameraControlMode::ROTATE_AROUND_POINT_FIX_UP_VECTOR;

    scene_camera.enableInput();

    {
        auto obj = FrustumLineMesh(scene_camera.proj, frustum_size, false);
        obj.SetVertexColor(vec4(1, 1, 1, 1));
        frustum_asset = std::make_shared<LineVertexColoredAsset>(obj);
    }

    crop_debug_size = ivec2(scene->scene_cameras[0].w * scene->dataset_params.render_scale,
                            scene->scene_cameras[0].h * scene->dataset_params.render_scale);
    crop_debug_rotation.setZero();
    crop_debug_rotation(0, 0) = 1;
    crop_debug_rotation(1, 1) = 1;
}

void SceneViewer::OptimizePoints()
{
    {
        ScopedTimerPrintLine tim("ReorderMorton64");
        scene->point_cloud.ReorderMorton64();
    }
    {
        ScopedTimerPrintLine tim("RandomBlockShuffle");
        scene->point_cloud.RandomBlockShuffle(default_point_block_size);
    }

    UpdatePointCloudBuffer();
}


void SceneViewer::SetRandomPointColor()
{
    for (auto& c : scene->point_cloud.color)
    {
        vec3 nc     = Random::MatrixUniform<vec3>(0, 1);
        c.head<3>() = nc;
    }
    UpdatePointCloudBuffer();
}


void SceneViewer::RemoveInvalidPoints()
{
    console << "RemoveInvalidPoints" << std::endl;
    std::vector<int> to_erase;
    for (int i = 0; i < scene->point_cloud.NumVertices(); ++i)
    {
        if (scene->point_cloud.color[i].x() <= 0)
        {
            to_erase.push_back(i);
        }
    }

    scene->point_cloud.EraseVertices(to_erase);
    UpdatePointCloudBuffer();
}
void SceneViewer::Select(Ray r)
{
    float best_t = 10000;
    int best_id  = -1;


    for (int i = 0; i < scene->frames.size(); ++i)
    {
        auto& c = scene->frames[i];
        Sphere s(c.pose.translation().cast<float>(), sphere_radius * 4);

        float t1, t2;

        if (Intersection::RaySphere(r, s, t1, t2))
        {
            if (t1 < best_t)
            {
                best_t  = t1;
                best_id = i;
            }
        }
    }
    selected_capture = best_id;
    std::cout << "Selected Frame " << selected_capture << std::endl;
}



void SceneViewer::imgui()
{
#ifndef MINIMAL_GUI

    if (ImGui::ListBoxHeader("###CameraModel", 4))
    {
        if (ImGui::Selectable("Default")) viewer_camera = VIEWER_CAMERA_STATE::DEFAULT;
        if (ImGui::Selectable("Pinhole")) viewer_camera = VIEWER_CAMERA_STATE::PINHOLE;
        if (ImGui::Selectable("OCam")) viewer_camera = VIEWER_CAMERA_STATE::OCAM;
        if (ImGui::Selectable("Spherical")) viewer_camera = VIEWER_CAMERA_STATE::SPHERICAL;
        if (ImGui::Selectable("Ortho")) viewer_camera = VIEWER_CAMERA_STATE::ORTHO;
        if (ImGui::Selectable("ExtraEval")) viewer_camera = VIEWER_CAMERA_STATE::EXTRA_EVAL;
        ImGui::InputFloat("pinhole_fx", &pinhole_fx);
        ImGui::InputFloat("pinhole_fy", &pinhole_fy);
        ImGui::InputFloat("pinhole_cx", &pinhole_cx);
        ImGui::InputFloat("pinhole_cy", &pinhole_cy);
        ImGui::ListBoxFooter();
    }
#endif
    if (ImGui::ListBoxHeader("###Frames", 10))
    {
        for (int i = 0; i < scene->frames.size(); ++i)
        {
            auto& f = scene->frames[i];

            auto str = "Frame " + std::to_string(i) + " exp: " + std::to_string(f.exposure_value) + " wp " +
                       std::to_string(f.white_balance(0)) + " " + std::to_string(f.white_balance(2));
            if (ImGui::Selectable(str.c_str(), i == selected_capture))
            {
                selected_capture = i;
                camera->setModelMatrix(f.OpenglModel());
                camera->updateFromModel();
            }
        }
        ImGui::ListBoxFooter();
    }
#ifndef MINIMAL_GUI
    if (scene->extra_eval_frames.size() > 0)
    {
        if (ImGui::ListBoxHeader("###ExtraEvalFrames", 10))
        {
            for (int i = 0; i < scene->extra_eval_frames.size(); ++i)
            {
                auto& f = scene->extra_eval_frames[i];

                auto str = "Frame " + std::to_string(i) + " exp: " + std::to_string(f.exposure_value) + " wp " +
                           std::to_string(f.white_balance(0)) + " " + std::to_string(f.white_balance(2));
                if (ImGui::Selectable(str.c_str(), i == selected_capture))
                {
                    selected_capture = i;
                    camera->setModelMatrix(f.OpenglModel());
                    camera->updateFromModel();
                    viewer_camera = VIEWER_CAMERA_STATE::EXTRA_EVAL;
                }
            }
            ImGui::ListBoxFooter();
        }
    }

    ImGui::Text("Scene Exposure: %f", scene->dataset_params.scene_exposure_value);

    if (ImGui::CollapsingHeader("Crop Transform"))
    {
        ImGui::Checkbox("use crop for rendering", &use_crop_for_rendering);
        if (ImGui::SliderInt("Crop Size", &crop_debug_size.x(), 32, 2048))
        {
            crop_debug_size.y() = crop_debug_size.x();
        }

        if (ImGui::SliderFloat("Zoom", &crop_debug_intrinsics.fx, 0.25f, 4.f))
        {
            crop_debug_intrinsics.fy = crop_debug_intrinsics.fx;
        }
        if (ImGui::SliderFloat("Rotate", &rot_angle, -M_PI, M_PI))
        {
            crop_debug_rotation(0, 0) = std::cos(rot_angle);
            crop_debug_rotation(0, 1) = -std::sin(rot_angle);
            crop_debug_rotation(1, 0) = std::sin(rot_angle);
            crop_debug_rotation(1, 1) = std::cos(rot_angle);
        }
        ImGui::SliderFloat("T_x", &crop_debug_intrinsics.cx, -scene->scene_cameras[0].w, scene->scene_cameras[0].w);
        ImGui::SliderFloat("T_y", &crop_debug_intrinsics.cy, -scene->scene_cameras[0].h, scene->scene_cameras[0].h);
        if (ImGui::Button("Random Crop"))
        {
            std::cout << scene->scene_cameras[0].w << " " << scene->scene_cameras[0].h << " "
                      << scene->dataset_params.render_scale << std::endl;
            crop_debug_intrinsics = RandomImageCrop(ivec2(scene->scene_cameras[0].w, scene->scene_cameras[0].h) / 2,
                                                    crop_debug_size, true, true, false, vec2(0.75, 1.5));

            {
                float rand_angle          = Random::sampleDouble(-M_PI, M_PI);
                crop_debug_rotation(0, 0) = std::cos(rand_angle);
                crop_debug_rotation(0, 1) = -std::sin(rand_angle);
                crop_debug_rotation(1, 0) = std::sin(rand_angle);
                crop_debug_rotation(1, 1) = std::cos(rand_angle);
            }
        }
    }

    if (ImGui::CollapsingHeader("AABB"))
    {
        if (ImGui::Checkbox("render custom aabb", &render_custom_aabb)) custom_aabb_dirty_flag = true;
        if (ImGui::SliderFloat3("min-vals", custom_aabb.min.data(), max_min_custom_aabb.x(), max_min_custom_aabb.y()))
            custom_aabb_dirty_flag = true;
        if (ImGui::SliderFloat3("max-vals", custom_aabb.max.data(), max_min_custom_aabb.x(), max_min_custom_aabb.y()))
            custom_aabb_dirty_flag = true;
        if (ImGui::Button("printf"))
        {
            std::cout << std::endl
                      << " " << custom_aabb.min.x() << " " << custom_aabb.min.y() << " " << custom_aabb.min.z()
                      << std::endl;
            std::cout << " " << custom_aabb.max.x() << " " << custom_aabb.max.y() << " " << custom_aabb.max.z()
                      << std::endl;
        }
        ImGui::InputFloat("min-vals", &max_min_custom_aabb.x());
        ImGui::InputFloat("max-vals", &max_min_custom_aabb.y());
        if (ImGui::SliderFloat4("color_aabb", custom_aabb_color.data(), 0, 1)) custom_aabb_dirty_flag = true;

        if (ImGui::Checkbox("use aabb to discard everything else", &use_custom_aabb_to_cut_out))
            custom_aabb_dirty_flag = true;
        ImGui::Checkbox("no env map", &no_env_map);
    }

    if (ImGui::CollapsingHeader("Environment Spheres"))
    {
        if (ImGui::Checkbox("render custom spheres", &render_custom_sphere)) custom_sphere_dirty_flag = false;
        ImGui::Checkbox("render experiment spheres", &render_experiment_sphere);
        if (ImGui::InputInt("custom_num_spheres", &custom_num_spheres)) custom_sphere_dirty_flag = true;
        if (ImGui::InputFloat("custom_inner_radius", &custom_inner_radius)) custom_sphere_dirty_flag = true;
        if (ImGui::InputFloat("custom_radius_factor", &custom_radius_factor)) custom_sphere_dirty_flag = true;
    }


    if (ImGui::CollapsingHeader("Rendering"))
    {
        ImGui::Checkbox("render_frustums", &render_frustums);

        if (ImGui::Button("Update Point Buffer"))
        {
            UpdatePointCloudBuffer();
        }

        if (gl_points)
        {
            gl_points->imgui();
        }
    }

    if (ImGui::CollapsingHeader("Augmentation"))
    {
        if (ImGui::Button("invert camera poses"))
        {
            for (auto& f : scene->frames)
            {
                f.pose = f.pose.inverse();
            }
        }


        if (ImGui::Button("duplicate"))
        {
            int n = scene->frames.size();
            for (int i = 0; i < n; ++i)
            {
                scene->frames.push_back(scene->frames[i]);
            }
        }

        static float downfac = 2;
        ImGui::SetNextItemWidth(100);
        ImGui::InputFloat("###downfac", &downfac, 0, 0, "%.5f");
        ImGui::SameLine();
        if (ImGui::Button("downsample points"))
        {
            scene->DownsamplePoints(downfac);
            OptimizePoints();
            scene->ComputeRadius();
            UpdatePointCloudBuffer();
        }

        static int dupfac = 2;
        ImGui::SetNextItemWidth(100);
        ImGui::InputInt("###dupfac", &dupfac);
        ImGui::SameLine();
        static float dupdis = 1;
        ImGui::SetNextItemWidth(100);
        ImGui::InputFloat("###dupdis", &dupdis);
        ImGui::SameLine();
        if (ImGui::Button("dup points"))
        {
            scene->DuplicatePoints(dupfac, dupdis);
            scene->ComputeRadius();
            OptimizePoints();
            UpdatePointCloudBuffer();
        }

        static float rcdis = 0.0005;
        ImGui::SetNextItemWidth(100);
        ImGui::InputFloat("###rcdis", &rcdis, 0, 0, "%.5f");
        ImGui::SameLine();
        if (ImGui::Button("remove close z"))
        {
            scene->RemoveClosePoints(rcdis);
            OptimizePoints();
            scene->ComputeRadius();
            UpdatePointCloudBuffer();
        }



        static float ldis = 0.02;
        ImGui::SetNextItemWidth(100);
        ImGui::InputFloat("###ldis", &ldis, 0, 0, "%.5f");
        ImGui::SameLine();
        if (ImGui::Button("remove lonely z"))
        {
            scene->RemoveLonelyPoints(5, ldis);
            OptimizePoints();
            scene->ComputeRadius();
            UpdatePointCloudBuffer();
        }

        static float doudis = 0.0002;
        ImGui::SetNextItemWidth(100);
        ImGui::InputFloat("###doudis", &doudis, 0, 0, "%.5f");
        ImGui::SameLine();
        if (ImGui::Button("remove close"))
        {
            int bef = scene->point_cloud.NumVertices();
            scene->point_cloud.RemoveDoubles(doudis);
            int aft = scene->point_cloud.NumVertices();
            OptimizePoints();
            scene->ComputeRadius();
            UpdatePointCloudBuffer();
            std::cout << "remove close dis " << doudis << " Points " << bef << " -> " << aft << std::endl;
        }

        static float sdev_noise = 0.1;
        ImGui::SetNextItemWidth(100);
        ImGui::InputFloat("###sdev_noise", &sdev_noise);
        ImGui::SameLine();
        if (ImGui::Button("pose noise"))
        {
            scene->AddPoseNoise(0, sdev_noise);
        }

        static float size_of_mask_in_ws = 0.1f;
        ImGui::InputFloat("###mask_size", &size_of_mask_in_ws);
        ImGui::SameLine();
        if (ImGui::Button("CreateMasks"))
        {
            CreateMasks(true, size_of_mask_in_ws);
        }
    }


    if (ImGui::CollapsingHeader("Point Cloud"))
    {
        static float point_noise = 0.01;
        ImGui::SetNextItemWidth(100);
        ImGui::InputFloat("###point_noise", &point_noise);
        ImGui::SameLine();
        if (ImGui::Button("point noise"))
        {
            scene->AddPointNoise(point_noise);
            UpdatePointCloudBuffer();
        }


        if (ImGui::Button("color points by layout"))
        {
            vec4 c = Random::MatrixUniform<vec4>(0, 1);
            c(3)   = 1;

            for (int i = 0; i < scene->point_cloud.NumVertices(); ++i)
            {
                if (i % 256 == 0)
                {
                    c    = Random::MatrixUniform<vec4>(0, 1);
                    c(3) = 1;
                }
                scene->point_cloud.color[i] = c;
            }
            UpdatePointCloudBuffer();
        }

        if (ImGui::Button("OptimizePoints (memory layout)"))
        {
            OptimizePoints();
        }

        if (ImGui::Button("ComputeRadius"))
        {
            scene->ComputeRadius();
        }

        if (ImGui::Button("SetRandomPointColor")) SetRandomPointColor();
    }

    if (ImGui::Button("save scene"))
    {
        scene->Save();
    }

    if (ImGui::Button("save poses to poses_quatxyzw_transxyz.txt"))
    {
        std::vector<Sophus::SE3d> posesd;
        for (auto f : scene->frames)
        {
            SE3 p = f.pose;
            posesd.push_back(p);
        }
        auto file_pose = scene->scene_path + "/poses_quatxyzw_transxyz.txt";
        SceneData::SavePoses(posesd, file_pose);
    }


    if (ImGui::Button("save original points ply"))
    {
        std::string file = scene->file_dataset_base + "/point_cloud_exported.ply";
        Saiga::UnifiedModel(scene->point_cloud).Save(file);
    }
#endif
}

void SceneViewer::RenderMultiscalarSpheres(Camera* cam, bool recompute, int num_spheres, float inner_radius,
                                           float radius_factor, vec3 mid_point)
{
    static std::vector<std::shared_ptr<ColoredAsset>> spheres;
    // static std::vector<std::shared_ptr<LineVertexColoredAsset>> spheres;

    if (spheres.size() != num_spheres || recompute)
    {
        spheres.clear();
        for (int i = 0; i < num_spheres; ++i)
        {
            float rad   = inner_radius * pow(radius_factor, i);
            auto sphere = Saiga::IcoSphereMesh(Sphere(mid_point, rad), 2);
            for (auto n : sphere.normal)
            {
                n *= -1;
            }
            vec4 col;
            if (i > 4)
                col = (vec4(Saiga::linearRand(0, 1), Saiga::linearRand(0, 1), Saiga::linearRand(0, 1),
                            float(i + 1) / float(num_spheres)));
            if (i == 3) col = (vec4(1, 1, 1, float(i + 1) / float(num_spheres)));
            if (i == 2) col = (vec4(0, 0, 1, float(i + 1) / float(num_spheres)));
            if (i == 1) col = (vec4(0, 1, 0, float(i + 1) / float(num_spheres)));
            if (i == 0) col = vec4(1, 0, 0, 0.5);

            sphere.SetVertexColor(col);
            spheres.push_back(std::make_shared<ColoredAsset>(sphere));
            // spheres.push_back(std::make_shared<LineVertexColoredAsset>(sphere));
            if (i == 0)
                spheres[i]->forwardShader = shaderLoader.load<MVPColorShader>("asset/ColoredAssetWithAlpha.glsl");

            //        std::cout << "Create Sphere (" << rad << ") col:" << sphere.color[0] << std::endl;
        }
    }
    //  glEnable(GL_BLEND);
    //  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_CULL_FACE);


    for (int i = num_spheres - 1; i >= 0; --i)
    {
        auto sphere = spheres[i];

        // sphere->forwardShader->bind();

        if (i == 0)
        {
            glEnable(GL_BLEND);
            //  glDisable(GL_DEPTH_TEST);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            sphere->renderForward(cam, Saiga::mat4::Identity());
            //  glDisable(GL_BLEND);
            glEnable(GL_DEPTH_TEST);
        }
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

        sphere->renderForward(cam, Saiga::mat4::Identity());

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
}


void SceneViewer::RenderDebug(Camera* cam)
{
    // capture_asset->forwardShader   = shaderLoader.load<MVPColorShader>("asset/ColoredAssetWithAlpha.glsl");

    if (capture_asset->forwardShader->bind())
    {
        if (scene)
        {
            if (render_frustums)
            {
                for (int i = 0; i < scene->frames.size(); ++i)
                {
                    auto& c = scene->frames[i];
                    if (i == selected_capture)
                    {
                        capture_asset->forwardShader->uploadColor(exp2(scene->dataset_params.scene_exposure_value) *
                                                                  vec4(1, 1, 0, 1));
                    }
                    else
                    {
                        capture_asset->forwardShader->uploadColor(exp2(scene->dataset_params.scene_exposure_value) *
                                                                  c.display_color);
                    }
                    capture_asset->forwardShader->uploadModel(c.OpenglModel());
                    capture_asset->renderRaw();
                    glLineWidth(3);
                    frustum_asset->renderRaw();
                    glLineWidth(1);
                }
            }
        }
        // reset modular color
        capture_asset->forwardShader->uploadColor(exp2(scene->dataset_params.scene_exposure_value) * vec4(1, 1, 1, 1));

        capture_asset->forwardShader->unbind();
    }

    if (render_custom_aabb)
    {
        static std::shared_ptr<LineVertexColoredAsset> bounding_box_mesh;
        if (custom_aabb_dirty_flag)
        {
            UnifiedMesh bb_scene_unified_mesh = GridBoxLineMesh(custom_aabb);

            bb_scene_unified_mesh.SetVertexColor(
                vec4(custom_aabb_color.x(), custom_aabb_color.y(), custom_aabb_color.z(), custom_aabb_color.w()));
            bounding_box_mesh = std::make_shared<LineVertexColoredAsset>(bb_scene_unified_mesh);

            custom_aabb_dirty_flag = false;
        }

        if (bounding_box_mesh)
        {
            bounding_box_mesh->renderForward(camera, Saiga::mat4::Identity());
        }
    }
}
void SceneViewer::CreateMasks(bool mult_old_mask, float size_of_mask_in_ws)
{
    std::filesystem::path out_dir = scene->dataset_params.image_dir;
    out_dir                       = out_dir.parent_path().parent_path();
    out_dir                       = out_dir.append("masks/");
    std::filesystem::create_directories(out_dir);

    std::cout << "CreateMasks into " << out_dir << std::endl;


    // new

    UnifiedModel model_with_tris;
    {
        ScopedTimerPrintLine tim("Creating Quads");

        auto mesh_ = scene->point_cloud;
        {
            UnifiedMesh mesh_with_tris;
            for (int i = 0; i < mesh_.position.size(); ++i)
            {
                vec3 p = mesh_.position[i];
                vec3 n = mesh_.normal[i].normalized();

                // build tangent frame
                vec3 right = vec3(1, 0, 0);
                if (n == right) right = vec3(0, 1, 0);

                vec3 up = cross(n, right);
                right   = cross(n, up);

                vec3 p0 = p - up * size_of_mask_in_ws - right * size_of_mask_in_ws;
                vec3 p1 = p - up * size_of_mask_in_ws + right * size_of_mask_in_ws;
                vec3 p2 = p + up * size_of_mask_in_ws + right * size_of_mask_in_ws;
                vec3 p3 = p + up * size_of_mask_in_ws - right * size_of_mask_in_ws;


                int size_before_add = mesh_with_tris.position.size();
                ivec3 tri1          = ivec3(size_before_add + 0, size_before_add + 1, size_before_add + 2);
                ivec3 tri2          = ivec3(size_before_add + 2, size_before_add + 3, size_before_add + 0);

                mesh_with_tris.position.push_back(p0);
                mesh_with_tris.position.push_back(p1);
                mesh_with_tris.position.push_back(p2);
                mesh_with_tris.position.push_back(p3);
                mesh_with_tris.normal.push_back(n);
                mesh_with_tris.normal.push_back(n);
                mesh_with_tris.normal.push_back(n);
                mesh_with_tris.normal.push_back(n);

                mesh_with_tris.triangles.push_back(tri1);
                mesh_with_tris.triangles.push_back(tri2);
            }
            std::cout << mesh_with_tris.position.size() << "- " << mesh_with_tris.triangles.size() << std::endl;
            model_with_tris.mesh.push_back(mesh_with_tris);
        }
    }
    auto mesh = model_with_tris.CombinedMesh(VERTEX_POSITION);
    //\new

    // auto mesh = model.CombinedMesh(VERTEX_POSITION);


    auto triangles = mesh.first.TriangleSoup();
    for (auto& t : triangles)
    {
        t.ScaleUniform(1.0005);
    }

    SAIGA_ASSERT(!triangles.empty());

    AccelerationStructure::ObjectMedianBVH bvh;

    {
        ScopedTimerPrintLine tim("Creating BVH");
        bvh                  = AccelerationStructure::ObjectMedianBVH(triangles);
        bvh.triangle_epsilon = 0.00;
        bvh.bvh_epsilon      = 0.00;
    }


    std::vector<std::vector<vec3>> directions(scene->scene_cameras.size());
    std::vector<ImageView<vec3>> dirs(scene->scene_cameras.size());

    {
        ScopedTimerPrintLine tim("Unproject image");

        for (int ic = 0; ic < scene->scene_cameras.size(); ++ic)
        {
            auto cam        = scene->scene_cameras[ic];
            auto& dirs_data = directions[ic];
            auto& dir       = dirs[ic];

            dirs_data.resize(cam.h * cam.w);
            dir = ImageView<vec3>(cam.h, cam.w, dirs_data.data());


            if (cam.camera_model_type == CameraModel::PINHOLE_DISTORTION)
            {
#pragma omp parallel for
                for (int y = 0; y < cam.h; ++y)
                {
                    for (int x = 0; x < cam.w; ++x)
                    {
                        vec2 ip(x, y);
                        vec2 dist   = cam.K.unproject2(ip);
                        vec2 undist = undistortNormalizedPointSimple(dist, cam.distortion);
                        vec3 np     = vec3(undist(0), undist(1), 1);
                        dir(y, x)   = np.normalized();
                    }
                }
            }
            else if (cam.camera_model_type == CameraModel::OCAM)
            {
#pragma omp parallel for
                for (int y = 0; y < cam.h; ++y)
                {
                    for (int x = 0; x < cam.w; ++x)
                    {
                        Vec2 ip(x, y);
                        Vec3 np   = UnprojectOCam<double>(ip, 1, cam.ocam.AffineParams(), cam.ocam.poly_cam2world);
                        dir(y, x) = np.normalized().cast<float>();
                    }
                }
            }
            else
            {
                SAIGA_EXIT_ERROR("unknown camera model");
            }
        }
    }

    std::vector<std::string> mask_files;

    for (int i = 0; i < scene->frames.size(); ++i)
    {
        std::cout << "Process image " << i << std::endl;
        auto& f = scene->frames[i];
        TemplatedImage<unsigned char> mask_img(f.h, f.w);
        mask_img.makeZero();


        auto dir = dirs[f.camera_index];

        vec3 center = f.pose.translation().cast<float>();

        quat R = f.pose.unit_quaternion().cast<float>();

        ProgressBar bar(std::cout, "Render Mask", mask_img.h);
#pragma omp parallel for schedule(dynamic)
        for (int y = 0; y < mask_img.h; ++y)
        {
            for (int x = 0; x < mask_img.w; ++x)
            {
                vec3 d = dir(y, x);

                Ray r;
                r.origin    = center;
                r.direction = R * d;
                auto inter  = bvh.getClosest(r);
                if (inter.valid)
                {
                    mask_img(y, x) = 255;
                }
            }
            bar.addProgress(1);
        }

        if (mult_old_mask)
        {
            if (std::filesystem::exists(scene->dataset_params.mask_dir + f.mask_file))
            {
                Saiga::TemplatedImage<unsigned char> img_mask_large(scene->dataset_params.mask_dir + f.mask_file);

                for (int y = 0; y < mask_img.h; ++y)
                {
                    for (int x = 0; x < mask_img.w; ++x)
                    {
                        if (img_mask_large(y, x) == 0)
                        {
                            mask_img(y, x) = 0;
                        }
                    }
                }
            }
        }

        std::string mask_file = leadingZeroString(i, 5) + ".png";
        mask_files.push_back(mask_file);
        auto dst_file = out_dir.string() + "/" + mask_file;
        mask_img.save(dst_file);
    }

    std::ofstream ostream3(out_dir.string() + "/masks.txt");
    for (auto m : mask_files)
    {
        ostream3 << m << "\n";
    }
}
void SceneViewer::UpdatePointCloudBuffer()
{
    gl_points = nullptr;
}
