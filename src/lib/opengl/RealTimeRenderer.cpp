/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "RealTimeRenderer.h"

#include "saiga/colorize.h"
#include "saiga/opengl/imgui/imgui_opengl.h"

std::vector<std::string> getDirectoriesAndFollowSymlinks(std::string dir)
{
    std::vector<std::string> dirs;
    for (auto const& dir_entry : std::filesystem::directory_iterator{dir})
    {
        if (dir_entry.is_directory() || dir_entry.is_symlink())  // && dir_entry.path()
        {
            // std::cout << dir_entry.path() << '\n';
            dirs.push_back(dir_entry.path().string());
        }
    }
    return dirs;
}

RealTimeRenderer::Experiment::Experiment(std::string dir, std::string name, std::string scene_name, bool render_able)
    : dir(dir), name(name)
{
    if (!std::filesystem::exists(dir + "params.ini"))
    {
        return;
    }

    Directory d(dir);
    auto ep_dirs = d.getDirectories();
    // auto ep_dirs = getDirectoriesAndFollowSymlinks(dir);

    ep_dirs.erase(std::remove_if(ep_dirs.begin(), ep_dirs.end(), [](auto str) { return !hasPrefix(str, "ep"); }),
                  ep_dirs.end());

    if (ep_dirs.empty()) return;
    std::sort(ep_dirs.begin(), ep_dirs.end());

    std::cout << "Found experiment " << dir << " with " << ep_dirs.size() << " epochs" << std::endl;


    for (auto ep_dir : ep_dirs)
    {
        EP ep;
        ep.name       = ep_dir;
        ep.dir        = dir + ep_dir;
        ep.scene_name = scene_name;
        ep.ep         = std::stoi(ep_dir.substr(2, 4));

        if (render_able)
        {
            if (!std::filesystem::exists(ep.dir + "/render_net.pth") ||
                !std::filesystem::exists(ep.dir + "/scene_" + scene_name + "_texture.pth"))
            {
                continue;
            }
        }

        eps.emplace_back(ep);
    }
}



RealTimeRenderer::RealTimeRenderer(std::shared_ptr<SceneData> scene) : scene(scene)
{
    // Directory dir(experiments_base);
    // auto ex_names = dir.getDirectories();

    // auto ex_names = getDirectoriesAndFollowSymlinks(experiments_base);
    // std::sort(ex_names.begin(), ex_names.end(), std::greater<std::string>());

    std::function<void(std::string, std::string, int)> find_experiments_in_subfolders_and_recurse =
        [&](std::string path, std::string ex_name_this_far, int recursion_depth)
    {
        if (recursion_depth <= 0) return;
        auto folders = getDirectoriesAndFollowSymlinks(path);
        std::sort(folders.begin(), folders.end(), std::greater<std::string>());
        for (auto n : folders)
        {
            if (n == "." || n == "..") continue;
            std::string ex_name_str = n.substr(n.find_last_of("/") + 1);
            // Experiment e(n + "/", ex_name_this_far + ex_name_str, scene->scene_name);
            Experiment e(n + "/", ex_name_str + " (" + ex_name_this_far + ")", scene->scene_name);
            if (!e.eps.empty()) experiments.push_back(e);

            std::cout << n << std::endl;
            // recurse
            std::string ex_name_shortened = ex_name_this_far;
            ex_name_shortened += (ex_name_str.length() > 5) ? (ex_name_str.substr(0, 5) + "...") : ex_name_str;
            ex_name_shortened += "/";
            find_experiments_in_subfolders_and_recurse(n, ex_name_shortened, recursion_depth - 1);
        };
    };
    find_experiments_in_subfolders_and_recurse(experiments_base, "", 4);

    std::sort(experiments.begin(), experiments.end(), [](Experiment a, Experiment b) { return a.name > b.name; });
    /*
        for (auto n : ex_names)
        {
            if (n == "." || n == "..") continue;
            std::string ex_name_str = n.substr(n.find_last_of("/") + 1);

            // Experiment e(experiments_base + "/" + n + "/", n, scene->scene_name);
            // Experiment e(ex_name_str + "/", n, scene->scene_name);
            Experiment e(n + "/", ex_name_str, scene->scene_name);
            if (!e.eps.empty())
            {
                experiments.push_back(e);
            }

            {
                // Directory dir_sub(experiments_base + "/" + n);
                // auto sub_ex_names = dir_sub.getDirectories();
                auto sub_ex_names = getDirectoriesAndFollowSymlinks(n);


                std::sort(sub_ex_names.begin(), sub_ex_names.end(), std::greater<std::string>());
                for (auto sub_n : sub_ex_names)
                {
                    if (sub_n == "." || sub_n == "..") continue;
                    std::string sub_ex_name_str = sub_n.substr(sub_n.find_last_of("/") + 1);

                    std::string path_shortened =
                        (ex_name_str.length() > 5) ? (ex_name_str.substr(0, 5) + "...") : ex_name_str;
                    // Experiment e(experiments_base + "/" + n + "/" + sub_n + "/", dir_name_shortened + "/" +
       sub_n,
                    //              scene->scene_name);
                    Experiment e2(sub_n + "/", path_shortened + "/" + sub_ex_name_str, scene->scene_name);
                    if (!e2.eps.empty())
                    {
                        experiments.push_back(e2);
                    }
                }
            }
        }
    */
    current_ex = 0;
    // Load last in list
    if (!experiments.empty()) current_ep = experiments[current_ex].eps.size() - 1;

    LoadNets();
    torch::set_num_threads(4);
}

template <typename T>
void warpPerspective(ImageView<T> src, ImageView<T> dst, IntrinsicsPinholef dst_2_src, Matrix<float, 2, 2> crop_rot)
{
    for (auto y : dst.rowRange())
    {
        for (auto x : dst.colRange())
        {
            vec2 p(x, y);
            vec2 h = vec2(dst.cols / 2, dst.rows / 2);
            p -= h;
            p = crop_rot.inverse() * p;
            p += h;

            vec2 ip  = dst_2_src.normalizedToImage(p);
            float dx = ip(0);
            float dy = ip(1);
            if (src.inImage(dy, dx))
            {
                dst(y, x) = src.inter(dy, dx);
            }
        }
    }
}

std::shared_ptr<Texture> RealTimeRenderer::getClosestGTImage(ImageInfo fd)
{
    // find closest gt image
    std::vector<std::pair<double, int>> err_t, err_r, score;

    for (int i = 0; i < scene->frames.size(); ++i)
    {
        auto& f  = scene->frames[i];
        auto err = translationalError(fd.pose, f.pose);

        Vec3 dir1 = fd.pose.so3() * Vec3(0, 0, 1);
        Vec3 dir2 = f.pose.so3() * Vec3(0, 0, 1);

        auto err_angle = degrees(acos(dir1.dot(dir2)));

        err_t.push_back({err, i});
        err_r.push_back({err_angle, i});
    }

    std::sort(err_t.begin(), err_t.end());
    std::sort(err_r.begin(), err_r.end());

    for (int i = 0; i < err_t.size(); ++i)
    {
        score.push_back({0, i});
    }

    for (int i = 0; i < err_t.size(); ++i)
    {
        score[err_t[i].second].first += i;
        score[err_r[i].second].first += i;
    }
    std::sort(score.begin(), score.end());

    int best_idx = score.front().second;

    static IntrinsicsPinholef last_crop;
    if (best_idx != current_best_gt || !(last_crop == fd.crop_transform))
    {
        if (current_best_gt != -1 && best_gt_counter < 20)
        {
            // smooth out a bit so we change only after every 30 frames
            best_gt_counter++;
        }
        else
        {
            auto& f = scene->frames[best_idx];
            //   console << "Current Be st (img,cam) = (" << f.image_index << "," << f.camera_index
            //           << ") EV: " << f.exposure_value << std::endl;
            Image img;


            if (img.load(scene->dataset_params.image_dir + "/" + f.target_file))
            {
                Saiga::TemplatedImage<ucvec3> gt_crop(fd.h, fd.w);
                auto dst_2_src = fd.crop_transform.inverse();
                warpPerspective(img.getImageView<ucvec3>(), gt_crop.getImageView(), dst_2_src, fd.crop_rotation);
                //    best_gt_texture = std::make_shared<Texture>(img);
                best_gt_texture = std::make_shared<Texture>(gt_crop);
            }
            else
            {
                std::cout << "Failed to load Ground Truth image. Check if 'image_dir' in dataset.ini is correct!"
                          << std::endl;
            }
            current_best_gt = best_idx;
            best_gt_counter = 0;
        }
    }
    return best_gt_texture;
}

inline void centeredPosSize(const vec2& win_size, const vec2& tex_size, vec2& out_img_size, vec2& out_img_pos)
{
    float ar_win = win_size.x() / win_size.y();
    float ar_tex = tex_size.x() / tex_size.y();
    if (ar_tex > ar_win)
    {  // width is fitting
        out_img_size = vec2(win_size.x(), win_size.x() * (1 / ar_tex));
    }
    else
    {  // height is fitting
        out_img_size = vec2(win_size.y() * ar_tex, win_size.y());
    }
    out_img_pos = vec2((win_size.x() - out_img_size.x()) / 2, (win_size.y() - out_img_size.y()) / 2);
}

void center_tex_in_imgui(std::shared_ptr<Texture2D> tex, vec2& out_img_size, vec2& out_img_pos, bool flip = false)
{
    ivec2 tex_size = ivec2(tex.get()->getWidth(), tex.get()->getHeight());
    if (flip) std::swap(tex_size.x(), tex_size.y());
    ivec2 win_size = ivec2(ImGui::GetWindowContentRegionMax().x, ImGui::GetWindowContentRegionMax().y);
    if (flip) std::swap(win_size.x(), win_size.y());

    centeredPosSize(win_size, tex_size, out_img_size, out_img_pos);
}

void RealTimeRenderer::Forward(ImageInfo fd, int rotate_result_90_deg, vec3 debug_dir)
{
    if (experiments.empty()) return;

    timer_system.BeginFrame();



    mouse_on_view = false;
    if (ImGui::Begin("Neural View"))
    {
        ImGui::BeginChild("neural_child", ImVec2(0, 0), false, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoMove);
        mouse_on_view |= ImGui::IsWindowHovered();


        vec2 out_img_size, out_img_pos;
        Render(fd, debug_dir);

        if (use_gl_tonemapping)
        {
            if (rotate_result_90_deg)
            {
                center_tex_in_imgui(output_texture_ldr, out_img_size, out_img_pos, true);

                ImGui::TextureRotate90(output_texture_ldr.get(), ImGui::GetWindowContentRegionMax(), false,
                                       rotate_result_90_deg > 0 ? true : false);
            }
            else
            {
                center_tex_in_imgui(output_texture_ldr, out_img_size, out_img_pos, false);

                ImGui::Texture(output_texture_ldr.get(), ImGui::GetWindowContentRegionMax(), false);
            }
        }
        else
        {
            if (rotate_result_90_deg)
            {
                center_tex_in_imgui(output_texture, out_img_size, out_img_pos, true);

                ImGui::TextureRotate90(output_texture.get(), ImGui::GetWindowContentRegionMax(), false,
                                       rotate_result_90_deg > 0 ? true : false);
            }
            else
            {
                center_tex_in_imgui(output_texture, out_img_size, out_img_pos, false);

                ImGui::Texture(output_texture.get(), ImGui::GetWindowContentRegionMax(), false);
            }
        }

        ImGui::EndChild();
    }
    ImGui::End();

    if (ImGui::Begin("Debug View"))
    {
        vec2 out_img_size, out_img_pos;

        ImGui::BeginChild("dbg_child", ImVec2(0, 0), false, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoMove);
        mouse_on_view |= ImGui::IsWindowHovered();
        if (render_color)
        {
            RenderColor(fd, color_flags, debug_dir);
            if (rotate_result_90_deg)
            {
                center_tex_in_imgui(output_color, out_img_size, out_img_pos, true);

                ImGui::TextureRotate90(output_color.get(), ImGui::GetWindowContentRegionMax(), false,
                                       rotate_result_90_deg > 0 ? true : false);
            }
            else
            {
                center_tex_in_imgui(output_color, out_img_size, out_img_pos, false);

                ImGui::Texture(output_color.get(), ImGui::GetWindowContentRegionMax(), false);
            }
        }

        ImGui::EndChild();
    }
    ImGui::End();


    {
        vec2 out_img_size, out_img_pos;

        getClosestGTImage(fd);
        if (ImGui::Begin("Closest Ground Truth"))
        {
            ImGui::BeginChild("gt_child", ImVec2(0, 0), false, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoMove);
            if (best_gt_texture)
            {
                if (rotate_result_90_deg)
                {
                    center_tex_in_imgui(best_gt_texture, out_img_size, out_img_pos, true);

                    ImGui::TextureRotate90(best_gt_texture.get(), ImGui::GetWindowContentRegionMax(), false,
                                           rotate_result_90_deg > 0 ? true : false);
                }
                else
                {
                    center_tex_in_imgui(best_gt_texture, out_img_size, out_img_pos, false);

                    ImGui::Texture(best_gt_texture.get(), ImGui::GetWindowContentRegionMax(), false);
                }
            }
            ImGui::EndChild();
        }
        ImGui::End();
    }

    timer_system.EndFrame();
}
void RealTimeRenderer::imgui()
{
    if (experiments.empty())
    {
        // std::cout << "no experiments :(" << std::endl;
        return;
    }

    static AABB last_aabb;
    static torch::Tensor last_pos, last_norm, last_org_col, last_index, last_org_ind;
    if (reset_new_ex) last_aabb = AABB();
    if (!last_pos.defined() || reset_new_ex)
    {
        last_pos     = ns->point_cloud_cuda->t_position;
        last_norm    = ns->point_cloud_cuda->t_normal;
        last_org_col = ns->point_cloud_cuda->t_original_color;
        last_index   = ns->point_cloud_cuda->t_index;
        //    last_org_ind = ns->point_cloud_cuda->t_original_index;
        reset_new_ex = false;
    }

    if (custom_discard_aabb.maxSize() > 0.01)
    {
        if (custom_discard_aabb.min != last_aabb.min || custom_discard_aabb.max != last_aabb.max)
        {
            last_aabb = custom_discard_aabb;

            ns->point_cloud_cuda->t_position       = last_pos;
            ns->point_cloud_cuda->t_normal         = last_norm;
            ns->point_cloud_cuda->t_original_color = last_org_col;
            ns->point_cloud_cuda->t_index          = last_index;
            //   ns->point_cloud_cuda->t_original_index = last_org_ind;

            torch::NoGradGuard ngg;

            torch::Tensor keep = ((ns->point_cloud_cuda->t_position.slice(1, 0, 1) > custom_discard_aabb.min.x()) &
                                  (ns->point_cloud_cuda->t_position.slice(1, 1, 2) > custom_discard_aabb.min.y()) &
                                  (ns->point_cloud_cuda->t_position.slice(1, 2, 3) > custom_discard_aabb.min.z()) &
                                  (ns->point_cloud_cuda->t_position.slice(1, 0, 1) < custom_discard_aabb.max.x()) &
                                  (ns->point_cloud_cuda->t_position.slice(1, 1, 2) < custom_discard_aabb.max.y()) &
                                  (ns->point_cloud_cuda->t_position.slice(1, 2, 3) < custom_discard_aabb.max.z()))
                                     .squeeze();


            ns->point_cloud_cuda->RemoveSelected(keep);
        }
    }

    if (ImGui::Begin("Neural Renderer"))
    {
#ifndef MINIMAL_GUI
        if (ImGui::Button("render all extras"))
        {
            for (int i = 0; i < scene->extra_eval_frames.size(); ++i)
            {
                ImageInfo fd;
                fd.crop_rotation.setZero();
                fd.crop_rotation(0, 0) = 1;
                fd.crop_rotation(1, 1) = 1;
                fd.camera_model_type   = CameraModel::PINHOLE_DISTORTION;
                fd.w                   = scene->extra_eval_cameras[0].w;
                fd.h                   = scene->extra_eval_cameras[0].h;
                fd.K                   = scene->extra_eval_cameras[0].K;
                fd.distortion          = scene->extra_eval_cameras[0].distortion;
                // fd.crop_transform      = fd.crop_transform.scale(scene->dataset_params.render_scale);
                fd.pose = scene->extra_eval_frames[i].pose;

                //  fd.exposure_value = 11.f;

                Render(fd, vec3(0, 0, 0), 1 << 9);
                DownloadRender().save("out_extra_eval_" + std::to_string(i) + ".png");
                std::cout << "Extra Render " << i << std::endl;
            }
        }

        if (ImGui::Button("save points ply"))
        {
            // auto ns_copy = ns.get();

            // ns_copy->RemovePoints()
            std::cout << "Remove Points" << std::endl;

            auto indices_to_remove =
                torch::where(ns->texture->confidence_value_of_point.squeeze() < stability_cutoff_value, 1, 0).nonzero();
            // auto indices_to_remove =
            // torch::argwhere(tex->confidence_value_of_point.squeeze() <
            // params->points_adding_params.removal_confidence_cutoff);

            if (indices_to_remove.size(0) > 0)
            {
                ns->RemovePoints(indices_to_remove, false);
            }
            std::string file = scene->file_dataset_base + "/point_cloud_exported.ply";
            auto point_cloud = ns->point_cloud_cuda->Mesh();
            // SceneData::RemovePointsInCloseArea(point_cloud, ns->scene->frames, 0.01);
            Saiga::UnifiedModel(point_cloud).Save(file);
            // ns->scene->frames
            std::cout << "export to " << file << std::endl;
            exit(1);
        }

        if (ImGui::Button("save only added points ply"))
        {
            std::string file = scene->file_dataset_base + "/point_cloud_exported.ply";
            std::cout << "Remove Points" << std::endl;

            auto indices_to_remove =
                torch::where(ns->texture->confidence_value_of_point.squeeze() < stability_cutoff_value, 1, 0).nonzero();
            // auto indices_to_remove =
            // torch::argwhere(tex->confidence_value_of_point.squeeze() <
            // params->points_adding_params.removal_confidence_cutoff);



            if (indices_to_remove.size(0) > 0)
            {
                ns->RemovePoints(indices_to_remove);
            }
            auto mesh = ns->point_cloud_cuda->Mesh();
            std::vector<int> indices_to_remove2;
            for (int i = 0; i < mesh.color.size(); ++i)
            {
                if (mesh.color[i].w() != 0)
                {
                    indices_to_remove2.push_back(i);
                }
            }
            mesh.EraseVertices(indices_to_remove2);
            Saiga::UnifiedModel(mesh).Save(file);

            std::cout << "export to " << file << std::endl;
            exit(1);
        }
        if (ImGui::ListBoxHeader("###RenderModes", 5))
        {
            std::vector<std::string> render_modes_str = {"DT", "Blend", "FuzzyBlend", "BilinearBlend", "TiledBB"};
            for (int i = 0; i < render_modes_str.size(); ++i)
            {
                const bool is_selected = params->render_params.render_mode == i;
                if (is_selected)
                {
                    ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(85, 107, 47, 255));
                }
                if (ImGui::Selectable(render_modes_str[i].c_str(), is_selected))
                {
                    params->render_params.render_mode = i;
                }
                if (is_selected)
                {
                    ImGui::PopStyleColor();
                }
            }
            ImGui::ListBoxFooter();
        }
        ImGui::SetNextItemWidth(600);
        if (ImGui::ListBoxHeader("###experiments", 10))
        {
            for (int i = 0; i < experiments.size(); ++i)
            {
                if (ImGui::Selectable(experiments[i].name.c_str(), current_ex == i))
                {
                    current_ex = i;
                    current_ep = experiments[i].eps.size() - 1;
                    LoadNets();
                }
            }
            ImGui::ListBoxFooter();
        }
        // ImGui::SameLine();
        ImGui::SetNextItemWidth(600);
        if (ImGui::ListBoxHeader("###Eps", 10))
        {
            auto& ex = experiments[current_ex];
            for (int i = 0; i < ex.eps.size(); ++i)
            {
                if (ImGui::Selectable(ex.eps[i].name.c_str(), current_ep == i))
                {
                    current_ep = i;
                    LoadNets();
                }
            }
            ImGui::ListBoxFooter();
        }

        ImGui::Checkbox("use_center_tensor", &use_center_tensor);
        ImGui::Checkbox("use_custom_camera", &use_custom_camera);
#endif  // MINIMAL_GUI

        if (pipeline)
        {
#ifndef MINIMAL_GUI

            if (ImGui::CollapsingHeader("Params"))
            {
                pipeline->params->imgui();
            }
#endif
            ImGui::SliderInt("color_layer", &color_layer, 0, ns->params->net_params.num_layers - 1);

#ifndef MINIMAL_GUI

            ImGui::Checkbox("enable_response", &ns->camera->params.enable_response);
            ImGui::Checkbox("enable_white_balance", &ns->camera->params.enable_white_balance);
#endif
            ImGui::SliderInt("color_flags", &color_flags, 0, ns->params->pipeline_params.num_texture_channels / 4);
            ImGui::SliderFloat("color_scale", &color_scale, 0, 16);

#ifndef MINIMAL_GUI

            ImGui::SliderInt("manuel timestep", &manuel_timestep, -1, ns->poses->poses_se3.size(0));

            ImGui::Checkbox("invert debug in red", &invert_colors_and_red);

            ImGui::Checkbox("Better Visualization of Points", &use_visualize_confidence_as_full);

            ImGui::Checkbox("use_discard_in_main_render_window", &use_discard_in_main_render_window);

            ImGui::SliderFloat("Cutoff Value", &stability_cutoff_value, 0, 1);
            ImGui::InputFloat("cutoff multiplier", &stability_cutoff_value_multiplier);

            ImGui::SliderInt("current eval epoch", &current_eval_epoch, -1, 6000);

            ImGui::SliderFloat("fixed_confidence", &fixed_confidence, 0, 1);

            ImGui::SliderFloat3("new points color", point_spawn_debug_color.data(), 0, 1);



            // static int cam_model = int(scene->dataset_params.camera_model);
            // ImGui::SliderInt("camera model", &cam_model, 0, 2);
            // switch (cam_model)
            //{
            //     case 0:
            //     {
            //         scene->dataset_params.camera_model =
            //         CameraModel::PINHOLE_DISTORTION; break;
            //     }
            //     case 1:
            //     {
            //         scene->dataset_params.camera_model = CameraModel::OCAM;
            //         break;
            //     }
            //     case 2:
            //     {
            //         scene->dataset_params.camera_model = CameraModel::SPHERICAL;
            //         break;
            //     }
            // }

            ImGui::Checkbox("enable_vignette", &ns->camera->params.enable_vignette);
            ImGui::Checkbox("enable_exposure", &ns->camera->params.enable_exposure);


            if (ImGui::Button("write optimized camera poses"))
            {
                auto poses = ns->poses->Download();
                std::ofstream strm("poses_quat_wxyz_pos_xyz.txt");
                for (auto pf : poses)
                {
                    SE3 p  = pf.cast<double>().inverse();
                    auto q = p.unit_quaternion();
                    auto t = p.translation();

                    strm << std::setprecision(8) << std::scientific << q.w() << " " << q.x() << " " << q.y() << " "
                         << q.z() << " " << t.x() << " " << t.y() << " " << t.z() << "\n";
                }
            }
            static float max_dens = 5;
            ImGui::SetNextItemWidth(100);
            ImGui::InputFloat("###max_dens", &max_dens);
            ImGui::SameLine();
            if (ImGui::Button("Vis. density"))
            {
                auto pc = scene->point_cloud;
                SAIGA_ASSERT(pc.data.size() == pc.color.size());

                for (int i = 0; i < pc.data.size(); ++i)
                {
                    float f               = pc.data[i](0) / max_dens;
                    vec3 c                = colorizeFusion(f);
                    pc.color[i].head<3>() = c;
                    pc.color[i](3)        = 1;
                }

                debug_color_texture = NeuralPointTexture(pc);
                debug_color_texture->to(device);
            }
            if (ImGui::Button("Vis. color"))
            {
                debug_color_texture = NeuralPointTexture(scene->point_cloud);
                debug_color_texture->to(device);
            }

            ImGui::Checkbox("use_gl_tonemapping", &use_gl_tonemapping);
            if (use_gl_tonemapping)
            {
                ImGui::Checkbox("use_bloom", &use_bloom);
                if (use_bloom)
                {
                    bloom.imgui();
                }
            }

            ImGui::Checkbox("channels_last", &pipeline->params->net_params.channels_last);
            if (ImGui::Checkbox("half_float", &pipeline->params->net_params.half_float))
            {
                auto target_type = pipeline->params->net_params.half_float ? torch::kFloat16 : torch::kFloat32;

                pipeline->render_network->to(target_type);
                ns->camera->to(target_type);
            }
            if (ImGui::Button("random shuffle"))
            {
                scene->point_cloud.RandomShuffle();
                pipeline->Train(false);
            }

            if (ImGui::Button("save render gl tonemap"))
            {
                output_texture_ldr->download(output_image_ldr.data());
                output_image_ldr.save("out_render_gl.png");
            }

            if (ImGui::Button("save render"))
            {
                DownloadRender().save("out_render.png");
            }

            if (ImGui::Button("save gt"))
            {
                DownloadGt().save("out_gt.png");
            }


            if (ImGui::Button("save debug"))
            {
                TemplatedImage<ucvec4> tmp(output_color->getHeight(), output_color->getWidth());

                output_color->bind();
                glGetTexImage(output_color->getTarget(), 0, GL_RGBA, GL_UNSIGNED_BYTE, tmp.data());
                assert_no_glerror();
                output_color->unbind();

                tmp.save("out_debug_" + std::to_string(color_layer) + ".png");
            }

            if (ImGui::Button("morton shuffle"))
            {
                scene->point_cloud.ReorderMorton64();
                pipeline->Train(false);
            }

            if (ImGui::Button("RandomBlockShuffle shuffle"))
            {
                scene->point_cloud.RandomBlockShuffle(256);
                pipeline->Train(false);
            }

            if (ImGui::Button("Add random points from png stack"))
            {
                // AddNewRandomPointsFromCTStack(int num_points_to_add, std::string
                // path, float ct_volume_scale, vec3 ct_volume_translation)
                ns->AddNewRandomPointsFromCTStack(10000, "ct_png_stack/cornell_test/png", 5.f, vec3(0, 0, 1));
                debug_color_texture = nullptr;
            }
            /*
            if (ImGui::Button("Add random points from hdr volume"))
            {
                // AddNewRandomPointsFromCTStack(int num_points_to_add, std::string
                // path, float ct_volume_scale, vec3 ct_volume_translation)
                auto hdr_img_path           = "ct_png_stack/hdr_playground/volume.hdr";
                torch::Tensor volume_tensor = LoadHDRImageTensor(hdr_img_path);

                ns->AddNewRandomPointsFromCTHdr(volume_tensor, 20, 1, vec3(0, 0, 0), scene->point_cloud.BoundingBox());
                debug_color_texture = nullptr;
            }
            */
            static int num_spheres         = 1;
            static float inner_radius      = 20.f;
            static float env_radius_factor = 20.f;
            static int num_points          = 10000;

            ImGui::InputInt("num_spheres", &num_spheres);
            ImGui::InputFloat("inner_rad", &inner_radius);
            ImGui::InputFloat("env_radius_factor", &env_radius_factor);
            ImGui::InputInt("num_points", &num_points);

            if (ImGui::Button("Add random points for environment map"))
            {
                ns->AddNewRandomForEnvSphere(num_spheres, inner_radius, env_radius_factor, num_points, false);
                debug_color_texture = nullptr;
            }
            if (ImGui::Button("Add random points boxes"))
            {
                ns->AddNewRandomPointsInValuefilledBB(100000, 0.05);
                debug_color_texture = nullptr;
            }
            if (ImGui::CollapsingHeader("PointMov"))
            {
                static torch::Tensor pos_save;
                static float amplitude   = 1.f;
                static float frequency   = 0.5f;
                static int axis_for_amp  = 0;
                static int axis_for_freq = 1;

                ImGui::SliderFloat("amplitude", &amplitude, 0, 4.f);
                ImGui::SliderFloat("frequency", &frequency, 0, 4.f);
                ImGui::SliderInt("axis for amplitude", &axis_for_amp, 0, 2);
                ImGui::SliderInt("axis for frequency", &axis_for_freq, 0, 2);

                if (ImGui::Button("Move"))
                {
                    static torch::Tensor pos_save;

                    torch::NoGradGuard ngg;
                    if (!pos_save.defined())
                    {
                        pos_save = ns->point_cloud_cuda->t_position.clone();
                    }

                    ns->point_cloud_cuda->t_position = pos_save.clone();

                    // std::cout << TensorInfo(ns->point_cloud_cuda->t_position)
                    //           <<
                    //           TensorInfo(ns->point_cloud_cuda->t_position.slice(1,
                    //           0, 1))
                    //           <<
                    //           TensorInfo(torch::sin(ns->point_cloud_cuda->t_position.slice(1,
                    //           1, 2))) << std::endl;
                    ns->point_cloud_cuda->t_position.slice(1, axis_for_amp, axis_for_amp + 1) +=
                        torch::sin(ns->point_cloud_cuda->t_position.slice(1, axis_for_freq, axis_for_freq + 1) *
                                   frequency) *
                        amplitude;
                }
            }
#endif
        }
    }
    ImGui::End();

    timer_system.Imgui();
}

void RealTimeRenderer::overrideForCustomCam(ImageInfo& fd, float& old_cutoff)
{
    rt_extrinsics->SetPose(0, fd.pose);
    old_cutoff = params->render_params.dist_cutoff;
    // if (scene->scene_cameras.begin()->camera_model_type == CameraModel::PINHOLE_DISTORTION)
    if (fd.camera_model_type == CameraModel::PINHOLE_DISTORTION)
    {
        params->render_params.dist_cutoff = fd.distortion.MonotonicThreshold();
        // std::cout << "PINHOLE " << fd.K << " ---- " << fd.distortion << std::endl;
        rt_intrinsics->SetPinholeIntrinsics(0, fd.K, fd.distortion);
    }
    // else if (scene->scene_cameras.begin()->camera_model_type == CameraModel::OCAM)
    else if (fd.camera_model_type == CameraModel::OCAM)

    {
        params->render_params.dist_cutoff = 0.15f;
        rt_intrinsics->SetOcamIntrinsics(0, fd.ocam);
    }
    else
    {
        rt_intrinsics->SetPinholeIntrinsics(0, fd.K, fd.distortion);
    }
    std::swap(ns->intrinsics, rt_intrinsics);
    std::swap(ns->poses, rt_extrinsics);

    // batch.front()->img.camera_index      = 0;
    // batch.front()->img.image_index       = 0;
    // batch.front()->img.camera_model_type = scene->dataset_params.camera_model;
}

torch::Tensor RealTimeRenderer::Render(ImageInfo fd, vec3 debug_refl_dir, int border_to_render)
{
    if (fixed_confidence > 0.f)
    {
        std::cout << "CAREFUL: FIXED CONFIDENCE with value" << fixed_confidence << std::endl;
        float inv_sigmoid                 = 1.f / 10.f * (std::log(fixed_confidence) - std::log(1 - fixed_confidence));
        ns.get()->texture->confidence_raw = (torch::full_like(ns.get()->texture->confidence_raw, inv_sigmoid));
    }

    ivec2 original_resolution = ivec2(fd.w, fd.h);  //+ ivec2(border_to_render * 2, border_to_render * 2);



    fd.w += 2 * border_to_render;
    fd.h += 2 * border_to_render;
    // fd.K.cx += border_to_render;
    // fd.K.cy += border_to_render;

    SAIGA_ASSERT(pipeline);
    if (!pipeline) return torch::Tensor();

    std::vector<NeuralTrainData> batch(1);
    batch.front() = std::make_shared<TorchFrameData>();

    batch.front()->img              = fd;
    batch.front()->img.camera_index = 0;
    batch.front()->img.image_index  = 0;
    // batch.front()->img.camera_model_type = scene->dataset_params.camera_model;

    if (current_best_gt != -1)
    {
        auto& f = scene->frames[current_best_gt];
        SAIGA_ASSERT(current_best_gt == f.image_index);
        batch.front()->img.camera_index = f.camera_index;
        batch.front()->img.image_index  = f.image_index;
        batch.front()->timestep =
            torch::from_blob(&f.camera_index, {1}, torch::TensorOptions().dtype(torch::kLong)).cuda().to(torch::kFloat);
    }
    // if (manuel_timestep != -1)


    if (!uv_tensor.defined() || uv_tensor.size(1) != fd.h || uv_tensor.size(2) != fd.w)
    {
        auto uv_image    = InitialUVImage(fd.h, fd.w);
        uv_tensor        = ImageViewToTensor(uv_image.getImageView()).to(device);
        uv_tensor_center = torch::zeros_like(uv_tensor);
    }


    if (!direction_tensor.defined() || direction_tensor.size(1) != fd.h || direction_tensor.size(2) != fd.w)
    {
        auto& cam = scene->frames[batch.front()->img.image_index];
        // auto direction_img =
        //     InitialDirectionImage(cam.w * scene->dataset_params.render_scale,
        //                           cam.h * scene->dataset_params.render_scale, cam.camera_model_type,
        //                           cam.K, cam.distortion, cam.ocam.cast<double>());

        auto direction_img =
            InitialDirectionImage(fd.w, fd.h, cam.camera_model_type, cam.K, cam.distortion, cam.ocam.cast<double>());

        direction_tensor = ImageViewToTensor(direction_img.getImageView()).to(device);
    }

    auto d_tensor = direction_tensor.clone();
    if (debug_refl_dir.x() != 0 && debug_refl_dir.y() != 0 && debug_refl_dir.z() != 0)
    {
        //        PrintTensorInfo(d_tensor.slice(0, 2, 3));
        d_tensor.slice(0, 2, 3) += (-1 + debug_refl_dir.z());
        d_tensor.slice(0, 0, 1) += debug_refl_dir.x();
        d_tensor.slice(0, 1, 2) += debug_refl_dir.y();
        //   std::cout << debug_refl_dir << std::endl;
    }

    if (original_resolution.y() != output_image.h || original_resolution.x() != output_image.w)
    {
        std::cout << "RESIZE" << std::endl;
        output_image.create(original_resolution.y(), original_resolution.x());
        output_image.getImageView().set(vec4(1, 1, 1, 1));
        output_image_ldr.create(output_image.dimensions());
        output_texture     = std::make_shared<Texture>(output_image);
        output_texture_ldr = std::make_shared<Texture>(output_image_ldr);

        texure_interop = std::make_shared<Saiga::CUDA::Interop>();
        texure_interop->initImage(output_texture->getId(), output_texture->getTarget());

        std::cout << "Setting Neural Render Size to " << original_resolution.x() << "x" << original_resolution.y()
                  << std::endl;
    }

    batch.front()->uv = use_center_tensor ? uv_tensor_center : uv_tensor;
    SAIGA_ASSERT(batch.front()->uv.size(1) == fd.h);
    SAIGA_ASSERT(batch.front()->uv.size(2) == fd.w);
    batch.front()->direction = d_tensor;
    SAIGA_ASSERT(batch.front()->direction.size(1) == fd.h);
    SAIGA_ASSERT(batch.front()->direction.size(2) == fd.w);

    float old_cutoff = 0;
    if (use_custom_camera)
    {
        overrideForCustomCam(fd, old_cutoff);
        batch.front()->img.camera_index      = 0;
        batch.front()->img.image_index       = 0;
        batch.front()->img.camera_model_type = fd.camera_model_type;
    }
    pipeline->params->pipeline_params.skip_sensor_model = use_gl_tonemapping;

    long man_timestep = manuel_timestep;
    batch.front()->timestep =
        torch::from_blob(&man_timestep, {1}, torch::TensorOptions().dtype(torch::kLong)).cuda().to(torch::kFloat);


    auto debug_weight_color_old = pipeline->render_module->params->render_params.debug_weight_color;
    auto debug_listlength_old   = pipeline->render_module->params->render_params.debug_max_list_length;

    pipeline->render_module->params->render_params.debug_weight_color = false;


    pipeline->render_module->params->render_params.test_refl_x = debug_refl_dir.x();
    pipeline->render_module->params->render_params.test_refl_y = debug_refl_dir.y();
    pipeline->render_module->params->render_params.test_refl_z = debug_refl_dir.z();

    pipeline->render_module->params->pipeline_params.enable_environment_map =
        render_env_map && !params->pipeline_params.environment_map_params.use_points_for_env_map;


    pipeline->render_module->params->render_params.viewer_only = true;

    if (stability_cutoff_value != 10.f)
    {
        pipeline->render_module->params->render_params.stability_cutoff_value =
            stability_cutoff_value * stability_cutoff_value_multiplier;
    }

    ns->intrinsics->AddToOpticalCenter(border_to_render);
    torch::Tensor x;
    {
        auto timer                 = timer_system.Measure("Forward");
        auto neural_exposure_value = fd.exposure_value - scene->dataset_params.scene_exposure_value;
        auto f_result = pipeline->Forward(*ns, batch, {}, false, current_eval_epoch, false, neural_exposure_value,
                                          fd.white_balance);
        x             = f_result.x;

        x = CenterCrop2D(x, border_to_render).contiguous();

        if (x.size(0) > 1)
        {
            static int which_channel = 0;
            //  ImGui::InputInt("which channel", &which_channel, 0, 2);
            x = x.slice(0, which_channel, which_channel + 1);
        }

        // batchsize == 1 !
        SAIGA_ASSERT(x.dim() == 4 && x.size(0) == 1);
        SAIGA_ASSERT(x.size(1) == 3);
    }
    ns->intrinsics->AddToOpticalCenter(-border_to_render);

    auto timer = timer_system.Measure("Post Process");
    x          = x.squeeze();



    // x has size [c, h, w]
    torch::Tensor alpha_channel =
        torch::ones({1, x.size(1), x.size(2)}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat));
    x = torch::cat({x, alpha_channel}, 0);


    x = x.permute({1, 2, 0});
    // x = x.clamp(0.f, 1.f);
    x = x.contiguous();

    texure_interop->mapImage();
    CHECK_CUDA_ERROR(cudaMemcpy2DToArray(texure_interop->array, 0, 0, x.data_ptr(), x.stride(0) * sizeof(float),
                                         x.size(1) * x.size(2) * sizeof(float), x.size(0), cudaMemcpyDeviceToDevice));
    texure_interop->unmap();

    if (use_gl_tonemapping)
    {
        tone_mapper.MapLinear(output_texture.get());
        if (use_bloom)
        {
            bloom.Render(output_texture.get());
        }
        tone_mapper.Map(output_texture.get(), output_texture_ldr.get());
    }

    pipeline->render_module->params->render_params.debug_weight_color    = debug_weight_color_old;
    pipeline->render_module->params->render_params.debug_max_list_length = debug_listlength_old;

    if (use_custom_camera)
    {
        std::swap(ns->intrinsics, rt_intrinsics);
        std::swap(ns->poses, rt_extrinsics);
        params->render_params.dist_cutoff = old_cutoff;
    }


    SAIGA_ASSERT(x.is_contiguous() && x.is_cuda());
    return x;
}



void RealTimeRenderer::SetupRenderedDebugColor()
{
    torch::NoGradGuard ngg;

    static float last_fixed_conv                      = 0.f;
    static bool last_invert_colors_and_red            = false;
    static bool last_use_visualize_confidence_as_full = false;


    if (!debug_color_texture || last_fixed_conv != fixed_confidence ||
        last_invert_colors_and_red != invert_colors_and_red ||
        use_visualize_confidence_as_full != last_use_visualize_confidence_as_full ||
        point_spawn_debug_color != prev_debug_col)
    {
        std::cout << "create debug tex " << std::endl;

        debug_color_texture = NeuralPointTexture(scene->point_cloud, 4, ns->point_cloud_cuda->Indices().size());
        debug_color_texture->texture          = ns->point_cloud_cuda->t_original_color.permute({1, 0}).clone();
        debug_color_texture->background_color = debug_color_texture->background_color_raw;
        if (point_spawn_debug_color != prev_debug_col)
        {
            auto new_points = debug_color_texture->texture.slice(0, 3, 4).lt(0.5);

            std::cout << TensorInfo(new_points) << std::endl;

            debug_color_texture->texture.slice(0, 0, 1).index_put_({new_points}, point_spawn_debug_color.x());
            debug_color_texture->texture.slice(0, 1, 2).index_put_({new_points}, point_spawn_debug_color.y());
            debug_color_texture->texture.slice(0, 2, 3).index_put_({new_points}, point_spawn_debug_color.z());
            // debug_color_texture->texture =
            // ns->point_cloud_cuda->t_original_color.permute({1, 0}).clone();

            prev_debug_col = point_spawn_debug_color;
        }


        torch::Tensor conv_tex;
        conv_tex = ns.get()->texture->confidence_value_of_point.clone();
        if (invert_colors_and_red)
        {
            debug_color_texture->texture = torch::max(debug_color_texture->texture) - debug_color_texture->texture;
            last_invert_colors_and_red   = invert_colors_and_red;
        }

        if (use_visualize_confidence_as_full)
        {
            conv_tex = torch::where(conv_tex > stability_cutoff_value, torch::ones_like(conv_tex),
                                    torch::zeros_like(conv_tex));
            last_use_visualize_confidence_as_full = use_visualize_confidence_as_full;
        }
        debug_color_texture->confidence_value_of_point = conv_tex.to(device);

        if (fixed_confidence > 0.f)
        {
            float inv_sigmoid = 1.f / 10.f * (std::log(fixed_confidence) - std::log(1 - fixed_confidence));
            debug_color_texture->confidence_raw =
                (torch::full_like(debug_color_texture->confidence_value_of_point, inv_sigmoid));
            debug_color_texture->confidence_value_of_point =
                (torch::full_like(debug_color_texture->confidence_value_of_point, fixed_confidence));
            last_fixed_conv = fixed_confidence;
        }

        debug_color_texture->to(device);
        debug_color_texture_texture = debug_color_texture->texture;
    }

    /*
    static int last_bb_debug_show = 0;
    if (current_bb_debug_show != 0 && last_bb_debug_show != current_bb_debug_show)
    {
        if (current_bb_debug_show == 1)
        {
            torch::Tensor cell_ids_per_point = ns->point_cloud_cuda->DebugBBIndexToCol();
            debug_color_texture->texture     = cell_ids_per_point.permute({1, 0});
            //  debug_color_texture->texture =cell_ids_per_point.clone().repeat({4,1});
        }
        else if (current_bb_debug_show == 2)
        {
            // torch::Tensor cell_ids_per_point = ns->point_cloud_cuda->GetPerPointBBIndex();
            // debug_color_texture->texture
            // =torch::ones_like(cell_ids_per_point).index_put_({(cell_ids_per_point!=-1)},0).repeat({4,1});
    std::cout
            // << torch::count_nonzero(cell_ids_per_point+1) << TensorInfo(cell_ids_per_point) << std::endl;
            torch::Tensor cell_vals_per_point = ns->point_cloud_cuda->GetPerPointBBValue();
            debug_color_texture->texture      = cell_vals_per_point.clone().repeat({4, 1}) /
    cell_vals_per_point.max();
        }
        else if (current_bb_debug_show == 3)
        {
            torch::Tensor cell_ids_per_point = ns->point_cloud_cuda->GetPerPointBBIndex();
            debug_color_texture->texture     = cell_ids_per_point.repeat({4, 1}) / cell_ids_per_point.max();
        }

        last_bb_debug_show = current_bb_debug_show;
    }
    else if (current_bb_debug_show == 0)
    {
        if (last_state_col != current_state_debug_show || last_state_conv != discard_with_state ||
            use_which_layer != last_which_layer)
        {
            torch::Tensor col_tex;
            torch::Tensor conv_tex;
            switch (current_state_debug_show)
            {
                case 0:
                {
                    // color tex
                    col_tex = debug_color_texture_texture;
                    break;
                }
                case 1:
                {
                    // color tex
                    auto indices_to_remove =
                        torch::where(ns->texture->confidence_value_of_point.squeeze() < stability_cutoff_value, 1,
    0) .nonzero(); col_tex = ns.get()->texture->confidence_value_of_point.clone().repeat({4, 1}); break;
                }
                case 11:
                {
                    break;
                }
                case 12:
                {
                    // length of feature ved
                    col_tex =
                        torch::sum(torch::abs(ns->texture->texture) / ns->texture->texture.size(0), 0).repeat({4,
    1}); break;
                }
                case 13:
                {
                    // length of 4 features
                    col_tex = torch::sum(torch::abs(ns->texture->texture.slice(0, 0, 4)) / 4, 0).repeat({4, 1});
                    break;
                }
                case 14:
                {
                    // length of 4 features
                    col_tex = torch::sum(torch::abs(ns->texture->texture.slice(0, 4, ns->texture->texture.size(0)))
    / (ns->texture->texture.size(0) - 4), 0) .repeat({4, 1}); break;
                }
                case 15:
                {
                    // diff to background color
                    auto t  = torch::abs(ns.get()->texture->texture -
    ns.get()->texture->background_color.unsqueeze(1)); col_tex = t; break;
                }
                case 16:
                {
                    // length of 4 features
                    col_tex =
                        0.5 * (torch::sum(torch::abs(ns->texture->texture.slice(0, 0, 4)) / 4, 0).repeat({4, 1}) +
                               torch::sum(torch::abs(ns->texture->texture.slice(0, 4, ns->texture->texture.size(0)))
    / (ns->texture->texture.size(0) - 4), 0) .repeat({4, 1})); break;
                }
            }
            switch (discard_with_state)
            {
                case 0:
                case 1:
                {
                    // conv_tex = torch::ones_like(debug_color_texture->loss_of_images_proj_on_points);
                    conv_tex = ns.get()->texture->confidence_value_of_point.clone();

                    break;
                }
                case 11:
                {
                    break;
                }
                case 12:
                {
                    // length of feature ved
                    conv_tex =
                        torch::sum(torch::abs(ns->texture->texture) / ns->texture->texture.size(0), 0).repeat({4,
    1}); break;
                }
                case 13:
                {
                    // length of 4 features
                    conv_tex = torch::sum(torch::abs(ns->texture->texture.slice(0, 0, 4)) / 4, 0).repeat({4, 1});
                    break;
                }
                case 14:
                {
                    // length of 4 features
                    conv_tex = torch::sum(torch::abs(ns->texture->texture.slice(0, 4, ns->texture->texture.size(0)))
    / (ns->texture->texture.size(0) - 4), 0) .repeat({4, 1}); break;
                }
                case 15:
                {
                    // diff to background color
                    auto t = torch::abs(ns.get()->texture->texture -
    ns.get()->texture->background_color.unsqueeze(1)); conv_tex = t; break;
                }
            }
            debug_color_texture->texture = col_tex;  // * grad_mul_factor;
            if (invert_colors_and_red)
            {
                debug_color_texture->texture = torch::max(debug_color_texture->texture) -
    debug_color_texture->texture;
            }

            if (use_visualize_confidence_as_full)
            {
                conv_tex = torch::where(conv_tex > stability_cutoff_value, torch::ones_like(conv_tex),
                                        torch::zeros_like(conv_tex));
            }
            debug_color_texture->confidence_value_of_point = conv_tex;
        }
        */
    {
        static torch::Tensor old_conf;
        bool init_conv_fallback_tex = true;
        if (init_conv_fallback_tex && !use_discard_in_main_render_window)
        {
            old_conf               = ns->texture->confidence_value_of_point.clone();
            init_conv_fallback_tex = false;
        }
        if (use_discard_in_main_render_window)
        {
            ns->texture->confidence_value_of_point = debug_color_texture->confidence_value_of_point.clone();
        }
        else
        {
            ns->texture->confidence_value_of_point = old_conf;
        }
    }
}



void RealTimeRenderer::RenderColor(ImageInfo fd, int flags, vec3 debug_dir)
{
    SetupRenderedDebugColor();

    NeuralRenderInfo nri;
    nri.scene        = ns.get();
    nri.num_layers   = nri.scene->params->net_params.num_layers;  // color_layer; //
    nri.timer_system = nullptr;

    fd.image_index  = 0;
    fd.camera_index = 0;

    auto old_tex = nri.scene->texture;
    auto old_env = nri.scene->environment_map;

    nri.images.push_back(fd);

    if (current_best_gt != -1)
    {
        auto& f                         = scene->frames[current_best_gt];
        nri.images.front().camera_index = f.camera_index;
        nri.images.front().image_index  = f.image_index;
    }

    params->render_params.viewer_only = true;

    float old_cutoff = 0;
    if (use_custom_camera)
    {
        overrideForCustomCam(fd, old_cutoff);

        nri.images.front().camera_index      = 0;
        nri.images.front().image_index       = 0;
        nri.images.front().camera_model_type = fd.camera_model_type;
    }

    nri.params         = pipeline->render_module->params->render_params;
    nri.params.dropout = false;

    nri.params.test_refl_x = debug_dir.x();
    nri.params.test_refl_y = debug_dir.y();
    nri.params.test_refl_z = debug_dir.z();
    if (flags == 0)
    {
        // Render the point color
        nri.params.num_texture_channels = 4;
        nri.scene->texture              = debug_color_texture;

        nri.scene->environment_map = nullptr;
    }

    if (stability_cutoff_value != 10.f)
        pipeline->render_module->params->render_params.stability_cutoff_value =
            stability_cutoff_value * stability_cutoff_value_multiplier;

    nri.train         = false;
    nri.current_epoch = current_eval_epoch;
    // torch::Tensor x            = pipeline->render_module->forward(&nri).first.back();
    torch::Tensor x            = pipeline->render_module->forward(&nri).first[color_layer];
    nri.scene->texture         = old_tex;
    nri.scene->environment_map = old_env;

    x = x.squeeze();
    x = x.permute({1, 2, 0});


    // if (current_state_debug_show > 0)
    if (false)
    {
        // if(params->pipeline_params.use_point_adding_and_removing_module)
        {
            // use channel one confidence
            auto conf_c = x.slice(2, 0, 1);

            auto conf_r = conf_c.cpu();
            auto a      = (float*)conf_r.data_ptr();

            for (int i = 0; i < conf_r.sizes()[0] * conf_r.sizes()[1]; i++)
            {
                if (a[i] <= 0.01)
                {
                    a[i] = 1.0;
                }
            }
            conf_r = conf_r.cuda();
            x      = torch::cat({conf_c, conf_c, conf_c, torch::ones_like(conf_c)}, 2);
        }
    }
    else if (flags >= 1 || nri.params.add_depth_to_network)
    {
        x = x.slice(2, (flags - 1) * 4, flags * 4);
        x = x.abs();
    }
    if (invert_colors_and_red)
    {
        auto c = x.slice(2, 0, 1);
        x      = torch::cat({c, torch::zeros_like(c), torch::zeros_like(c), torch::ones_like(c)}, 2);
    }

    x = x * (1.f / color_scale);
    x = x.clamp(0.f, 1.f);

    x.slice(2, 3, 4).fill_(1);

    x = x.contiguous();

    if (!output_color || x.size(0) != output_color->getHeight() || x.size(1) != output_color->getWidth())
    {
        TemplatedImage<vec4> tmp(x.size(0), x.size(1));
        output_color = std::make_shared<Texture>(tmp);
        output_color->setFiltering(GL_NEAREST);

        color_interop = std::make_shared<Saiga::CUDA::Interop>();
        color_interop->initImage(output_color->getId(), output_color->getTarget());

        std::cout << "Setting Debug Output Size to " << x.size(1) << "x" << x.size(0) << std::endl;
    }

    color_interop->mapImage();
    CHECK_CUDA_ERROR(cudaMemcpy2DToArray(color_interop->array, 0, 0, x.data_ptr(), x.stride(0) * sizeof(float),
                                         x.size(1) * x.size(2) * sizeof(float), x.size(0), cudaMemcpyDeviceToDevice));
    color_interop->unmap();


    if (use_custom_camera)
    {
        std::swap(ns->intrinsics, rt_intrinsics);
        std::swap(ns->poses, rt_extrinsics);
        params->render_params.dist_cutoff = old_cutoff;
    }
}


void RealTimeRenderer::LoadNets()
{
    if (experiments.empty()) return;

    ns                  = nullptr;
    pipeline            = nullptr;
    debug_color_texture = nullptr;

    auto ex = experiments[current_ex];
    auto ep = ex.eps[current_ep];

    std::cout << "loading checkpoint " << ex.name << " -> " << ep.name << std::endl;

    params = std::make_shared<CombinedParams>(ex.dir + "/params.ini");

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    if (deviceProp.major > 6)
    {
        std::cout << "Using half_float inference" << std::endl;
        params->net_params.half_float = true;
    }

    params->pipeline_params.train             = false;
    params->render_params.render_outliers     = false;
    params->train_params.checkpoint_directory = ep.dir;

    params->train_params.loss_vgg = 0;
    params->train_params.loss_l1  = 0;
    params->train_params.loss_mse = 0;

    prev_debug_col = vec3(1, 1, 1);

    current_best_gt = -1;
    reset_new_ex    = true;
    pipeline        = std::make_shared<NeuralPipeline>(params);
    pipeline->Train(false);
    pipeline->timer_system = &timer_system;

    pipeline->render_network->eval();

    ns = std::make_shared<NeuralScene>(scene, params, true);
    std::cout << "Loaded Neural Scene." << std::endl;
    ns->to(device);
    ns->Train(0, false);

    rt_intrinsics = IntrinsicsModule(scene->scene_cameras.front().K);
    rt_extrinsics = PoseModule(scene->frames[0].pose);

    if (ns->camera->exposures_values.defined())
    {
        auto ex_cpu = ns->camera->exposures_values.cpu();
        SAIGA_ASSERT(scene->frames.size() == ex_cpu.size(0));
        for (int i = 0; i < scene->frames.size(); ++i)
        {
            float ex                        = ex_cpu[i][0][0][0].item().toFloat();
            scene->frames[i].exposure_value = ex + scene->dataset_params.scene_exposure_value;
        }
    }
    current_eval_epoch = ep.ep;
    {
        // select render mode
        int mode_select         = 0;
        int diff_to_start_epoch = 1000000;
        for (int i = 0; i < params->pipeline_params.render_modes_start_epochs.size(); ++i)
        {
            int start_ep_m = params->pipeline_params.render_modes_start_epochs[i];
            if (current_eval_epoch >= start_ep_m)
            {
                if (current_eval_epoch - start_ep_m < diff_to_start_epoch)
                {
                    diff_to_start_epoch = current_eval_epoch - start_ep_m;
                    mode_select         = i;
                }
            }
        }
        params->render_params.render_mode = mode_select;
    }
}