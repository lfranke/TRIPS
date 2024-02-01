/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "TrainLiveViewer.h"
#ifdef LIVE_VIEWER

#    include "saiga/vision/torch/TorchHelper.h"

#    include "config.h"

#    include <csignal>
extern Saiga::Camera* camera;

void set_imgui_dark_theme();

LossBufferWithGraph::LossBufferWithGraph(int size_of_buffer, std::string name)
    : size_of_buffer(size_of_buffer), name(name)
{
    build();
}
void LossBufferWithGraph::build()
{
    m_loss_graph         = std::vector<float>(size_of_buffer, 0);
    m_loss_graph_samples = 0;
}
void LossBufferWithGraph::imgui(float size_of_widget)
{
    ImGui::PlotLines(("loss graph (" + name + ")").c_str(), m_loss_graph.data(),
                     std::min(m_loss_graph_samples, int(m_loss_graph.size())),
                     (m_loss_graph_samples < m_loss_graph.size()) ? 0 : (m_loss_graph_samples % m_loss_graph.size()), 0,
                     FLT_MAX, FLT_MAX, ImVec2(0, size_of_widget));
    ImGui::SameLine();
    ImGui::Text(std::to_string(m_loss_graph[m_loss_graph_samples - 1]).c_str());
}
void LossBufferWithGraph::addLossSample(float loss_sample)
{
    m_loss_graph[m_loss_graph_samples++ % m_loss_graph.size()] = loss_sample;
}


TrainViewer::TrainViewer(const Saiga::WindowParameters& windowParameters,
                         const Saiga::OpenGLParameters& openglParameters, const RenderingParameters& rendererParameters)
    : StandaloneWindow(windowParameters, openglParameters, rendererParameters),
      loss_batches(size_of_loss_buffer, "batches"),
      loss_epochs(500, "epochs"),
      val_img_loss_lpips(500, "lpips"),
      val_img_loss_ssim(500, "ssim"),
      val_img_loss_vgg(500, "vgg"),
      val_img_loss_l1(500, "l1"),
      val_img_loss_l2(500, "l2"),
      val_img_loss_psnr(500, "psnr")
{
    TemplatedImage<vec4> output_image;
    output_image.create(1080, 1920);
    output_image.getImageView().set(vec4(1, 1, 1, 1));
    output_texture_render = std::make_shared<Texture>(output_image);

    set_imgui_dark_theme();

    auto editor_layout = std::make_unique<EditorLayoutL>();
    editor_layout->RegisterImguiWindow("Neural View", EditorLayoutL::WINDOW_POSITION_3DVIEW);
    editor_layout->RegisterImguiWindow("Options", EditorLayoutL::WINDOW_POSITION_BOTTOM);
    editor_layout->RegisterImguiWindow("Log", EditorLayoutL::WINDOW_POSITION_BOTTOM);
    editor_gui.SetLayout(std::move(editor_layout));
}

void TrainViewer::update(float dt)
{
    scene_camera.update(dt);
}


void TrainViewer::keyPressed(int key, int scancode, int mods)
{
    Saiga::glfw_KeyListener::keyPressed(key, scancode, mods);

    switch (key)
    {
        case GLFW_KEY_W:
        {
            use_custom_camera = true;
            break;
        }
        case GLFW_KEY_S:
        {
            use_custom_camera = true;
            break;
        }
        case GLFW_KEY_A:
        {
            use_custom_camera = true;
            break;
        }
        case GLFW_KEY_D:
        {
            use_custom_camera = true;
            break;
        }
        case GLFW_KEY_ESCAPE:
        {
            window->close();
            // raise signal, train signal handler will catch it and save the current state
            std::raise(SIGINT);
            break;
        }
    }
}
void TrainViewer::setupCamera(std::shared_ptr<SceneData>& scene, int frame_index)
{
    // std::cout << "Set cam to " << frame_index << std::endl;
    ::camera = &scene_camera;
    auto& f  = scene->frames[frame_index];
    camera->setModelMatrix(f.OpenglModel());
    camera->updateFromModel();
    window->setCamera(camera);
    if (!rt_intrinsics)
    {
        rt_intrinsics = IntrinsicsModule(scene->scene_cameras.front().K);
    }
    if (!rt_extrinsics) rt_extrinsics = PoseModule(scene->frames[0].pose);
}

void TrainViewer::newFrameArrived(float dt)
{
    update(dt);
    // updating.interpolate(dt, interpolation);
    window->update(dt);
    window->render();
    window->swap();
    interpolate(dt, 0.f);
}
void TrainViewer::interpolate(float dt, float interpolation)
{
    if (renderer->use_mouse_input_in_3dview || mouse_on_view)
    {
        scene_camera.interpolate(dt, interpolation);
        // std::cout << scene_camera.view << std::endl;
    }
    mouse_on_view = false;
}

void TrainViewer::render(Saiga::RenderInfo render_info)
{
    mouse_on_view = false;

    if (render_info.render_pass == Saiga::RenderPass::GUI)
    {
        if (ImGui::Begin("Neural View"))
        {
            ImGui::BeginChild("neural_child", ImVec2(0, 0), false,
                              ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoMove);

            ImGui::Texture(output_texture_render.get(), ImGui::GetWindowContentRegionMax(), false);
            mouse_on_view = ImGui::IsItemHovered();
            ImGui::EndChild();
        }
        ImGui::End();
        imgui();
        // if (ImGui::Begin("Ops")) ImGui::End();
    }
}
void TrainViewer::imgui()
{
    if (ImGui::Begin("Options"))
    {
        ImGui::InputInt("update interval", &update_interval);
        if (ImGui::SliderInt("view_num", &view_num, 0, max_images - 1))
        {
            // use_custom_camera = false;
            // std::cout << "Switch to view " << view_num << std::endl;
            setupCamera(last_scene, view_num);
        }
        ImGui::Checkbox("own movement", &use_custom_camera);
        loss_batches.imgui();
        loss_epochs.imgui();

        if (ImGui::Button("Restart Training")) restart_training = true;

        ImGui::Text("losses this image:");
        ImGui::SameLine();
        ImGui::InputInt("samples per epoch", &loss_samples_per_epoch);

        val_img_loss_lpips.imgui();
        val_img_loss_psnr.imgui();
        val_img_loss_ssim.imgui();
        val_img_loss_vgg.imgui();
        val_img_loss_l1.imgui();
        val_img_loss_l2.imgui();
    }
    ImGui::End();
}
void TrainViewer::addValLoss(LossResult loss)
{
    val_img_loss_lpips.addLossSample(loss.loss_lpips);
    val_img_loss_ssim.addLossSample(loss.loss_ssim);
    val_img_loss_vgg.addLossSample(loss.loss_vgg);
    val_img_loss_l1.addLossSample(loss.loss_l1);
    val_img_loss_l2.addLossSample(loss.loss_mse);
    val_img_loss_psnr.addLossSample(loss.loss_psnr);
}

bool TrainViewer::checkForRestart()
{
    if (restart_training)
    {
        restart_training = false;

        // reset plots
        loss_batches.build();
        loss_epochs.build();
        val_img_loss_lpips.build();
        val_img_loss_ssim.build();
        val_img_loss_vgg.build();
        val_img_loss_l1.build();
        val_img_loss_l2.build();
        val_img_loss_psnr.build();
        return true;
    }
    return restart_training;
}
void TrainViewer::addLossSample(float val)
{
    loss_batches.addLossSample(val);
}

void TrainViewer::addEpochLoss(float val)
{
    loss_epochs.addLossSample(val);

    // reset loss sample count
    sample_this_epoch = 0;
}



ImageInfo TrainViewer::CurrentFrameData(std::shared_ptr<NeuralScene>& neural_scene, int frame_idx)
{
    auto& scene = neural_scene->scene;
    last_scene  = scene;
    ImageInfo fd;
    fd.crop_rotation.setZero();
    fd.crop_rotation(0, 0) = 1;
    fd.crop_rotation(1, 1) = 1;

    auto frame     = scene->frames[frame_idx];
    auto scene_cam = scene->scene_cameras[frame.camera_index];

    fd.w = scene_cam.w;  // * scene->dataset_params.render_scale;
    fd.h = scene_cam.h;  // * scene->dataset_params.render_scale;
    fd.K = scene_cam.K;

    fd.camera_model_type = scene_cam.camera_model_type;
    fd.distortion        = scene_cam.distortion;
    fd.ocam              = scene_cam.ocam.cast<float>();

    fd.camera_index = frame.camera_index;
    fd.image_index  = frame_idx;

    float downsample_fac = 1.f;

    fd.w *= scene->dataset_params.render_scale * downsample_fac;
    fd.h *= scene->dataset_params.render_scale * downsample_fac;

    fd.crop_transform = fd.crop_transform.scale(scene->dataset_params.render_scale * downsample_fac);
    // fd.pose           = scene->frames[frame_idx].pose;
    // auto pose = neural_scene->poses->Download(frame_idx).inverse();
    // fd.pose   = pose;

    if (use_custom_camera)
    {
        fd.pose = Sophus::SE3f::fitToSE3(scene_camera.model * GL2CVView()).cast<double>();
    }

    return fd;
}


void TrainViewer::checkAndUpdateTexture(std::shared_ptr<NeuralPipeline> pipeline,
                                        std::shared_ptr<NeuralScene> neural_scene, int epoch_id)
{
    static std::chrono::time_point<std::chrono::system_clock> last_update = std::chrono::system_clock::now();

    std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();

    float time_since_last_update = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_update).count();
    if (time_since_last_update > update_interval)
    {
        SAIGA_OPTIONAL_TIME_MEASURE("RenderLiveViewer", pipeline->timer_system);

        pipeline->params->render_params.viewer_only = true;
        pipeline->params->pipeline_params.train     = false;
        // select render mode
        int mode_select         = 0;
        int diff_to_start_epoch = 1000000;
        for (int i = 0; i < pipeline->params->pipeline_params.render_modes_start_epochs.size(); ++i)
        {
            int start_ep_m = pipeline->params->pipeline_params.render_modes_start_epochs[i];
            if (epoch_id >= start_ep_m)
            {
                if (epoch_id - start_ep_m < diff_to_start_epoch)
                {
                    diff_to_start_epoch = epoch_id - start_ep_m;
                    mode_select         = i;
                }
            }
        }
        pipeline->params->render_params.render_mode = mode_select;

        const int image_idx = view_num;
        auto fd             = CurrentFrameData(neural_scene, image_idx);  // scene->Frame(image_idx);
        int camera_id       = fd.camera_index;
        std::vector<NeuralTrainData> live_view_batch(1);
        // live_view_batch.front() = std::make_shared<TorchFrameData>();

        live_view_batch.front() = std::make_shared<TorchFrameData>();

        live_view_batch.front()->img              = fd;
        live_view_batch.front()->img.camera_index = camera_id;
        live_view_batch.front()->camera_index     = torch::full({1}, camera_id);
        live_view_batch.front()->img.image_index  = image_idx;

        static torch::Tensor uv_tensor;
        if (!uv_tensor.defined() || uv_tensor.size(1) != fd.h || uv_tensor.size(2) != fd.w)
        {
            auto uv_image = InitialUVImage(fd.h, fd.w);
            uv_tensor     = ImageViewToTensor(uv_image.getImageView()).to(torch::kCUDA);
        }
        static torch::Tensor direction_tensor;
        if (!direction_tensor.defined() || direction_tensor.size(1) != fd.h || direction_tensor.size(2) != fd.w)
        {
            auto& cam = neural_scene->scene->frames[image_idx];
            auto direction_img =
                InitialDirectionImage(cam.w * neural_scene->scene->dataset_params.render_scale,
                                      cam.h * neural_scene->scene->dataset_params.render_scale, cam.camera_model_type,
                                      cam.K, cam.distortion, cam.ocam.cast<double>());
            direction_tensor = ImageViewToTensor(direction_img.getImageView()).to(torch::kCUDA);
        }
        live_view_batch.front()->direction = direction_tensor;
        //  PrintTensorInfo(live_view_batch.front()->direction);
        //  std::cout << fd.h << std::endl;

        SAIGA_ASSERT(live_view_batch.front()->direction.size(1) == fd.h);
        SAIGA_ASSERT(live_view_batch.front()->direction.size(2) == fd.w);
        live_view_batch.front()->uv = uv_tensor;
        long man_timestep           = 0;
        live_view_batch.front()->timestep =
            torch::from_blob(&man_timestep, {1}, torch::TensorOptions().dtype(torch::kLong)).cuda().to(torch::kFloat);

        // bool loss_stats = !use_custom_camera && sample_this_epoch < loss_samples_per_epoch &&
        //                   fd.camera_model_type != CameraModel::OCAM;

        bool loss_stats = false;

        float old_cutoff = 0;
        if (use_custom_camera)
        {
            rt_extrinsics->SetPose(0, fd.pose);
            old_cutoff = pipeline->params->render_params.dist_cutoff;
            // if (scene->scene_cameras.begin()->camera_model_type == CameraModel::PINHOLE_DISTORTION)
            if (fd.camera_model_type == CameraModel::PINHOLE_DISTORTION)
            {
                pipeline->params->render_params.dist_cutoff = fd.distortion.MonotonicThreshold();
                // std::cout << "PINHOLE " << fd.K << " ---- " << fd.distortion << std::endl;
                rt_intrinsics->SetPinholeIntrinsics(0, fd.K, fd.distortion);
            }
            // else if (scene->scene_cameras.begin()->camera_model_type == CameraModel::OCAM)
            else if (fd.camera_model_type == CameraModel::OCAM)

            {
                pipeline->params->render_params.dist_cutoff = 0.15f;
                rt_intrinsics->SetOcamIntrinsics(0, fd.ocam);
            }
            else
            {
                rt_intrinsics->SetPinholeIntrinsics(0, fd.K, fd.distortion);
            }
            std::swap(neural_scene->intrinsics, rt_intrinsics);
            std::swap(neural_scene->poses, rt_extrinsics);
            live_view_batch.front()->img.camera_index = 0;
            live_view_batch.front()->img.image_index  = 0;

            live_view_batch.front()->img.camera_model_type = fd.camera_model_type;
        }
        else if (loss_stats)
        {
            const auto fd = neural_scene->scene->Frame(view_num);

            Saiga::TemplatedImage<ucvec3> img_gt_large(neural_scene->scene->dataset_params.image_dir + "/" +
                                                       fd.target_file);

            live_view_batch.front()->target = ImageViewToTensor(img_gt_large.getImageView());
            TemplatedImage<unsigned char> mask(live_view_batch.front()->img.h, live_view_batch.front()->img.w);
            mask.getImageView().set(255);
            live_view_batch.front()->target_mask = ImageViewToTensor(mask.getImageView());
            live_view_batch.front()->to(torch::kCUDA);
            ++sample_this_epoch;
        }
        //  live_view_batch.front()->print();

        pipeline->render_module->train(false);

        // neural_renderer->tone_mapper.params_dirty |= renderer->tone_mapper.params_dirty;
        auto neural_exposure_value = fd.exposure_value;  //- neural_scene->scene->dataset_params.scene_exposure_value;
        // auto neural_exposure_value = neural_scene->scene->dataset_params.scene_exposure_value;

        auto f_result = pipeline->Forward(*neural_scene, live_view_batch, {}, loss_stats, epoch_id, false  //);
                                          ,
                                          neural_exposure_value, fd.white_balance);
        torch::Tensor result_live_viewer = f_result.x;
        result_live_viewer               = result_live_viewer.squeeze();

        //  f_result.float_loss.Print();

        torch::Tensor alpha_channel = torch::ones({1, result_live_viewer.size(1), result_live_viewer.size(2)},
                                                  torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat));
        result_live_viewer          = torch::cat({result_live_viewer, alpha_channel}, 0);


        if (loss_stats) addValLoss(f_result.float_loss);

        result_live_viewer = result_live_viewer.permute({1, 2, 0});
        // x = x.clamp(0.f, 1.f);
        result_live_viewer = result_live_viewer.contiguous();

        static std::shared_ptr<Saiga::CUDA::Interop> texture_interop = std::make_shared<Saiga::CUDA::Interop>();
        // static std::shared_ptr<Texture> output_texture;
        static TemplatedImage<vec4> output_image;
        if (fd.h != output_image.h || fd.w != output_image.w)
        {
            // output_image.create(1080, 1920);
            output_image.create(fd.h, fd.w);
            output_image.getImageView().set(vec4(1, 1, 1, 1));
            output_texture_render = std::make_shared<Texture>(output_image);


            //  output_texture = std::make_shared<Texture>(output_image);
            texture_interop->initImage(output_texture_render->getId(), output_texture_render->getTarget());
        }

        texture_interop->mapImage();
        CHECK_CUDA_ERROR(cudaMemcpy2DToArray(texture_interop->array, 0, 0, result_live_viewer.data_ptr(),
                                             result_live_viewer.stride(0) * sizeof(float),
                                             result_live_viewer.size(1) * result_live_viewer.size(2) * sizeof(float),
                                             result_live_viewer.size(0), cudaMemcpyDeviceToDevice));
        texture_interop->unmap();

        pipeline->render_module->train(true);
        pipeline->params->render_params.viewer_only = false;
        pipeline->params->pipeline_params.train     = true;


        if (use_custom_camera)
        {
            std::swap(neural_scene->intrinsics, rt_intrinsics);
            std::swap(neural_scene->poses, rt_extrinsics);

            pipeline->params->render_params.dist_cutoff = old_cutoff;
        }

        newFrameArrived(time_since_last_update / 1000.f);
        last_update = std::chrono::system_clock::now();
    }
}

void set_imgui_dark_theme()
{
    ImGui::StyleColorsDark();

    auto& colors              = ImGui::GetStyle().Colors;
    colors[ImGuiCol_WindowBg] = ImVec4{0.1f, 0.105f, 0.11f, 1.0f};

    // Headers
    colors[ImGuiCol_Header]        = ImVec4{0.2f, 0.205f, 0.21f, 1.0f};
    colors[ImGuiCol_HeaderHovered] = ImVec4{0.3f, 0.305f, 0.31f, 1.0f};
    colors[ImGuiCol_HeaderActive]  = ImVec4{0.15f, 0.1505f, 0.151f, 1.0f};

    // Buttons
    colors[ImGuiCol_Button]        = ImVec4{0.2f, 0.205f, 0.21f, 1.0f};
    colors[ImGuiCol_ButtonHovered] = ImVec4{0.3f, 0.305f, 0.31f, 1.0f};
    colors[ImGuiCol_ButtonActive]  = ImVec4{0.15f, 0.1505f, 0.151f, 1.0f};

    // Frame BG
    colors[ImGuiCol_FrameBg]        = ImVec4{0.2f, 0.205f, 0.21f, 1.0f};
    colors[ImGuiCol_FrameBgHovered] = ImVec4{0.3f, 0.305f, 0.31f, 1.0f};
    colors[ImGuiCol_FrameBgActive]  = ImVec4{0.15f, 0.1505f, 0.151f, 1.0f};

    // Tabs
    colors[ImGuiCol_Tab]                = ImVec4{0.15f, 0.1505f, 0.151f, 1.0f};
    colors[ImGuiCol_TabHovered]         = ImVec4{0.38f, 0.3805f, 0.381f, 1.0f};
    colors[ImGuiCol_TabActive]          = ImVec4{0.28f, 0.2805f, 0.281f, 1.0f};
    colors[ImGuiCol_TabUnfocused]       = ImVec4{0.15f, 0.1505f, 0.151f, 1.0f};
    colors[ImGuiCol_TabUnfocusedActive] = ImVec4{0.2f, 0.205f, 0.21f, 1.0f};

    // Title
    colors[ImGuiCol_TitleBg]          = ImVec4{0.15f, 0.1505f, 0.151f, 1.0f};
    colors[ImGuiCol_TitleBgActive]    = ImVec4{0.15f, 0.1505f, 0.151f, 1.0f};
    colors[ImGuiCol_TitleBgCollapsed] = ImVec4{0.15f, 0.1505f, 0.151f, 1.0f};


    ImGuiStyle& style     = ImGui::GetStyle();
    style.FrameBorderSize = 1.f;
    style.FramePadding    = ImVec2(5.f, 2.f);

    ImGuiIO& io = ImGui::GetIO();
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
    {
        style.WindowRounding        = 0.f;
        colors[ImGuiCol_WindowBg].w = 1.f;
    }
}
#endif