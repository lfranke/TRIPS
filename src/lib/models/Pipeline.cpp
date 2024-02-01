/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Pipeline.h"

#include <c10/cuda/CUDACachingAllocator.h>

CUDA::CudaTimerSystem* timer_for_nets;


NeuralPipeline::NeuralPipeline(std::shared_ptr<CombinedParams> _params) : params(_params)
{
    params->Check();
    if (params->net_params.network_version == "MultiScaleUnet2d")
    {
        // render_network = std::make_shared<DerivedRenderNet<MultiScaleUnet2d>>(params->net_params);
        render_network = std::make_shared<DerivedRenderNet<MultiScaleUnet2d>>(params->net_params);
    }
    else if (params->net_params.network_version == "MultiScaleUnet2dSlim")
    {
        render_network = std::make_shared<DerivedRenderNet<MultiScaleUnet2dSlim>>(params->net_params);
    }
    else if (params->net_params.network_version == "MultiScaleUnet2dUltraSlim")
    {
        render_network = std::make_shared<DerivedRenderNet<MultiScaleUnet2dUltraSlim>>(params->net_params);
    }
    else if (params->net_params.network_version == "MultiScaleUnet2dDecOnly")
    {
        render_network = std::make_shared<DerivedRenderNet<MultiScaleUnet2dDecOnly>>(params->net_params);
    }
    else if (params->net_params.network_version == "MultiScaleUnet2dDecOnlySmall")
    {
        render_network = std::make_shared<DerivedRenderNet<MultiScaleUnet2dDecOnlySmall>>(params->net_params);
    }
    else if (params->net_params.network_version == "MultiScaleUnet2dDecOnlySmallFixed")
    {
        render_network = std::make_shared<DerivedRenderNet<MultiScaleUnet2dDecOnlySmallFixed>>(params->net_params);
    }
    else if (params->net_params.network_version == "MultiScaleUnet2dSphericalHarmonics")
    {
        render_network = std::make_shared<DerivedRenderNet<MultiScaleUnet2dSphericalHarmonics>>(params->net_params);
    }
    else if (params->net_params.network_version == "MultiScaleUnet2dSphericalHarmonicsFixed")
    {
        render_network =
            std::make_shared<DerivedRenderNet<MultiScaleUnet2dSphericalHarmonicsFixed>>(params->net_params);
    }
    else if (params->net_params.network_version == "MultiSphericalHarmonics")
    {
        render_network = std::make_shared<DerivedRenderNet<MultiSphericalHarmonics>>(params->net_params);
    }
    else if (params->net_params.network_version == "MultiScaleUnet2dSphericalHarmonicsInL2Fixed")
    {
        render_network =
            std::make_shared<DerivedRenderNet<MultiScaleUnet2dSphericalHarmonicsInL2Fixed>>(params->net_params);
    }


    else
    {
        SAIGA_ASSERT(false);
    }

    if (!params->optimizer_params.fix_dynamic_refinement)
    {
        // refinement_module = RefinementNet(params->net_params);
        dynamic_refinement_module = DynamicRefinementMLP();
    }


    if (params->optimizer_params.network_checkpoint_directory != "")
    {
        std::cout << "Resuming network from " << params->optimizer_params.network_checkpoint_directory << std::endl;
        LoadCheckpoint(params->optimizer_params.network_checkpoint_directory);
    }
    // PrintModelParamsCompact(render_network.get());
    LoadCheckpoint(params->train_params.checkpoint_directory);

    if (params->net_params.half_float)
    {
        render_network->to(torch::kFloat16);
    }
    else
    {
        render_network->to(torch::kFloat32);
    }
    std::cout << "RENDER NETWORK: NUM PARAMETERS: " << render_network->parameters().size() << std::endl;
    // render_network->PrintModelParamsCompact();
    {
        size_t sum = 0;

        for (auto& t : render_network->parameters())
        {
            size_t local_sum = 1;
            for (auto i : t.sizes())
            {
                local_sum *= i;
            }
            sum += local_sum;
        }
        std::cout << "Total Model Params: " << sum << std::endl;
    }
    render_network->train();
    render_network->to(device);
    if (!params->optimizer_params.fix_dynamic_refinement)
    {
        dynamic_refinement_module->train();
        dynamic_refinement_module->to(device);
    }


    if (params->pipeline_params.train)
    {

        std::cout << "Loading VGG from: " << params->train_params.vgg_path << std::endl;
        loss_vgg =
            std::make_shared<Saiga::PretrainedVGG19Loss>(Saiga::PretrainedVGG19Loss(params->train_params.vgg_path));

        loss_vgg->eval();
        loss_vgg->to(device);


        loss_lpips.module.eval();
        loss_lpips.module.to(device);

        loss_ssim->eval();
        loss_ssim->to(device);

        if (!params->optimizer_params.fix_render_network)
        {
            // In half precision the default eps of 1e-8 is rounded to 0
            double adam_eps = params->net_params.half_float ? 1e-4 : 1e-8;
            // if (params->optimizer_params.use_myadam_everywhere)
            //{
            //     using TexOpt   = torch::optim::MyAdamOptions;
            //     using TexOptim = torch::optim::MyAdam;
            //     std::vector<torch::optim::OptimizerParamGroup> g;
            //     auto opt_t = std::make_unique<TexOpt>(params->optimizer_params.lr_render_network);
            //     opt_t->eps(adam_eps);
            //     std::vector<torch::Tensor> ts = render_network->parameters();
            //     g.emplace_back(ts, std::move(opt_t));
            //     render_optimizer = std::make_shared<TexOptim>(g,
            //     TexOpt(params->optimizer_params.lr_render_network)); if
            //     (!params->optimizer_params.fix_dynamic_refinement)
            //     {
            //         std::vector<torch::optim::OptimizerParamGroup> g2 =
            //         {dynamic_refinement_module->parameters()}; refinement_optimizer =
            //         std::make_shared<torch::optim::MyAdam>(
            //             g2, torch::optim::MyAdamOptions().lr(params->optimizer_params.lr_dynamic_refinement));
            //     }
            // }
            // else
            {
                std::vector<torch::optim::OptimizerParamGroup> g = {render_network->parameters()};
                render_optimizer                                 = std::make_shared<torch::optim::Adam>(
                    g, torch::optim::AdamOptions().lr(params->optimizer_params.lr_render_network).eps(adam_eps));

                if (!params->optimizer_params.fix_dynamic_refinement)
                {
                    std::vector<torch::optim::OptimizerParamGroup> g2 = {dynamic_refinement_module->parameters()};
                    refinement_optimizer                              = std::make_shared<torch::optim::SGD>(
                        g2, torch::optim::SGDOptions(params->optimizer_params.lr_dynamic_refinement));
                }
            }
        }
        else
        {
            render_network->eval();
        }
    }
}


void evaluateSH(torch::Tensor& positions_in, torch::Tensor pose_in, torch::Tensor& texture_in,
                torch::Tensor& texture_out, int sh_bands = 3)
{
    static std::shared_ptr<SphericalHarmonicsEncoding> enc;
    if (!enc)
    {
        enc = std::make_shared<SphericalHarmonicsEncoding>(sh_bands);
        enc->module->to(torch::kCUDA);
    }

    torch::Tensor norm_dir = torch::nn::functional::normalize(positions_in.slice(1, 0, 3) - pose_in,
                                                              torch::nn::functional::NormalizeFuncOptions().dim(1));

    auto sh_enc = enc->forward(norm_dir);

    int feature_out = texture_in.size(0) / (sh_bands * sh_bands);
    auto t_per      = texture_in.permute({1, 0});

    auto enc_repeated = sh_enc.repeat_interleave(feature_out, 1);

    auto tex = (t_per * enc_repeated);

    texture_out = tex.reshape({tex.size(0), feature_out, sh_bands * sh_bands}).sum(-1).permute({1, 0});
}



ForwardResult NeuralPipeline::Forward(NeuralScene& scene, std::vector<NeuralTrainData>& batch,
                                      torch::Tensor global_mask, bool loss_statistics, int current_epoch,
                                      bool keep_image, float fixed_exposure, vec3 fixed_white_balance)
{
    TimerSystem::TimeData* unet_backwards_timer;
    TimerSystem::TimeData* camera_backwards_timer;
    TimerSystem::TimeData* loss_backwards_timer;
    TimerSystem::TimeData* render_backwards_timer;
    TimerSystem::TimeData* prep_backwards_timer;
    // timer_system = nullptr;
    if (timer_system)
    {
        unet_backwards_timer   = timer_system->CreateNewTimer("Unet Backwards");
        camera_backwards_timer = timer_system->CreateNewTimer("Camera Backwards");
        loss_backwards_timer   = timer_system->CreateNewTimer("Loss Backwards");
        render_backwards_timer = timer_system->CreateNewTimer("Render Backwards");
        prep_backwards_timer   = timer_system->CreateNewTimer("Preparation Backwards");
    }
    if (!params->render_params.viewer_only)
    {
        // select render mode
        int mode_select         = 0;
        int diff_to_start_epoch = 1000000;
        for (int i = 0; i < params->pipeline_params.render_modes_start_epochs.size(); ++i)
        {
            int start_ep_m = params->pipeline_params.render_modes_start_epochs[i];
            if (current_epoch >= start_ep_m)
            {
                if (current_epoch - start_ep_m < diff_to_start_epoch)
                {
                    diff_to_start_epoch = current_epoch - start_ep_m;
                    mode_select         = i;
                }
            }
        }
        params->render_params.render_mode = mode_select;
    }
    render_module->params = params;



    std::vector<torch::Tensor> neural_images;
    std::vector<torch::Tensor> masks;
    {
        // SAIGA_OPTIONAL_TIME_MEASURE("Clamp on confidence", timer_system);
        //  scene.texture->confidence_value_of_point = torch::clamp(scene.texture->confidence_value_of_point,0,1);
        SAIGA_OPTIONAL_TIME_MEASURE("Sigmoid on confidence", timer_system);
        float narrowing_fac =
            current_epoch > 0 ? params->points_adding_params.sigmoid_narrowing_factor * current_epoch : 0;
        scene.texture->PrepareConfidence(narrowing_fac, params->net_params.num_input_layers);
    }
    {
        SAIGA_OPTIONAL_TIME_MEASURE("Prep Tex", timer_system);
        scene.texture->PrepareTexture(params->pipeline_params.non_subzero_texture);
    }
    if (params->pipeline_params.num_spherical_harmonics_bands_per_point > 0)
    {
        // not implemented
        SAIGA_ASSERT(params->train_params.batch_size == 1);
        vec3 pose_cpu = scene.poses->Download(batch.front()->img.image_index).translation().cast<float>();
        torch::Tensor pose =
            torch::from_blob(&pose_cpu, {3}, torch::TensorOptions().dtype(torch::kFloat)).clone().cuda();
        evaluateSH(scene.point_cloud_cuda->t_position, pose, scene.texture->texture_raw, scene.texture->texture,
                   params->pipeline_params.num_spherical_harmonics_bands_per_point);
        // PrintTensorInfo(scene.texture->texture);
    }

    std::function<torch::autograd::Variable(torch::autograd::Variable)> timer_grad_for_conf_raw(
        [prep_backwards_timer](torch::autograd::Variable grad)
        {
            // std::cout << "compute grad for neural img" << std::endl;
            //   prep_backwards_timer->Stop();
            return grad;
        });
    if (render_module->is_training()) scene.texture->confidence_raw.register_hook(timer_grad_for_conf_raw);

    if (!params->optimizer_params.fix_dynamic_refinement &&
        current_epoch > params->train_params.lock_dynamic_refinement_epochs)
    {
        SAIGA_OPTIONAL_TIME_MEASURE("Dynamic Refinement", timer_system);
        std::vector<torch::Tensor> dyn_r;
        for (auto& b : batch)
        {
            auto timesteps = b->timestep.repeat({scene.point_cloud_cuda->t_position.size(0), 1});
            auto pos       = scene.point_cloud_cuda->t_position.slice(1, 0, 3).clone();
            // std::cout << TensorInfo(timesteps);
            // std::cout << TensorInfo(pos);

            auto key = torch::cat({pos, timesteps}, 1);
            // std::cout << TensorInfo(key);
            auto x = dynamic_refinement_module->forward(key) * 0.01;
            dyn_r.push_back(x.unsqueeze(0));
        }
        scene.dynamic_refinement_t = torch::cat(dyn_r, 0);
        // std::cout << TensorInfo(scene.dynamic_refinement_t) << std::endl;
    }
    else
    {
        scene.dynamic_refinement_t = torch::zeros({1}).cuda();
    }

    std::function<torch::autograd::Variable(torch::autograd::Variable)> timer_grad_for_conf_perp(
        [prep_backwards_timer, render_backwards_timer](torch::autograd::Variable grad)
        {
            // std::cout << "compute grad for neural img" << std::endl;
            if (render_backwards_timer) render_backwards_timer->Stop();
            //     prep_backwards_timer->Start();
            return grad;
        });
    if (render_module->is_training() && timer_system)
        scene.texture->confidence_value_of_point.register_hook(timer_grad_for_conf_perp);
    {
        SAIGA_OPTIONAL_TIME_MEASURE("Render", timer_system);
        std::tie(neural_images, masks) = render_module->forward(scene, batch, current_epoch, timer_system);
        SAIGA_ASSERT(neural_images.size() == params->net_params.num_input_layers);
    }



    if (params->net_params.channels_last)
    {
        SAIGA_OPTIONAL_TIME_MEASURE("channels_last", timer_system);
        for (auto& t : neural_images)
        {
            t = t.to(t.options(), false, false, torch::MemoryFormat::ChannelsLast);
        }
    }

    if (params->net_params.half_float)
    {
        SAIGA_OPTIONAL_TIME_MEASURE("half_float", timer_system);
        for (auto& t : neural_images)
        {
            t = t.to(torch::kFloat16);
        }
        for (auto& t : masks)
        {
            t = t.to(torch::kFloat16);
        }
        for (auto& b : batch)
        {
            b->uv = b->uv.to(torch::kFloat16);
        }
    }


    std::function<torch::autograd::Variable(torch::autograd::Variable)> timer_grad_for_neural_image(
        [render_backwards_timer, unet_backwards_timer](torch::autograd::Variable grad)
        {
            // std::cout << "compute grad for neural img" << std::endl;
            if (unet_backwards_timer) unet_backwards_timer->Stop();
            if (render_backwards_timer) render_backwards_timer->Start();
            return grad;
        });
    if (render_module->is_training()) neural_images.front().register_hook(timer_grad_for_neural_image);


#if 0
    {
        std::cout << "> torch forward" << std::endl;
        std::vector<torch::Tensor> render_result = RenderPointCloud(*render_data);
        for (auto i : render_result)
        {
            PrintTensorInfo(i);
        }

        std::cout << "> my forward" << std::endl;
        std::vector<torch::Tensor> input2 = BlendPointCloud(render_data);
        for (auto i : input2)
        {
            PrintTensorInfo(i);
        }

        std::cout << "> diff forward" << std::endl;
        for (int i = 0; i < render_result.size(); ++i)
        {
            PrintTensorInfo(render_result[i] - input2[i]);
            //            SAIGA_ASSERT((render_result[i] - input2[i]).sum().item().toFloat() == 0);
        }

        if (scene.texture->texture.requires_grad() && torch::GradMode::is_enabled())
        {
            std::cout << "> torch backward" << std::endl;
            auto l1 = torch::zeros({1}, render_result.front().options());
            for (auto& i : render_result) l1 += i.sum();
            l1.backward();
            auto grad1 = scene.texture->texture.grad().clone();
            PrintTensorInfo(grad1);
            PrintTensorInfo(grad1.slice(1, 0, 1));
            PrintTensorInfo(grad1.slice(1, 1, 923745934857));
            std::cout << grad1.slice(1, 0, 1) << std::endl;
            scene.texture->texture.mutable_grad().zero_();

            std::cout << "> my backward" << std::endl;
            auto l2 = torch::zeros({1}, input2.front().options());
            for (auto& i : input2) l2 += i.sum();
            l2.backward();
            auto grad2 = scene.texture->texture.grad().clone();
            PrintTensorInfo(grad2);
            PrintTensorInfo(grad2.slice(1, 0, 1));
            PrintTensorInfo(grad2.slice(1, 1, 923745934857));
            std::cout << grad2.slice(1, 0, 1) << std::endl;

            std::cout << "> diff backward" << std::endl;
            PrintTensorInfo(grad1 - grad2);
            //            SAIGA_ASSERT((grad1 - grad2).sum().item().toFloat() == 0);
        }
        exit(0);
    }
#endif


    torch::Tensor local_mask;
    torch::Tensor frame_index;
    torch::Tensor full_target;
    torch::Tensor uv;
    torch::Tensor scale;
    torch::Tensor direction;
    {
        SAIGA_OPTIONAL_TIME_MEASURE("Stack", timer_system);

        // 2. Stack into batch
        std::vector<torch::Tensor> mask_list;
        std::vector<torch::Tensor> target_list;
        std::vector<torch::Tensor> uv_list;
        std::vector<torch::Tensor> direction_list;
        std::vector<torch::Tensor> scale_list;
        std::vector<torch::Tensor> frame_index_list;
        // multple renderings of the same image crop
        for (int i = 0; i < batch.size(); ++i)
        {
            auto& pd = batch[i];
            // for (auto& pd : batch)
            //{
            //   std::cout << TensorInfo(pd->uv) << std::endl;
            SAIGA_ASSERT(pd->uv.dim() == 3);

            if (pd->target.defined())
            {
                target_list.push_back((pd->target));
                mask_list.push_back((pd->target_mask));
            }

            if (pd->scale.defined())
            {
                scale_list.push_back((pd->scale));
            }
            uv_list.push_back((pd->uv));
            //  std::cout << pd->img.image_index << std::endl;
            if (pd->direction.defined() && !params->pipeline_params.skip_neural_render_network)
            {
                // PrintTensorInfo(scene.poses->poses_se3);
                auto pose = scene.poses->poses_se3.slice(0, pd->img.image_index, pd->img.image_index + 1);
                auto t = ProjectDirectionsToWS(pd->direction, pose);
                //// if (params->net_params.half_float) t = t.to(torch::kFloat16);
                direction_list.push_back(t);
            }
            if (pd->camera_index.defined()) frame_index_list.push_back((pd->camera_index));
        }
        if (!target_list.empty()) full_target = torch::stack(target_list);
        if (!mask_list.empty()) local_mask = torch::stack(mask_list);
        if (!scale_list.empty()) scale = torch::stack(scale_list);
        uv = torch::stack(uv_list);
        if (!frame_index_list.empty()) frame_index = torch::stack(frame_index_list);
        if (!direction_list.empty()) direction = torch::stack(direction_list);
        //   std::cout << "directionstack: " << TensorInfo(direction) << std::endl;
    }


    if (params->pipeline_params.log_texture)
    {
        SAIGA_OPTIONAL_TIME_MEASURE("log_texture", timer_system);

        for (auto& n : neural_images)
        {
            n = torch::exp(n);
        }
    }

#if 0
    if (params->pipeline_params.use_refinement_step){
        SAIGA_OPTIONAL_TIME_MEASURE("RefinementNet", timer_system);

        using namespace torch::indexing;
        int max_h = neural_images[0].sizes()[2];
        int max_w = neural_images[0].sizes()[3];

        //torch::Tensor view_q = torch::Tensor({neural_images[0].sizes()[0],3,neural_images[0].sizes()[2], neural_images[0].sizes()[3]}, );
        std::vector<torch::Tensor> directions;
        for(auto b : batch){
            int cam_ind = b->img.camera_index;
            Sophus::SE3f V = scene.scene->frames[cam_ind].pose.template cast<float>();
            vec3 dir = V*vec3(0,0,1);
            //auto x = torch::full({1,max_h,max_w}, float(dir.x()), torch::TensorOptions().dtype(torch::kFloat32));
            torch::Tensor d = torch::cat({torch::full({1,max_h,max_w},dir.x()),
                                          torch::full({1,max_h,max_w},dir.y()),
                                          torch::full({1,max_h,max_w},dir.z())},0).cuda().type_as(neural_images[0]);

            directions.push_back(d);
        }
        auto dirs = torch::stack(directions,0);

        for(int i=0; i< neural_images.size();++i)
        {
            torch::Tensor layer = torch::full({neural_images[i].sizes()[0],1,neural_images[i].sizes()[2],neural_images[i].sizes()[3]},float(i)).cuda().type_as(neural_images[i]);

            auto ni_view_layer = torch::cat({neural_images[i],dirs.index({{Slice(),Slice(),Slice(0,neural_images[i].sizes()[2]),Slice(0,neural_images[i].sizes()[3])}}), layer},1);

            neural_images[i] = refinement_module->forward(ni_view_layer);

        }
    }

#endif

    torch::Tensor x;
    if (params->pipeline_params.skip_neural_render_network)
    {
        x = neural_images.front();
        SAIGA_ASSERT(x.size(1) == 3);
    }
    else if (params->pipeline_params.skip_neural_render_network_but_add_layers)
    {
        SAIGA_OPTIONAL_TIME_MEASURE("Add Layers", timer_system);

        std::vector<double> scale     = {2.0, 2.0};
        auto up_func                  = torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(scale).mode(
            torch::kNearest));  //.mode(torch::kBilinear).align_corners(false));
        const int num_layers_upsample = params->net_params.num_input_layers;
        const int last_used_layer_idx = 0;

        //   torch::Tensor mask;
        for (int i = num_layers_upsample - 1; i >= last_used_layer_idx; --i)
        {
            if (i != num_layers_upsample - 1)
            {
                x = up_func->forward(x);
                x = torch::nn::functional::pad(
                    x, torch::nn::functional::PadFuncOptions(
                           {0, 0, neural_images[i].size(2) - x.size(2), neural_images[i].size(3) - x.size(3)})
                           .mode(torch::kConstant));
                x = x + neural_images[i];

                //       mask = up_func->forward(mask);
                // auto m2 = torch::where(neural_images[i] != 0, torch::ones_like(neural_images[i]),
                //                        torch::zeros_like(neural_images[i]));
                //       mask    = mask + m2;
            }
            else
            {
                x = neural_images[i];
                //       mask = torch::where(neural_images[i] != 0, torch::ones_like(neural_images[i]),
                //                           torch::zeros_like(neural_images[i]));
            }
        }
        // auto m_nonzero = torch::where(mask != 0, mask, torch::ones_like(mask));
        // x              = x / m_nonzero;
    }
    else
    {
        SAIGA_OPTIONAL_TIME_MEASURE("Unet", timer_system);
        SAIGA_ASSERT(!neural_images.empty());
        SAIGA_ASSERT(neural_images.size() == params->net_params.num_input_layers);
        SAIGA_ASSERT(neural_images.front().size(1) == params->net_params.num_input_channels);

        if (params->net_params.network_version == "MultiScaleUnet2dSphericalHarmonics" ||
            params->net_params.network_version == "MultiScaleUnet2dSphericalHarmonicsFixed" ||
            params->net_params.network_version == "MultiScaleUnet2dSphericalHarmonicsInL2Fixed" ||
            params->net_params.network_version == "MultiSphericalHarmonics")
        {
            neural_images.push_back(direction);
        }

        // std::cout << neural_images.size() << std::endl;

        if (timer_system) timer_for_nets = timer_system;

        if (params->render_params.output_background_mask)
        {
            x = render_network->forward(neural_images, masks);
        }
        else
        {
            x = render_network->forward(neural_images);
        }
        SAIGA_ASSERT(x.size(1) == params->net_params.num_output_channels);
        SAIGA_ASSERT(x.size(1) == 3);
    }

    std::function<torch::autograd::Variable(torch::autograd::Variable)> timer_grad_for_x(
        [unet_backwards_timer, camera_backwards_timer](torch::autograd::Variable grad)
        {
            // std::cout << "compute grad for x" << std::endl;
            if (camera_backwards_timer) camera_backwards_timer->Stop();
            if (unet_backwards_timer) unet_backwards_timer->Start();

            return grad;
        });
    if (render_module->is_training()) x.register_hook(timer_grad_for_x);


    torch::Tensor target;

    if ((x.size(2) != neural_images.front().size(2) || x.size(3) != neural_images.front().size(3)))
    {
        SAIGA_OPTIONAL_TIME_MEASURE("crop", timer_system);

        // The unet has cropped a few pixels because the input wasn't divisible by 16
        if (uv.defined())
        {
            uv = CenterCrop2D(uv, x.sizes());
        }
        if (full_target.defined())
        {
            target = CenterCrop2D(full_target, x.sizes());
        }
        if (global_mask.defined())
        {
            global_mask = CenterCrop2D(global_mask, x.sizes());
        }

        if (local_mask.defined())
        {
            local_mask = CenterCrop2D(local_mask, x.sizes());
        }
    }
    else
    {
        target = full_target;
        if (full_target.defined())
            if (x.size(2) != target.size(2) || x.size(3) != target.size(3))
            {
                target = CenterCrop2D(full_target, x.sizes());
            }
    }



    if (params->pipeline_params.log_render)
    {
        SAIGA_OPTIONAL_TIME_MEASURE("log_render", timer_system);
        x = torch::exp2(x);
    }

    if (params->net_params.half_float)
    {
        SAIGA_ASSERT(x.dtype() == torch::kFloat16);
    }


    if (!params->pipeline_params.skip_sensor_model)
    {
        SAIGA_OPTIONAL_TIME_MEASURE("Sensor Model", timer_system);
        x = scene.camera->forward(x, frame_index, uv, scale, fixed_exposure, fixed_white_balance);
    }

    if (params->net_params.half_float)
    {
        SAIGA_ASSERT(x.dtype() == torch::kFloat16);
        x = x.to(torch::kFloat32);
    }

    if (params->points_adding_params.use_grid_loss)
    {
        SAIGA_OPTIONAL_TIME_MEASURE("Hook gradients", timer_system);

        std::function<torch::autograd::Variable(torch::autograd::Variable)> copy_gradient_on_backward(
            [this](torch::autograd::Variable grad)
            {
                //   std::cout << TensorInfo(grad) << std::endl;
                this->render_module->cache->gradient_of_forward_pass_x = grad.clone().cuda();
                //   std::cout << TensorInfo(this->gradient_of_forward_pass_x) << std::endl;
                return grad;
            });
        if (render_module->is_training()) x.register_hook(copy_gradient_on_backward);
    }

    ForwardResult fr;
    fr.x      = x;
    fr.target = target;

    //  fr.x = batch[0]->direction.unsqueeze(0);
    // fr.x = direction;

    std::function<torch::autograd::Variable(torch::autograd::Variable)> timer_grad_for_frx(
        [camera_backwards_timer, loss_backwards_timer](torch::autograd::Variable grad)
        {
            // std::cout << "compute grad for x" << std::endl;
            if (loss_backwards_timer) loss_backwards_timer->Stop();
            if (camera_backwards_timer) camera_backwards_timer->Start();

            return grad;
        });
    if (render_module->is_training()) fr.x.register_hook(timer_grad_for_frx);

    // Eval loss only if required
    torch::Tensor lt_vgg, lt_l1, lt_mse, lt_ssim, lt_lpips;
    if (!params->render_params.viewer_only || loss_statistics)
    {
        SAIGA_OPTIONAL_TIME_MEASURE("Compute Loss", timer_system);

        if (loss_statistics || params->train_params.loss_vgg > 0 || params->train_params.loss_l1 > 0 ||
            params->train_params.loss_mse > 0 || params->train_params.loss_ssim > 0 ||
            params->train_params.loss_lpips > 0)
        {
            SAIGA_OPTIONAL_TIME_MEASURE("Masking", timer_system);

            SAIGA_ASSERT(target.defined());
            fr.loss = torch::zeros({1}, torch::TensorOptions().device(device));
            if (global_mask.defined())
            {
                x      = x * global_mask;
                target = target * global_mask;
            }

            if (local_mask.defined())
            {
                // std::cout << TensorInfo(x) << TensorInfo(local_mask) << TensorInfo(target) << std::endl;
                x      = x * local_mask;
                target = target * local_mask;
            }
        }

        fr.float_loss.count = 1;

        // auto x_crop = x;
        // auto t_crop = target;
        // if (params->train_params.crop_for_loss)
        //{
        //     x_crop = CenterCrop2D(x, 16);
        //     t_crop = CenterCrop2D(target, 16);
        // }
        //  PrintTensorInfo(x_crop);
        //  PrintTensorInfo(t_crop);

        if (loss_statistics ||
            (params->train_params.loss_vgg > 0 && current_epoch > params->train_params.only_start_vgg_after_epochs))
        {
            SAIGA_OPTIONAL_TIME_MEASURE("VGG", timer_system);
            SAIGA_ASSERT(fr.loss.defined());
            // lt_vgg = loss_vgg->forward(x_crop, t_crop);
            lt_vgg = loss_vgg->forward(x, target);
            fr.loss += lt_vgg * params->train_params.loss_vgg;
            fr.float_loss.loss_vgg = lt_vgg.item().toFloat();
        }
        if (loss_statistics || params->train_params.loss_l1 > 0)
        {
            SAIGA_OPTIONAL_TIME_MEASURE("L1", timer_system);

            lt_l1 = torch::l1_loss(x, target);
            fr.loss += lt_l1 * params->train_params.loss_l1;
            fr.float_loss.loss_l1 = lt_l1.item().toFloat();
        }
        if (loss_statistics || params->train_params.loss_mse > 0)
        {
            SAIGA_OPTIONAL_TIME_MEASURE("MSE", timer_system);

            lt_mse = torch::mse_loss(x, target);
            fr.loss += lt_mse * params->train_params.loss_mse;
            fr.float_loss.loss_mse = lt_mse.item().toFloat();
        }
        if (loss_statistics || params->train_params.loss_ssim > 0)
        {
            SAIGA_OPTIONAL_TIME_MEASURE("SSIM", timer_system);

            lt_ssim = (1 - loss_ssim->forward(x, target)) / 2.f;
            fr.loss += lt_ssim * params->train_params.loss_ssim;
            fr.float_loss.loss_ssim = lt_ssim.item().toFloat();
        }
        if (loss_statistics || params->train_params.loss_lpips > 0)
        {
            SAIGA_OPTIONAL_TIME_MEASURE("LPIPS", timer_system);

            lt_lpips = loss_lpips.forward(x, target).sum();
            fr.loss += lt_lpips * params->train_params.loss_lpips;
            fr.float_loss.loss_lpips = lt_lpips.item().toFloat();
        }
        if (loss_statistics)
        {
            auto i                  = x.clamp(0, 1);
            auto t                  = target.clamp(0, 1);
            fr.float_loss.loss_psnr = loss_psnr->forward(i, t).item().toFloat();
        }
    }
    if (!params->render_params.viewer_only)
    {
        SAIGA_OPTIONAL_TIME_MEASURE("Stats and bookkeeping", timer_system);

        if (loss_statistics)
        {
            auto i                  = x.clamp(0, 1);
            auto t                  = target.clamp(0, 1);
            fr.float_loss.loss_psnr = loss_psnr->forward(i, t).item().toFloat();
            // fr.float_loss.loss_lpips = loss_lpips.forward(i, t).item().toFloat();
        }

        if (fr.loss.defined())
        {
            fr.float_loss.loss_float = fr.loss.item().toFloat();

            auto param_loss = params->optimizer_params.response_smoothness * scene.camera->ParamLoss(frame_index);
            fr.float_loss.loss_float_param = param_loss.item().toFloat();
            fr.loss += param_loss;

            if (params->pipeline_params.verbose_eval)
            {
                int index = frame_index.item().toLong();
                std::cout << "frame index " << index << std::endl;
                std::cout << "loss " << fr.float_loss.loss_float << std::endl;
                std::cout << "====" << std::endl;
            }

            if (!std::isfinite(fr.float_loss.loss_float))
            {
                for (auto i : neural_images)
                {
                    PrintTensorInfo(i);
                }
                std::cout << std::endl;
                PrintTensorInfo(uv);
                PrintTensorInfo(fr.loss);
                PrintTensorInfo(x);
                PrintTensorInfo(global_mask);
                PrintTensorInfo(target);
                std::cout << std::endl;
                std::cout << "Scene:" << std::endl;
                scene.Log("debug/");
                Log("debug/");
                throw std::runtime_error("Loss not finite :(");
            }
        }

        //    //4.5 save error image for grid projection
        //    if(render_module->is_training())
        //    {
        //      //  std::cout << TensorInfo(target) << ".. " << TensorInfo(x) << std::endl;
        //        this->render_module->cache->l1_error_image = torch::abs(target-x).clone();
        //    }

        // 5. Convert to image for visualization and debugging
        if (keep_image)
        {
            auto x_full = CenterEmplace(x, torch::ones_like(full_target));

            for (int i = 0; i < batch.size(); ++i)
            {
                fr.outputs.push_back(Saiga::TensorToImage<ucvec3>(x_full[i]));
                fr.targets.push_back(Saiga::TensorToImage<ucvec3>(full_target[i]));

                //            fr.outputs.push_back(Saiga::TensorToImage<ucvec3>(x[i]));
                //            fr.targets.push_back(Saiga::TensorToImage<ucvec3>(target[i]));
                fr.image_ids.push_back(frame_index[i].item().toLong());
            }
        }
    }
    std::function<torch::autograd::Variable(torch::autograd::Variable)> timer_grad_for_frloss(
        [loss_backwards_timer](torch::autograd::Variable grad)
        {
            // std::cout << "compute grad for x" << std::endl;
            if (loss_backwards_timer) loss_backwards_timer->Start();

            return grad;
        });
    if (render_module->is_training()) fr.loss.register_hook(timer_grad_for_frloss);
    return fr;
}
void NeuralPipeline::OptimizerStep(int epoch_id)
{
    if (render_optimizer)
    {
        render_optimizer->step();
        render_optimizer->zero_grad();
    }
    if (refinement_optimizer && epoch_id > params->train_params.lock_dynamic_refinement_epochs)
    {
        refinement_optimizer->step();
        refinement_optimizer->zero_grad();
    }
}

void NeuralPipeline::OptimizerClear(int epoch_id)
{
    if (render_optimizer)
    {
        render_optimizer->zero_grad();
    }
    if (refinement_optimizer)
    {
        refinement_optimizer->zero_grad();
    }
}
void NeuralPipeline::UpdateLearningRate(double factor)
{
    if (render_optimizer)
    {
        UpdateOptimLR(render_optimizer.get(), factor);
    }
    if (refinement_optimizer)
    {
        UpdateOptimLR(refinement_optimizer.get(), factor);
    }
}

void NeuralPipeline::Train(bool train)
{
    render_module = nullptr;
    c10::cuda::CUDACachingAllocator::emptyCache();
    render_module = PointRenderModule(params);

    if (render_optimizer)
    {
        render_optimizer->zero_grad();
    }

    render_module->train(train);
    if (params->optimizer_params.fix_render_network)
    {
        render_network->train(train);
    }
    else
    {
        render_network->train(train);
    }
}
void NeuralPipeline::Log(const std::string& dir) {}
