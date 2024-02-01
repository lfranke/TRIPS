/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/core/util/file.h"
#include "saiga/cuda/imgui_cuda.h"
#include "saiga/vision/torch/ImageSimilarity.h"
#include "saiga/vision/torch/TorchHelper.h"
#include "saiga/vision/torch/VGGLoss.h"

#include "data/NeuralScene.h"
#include "models/MyAdam.h"
#include "models/Networks.h"
#include "models/NeuralRefinement.h"
#include "models/NeuralTexture.h"
#include "models/Pipeline.h"
#include "models/mlp.h"
#include "rendering/PointRenderer.h"
#include "rendering/RenderModule.h"
using namespace Saiga;

struct LossResult
{
    float loss_vgg   = 0;
    float loss_l1    = 0;
    float loss_mse   = 0;
    float loss_psnr  = 0;
    float loss_ssim  = 0;
    float loss_lpips = 0;

    float loss_float       = 0;
    float loss_float_param = 0;

    int count = 0;

    LossResult& operator+=(const LossResult& other)
    {
        loss_vgg += other.loss_vgg;
        loss_l1 += other.loss_l1;
        loss_mse += other.loss_mse;
        loss_psnr += other.loss_psnr;
        loss_ssim += other.loss_ssim;
        loss_lpips += other.loss_lpips;
        loss_float += other.loss_float;
        loss_float_param += other.loss_float_param;
        count += other.count;
        return *this;
    }

    LossResult& operator/=(float value)
    {
        loss_vgg /= value;
        loss_l1 /= value;
        loss_mse /= value;
        loss_psnr /= value;
        loss_ssim /= value;
        loss_lpips /= value;
        loss_float /= value;
        loss_float_param /= value;
        return *this;
    }

    LossResult Average()
    {
        LossResult cpy = *this;
        cpy /= count;
        return cpy;
    }

    void AppendToFile(const std::string& file, int epoch)
    {
        std::ofstream strm(file, std::ios_base::app);

        Table tab({10, 15, 15, 15, 15, 15, 15, 15}, strm, ',');
        if (epoch == 0)
        {
            tab << "ep"
                << "vgg"
                << "lpips"
                << "l1"
                << "psrn"
                << "ssim"
                << "param"
                << "count";
        }
        tab << epoch << loss_vgg << loss_lpips << loss_l1 << loss_psnr << loss_ssim << loss_float_param << count;
    }

    void Print()
    {
        console << "Param " << loss_float_param << " VGG " << loss_vgg << " L1 " << loss_l1 << " MSE " << loss_mse
                << " PSNR " << loss_psnr << " SSIM " << loss_ssim << " LPIPS " << loss_lpips << " count " << count
                << std::endl;
    }
};

struct ForwardResult
{
    std::vector<TemplatedImage<ucvec3>> outputs;
    std::vector<TemplatedImage<ucvec3>> targets;
    std::vector<int> image_ids;

    torch::Tensor x;
    torch::Tensor loss;
    torch::Tensor target;

    LossResult float_loss;
};


class RenderNet
{
   public:
    RenderNet() {}
    //    RenderNet(MultiScaleUnet2dParams params): multiScaleUnet2d(std::make_shared<MultiScaleUnet2d>(params)){}
    virtual void to(torch::DeviceType d) = 0;
    virtual void to(torch::ScalarType d) = 0;

    virtual void eval()        = 0;
    virtual void train()       = 0;
    virtual void train(bool t) = 0;
    virtual bool valid()       = 0;

    virtual void save(std::string path) = 0;
    virtual void load(std::string path) = 0;

    virtual std::vector<torch::Tensor> parameters() = 0;

    virtual at::Tensor forward(ArrayView<torch::Tensor> inputs)                                 = 0;
    virtual at::Tensor forward(ArrayView<torch::Tensor> inputs, ArrayView<torch::Tensor> masks) = 0;
};

template <class T>
class DerivedRenderNet : public RenderNet
{
   public:
    std::shared_ptr<T> network = nullptr;
    DerivedRenderNet(MultiScaleUnet2dParams params) : network(std::make_shared<T>(params)) {}

    virtual void to(torch::DeviceType d) { (*network)->to(d); }
    virtual void to(torch::ScalarType d) { (*network)->to(d); }
    virtual void eval() { (*network)->eval(); }
    virtual void train() { (*network)->train(); }
    virtual void train(bool t) { (*network)->train(t); }
    virtual bool valid() { return network != nullptr; };

    virtual std::vector<torch::Tensor> parameters() { return (*network)->parameters(); };



    virtual at::Tensor forward(ArrayView<torch::Tensor> inputs) { return (*network)->forward(inputs); }
    virtual at::Tensor forward(ArrayView<torch::Tensor> inputs, ArrayView<torch::Tensor> masks)
    {
        return (*network)->forward(inputs, masks);
    }

    virtual void save(std::string path) { torch::save(*network, path); }

    virtual void load(std::string path)
    {
        if (*network && std::filesystem::exists(path))
        {
            std::cout << "Load Checkpoint render" << std::endl;
            torch::load(*network, path);
        }
    }
};

using RenderNetwork       = RenderNet;
using RenderNetworkParams = MultiScaleUnet2dParams;

class NeuralPipeline
{
   public:
    NeuralPipeline(std::shared_ptr<CombinedParams> params);

    void Train(bool train);

    void SaveCheckpoint(const std::string& dir)
    {
        render_network->save(dir + "/render_net.pth");
        if (dynamic_refinement_module)
        {
            torch::save(dynamic_refinement_module, dir + "/dynamic_refinement_module.pth");
        }
    }
    void LoadCheckpoint(const std::string& dir)
    {
        render_network->load(dir + "/render_net.pth");
        if (dynamic_refinement_module && std::filesystem::exists(dir + "/dynamic_refinement_module.pth"))
        {
            torch::load(dynamic_refinement_module, dir + "/dynamic_refinement_module.pth");
        }
    }

    // void SaveCheckpoint(const std::string& dir) { torch::save(render_network, dir + "/render_net.pth"); }
    // void LoadCheckpoint(const std::string& dir)
    // {
    //     if (render_network.valid() && std::filesystem::exists(dir + "/render_net.pth"))
    //     {
    //         std::cout << "Load Checkpoint render" << std::endl;
    //         torch::load(render_network, dir + "/render_net.pth");
    //     }
    // }

    void Log(const std::string& dir);

    void OptimizerStep(int epoch_id);
    void OptimizerClear(int epoch_id);
    void UpdateLearningRate(double factor);

    ForwardResult Forward(NeuralScene& scene, std::vector<NeuralTrainData>& batch, torch::Tensor global_mask,
                          bool loss_statistics, int current_epoch = -1, bool keep_image = false,
                          float fixed_exposure     = std::numeric_limits<float>::infinity(),
                          vec3 fixed_white_balance = vec3(std::numeric_limits<float>::infinity(), 0, 0));

    std::shared_ptr<RenderNetwork> render_network = nullptr;
    torch::DeviceType device                      = torch::kCUDA;

    // RefinementNet refinement_module = nullptr;
    DynamicRefinementMLP dynamic_refinement_module = nullptr;

    PointRenderModule render_module = nullptr;
    std::shared_ptr<CombinedParams> params;
    CUDA::CudaTimerSystem* timer_system = nullptr;

    std::shared_ptr<torch::optim::Optimizer> render_optimizer;
    std::shared_ptr<torch::optim::Optimizer> refinement_optimizer;

    // Loss stuff
    std::shared_ptr<PretrainedVGG19Loss> loss_vgg = nullptr;
    //    std::shared_ptr<PretrainedVGGLpipsLoss> loss_vgg = nullptr;
    PSNR loss_psnr   = PSNR(0, 1);
    LPIPS loss_lpips = LPIPS("loss/traced_lpips.pt");
    SSIM loss_ssim   = SSIM();
};