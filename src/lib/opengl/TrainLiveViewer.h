/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#pragma once
// #undef LIVE_VIEWER
//  #ifndef ADOP_HEADLESS
//  #    define LIVE_VIEWER
//  #endif
#ifdef LIVE_VIEWER

#    include "saiga/core/glfw/all.h"
#    include "saiga/cuda/CudaInfo.h"
#    include "saiga/cuda/imgui_cuda.h"
#    include "saiga/cuda/interop.h"
#    include "saiga/opengl/rendering/deferredRendering/deferredRendering.h"
#    include "saiga/opengl/rendering/empty_renderer.h"
#    include "saiga/opengl/rendering/forwardRendering/forwardRendering.h"
#    include "saiga/opengl/rendering/renderer.h"
#    include "saiga/opengl/window/WindowTemplate.h"
#    include "saiga/opengl/window/glfw_window.h"

#    include "data/Dataset.h"
#    include "models/Pipeline.h"

#    include <thread>


struct LossBufferWithGraph
{
    LossBufferWithGraph(int size_of_buffer, std::string name);
    void build();
    void imgui(float size_of_widget = 50.f);
    void addLossSample(float loss_sample);

    std::vector<float> m_loss_graph;
    int m_loss_graph_samples = 0;

    int size_of_buffer;
    std::string name;
};

class TrainViewer : public Saiga::StandaloneWindow<Saiga::WindowManagement::GLFW, Saiga::EmptyRenderer>,
                    public Saiga::glfw_KeyListener,
                    public Saiga::glfw_ResizeListener,
                    public Saiga::glfw_MouseListener,
                    public Saiga::glfw_ExternalDropListener

{
   public:
    TrainViewer(const Saiga::WindowParameters& windowParameters, const Saiga::OpenGLParameters& openglParameters,
                const RenderingParameters& rendererParameters);


    void newFrameArrived(float dt = 0.01);

    virtual void update(float dt) override;
    virtual void render(Saiga::RenderInfo render_info) override;
    virtual void keyPressed(int key, int scancode, int mods) override;
    virtual void interpolate(float dt, float interpolation) override;


    // this call computes the view during training
    void checkAndUpdateTexture(std::shared_ptr<NeuralPipeline> pipeline, std::shared_ptr<NeuralScene> neural_scene,
                               int epoch_id);


    ImageInfo CurrentFrameData(std::shared_ptr<NeuralScene>& neural_scene, int frame_idx);

    void imgui();
    void setDataloaderMaxImages(int max) { max_images = max; }

    void addLossSample(float val);
    void addEpochLoss(float val);

    void setupCamera(std::shared_ptr<SceneData>& scene, int frame_index = 0);
    bool checkForRestart();

    void addValLoss(LossResult loss);

   private:
    int view_num = 0;
    // interval in ms
    int update_interval = 250;
    std::shared_ptr<Texture> output_texture_render;
    int max_images                = 1;
    const int size_of_loss_buffer = 10000;

    LossBufferWithGraph loss_batches;
    LossBufferWithGraph loss_epochs;


    LossBufferWithGraph val_img_loss_lpips;
    LossBufferWithGraph val_img_loss_ssim;
    LossBufferWithGraph val_img_loss_vgg;
    LossBufferWithGraph val_img_loss_l1;
    LossBufferWithGraph val_img_loss_l2;
    LossBufferWithGraph val_img_loss_psnr;
    int sample_this_epoch      = 0;
    int loss_samples_per_epoch = 1;

    bool restart_training = false;

    bool use_custom_camera = false;

    bool mouse_on_view             = false;
    IntrinsicsModule rt_intrinsics = nullptr;
    PoseModule rt_extrinsics       = nullptr;
    Glfw_Camera<PerspectiveCamera> scene_camera;
    std::shared_ptr<SceneData> last_scene;



    // std::shared_ptr<LineVertexColoredAsset> spline_mesh;
    // SplinePath camera_spline;

    // ViewMode view_mode;
    // std::shared_ptr<DirectionalLight> sun;
    // TextureDisplay display;

    // std::unique_ptr<Framebuffer> target_framebuffer;

    // video recording variables
    // std::string recording_dir;
};

#endif