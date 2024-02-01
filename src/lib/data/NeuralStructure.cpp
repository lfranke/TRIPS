/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "NeuralStructure.h"

#include "rendering/PointRenderer.h"


PoseModuleImpl::PoseModuleImpl(std::shared_ptr<SceneData> scene)
{
    SAIGA_ASSERT(!scene->frames.empty());

    tangent_poses = torch::zeros({(long)scene->frames.size(), 6L},
                                 torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA))
                        .clone();
    tangent_poses = tangent_poses.set_requires_grad(true);

    std::vector<Sophus::SE3d> pose_data;
    for (auto& f : scene->frames)
    {
        pose_data.push_back(f.pose.inverse());
    }
    static_assert(sizeof(Sophus::SE3d) == sizeof(double) * 8);

    poses_se3 =
        torch::from_blob(pose_data.data(), {(long)pose_data.size(), 8L}, torch::TensorOptions().dtype(torch::kFloat64))
            .clone()
            .cuda();

    // The last element is just padding
    poses_se3.slice(1, 7, 8).zero_();
    register_parameter("tangent_poses", tangent_poses);
    register_buffer("poses_se3", poses_se3);
}

PoseModuleImpl::PoseModuleImpl(Sophus::SE3d pose)
{
    pose      = pose.inverse();
    poses_se3 = torch::from_blob(pose.data(), {1L, 8L}, torch::TensorOptions().dtype(torch::kFloat64)).clone().cuda();
    tangent_poses = torch::zeros({1L, 6L}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA)).clone();
    register_buffer("poses_se3", poses_se3);
    register_parameter("tangent_poses", tangent_poses);
}

void PoseModuleImpl::ApplyTangent()
{
    ApplyTangentToPose(tangent_poses, poses_se3);
}
std::vector<Sophus::SE3d> PoseModuleImpl::Download()
{
    torch::Tensor cp = poses_se3.contiguous().cpu();

    std::vector<Sophus::SE3d> result(cp.size(0));

    memcpy(result[0].data(), cp.data_ptr(), sizeof(Sophus::SE3d) * result.size());

    return result;
}

Sophus::SE3d PoseModuleImpl::Download(int index_pose)
{
    torch::Tensor cp = poses_se3.slice(0, index_pose, index_pose + 1).contiguous().cpu();

    std::vector<Sophus::SE3d> result(1);

    memcpy(result[0].data(), cp.data_ptr(), sizeof(Sophus::SE3d));

    return result[0];
}
void PoseModuleImpl::SetPose(int id, Sophus::SE3d pose)
{
    pose = pose.inverse();
    torch::NoGradGuard ngg;
    auto new_pose =
        torch::from_blob(pose.data(), {1L, 8L}, torch::TensorOptions().dtype(torch::kFloat64)).clone().cuda();
    poses_se3.slice(0, id, id + 1) = new_pose;
}

IntrinsicsModuleImpl::IntrinsicsModuleImpl(std::shared_ptr<SceneData> scene)
{
    SAIGA_ASSERT(!scene->scene_cameras.empty());

    bool only_pinhole       = true;
    int size_of_data_buffer = 13;
    for (auto& cam : scene->scene_cameras)
    {
        if (cam.camera_model_type != CameraModel::PINHOLE_DISTORTION)
        {
            only_pinhole        = false;
            size_of_data_buffer = max(size_of_data_buffer, cam.ocam.NumProjectParams());
        }
    }

    std::vector<float> intrinsics_data(size_of_data_buffer * scene->scene_cameras.size(), 0);


    for (int i = 0; i < scene->scene_cameras.size(); ++i)
    {
        auto& cam  = scene->scene_cameras[i];
        float* ptr = intrinsics_data.data() + (size_of_data_buffer * i);

        if (cam.camera_model_type == CameraModel::PINHOLE_DISTORTION)
        {
            vec5 a = cam.K.coeffs();
            vec8 b = cam.distortion.Coeffs();

            Eigen::Matrix<float, 13, 1> data;

            data.head<5>() = a;
            data.tail<8>() = b;

            for (int j = 0; j < data.size(); ++j)
            {
                ptr[j] = data[j];
            }
            std::cout << "Pinhole Intrinsics: " << cam.K << " -- " << cam.distortion << std::endl;
            //   intrinsics_data.push_back(data);
        }
        else if (cam.camera_model_type == CameraModel::OCAM)
        {
            auto c = scene->scene_cameras[i].ocam;

            auto params = c.ProjectParams();

            for (int j = 0; j < params.size(); ++j)
            {
                ptr[j] = params[j];
            }
        }
        else
        {
            SAIGA_EXIT_ERROR("unknown camera model");
        }
    }
    intrinsics = torch::from_blob(intrinsics_data.data(), {(long)scene->scene_cameras.size(), size_of_data_buffer},
                                  torch::TensorOptions().dtype(torch::kFloat32))
                     .clone()
                     .cuda();
    PrintTensorInfo(intrinsics);
    register_parameter("intrinsics", intrinsics);
    /*

        if (scene->dataset_params.camera_model == CameraModel::PINHOLE_DISTORTION)
        {
            std::vector<Eigen::Matrix<float, 13, 1>> intrinsics_data;


            for (auto& cam : scene->scene_cameras)
            {
                vec5 a = cam.K.coeffs();
                vec8 b = cam.distortion.Coeffs();


                Eigen::Matrix<float, 13, 1> data;

                data.head<5>() = a;
                data.tail<8>() = b;

                intrinsics_data.push_back(data);
            }



            intrinsics = torch::from_blob(intrinsics_data.data(), {(long)intrinsics_data.size(), 13L},
                                          torch::TensorOptions().dtype(torch::kFloat32))
                             .clone()
                             .cuda();


            register_parameter("intrinsics", intrinsics);
            std::cout << "Pinhole Intrinsics:" << std::endl;
            PrintTensorInfo(intrinsics);
        }
        else if (scene->dataset_params.camera_model == CameraModel::OCAM)
        {
            std::cout << "ocam Intrinsics:" << std::endl;

            long count = scene->scene_cameras.front().ocam.NumProjectParams();

            std::vector<float> intrinsics_data(count * scene->scene_cameras.size());

            for (int i = 0; i < scene->scene_cameras.size(); ++i)
            {
                auto c = scene->scene_cameras[i].ocam;
                SAIGA_ASSERT(c.NumProjectParams() == count);

                float* ptr = intrinsics_data.data() + (count * i);

                auto params = c.ProjectParams();

                for (int j = 0; j < params.size(); ++j)
                {
                    ptr[j] = params[j];
                }
            }

            intrinsics = torch::from_blob(intrinsics_data.data(), {(long)scene->scene_cameras.size(), count},
                                          torch::TensorOptions().dtype(torch::kFloat32))
                             .clone()
                             .cuda();
            PrintTensorInfo(intrinsics);
            register_parameter("intrinsics", intrinsics);
        }
        else
        {
            SAIGA_EXIT_ERROR("unknown camera model");
        }
        */
}

IntrinsicsModuleImpl::IntrinsicsModuleImpl(IntrinsicsPinholef K)
{
    Eigen::Matrix<float, 13, 1> data;
    data.setZero();
    data.head<5>() = K.coeffs();
    intrinsics = torch::from_blob(data.data(), {1L, 13L}, torch::TensorOptions().dtype(torch::kFloat32)).clone().cuda();

    register_parameter("intrinsics", intrinsics);
}

std::vector<IntrinsicsPinholef> IntrinsicsModuleImpl::DownloadK()
{
    torch::Tensor cp = intrinsics.clone().slice(1, 0, 5).contiguous().cpu();
    std::vector<Eigen::Matrix<float, 5, 1>> intrinsics_data(cp.size(0));
    memcpy(intrinsics_data[0].data(), cp.data_ptr(), sizeof(Eigen::Matrix<float, 5, 1>) * intrinsics_data.size());


    std::vector<IntrinsicsPinholef> result;
    for (auto v : intrinsics_data)
    {
        IntrinsicsPinholef f = v.head<5>().eval();
        result.push_back(f);
    }
    return result;
}

std::vector<Distortionf> IntrinsicsModuleImpl::DownloadDistortion()
{
    torch::Tensor cp = intrinsics.contiguous().cpu();
    std::vector<Eigen::Matrix<float, 13, 1>> intrinsics_data(cp.size(0));
    memcpy(intrinsics_data[0].data(), cp.data_ptr(), sizeof(Eigen::Matrix<float, 13, 1>) * intrinsics_data.size());


    std::vector<Distortionf> result;
    for (auto v : intrinsics_data)
    {
        Distortionf f = v.tail<8>().eval();
        result.push_back(f);
    }

    return result;
}

/*
std::vector<Distortionf> IntrinsicsModuleImpl::DownloadOcam(int width, int height)
{
    torch::Tensor cp = intrinsics.contiguous().cpu();
    std::vector<std::vector<float>> intrinsics_data(cp.size(0));
    for(auto& v: intrinsics_data){
        v.resize(cp.size(1));
    }
    memcpy(intrinsics_data[0].data(), cp.data_ptr(), sizeof(float) * cp.size(1) * intrinsics_data.size());


    std::vector<OCam<float>> result;
    for (auto v : intrinsics_data)
    {
        OCam<float> f = OCam<float>(width,height, vec5(v[0],v[1],v[2],v[3],v[4]), std::vector<float>(v.begin()+5,
v.end())); result.push_back(f);
    }

    return result;
}*/

void IntrinsicsModuleImpl::SetPinholeIntrinsics(int id, IntrinsicsPinholef K, Distortionf dis)
{
    Eigen::Matrix<float, 13, 1> intrinsics_data;

    intrinsics_data.head<5>() = K.coeffs();
    intrinsics_data.tail<8>() = dis.Coeffs();


    auto new_intr = torch::from_blob(intrinsics_data.data(), {1L, 13L}, torch::TensorOptions().dtype(torch::kFloat32))
                        .clone()
                        .cuda();

    torch::NoGradGuard ngg;
    // intrinsics.slice(0, id, id + 1) = new_intr;
    intrinsics = new_intr;
}
void IntrinsicsModuleImpl::AddToOpticalCenter(float x)
{
    // add to cx and cy, mostly for edge case handling
    intrinsics.slice(1, 2, 4) += x;
}
void IntrinsicsModuleImpl::SetOcamIntrinsics(int id, OCam<float> ocam)
{
    // Eigen::Matrix<float, 13, 1> intrinsics_data;
    long count = ocam.NumProjectParams();

    std::vector<float> intrinsics_data(count);
    {
        auto params = ocam.ProjectParams();
        for (int i = 0; i < params.size(); ++i)
        {
            intrinsics_data[i] = params[i];
        }
    }

    auto new_intr =
        torch::from_blob(intrinsics_data.data(), {(long)1, count}, torch::TensorOptions().dtype(torch::kFloat32))
            .clone()
            .cuda();

    torch::NoGradGuard ngg;
    // intrinsics.slice(0, id, id + 1) = new_intr;
    intrinsics = new_intr;
    //  PrintTensorInfo(intrinsics.slice(0, id, id + 1) );
}
