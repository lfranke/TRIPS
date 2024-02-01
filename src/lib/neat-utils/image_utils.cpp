/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "image_utils.h"

TemplatedImage<float> UndistortImage(ImageView<float> img, IntrinsicsPinholef K, Distortionf D,
                                     IntrinsicsPinholef target_K, ivec2 sizes_image, float scene_scale,
                                     unsigned int border_pixels)
{
    TemplatedImage<float> cp(sizes_image.y(), sizes_image.x());
    cp.create(sizes_image.y(), sizes_image.x());
    cp.makeZero();
#pragma omp parallel for
    for (int i_y = 0; i_y < cp.h; ++i_y)
    {
        for (int j_x = 0; j_x < cp.w; ++j_x)
        {
            vec2 p(j_x, i_y);

            p = target_K.unproject2(p);
            p = distortNormalizedPoint(p, D);
            p = K.normalizedToImage(p);

            p *= scene_scale;
            // if (p.y() < cp.h && p.y() >= 0 && p.x() < cp.w && p.x() >= 0)
            //   if(img.inImage(ivec2(p.x(),p.y())))
            // leave edges black
            if (p.y() >= border_pixels && p.y() < (img.height - border_pixels) && p.x() >= border_pixels &&
                p.x() <= (img.w - border_pixels))
            {
                cp(i_y, j_x) = img.inter(p.y(), p.x());
            }
        }
    }

    return cp;
}

template <typename T>
TemplatedImage<T> UndistortOCAMImage(ImageView<T> img, IntrinsicsPinholef K, IntrinsicsPinholef targetK,
                                     OCam<double> ocam, vec2 size_of_ocam_target_image, float scene_scale,
                                     unsigned int border_pixels)
{
    vec2 sizes_img = size_of_ocam_target_image;
    TemplatedImage<T> cp(sizes_img.y(), sizes_img.x());
    cp.create(sizes_img.y(), sizes_img.x());
    cp.makeZero();
    auto oc = ocam.cast<float>();

#pragma omp parallel for
    for (int i_y = 0; i_y < cp.h; ++i_y)
    {
        for (int j_x = 0; j_x < cp.w; ++j_x)
        {
            // vec3 px(j_x, i_y, 1);
            vec2 px = targetK.unproject2(vec2(j_x, i_y));

            // px = targetK.unproject(vec2(px.x(), px.y()), px.z());

            vec3 p_s = vec3(px.x(), px.y(), 1);
            vec2 p   = oc.Project(p_s);
            p *= scene_scale;

            // leave edges black
            if (p.y() >= border_pixels && p.y() < (img.height - border_pixels) && p.x() >= border_pixels &&
                p.x() <= (img.w - border_pixels))
            {
                cp(i_y, j_x) = img.inter(p.y(), p.x());
            }
        }
    }

    return cp;
}

template TemplatedImage<vec3> UndistortOCAMImage(ImageView<vec3> img, IntrinsicsPinholef K, IntrinsicsPinholef targetK,
                                                 OCam<double> ocam, vec2 size_of_ocam_target_image, float scene_scale,
                                                 unsigned int border_pixels);

template TemplatedImage<float> UndistortOCAMImage(ImageView<float> img, IntrinsicsPinholef K,
                                                  IntrinsicsPinholef targetK, OCam<double> ocam,
                                                  vec2 size_of_ocam_target_image, float scene_scale,
                                                  unsigned int border_pixels);

template TemplatedImage<ucvec3> UndistortOCAMImage(ImageView<ucvec3> img, IntrinsicsPinholef K,
                                                   IntrinsicsPinholef targetK, OCam<double> ocam,
                                                   vec2 size_of_ocam_target_image, float scene_scale,
                                                   unsigned int border_pixels);

torch::Tensor process_l1_image(torch::Tensor l1_img)
{
    l1_img *= 1.3;
    l1_img -= 0.3;
    l1_img.clamp_(0, 1);
    return l1_img;
}
torch::Tensor process_ssim_image(torch::Tensor ssim_map)
{
    ssim_map *= 1.3;
    ssim_map -= 0.3;
    ssim_map.clamp_(0, 1);
    return ssim_map;
}

torch::Tensor process_l2_image(torch::Tensor l2_map)
{
    const float fac = 0.001;
    l2_map *= (1.f + fac);
    l2_map -= fac;
    l2_map.clamp_(0, 1);
    return l2_map;
}

void write16bitImg(TemplatedImage<float> img, std::string path)
{
    TemplatedImage<unsigned short> img_16b(img.h, img.w);
    img_16b.create(img.h, img.w);
    img_16b.makeZero();
    for (int y = 0; y < img.height; ++y)
    {
        for (int x = 0; x < img.width; ++x)
        {
            img_16b(y, x) = (unsigned short)(clamp(img(y, x) * 65535.f, 0, 65535.f));
        }
    }
    img_16b.save(path);
}