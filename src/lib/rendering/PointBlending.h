/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#pragma once
#include "saiga/core/math/Types.h"

#include "config.h"
using namespace Saiga;



#ifdef __CUDACC__
__inline__ __device__ float compute_Ta(float alpha, float alpha_accumulated, float* J_alpha = nullptr)
{
    if (alpha_accumulated + alpha < 1.f)
    {
        if (J_alpha) *J_alpha = 1.f;
        return alpha;
    }

    // clamp alpha to rest accumulated
    float alpha_remainder = __saturatef(1.f - alpha_accumulated);
    if (J_alpha)
    {
        if (alpha_accumulated >= 1.f)
        {
            *J_alpha = 0.f;
        }
        else
        {
            float remainder = 1.f - alpha_accumulated;
            *J_alpha        = 0.f;
        }
    }

    return alpha_remainder;
}

inline __device__ float compute_point_size_fac_d(float point_size_opt, int layer, int max_layers, float* grad = nullptr)
{
    int l_lower = 0;
    int l_upper = 0;
    if (point_size_opt > 1)
    {
        l_lower = float(min(max(0, int(floor(log2f(point_size_opt)))), max_layers - 1));
        l_upper = float(min(max(0, int(ceil(log2f(point_size_opt)))), max_layers - 1));
    }

    if (layer < l_lower)
    {
        if (grad) *grad = 0.f;
        return 1.f;
    }

    float s_i = pow(2, layer);

    if (l_upper == 0)
    {
        const float cutoff_value = 0.25f;
        if (grad) *grad = (1 - cutoff_value);
        return (1 - cutoff_value) * (point_size_opt) + 0.25;
    }
    else if (l_lower == l_upper)
    {
        if (grad) *grad = 0.f;
        return 1;
    }
    if (grad)
    {
        float J = -1.f / (float(1 << l_upper) - float(1 << l_lower));
        if (layer == l_upper) J = -J;
        *grad = J;
    }
    return 1.f - abs((point_size_opt) - float(s_i)) / (float(1 << l_upper) - float(1 << l_lower));
    // return 1.f - abs(log2f(point_size_opt) - float(layer));  //  new
    // return 1.f - abs((point_size_opt) - float(s_i));  // broken
}

inline __device__ float compute_point_size_fac(float point_size_opt, int layer, int max_layers, float* grad = nullptr)
{
    float log_ps     = log2f(point_size_opt);
    int layer_lower  = 0;
    int layer_higher = 0;
    if (point_size_opt > 1)
    {
        layer_lower  = float(min(max(0, int(floor(log_ps))), max_layers - 1));
        layer_higher = float(min(max(0, int(ceil(log_ps))), max_layers - 1));
    }

    if (layer < layer_lower)
    {
        if (grad) *grad = 0.f;
        return 1.f;
    }

    float layer_factor = 1.f;
    if (layer_higher == 0)
    {
        // small point
        // cutoff value: MAX_ELEMENTS alpha-blended: 16: ~0.25 -> alpha_dest=0.99
        float cutoff_value = 0.25f;
        //// similar to elu, but with cutoff
        /////!!!! other
        layer_factor = (1 - cutoff_value) * exp(point_size_opt - 1) + 0.25;
        if (grad) *grad = 0.27591f * point_size_opt;

        // const float cutoff_value = 0.25f;
        // if (grad) *grad = (1 - cutoff_value);
        // return (1 - cutoff_value) * (point_size_opt) + 0.25;
    }
    else if (layer_lower == layer_higher)
    {
        // ceil and floor equal, directly on point -- should be failure case, just assume linear grad
        if (grad) *grad = 1.f;
    }
    else
    {
        // normal interpolation case:
        // distance factor is (x - x_lower) / (x_higher-x_lower)
        // if layer == higher : distance_fac
        // if layer == lower : 1- distance_fac (typical bilinear interpolation)
        // float l_l = powf(2, layer_lower);
        // float l_h = powf(2, layer_higher);
        float l_l = float(1 << layer_lower);
        float l_h = float(1 << layer_higher);

        layer_factor = (point_size_opt - l_l) / (l_h - l_l);
        if (layer == layer_lower) layer_factor = 1.f - layer_factor;


        if (grad)
        {
            float J = 1.f / (l_h - l_l);
            if (layer == layer_lower) J = -J;
            *grad = J;
        }
        if (layer_lower == max_layers - 1)
        {
            // point larger than layers
            layer_factor = 1.f;
            // TODO still compute gradient as usual
            if (grad) *grad = 1;
        }
    }

    return layer_factor;
}

#endif


inline
#ifdef __CUDACC__
    __device__
#endif
    float
    compute_lfac(float plv, float layer, float* grad)
{
    float div = plv - layer;
    if (div <= 0.f)
    {
        *grad = 0.f;
        return 0.f;
    }
    else if (div >= 1.f)
    {
        *grad = 1.f / (plv * plv);
        return 1.f / plv;
    }
    if (layer == 0)
    {
        *grad = 1;
        return div;
    }
    *grad = div / (plv * plv);

    return div / plv;
}

inline
#ifdef __CUDACC__
    __device__
#endif
    float
    compute_lfac_nonorm(float plv, float layer, float* grad = nullptr)
{
    float div = plv - layer;
#if defined(__CUDACC__)
    float result = __saturatef(div);
#else
    float result = std::clamp(div, 0.f, 1.f);

#endif
    if (div <= 0.f)
    {
        if (grad) *grad = 0.f;
        return 0.f;
    }
    else if (div >= 1.f)
    {
        if (grad) *grad = 0.f;
        return 1.f;
    }
    if (grad) *grad = 1.f;

    return result;
}

inline HD float compute_blending_fac_wo_index(vec2 uv_pos, ivec2 g)
{
    return (1.f - fabsf(uv_pos.x() - g.x())) * (1 - fabsf(uv_pos.y() - g.y()));
}

inline HD vec4 compute_blending_fac(vec2 uv_pos, Matrix<float, 4, 2>* J_uv = nullptr)
{
    // derivative is 1 in relevant area
    // vec2 subpixel_pos = uv_pos - ivec2(__float2int_rd(uv_pos(0)), __float2int_rd(uv_pos(1)));
    vec2 subpixel_pos = uv_pos - uv_pos.array().floor();

    /*
     *  2        3
     *  _________
     *  |       |
     *  |  *    |
     *  |_______|
     *  0       1
     */

    vec4 blend_vec;
    blend_vec.setZero();
    blend_vec(0) = (1.f - subpixel_pos.x()) * (1.f - subpixel_pos.y());
    blend_vec(1) = subpixel_pos.x() * (1.f - subpixel_pos.y());
    blend_vec(2) = (1.f - subpixel_pos.x()) * subpixel_pos.y();
    blend_vec(3) = subpixel_pos.x() * subpixel_pos.y();

    if (J_uv)
    {
        auto& J = *J_uv;
        J.setZero();
        J(0, 0) = subpixel_pos.y() - 1;
        J(0, 1) = subpixel_pos.x() - 1;

        J(1, 0) = 1 - subpixel_pos.y();
        J(1, 1) = -subpixel_pos.x();


        J(2, 0) = -subpixel_pos.y();
        J(2, 1) = 1 - subpixel_pos.x();

        J(3, 0) = subpixel_pos.y();
        J(3, 1) = subpixel_pos.x();
    }

    return blend_vec;
}

inline HD int blend_fac_index(vec2 uv_pos, vec2 gid)
{
    int x_i = gid.x() < uv_pos.x() ? 0 : 1;
    int y_i = gid.y() < uv_pos.y() ? 0 : 2;
    return y_i + x_i;
}


// #define CHANNELS 1
template <typename desc_vec, int size_of_desc_vec>
inline HD desc_vec compute_blend_vec(float alpha_dest, float alpha_s, desc_vec color, desc_vec color_dest,
                                     Saiga::Matrix<double, size_of_desc_vec, 1>* J_alphasource              = nullptr,
                                     Saiga::Matrix<double, size_of_desc_vec, size_of_desc_vec>* J_color     = nullptr,
                                     Saiga::Matrix<double, size_of_desc_vec, 1>* J_alphadest                = nullptr,
                                     Saiga::Matrix<double, size_of_desc_vec, size_of_desc_vec>* J_colordest = nullptr)
{
    desc_vec blended_col = alpha_dest * alpha_s * color + color_dest;

    if (J_alphadest)
    {
        auto& J = *J_alphadest;
        for (int i = 0; i < size_of_desc_vec; ++i)
        {
            J(i, 0) = alpha_s * color[i];
        }
    }
    if (J_alphasource)
    {
        auto& J = *J_alphasource;
        for (int i = 0; i < size_of_desc_vec; ++i)
        {
            J(i, 0) = alpha_dest * color[i];
        }
    }
    if (J_color)
    {
        auto& J = *J_color;
        for (int i = 0; i < size_of_desc_vec; ++i)
        {
            J(i, i) = alpha_dest * alpha_s;
        }
    }
    if (J_colordest)
    {
        auto& J = *J_colordest;
        for (int i = 0; i < size_of_desc_vec; ++i) J(i, i) = 1;
    }

    return blended_col;
}


template <typename T>
inline HD float compute_blend(float alpha_dest, float alpha_s, T color, T color_dest, float* J_alphasource = nullptr,
                              float* J_color = nullptr, float* J_alphadest = nullptr, float* J_colordest = nullptr)
{
    float blended_col = alpha_dest * alpha_s * color + color_dest;

    if (J_alphadest)
    {
        *J_alphadest = alpha_s * color;
    }
    if (J_alphasource)
    {
        *J_alphasource = alpha_dest * color;
    }
    if (J_color)
    {
        *J_color = alpha_dest * alpha_s;
    }
    if (J_colordest)
    {
        *J_colordest = 1;
    }

    return blended_col;
}



inline HD float compute_new_alphadest(float alphadest_old, float alpha_s, float* J_alphasource = nullptr,
                                      float* J_alphadest_old = nullptr)
{
    float new_alphadest = (1 - alpha_s) * alphadest_old;
    if (J_alphadest_old)
    {
        *J_alphadest_old = (1 - alpha_s);
    }
    if (J_alphasource)
    {
        *J_alphasource = -alphadest_old;
    }
    return new_alphadest;
}



#define CHANNELS 1
inline HD float compute_blend_d(float alpha_dest, float alpha_s, float color, float color_dest,
                                Saiga::Matrix<double, 1, CHANNELS>* J_alphasource = nullptr,
                                Saiga::Matrix<double, 1, CHANNELS>* J_color       = nullptr,
                                Saiga::Matrix<double, 1, CHANNELS>* J_alphadest   = nullptr,
                                Saiga::Matrix<double, 1, CHANNELS>* J_colordest   = nullptr)
{
    float blended_col = alpha_dest * alpha_s * color + color_dest;

    if (J_alphadest)
    {
        auto& J = *J_alphadest;
        J(0, 0) = alpha_s * color;
    }
    if (J_alphasource)
    {
        auto& J = *J_alphasource;
        J(0, 0) = alpha_dest * color;
    }
    if (J_color)
    {
        auto& J = *J_color;
        J(0, 0) = alpha_dest * alpha_s;
    }
    if (J_colordest)
    {
        auto& J = *J_colordest;
        J(0, 0) = 1;
    }

    return blended_col;
}

inline HD float compute_new_alphadest_d(float alphadest_old, float alpha_s,
                                        Saiga::Matrix<double, 1, 1>* J_alphasource   = nullptr,
                                        Saiga::Matrix<double, 1, 1>* J_alphadest_old = nullptr)
{
    float new_alphadest = (1 - alpha_s) * alphadest_old;
    if (J_alphadest_old)
    {
        auto& J = *J_alphadest_old;
        J(0, 0) = (1 - alpha_s);
    }
    if (J_alphasource)
    {
        auto& J = *J_alphasource;
        J(0, 0) = -alphadest_old;
    }
    return new_alphadest;
}


template <typename desc_vec, int size_of_desc_vec>
inline HD float normalize_by_alphadest(float color, float alphadest, Saiga::Matrix<double, 1, 1>* J_alphadest = nullptr,
                                       Saiga::Matrix<double, size_of_desc_vec, 1>* J_color = nullptr)
{
    SAIGA_ASSERT(false);  // not implemented
    float result = color / (1 - alphadest);
    if (J_alphadest)
    {
        auto& J = *J_alphadest;
        J(0, 0) = color / ((1 - alphadest) * (1 - alphadest));
    }
    if (J_color)
    {
        auto& J = *J_color;
        J(0, 0) = 1.0 / (1 - alphadest);
    }
    return result;
}