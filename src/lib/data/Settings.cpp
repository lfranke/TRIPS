/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "Settings.h"

#include "saiga/core/imgui/imgui.h"

void CombinedParams::Check()
{
    if (net_params.conv_block == "partial" || net_params.conv_block == "partial_multi" ||
        (pipeline_params.enable_environment_map && !pipeline_params.environment_map_params.use_points_for_env_map) ||
        pipeline_params.cat_masks_to_color)
    {
        if (pipeline_params.render_modes_start_epochs[4] < 0) render_params.output_background_mask = true;
    }

    if (pipeline_params.skip_neural_render_network)
    {
        pipeline_params.num_texture_channels = 3;
        net_params.num_input_layers          = 1;
    }
    if (pipeline_params.skip_neural_render_network_but_add_layers)
    {
        pipeline_params.num_texture_channels = 3;
    }

    // render_params.normalize_grads                      = !optimizer_params.use_myadam_everywhere;
    net_params.num_input_channels                      = pipeline_params.num_texture_channels;
    render_params.num_texture_channels                 = pipeline_params.num_texture_channels;
    render_params.use_point_adding_and_removing_module = pipeline_params.use_point_adding_and_removing_module;
    render_params.use_environment_map =
        pipeline_params.enable_environment_map && !pipeline_params.environment_map_params.use_points_for_env_map;
    render_params.debug_max_list_length = points_adding_params.debug_max_list_length;

    render_params.use_layer_point_size = !optimizer_params.fix_point_size;

    if (!optimizer_params.fix_points || !optimizer_params.fix_poses || !optimizer_params.fix_intrinsics ||
        !optimizer_params.fix_dynamic_refinement)
    {
        render_params.need_point_gradient = true;
        std::cout << "POINT GRADIENTS ARE COMPUTED." << std::endl;
    }
    else
    {
        render_params.need_point_gradient = false;

        std::cout << "Point gradients omitted." << std::endl;
    }



    // SAIGA_ASSERT(!train_params.texture_color_init || pipeline_params.num_texture_channels == 3);

    if (pipeline_params.environment_map_params.cat_env_to_color)
    {
        net_params.num_input_channels += pipeline_params.environment_map_params.env_map_channels;
    }
    else
    {
        pipeline_params.environment_map_params.env_map_channels = pipeline_params.num_texture_channels;
    }

    if (pipeline_params.cat_masks_to_color)
    {
        net_params.num_input_channels += 1;
    }
    if (render_params.add_depth_to_network)
    {
        net_params.num_input_channels += 1;
    }
}
void CombinedParams::imgui()
{
    ImGui::Checkbox("render_points", &render_params.render_points);
    ImGui::Checkbox("render_outliers", &render_params.render_outliers);
    ImGui::Checkbox("drop_out_points_by_radius", &render_params.drop_out_points_by_radius);
    ImGui::SliderFloat("drop_out_radius_threshold", &render_params.drop_out_radius_threshold, 0, 5);
    ImGui::Checkbox("super_sampling", &render_params.super_sampling);

    ImGui::Checkbox("check_normal", &render_params.check_normal);


    ImGui::Checkbox("debug_weight_color", &render_params.debug_weight_color);
    ImGui::InputInt("debug_max_list_length", &render_params.debug_max_list_length);
    ImGui::Checkbox("debug_depth_color", &render_params.debug_depth_color);
    ImGui::SliderFloat("debug_max_weight", &render_params.debug_max_weight, 0, 100);
    ImGui::Checkbox("debug_print_num_rendered_points", &render_params.debug_print_num_rendered_points);

    ImGui::SliderFloat("dropout", &render_params.dropout, 0, 1);
    ImGui::SliderFloat("depth_accept", &render_params.depth_accept, 0, 0.1);
    ImGui::SliderFloat("depth_accept_blend", &render_params.depth_accept_blend, 0.001, 1);

    ImGui::SliderFloat("dist_cutoff", &render_params.dist_cutoff, 0, 1);

    ImGui::Checkbox("combine_lists", &render_params.combine_lists);
    ImGui::Checkbox("render_points_in_all_lower_resolutions", &render_params.render_points_in_all_lower_resolutions);
    ImGui::Checkbox("saturated_alpha_accumulation", &render_params.saturated_alpha_accumulation);

    ImGui::Separator();
}
