//! Config command handlers: desktop_config, resolution_config, quality_config, etc.

use crate::config::RenderConfig;
use crate::shared::RendererCommand;

use super::{CommandContext, CommandHandler, CommandResult};

/// Handles `desktop_config`. Updates view state and render config. Post-finalize only.
pub struct ConfigCommandHandler;

impl CommandHandler for ConfigCommandHandler {
    fn handle(&mut self, cmd: &RendererCommand, ctx: &mut CommandContext<'_>) -> CommandResult {
        if !ctx.session_flags.init_state.is_finalized() {
            return CommandResult::Ignored;
        }
        match cmd {
            RendererCommand::desktop_config(x) => {
                *ctx.render_config = RenderConfig {
                    near_clip: ctx.view_state.near_clip,
                    far_clip: ctx.view_state.far_clip,
                    desktop_fov: ctx.view_state.desktop_fov,
                    vsync: x.v_sync,
                    use_debug_uv: ctx.render_config.use_debug_uv,
                    use_pbr: ctx.render_config.use_pbr,
                    skinned_apply_mesh_root_transform: ctx
                        .render_config
                        .skinned_apply_mesh_root_transform,
                    skinned_use_root_bone: ctx.render_config.skinned_use_root_bone,
                    gpu_validation_layers: ctx.render_config.gpu_validation_layers,
                    ray_tracing_enabled: ctx.render_config.ray_tracing_enabled,
                    use_opengl: ctx.render_config.use_opengl,
                    use_dx12: ctx.render_config.use_dx12,
                    debug_skinned: ctx.render_config.debug_skinned,
                    debug_blendshapes: ctx.render_config.debug_blendshapes,
                    skinned_flip_handedness: ctx.render_config.skinned_flip_handedness,
                    rtao_enabled: ctx.render_config.rtao_enabled,
                    ray_traced_shadows_enabled: ctx.render_config.ray_traced_shadows_enabled,
                    ray_traced_shadows_use_compute: ctx
                        .render_config
                        .ray_traced_shadows_use_compute,
                    rt_soft_shadow_samples: ctx.render_config.rt_soft_shadow_samples,
                    rt_soft_shadow_cone_scale: ctx.render_config.rt_soft_shadow_cone_scale,
                    rt_shadow_atlas_half_resolution: ctx
                        .render_config
                        .rt_shadow_atlas_half_resolution,
                    rtao_strength: ctx.render_config.rtao_strength,
                    ao_radius: ctx.render_config.ao_radius,
                    frustum_culling: ctx.render_config.frustum_culling,
                    parallel_mesh_draw_prep_batches: ctx
                        .render_config
                        .parallel_mesh_draw_prep_batches,
                    log_collect_draw_batches_timing: ctx
                        .render_config
                        .log_collect_draw_batches_timing,
                    use_host_unlit_pilot: ctx.render_config.use_host_unlit_pilot,
                    fullscreen_filter_hook: ctx.render_config.fullscreen_filter_hook,
                    shader_debug_override: ctx.render_config.shader_debug_override,
                    use_native_ui_wgsl: ctx.render_config.use_native_ui_wgsl,
                    native_ui_unlit_shader_id: ctx.render_config.native_ui_unlit_shader_id,
                    native_ui_text_unlit_shader_id: ctx
                        .render_config
                        .native_ui_text_unlit_shader_id,
                    ui_unlit_property_ids: ctx.render_config.ui_unlit_property_ids.clone(),
                    ui_text_unlit_property_ids: ctx
                        .render_config
                        .ui_text_unlit_property_ids
                        .clone(),
                    native_ui_world_space: ctx.render_config.native_ui_world_space,
                    native_ui_overlay_stencil_pipelines: ctx
                        .render_config
                        .native_ui_overlay_stencil_pipelines,
                    log_native_ui_routing: ctx.render_config.log_native_ui_routing,
                    native_ui_routing_metrics: ctx.render_config.native_ui_routing_metrics,
                    native_ui_uivert_pbr_fallback: ctx.render_config.native_ui_uivert_pbr_fallback,
                    native_ui_force_shader_hint_registration: ctx
                        .render_config
                        .native_ui_force_shader_hint_registration,
                    native_ui_default_surface_blend: ctx
                        .render_config
                        .native_ui_default_surface_blend,
                    pbr_bind_host_material_properties: ctx
                        .render_config
                        .pbr_bind_host_material_properties,
                    pbr_host_color_property_id: ctx.render_config.pbr_host_color_property_id,
                    pbr_host_metallic_property_id: ctx.render_config.pbr_host_metallic_property_id,
                    pbr_host_smoothness_property_id: ctx
                        .render_config
                        .pbr_host_smoothness_property_id,
                    multi_material_submeshes: ctx.render_config.multi_material_submeshes,
                    log_multi_material_submesh_mismatch: ctx
                        .render_config
                        .log_multi_material_submesh_mismatch,
                };
                CommandResult::Handled
            }
            _ => CommandResult::Ignored,
        }
    }
}
