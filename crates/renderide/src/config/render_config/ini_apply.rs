//! Applies `configuration.ini` entries onto [`crate::config::RenderConfig`].

use super::{RenderConfig, ShaderDebugOverride};
use crate::assets::NativeUiSurfaceBlend;
use crate::config::ini::parse_bool;

/// Applies one `(section, key, value)` triple from `configuration.ini` to [`crate::config::RenderConfig`].
///
/// Unknown section/key pairs are ignored. Parse errors are logged to stderr and the logger.
pub(crate) fn apply_render_config_ini_entry(
    config: &mut RenderConfig,
    section: &str,
    key: &str,
    value: &str,
) {
    match (section, key) {
        ("camera", "near_clip") => match value.parse::<f32>() {
            Ok(v) => {
                config.near_clip = v;
                eprintln!("[renderide] ini: near_clip = {}", v);
                logger::info!("ini: near_clip = {}", v);
            }
            Err(_) => eprintln!("[renderide] ini: near_clip parse error (raw = {:?})", value),
        },
        ("camera", "far_clip") => match value.parse::<f32>() {
            Ok(v) => {
                config.far_clip = v;
                eprintln!("[renderide] ini: far_clip = {}", v);
                logger::info!("ini: far_clip = {}", v);
            }
            Err(_) => eprintln!("[renderide] ini: far_clip parse error (raw = {:?})", value),
        },
        ("camera", "desktop_fov") => match value.parse::<f32>() {
            Ok(v) => {
                config.desktop_fov = v;
                eprintln!("[renderide] ini: desktop_fov = {}", v);
                logger::info!("ini: desktop_fov = {}", v);
            }
            Err(_) => eprintln!(
                "[renderide] ini: desktop_fov parse error (raw = {:?})",
                value
            ),
        },
        ("display", "vsync") => {
            if let Some(v) = parse_bool(value) {
                config.vsync = v;
                eprintln!("[renderide] ini: vsync = {}", v);
                logger::info!("ini: vsync = {}", v);
            } else {
                eprintln!("[renderide] ini: vsync parse error (raw = {:?})", value);
            }
        }
        ("rendering", "use_debug_uv") => {
            if let Some(v) = parse_bool(value) {
                config.use_debug_uv = v;
                eprintln!("[renderide] ini: use_debug_uv = {}", v);
                logger::info!("ini: use_debug_uv = {}", v);
            } else {
                eprintln!(
                    "[renderide] ini: use_debug_uv parse error (raw = {:?})",
                    value
                );
            }
        }
        ("rendering", "use_pbr") => {
            if let Some(v) = parse_bool(value) {
                config.use_pbr = v;
                eprintln!("[renderide] ini: use_pbr = {}", v);
                logger::info!("ini: use_pbr = {}", v);
            } else {
                eprintln!("[renderide] ini: use_pbr parse error (raw = {:?})", value);
            }
        }

        ("rendering", "fullscreen_filter_hook") => {
            if let Some(v) = parse_bool(value) {
                config.fullscreen_filter_hook = v;
                eprintln!("[renderide] ini: fullscreen_filter_hook = {}", v);
                logger::info!("ini: fullscreen_filter_hook = {}", v);
            } else {
                eprintln!(
                    "[renderide] ini: fullscreen_filter_hook parse error (raw = {:?})",
                    value
                );
            }
        }
        ("rendering", "multi_material_submeshes") => {
            if let Some(v) = parse_bool(value) {
                config.multi_material_submeshes = v;
                eprintln!("[renderide] ini: multi_material_submeshes = {}", v);
                logger::info!("ini: multi_material_submeshes = {}", v);
            } else {
                eprintln!(
                    "[renderide] ini: multi_material_submeshes parse error (raw = {:?})",
                    value
                );
            }
        }
        ("rendering", "log_multi_material_submesh_mismatch") => {
            if let Some(v) = parse_bool(value) {
                config.log_multi_material_submesh_mismatch = v;
                eprintln!(
                    "[renderide] ini: log_multi_material_submesh_mismatch = {}",
                    v
                );
                logger::info!("ini: log_multi_material_submesh_mismatch = {}", v);
            } else {
                eprintln!(
                    "[renderide] ini: log_multi_material_submesh_mismatch parse error (raw = {:?})",
                    value
                );
            }
        }
        ("rendering", "shader_debug_override") => {
            let v = value.trim();
            if v.eq_ignore_ascii_case("none") || v.is_empty() {
                config.shader_debug_override = ShaderDebugOverride::None;
                eprintln!("[renderide] ini: shader_debug_override = None");
                logger::info!("ini: shader_debug_override = None");
            } else if v.eq_ignore_ascii_case("force_legacy_global_shading")
                || v.eq_ignore_ascii_case("legacy")
            {
                config.shader_debug_override = ShaderDebugOverride::ForceLegacyGlobalShading;
                eprintln!("[renderide] ini: shader_debug_override = force_legacy_global_shading");
                logger::info!("ini: shader_debug_override = force_legacy_global_shading");
            } else {
                eprintln!(
                    "[renderide] ini: shader_debug_override unknown (raw = {:?})",
                    value
                );
            }
        }
        ("rendering", "skinned_apply_mesh_root_transform") => {
            if let Some(v) = parse_bool(value) {
                config.skinned_apply_mesh_root_transform = v;
                eprintln!("[renderide] ini: skinned_apply_mesh_root_transform = {}", v);
                logger::info!("ini: skinned_apply_mesh_root_transform = {}", v);
            } else {
                eprintln!(
                    "[renderide] ini: skinned_apply_mesh_root_transform parse error (raw = {:?})",
                    value
                );
            }
        }
        ("rendering", "skinned_use_root_bone") => {
            if let Some(v) = parse_bool(value) {
                config.skinned_use_root_bone = v;
                eprintln!("[renderide] ini: skinned_use_root_bone = {}", v);
                logger::info!("ini: skinned_use_root_bone = {}", v);
            } else {
                eprintln!(
                    "[renderide] ini: skinned_use_root_bone parse error (raw = {:?})",
                    value
                );
            }
        }
        ("rendering", "gpu_validation_layers") => {
            if let Some(v) = parse_bool(value) {
                config.gpu_validation_layers = v;
                eprintln!("[renderide] ini: gpu_validation_layers = {}", v);
                logger::info!("ini: gpu_validation_layers = {}", v);
            } else {
                eprintln!(
                    "[renderide] ini: gpu_validation_layers parse error (raw = {:?})",
                    value
                );
            }
        }
        ("rendering", "ray_tracing_enabled") => {
            if let Some(v) = parse_bool(value) {
                config.ray_tracing_enabled = v;
                eprintln!("[renderide] ini: ray_tracing_enabled = {}", v);
                logger::info!("ini: ray_tracing_enabled = {}", v);
            } else {
                eprintln!(
                    "[renderide] ini: ray_tracing_enabled parse error (raw = {:?})",
                    value
                );
            }
        }
        ("rendering", "use_opengl") => {
            if let Some(v) = parse_bool(value) {
                config.use_opengl = v;
                eprintln!("[renderide] ini: use_opengl = {}", v);
                logger::info!("ini: use_opengl = {}", v);
            } else {
                eprintln!(
                    "[renderide] ini: use_opengl parse error (raw = {:?})",
                    value
                );
            }
        }
        ("rendering", "use_dx12") => {
            if let Some(v) = parse_bool(value) {
                config.use_dx12 = v;
                eprintln!("[renderide] ini: use_dx12 = {}", v);
                logger::info!("ini: use_dx12 = {}", v);
            } else {
                eprintln!("[renderide] ini: use_dx12 parse error (raw = {:?})", value);
            }
        }
        ("rendering", "debug_skinned") => {
            if let Some(v) = parse_bool(value) {
                config.debug_skinned = v;
                eprintln!("[renderide] ini: debug_skinned = {}", v);
                logger::info!("ini: debug_skinned = {}", v);
            } else {
                eprintln!(
                    "[renderide] ini: debug_skinned parse error (raw = {:?})",
                    value
                );
            }
        }
        ("rendering", "debug_blendshapes") => {
            if let Some(v) = parse_bool(value) {
                config.debug_blendshapes = v;
                eprintln!("[renderide] ini: debug_blendshapes = {}", v);
                logger::info!("ini: debug_blendshapes = {}", v);
            } else {
                eprintln!(
                    "[renderide] ini: debug_blendshapes parse error (raw = {:?})",
                    value
                );
            }
        }
        ("rendering", "skinned_flip_handedness") => {
            if let Some(v) = parse_bool(value) {
                config.skinned_flip_handedness = v;
                eprintln!("[renderide] ini: skinned_flip_handedness = {}", v);
                logger::info!("ini: skinned_flip_handedness = {}", v);
            } else {
                eprintln!(
                    "[renderide] ini: skinned_flip_handedness parse error (raw = {:?})",
                    value
                );
            }
        }
        ("rendering", "parallel_mesh_draw_prep_batches") => {
            if let Some(v) = parse_bool(value) {
                config.parallel_mesh_draw_prep_batches = v;
                eprintln!("[renderide] ini: parallel_mesh_draw_prep_batches = {}", v);
                logger::info!("ini: parallel_mesh_draw_prep_batches = {}", v);
            } else {
                eprintln!(
                    "[renderide] ini: parallel_mesh_draw_prep_batches parse error (raw = {:?})",
                    value
                );
            }
        }
        ("rendering", "log_collect_draw_batches_timing") => {
            if let Some(v) = parse_bool(value) {
                config.log_collect_draw_batches_timing = v;
                eprintln!("[renderide] ini: log_collect_draw_batches_timing = {}", v);
                logger::info!("ini: log_collect_draw_batches_timing = {}", v);
            } else {
                eprintln!(
                    "[renderide] ini: log_collect_draw_batches_timing parse error (raw = {:?})",
                    value
                );
            }
        }
        ("rendering", "rtao_enabled") => {
            if let Some(v) = parse_bool(value) {
                config.rtao_enabled = v;
                eprintln!("[renderide] ini: rtao_enabled = {}", v);
                logger::info!("ini: rtao_enabled = {}", v);
            } else {
                eprintln!(
                    "[renderide] ini: rtao_enabled parse error (raw = {:?})",
                    value
                );
            }
        }
        ("rendering", "ray_traced_shadows_enabled") => {
            if let Some(v) = parse_bool(value) {
                config.ray_traced_shadows_enabled = v;
                eprintln!("[renderide] ini: ray_traced_shadows_enabled = {}", v);
                logger::info!("ini: ray_traced_shadows_enabled = {}", v);
            } else {
                eprintln!(
                    "[renderide] ini: ray_traced_shadows_enabled parse error (raw = {:?})",
                    value
                );
            }
        }
        ("rendering", "ray_traced_shadows_use_compute") => {
            if let Some(v) = parse_bool(value) {
                config.ray_traced_shadows_use_compute = v;
                eprintln!("[renderide] ini: ray_traced_shadows_use_compute = {}", v);
                logger::info!("ini: ray_traced_shadows_use_compute = {}", v);
            } else {
                eprintln!(
                    "[renderide] ini: ray_traced_shadows_use_compute parse error (raw = {:?})",
                    value
                );
            }
        }
        ("rendering", "rt_soft_shadow_samples") => match value.parse::<u32>() {
            Ok(v) => {
                config.rt_soft_shadow_samples = v.clamp(1, 16);
                eprintln!(
                    "[renderide] ini: rt_soft_shadow_samples = {}",
                    config.rt_soft_shadow_samples
                );
                logger::info!(
                    "ini: rt_soft_shadow_samples = {}",
                    config.rt_soft_shadow_samples
                );
            }
            Err(_) => eprintln!(
                "[renderide] ini: rt_soft_shadow_samples parse error (raw = {:?})",
                value
            ),
        },
        ("rendering", "rt_soft_shadow_cone_scale") => match value.parse::<f32>() {
            Ok(v) => {
                config.rt_soft_shadow_cone_scale = v;
                eprintln!("[renderide] ini: rt_soft_shadow_cone_scale = {}", v);
                logger::info!("ini: rt_soft_shadow_cone_scale = {}", v);
            }
            Err(_) => eprintln!(
                "[renderide] ini: rt_soft_shadow_cone_scale parse error (raw = {:?})",
                value
            ),
        },
        ("rendering", "rt_shadow_atlas_half_resolution") => {
            if let Some(v) = parse_bool(value) {
                config.rt_shadow_atlas_half_resolution = v;
                eprintln!("[renderide] ini: rt_shadow_atlas_half_resolution = {}", v);
                logger::info!("ini: rt_shadow_atlas_half_resolution = {}", v);
            } else {
                eprintln!(
                    "[renderide] ini: rt_shadow_atlas_half_resolution parse error (raw = {:?})",
                    value
                );
            }
        }
        ("rendering", "rtao_strength") => match value.parse::<f32>() {
            Ok(v) => {
                config.rtao_strength = v;
                eprintln!("[renderide] ini: rtao_strength = {}", v);
                logger::info!("ini: rtao_strength = {}", v);
            }
            Err(_) => eprintln!(
                "[renderide] ini: rtao_strength parse error (raw = {:?})",
                value
            ),
        },
        ("rendering", "ao_radius") => match value.parse::<f32>() {
            Ok(v) => {
                config.ao_radius = v;
                eprintln!("[renderide] ini: ao_radius = {}", v);
                logger::info!("ini: ao_radius = {}", v);
            }
            Err(_) => eprintln!("[renderide] ini: ao_radius parse error (raw = {:?})", value),
        },
        ("rendering", "frustum_culling") => {
            if let Some(v) = parse_bool(value) {
                config.frustum_culling = v;
                eprintln!("[renderide] ini: frustum_culling = {}", v);
                logger::info!("ini: frustum_culling = {}", v);
            } else {
                eprintln!(
                    "[renderide] ini: frustum_culling parse error (raw = {:?})",
                    value
                );
            }
        }
        ("rendering", "use_native_ui_wgsl") => {
            if let Some(v) = parse_bool(value) {
                config.use_native_ui_wgsl = v;
                eprintln!("[renderide] ini: use_native_ui_wgsl = {}", v);
                logger::info!("ini: use_native_ui_wgsl = {}", v);
            } else {
                eprintln!(
                    "[renderide] ini: use_native_ui_wgsl parse error (raw = {:?})",
                    value
                );
            }
        }
        ("rendering", "native_ui_world_space") => {
            if let Some(v) = parse_bool(value) {
                config.native_ui_world_space = v;
                eprintln!("[renderide] ini: native_ui_world_space = {}", v);
                logger::info!("ini: native_ui_world_space = {}", v);
            } else {
                eprintln!(
                    "[renderide] ini: native_ui_world_space parse error (raw = {:?})",
                    value
                );
            }
        }
        ("rendering", "native_ui_overlay_stencil_pipelines") => {
            if let Some(v) = parse_bool(value) {
                config.native_ui_overlay_stencil_pipelines = v;
                eprintln!(
                    "[renderide] ini: native_ui_overlay_stencil_pipelines = {}",
                    v
                );
                logger::info!("ini: native_ui_overlay_stencil_pipelines = {}", v);
            } else {
                eprintln!(
                    "[renderide] ini: native_ui_overlay_stencil_pipelines parse error (raw = {:?})",
                    value
                );
            }
        }
        ("rendering", "log_native_ui_routing") => {
            if let Some(v) = parse_bool(value) {
                config.log_native_ui_routing = v;
                eprintln!("[renderide] ini: log_native_ui_routing = {}", v);
                logger::info!("ini: log_native_ui_routing = {}", v);
            } else {
                eprintln!(
                    "[renderide] ini: log_native_ui_routing parse error (raw = {:?})",
                    value
                );
            }
        }
        ("rendering", "log_ui_unlit_material_inventory") => {
            if let Some(v) = parse_bool(value) {
                config.log_ui_unlit_material_inventory = v;
                eprintln!("[renderide] ini: log_ui_unlit_material_inventory = {}", v);
                logger::info!("ini: log_ui_unlit_material_inventory = {}", v);
            } else {
                eprintln!(
                    "[renderide] ini: log_ui_unlit_material_inventory parse error (raw = {:?})",
                    value
                );
            }
        }
        ("rendering", "native_ui_routing_metrics") => {
            if let Some(v) = parse_bool(value) {
                config.native_ui_routing_metrics = v;
                eprintln!("[renderide] ini: native_ui_routing_metrics = {}", v);
                logger::info!("ini: native_ui_routing_metrics = {}", v);
            } else {
                eprintln!(
                    "[renderide] ini: native_ui_routing_metrics parse error (raw = {:?})",
                    value
                );
            }
        }
        ("rendering", "material_batch_wire_metrics") => {
            if let Some(v) = parse_bool(value) {
                config.material_batch_wire_metrics = v;
                eprintln!("[renderide] ini: material_batch_wire_metrics = {}", v);
                logger::info!("ini: material_batch_wire_metrics = {}", v);
            } else {
                eprintln!(
                    "[renderide] ini: material_batch_wire_metrics parse error (raw = {:?})",
                    value
                );
            }
        }
        ("rendering", "material_batch_persist_extended_payloads") => {
            if let Some(v) = parse_bool(value) {
                config.material_batch_persist_extended_payloads = v;
                eprintln!(
                    "[renderide] ini: material_batch_persist_extended_payloads = {}",
                    v
                );
                logger::info!("ini: material_batch_persist_extended_payloads = {}", v);
            } else {
                eprintln!(
                    "[renderide] ini: material_batch_persist_extended_payloads parse error (raw = {:?})",
                    value
                );
            }
        }
        ("rendering", "native_ui_uivert_pbr_fallback") => {
            if let Some(v) = parse_bool(value) {
                config.native_ui_uivert_pbr_fallback = v;
                eprintln!("[renderide] ini: native_ui_uivert_pbr_fallback = {}", v);
                logger::info!("ini: native_ui_uivert_pbr_fallback = {}", v);
            } else {
                eprintln!(
                    "[renderide] ini: native_ui_uivert_pbr_fallback parse error (raw = {:?})",
                    value
                );
            }
        }
        ("rendering", "native_ui_default_surface_blend") => {
            if let Some(v) = NativeUiSurfaceBlend::parse_ini(value) {
                config.native_ui_default_surface_blend = v;
                eprintln!("[renderide] ini: native_ui_default_surface_blend = {:?}", v);
                logger::info!("ini: native_ui_default_surface_blend = {:?}", v);
            } else {
                eprintln!(
                    "[renderide] ini: native_ui_default_surface_blend parse error (raw = {:?})",
                    value
                );
            }
        }
        ("rendering", "pbr_bind_host_material_properties") => {
            if let Some(v) = parse_bool(value) {
                config.pbr_bind_host_material_properties = v;
                eprintln!("[renderide] ini: pbr_bind_host_material_properties = {}", v);
                logger::info!("ini: pbr_bind_host_material_properties = {}", v);
            } else {
                eprintln!(
                    "[renderide] ini: pbr_bind_host_material_properties parse error (raw = {:?})",
                    value
                );
            }
        }
        ("rendering", "pbr_bind_host_main_texture") => {
            if let Some(v) = parse_bool(value) {
                config.pbr_bind_host_main_texture = v;
                eprintln!("[renderide] ini: pbr_bind_host_main_texture = {}", v);
                logger::info!("ini: pbr_bind_host_main_texture = {}", v);
            } else {
                eprintln!(
                    "[renderide] ini: pbr_bind_host_main_texture parse error (raw = {:?})",
                    value
                );
            }
        }
        _ => {
            apply_native_ui_property_ini(config, section, key, value);
            apply_world_unlit_property_ini(config, section, key, value);
        }
    }
}

/// Applies `[native_ui_unlit_properties]` / `[native_ui_text_unlit_properties]` keys.
fn apply_native_ui_property_ini(config: &mut RenderConfig, section: &str, key: &str, value: &str) {
    match section {
        "native_ui_unlit_properties" => {
            let Ok(v) = value.parse::<i32>() else {
                logger::warn!("native_ui_unlit_properties: skip bad int for key {}", key);
                return;
            };
            let ids = &mut config.ui_unlit_property_ids;
            match key {
                "tint" => ids.tint = v,
                "overlay_tint" => ids.overlay_tint = v,
                "cutoff" => ids.cutoff = v,
                "rect" => ids.rect = v,
                "main_tex_st" => ids.main_tex_st = v,
                "mask_tex_st" => ids.mask_tex_st = v,
                "main_tex" => ids.main_tex = v,
                "mask_tex" => ids.mask_tex = v,
                "alphaclip" => ids.alphaclip = v,
                "rectclip" => ids.rectclip = v,
                "overlay" => ids.overlay = v,
                "texture_normalmap" => ids.texture_normalmap = v,
                "texture_lerpcolor" => ids.texture_lerpcolor = v,
                "mask_texture_mul" => ids.mask_texture_mul = v,
                "mask_texture_clip" => ids.mask_texture_clip = v,
                "src_blend" => ids.src_blend = v,
                "dst_blend" => ids.dst_blend = v,
                "zwrite" => ids.zwrite = v,
                "cull" => ids.cull = v,
                "ztest" => ids.ztest = v,
                "color_mask" => ids.color_mask = v,
                "stencil_comp" => ids.stencil_comp = v,
                "stencil_ref" => ids.stencil_ref = v,
                "stencil_op" => ids.stencil_op = v,
                "stencil_write_mask" => ids.stencil_write_mask = v,
                "stencil_read_mask" => ids.stencil_read_mask = v,
                "offset_factor" => ids.offset_factor = v,
                "offset_units" => ids.offset_units = v,
                _ => {}
            }
        }
        "native_ui_text_unlit_properties" => {
            let Ok(v) = value.parse::<i32>() else {
                logger::warn!(
                    "native_ui_text_unlit_properties: skip bad int for key {}",
                    key
                );
                return;
            };
            let ids = &mut config.ui_text_unlit_property_ids;
            match key {
                "tint_color" => ids.tint_color = v,
                "overlay_tint" => ids.overlay_tint = v,
                "outline_color" => ids.outline_color = v,
                "background_color" => ids.background_color = v,
                "range" => ids.range = v,
                "face_dilate" => ids.face_dilate = v,
                "face_softness" => ids.face_softness = v,
                "outline_size" => ids.outline_size = v,
                "rect" => ids.rect = v,
                "font_atlas" => ids.font_atlas = v,
                "raster" => ids.raster = v,
                "sdf" => ids.sdf = v,
                "msdf" => ids.msdf = v,
                "outline" => ids.outline = v,
                "rectclip" => ids.rectclip = v,
                "overlay" => ids.overlay = v,
                "src_blend" => ids.src_blend = v,
                "dst_blend" => ids.dst_blend = v,
                _ => {}
            }
        }
        _ => {}
    }
}

/// Applies `[world_unlit_properties]` keys for world `Shader "Unlit"` WGSL.
fn apply_world_unlit_property_ini(
    config: &mut RenderConfig,
    section: &str,
    key: &str,
    value: &str,
) {
    if section != "world_unlit_properties" {
        return;
    }
    let Ok(v) = value.parse::<i32>() else {
        logger::warn!("world_unlit_properties: skip bad int for key {}", key);
        return;
    };
    let ids = &mut config.world_unlit_property_ids;
    match key {
        "tex" => ids.tex = v,
        "tex_st" => ids.tex_st = v,
        "color" => ids.color = v,
        "cutoff" => ids.cutoff = v,
        "mask_tex" => ids.mask_tex = v,
        "mask_tex_st" => ids.mask_tex_st = v,
        "texture_kw" => ids.texture_kw = v,
        "color_kw" => ids.color_kw = v,
        "texture_normalmap_kw" => ids.texture_normalmap_kw = v,
        "alphatest_kw" => ids.alphatest_kw = v,
        "mask_texture_mul" => ids.mask_texture_mul = v,
        "mask_texture_clip" => ids.mask_texture_clip = v,
        _ => {}
    }
}
