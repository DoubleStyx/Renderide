//! Debug HUD helpers: aggregated shader route lines and native UI material/texture warnings.

use std::collections::{HashMap, HashSet};

/// Per `(request, message)` aggregation for HUD lines: draw count and distinct mesh ids.
#[derive(Default)]
struct ShaderHudAccum {
    draws: usize,
    meshes: HashSet<i32>,
}

/// Increments draw count and records `mesh_asset_id` for a grouped warning or route line.
fn accumulate_shader_hud_line(
    map: &mut HashMap<(String, String), ShaderHudAccum>,
    request: String,
    message: String,
    mesh_asset_id: i32,
) {
    let entry = map.entry((request, message)).or_default();
    entry.draws += 1;
    entry.meshes.insert(mesh_asset_id);
}

/// Default WGSL path label for a pipeline variant when no host shader path override exists.
fn shader_source_label(variant: crate::gpu::PipelineVariant) -> &'static str {
    use crate::gpu::PipelineVariant;

    match variant {
        PipelineVariant::Material { .. } => "RENDERIDESHADERS/world/unlit.wgsl",
        PipelineVariant::NativeUiUnlit { .. } | PipelineVariant::NativeUiUnlitStencil { .. } => {
            "RENDERIDESHADERS/ui/ui_unlit.wgsl"
        }
        PipelineVariant::NativeUiTextUnlit { .. }
        | PipelineVariant::NativeUiTextUnlitStencil { .. } => {
            "RENDERIDESHADERS/ui/ui_text_unlit.wgsl"
        }
        PipelineVariant::Pbr
        | PipelineVariant::PbrMRT
        | PipelineVariant::SkinnedPbr
        | PipelineVariant::SkinnedPbrMRT => "RENDERIDESHADERS/pbr/pbs_metallic.wgsl",
        PipelineVariant::PbrHostAlbedo => "RENDERIDESHADERS/pbr/pbs_metallic_host_albedo.wgsl",
        PipelineVariant::NormalDebug => "RENDERIDESHADERS/debug/normal_debug.wgsl",
        PipelineVariant::UvDebug => "RENDERIDESHADERS/debug/uv_debug.wgsl",
        PipelineVariant::NormalDebugMRT => "RENDERIDESHADERS/debug/normal_debug_mrt.wgsl",
        PipelineVariant::UvDebugMRT => "RENDERIDESHADERS/debug/uv_debug_mrt.wgsl",
        PipelineVariant::Skinned => "RENDERIDESHADERS/skinned/skinned.wgsl",
        PipelineVariant::SkinnedMRT => "RENDERIDESHADERS/skinned/skinned_mrt.wgsl",
        PipelineVariant::OverlayStencilContent
        | PipelineVariant::OverlayStencilMaskWrite
        | PipelineVariant::OverlayStencilMaskClear => {
            "RENDERIDESHADERS/overlay/overlay_stencil.wgsl"
        }
        PipelineVariant::OverlayStencilSkinned
        | PipelineVariant::OverlayStencilMaskWriteSkinned
        | PipelineVariant::OverlayStencilMaskClearSkinned => {
            "RENDERIDESHADERS/skinned/skinned.wgsl"
        }
        PipelineVariant::OverlayNoDepthNormalDebug => "RENDERIDESHADERS/debug/normal_debug.wgsl",
        PipelineVariant::OverlayNoDepthUvDebug => "RENDERIDESHADERS/debug/uv_debug.wgsl",
        PipelineVariant::OverlayNoDepthSkinned => "RENDERIDESHADERS/skinned/skinned.wgsl",
        PipelineVariant::PbrRayQuery
        | PipelineVariant::PbrMRTRayQuery
        | PipelineVariant::SkinnedPbrRayQuery
        | PipelineVariant::SkinnedPbrMRTRayQuery => "inline:pbr_ray_query_pending_file_migration",
    }
}

/// Host shader display name from registry (`unity_shader_name` or upload field), or `<none>`.
fn shader_request_label(
    registry: &crate::assets::AssetRegistry,
    shader_asset_id: Option<i32>,
) -> String {
    shader_asset_id
        .and_then(|id| registry.get_shader(id))
        .and_then(|shader| {
            shader
                .unity_shader_name
                .clone()
                .or_else(|| shader.upload_file_field().map(str::to_string))
        })
        .unwrap_or_else(|| "<none>".to_string())
}

/// Resolved WGSL path for HUD: host-relative path when mapped, else [`shader_source_label`].
fn shader_source_label_for_request(request: &str, variant: crate::gpu::PipelineVariant) -> String {
    if let Some(rel) = crate::assets::resolve_renderide_shader_rel_path(Some(request)) {
        return format!("RENDERIDESHADERS/{}", rel.replace('\\', "/"));
    }
    shader_source_label(variant).to_string()
}

/// Builds sorted HUD lines for host shader requests vs resolved WGSL paths, plus unsupported-program fallbacks.
pub fn summarize_shader_routes(
    draw_batches: &[crate::render::SpaceDrawBatch],
    registry: &crate::assets::AssetRegistry,
) -> (Vec<String>, Vec<String>) {
    let mut routes: HashMap<(String, String), ShaderHudAccum> = HashMap::new();
    let mut fallbacks: HashMap<(String, String), ShaderHudAccum> = HashMap::new();

    for batch in draw_batches {
        for draw in &batch.draws {
            let request = shader_request_label(registry, draw.shader_key.host_shader_asset_id);
            let loaded = shader_source_label_for_request(&request, draw.pipeline_variant);
            let key = (request.clone(), loaded.clone());
            let entry = routes.entry(key).or_default();
            entry.draws += 1;
            entry.meshes.insert(draw.mesh_asset_id);

            let is_host_fallback = draw
                .shader_key
                .host_shader_asset_id
                .and_then(|id| registry.get_shader(id))
                .map(|shader| shader.program == crate::assets::EssentialShaderProgram::Unsupported)
                .unwrap_or(false);
            if is_host_fallback {
                let fallback_entry = fallbacks.entry((request, loaded)).or_default();
                fallback_entry.draws += 1;
                fallback_entry.meshes.insert(draw.mesh_asset_id);
            }
        }
    }

    let mut route_lines: Vec<_> = routes
        .into_iter()
        .map(|((request, loaded), accum)| {
            (
                accum.draws,
                format!(
                    "draws {:>4}  meshes {:>4}  |  {} -> {}",
                    accum.draws,
                    accum.meshes.len(),
                    request,
                    loaded
                ),
            )
        })
        .collect();
    route_lines.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));

    let mut fallback_lines: Vec<_> = fallbacks
        .into_iter()
        .map(|((request, loaded), accum)| {
            (
                accum.draws,
                format!(
                    "draws {:>4}  meshes {:>4}  |  {} -> {}",
                    accum.draws,
                    accum.meshes.len(),
                    request,
                    loaded
                ),
            )
        })
        .collect();
    fallback_lines.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));

    (
        route_lines
            .into_iter()
            .take(8)
            .map(|(_, line)| line)
            .collect(),
        fallback_lines
            .into_iter()
            .take(5)
            .map(|(_, line)| line)
            .collect(),
    )
}

/// Builds warning lines for native UI draws (missing textures, unmapped property ids, unsupported state).
pub fn summarize_shader_warnings(
    draw_batches: &[crate::render::SpaceDrawBatch],
    registry: &crate::assets::AssetRegistry,
    render_config: &crate::config::RenderConfig,
    gpu: &crate::gpu::GpuState,
) -> Vec<String> {
    use crate::assets::{
        MaterialPropertyLookupIds, MaterialPropertyStore, MaterialPropertyValue,
        texture2d_asset_id_from_packed, ui_text_unlit_material_uniform, ui_unlit_material_uniform,
        unpack_host_texture_packed,
    };
    use crate::gpu::PipelineVariant;

    fn has_float_flag(
        store: &MaterialPropertyStore,
        lookup: MaterialPropertyLookupIds,
        pid: i32,
    ) -> bool {
        if pid < 0 {
            return false;
        }
        matches!(
            store.get_merged(lookup, pid),
            Some(MaterialPropertyValue::Float(v)) if *v >= 0.5
        )
    }

    fn float_value(
        store: &MaterialPropertyStore,
        lookup: MaterialPropertyLookupIds,
        pid: i32,
    ) -> Option<f32> {
        if pid < 0 {
            return None;
        }
        match store.get_merged(lookup, pid) {
            Some(MaterialPropertyValue::Float(v)) => Some(*v),
            _ => None,
        }
    }

    fn packed_texture_label(packed: i32) -> String {
        if packed == 0 {
            return "packed=0 (none)".to_string();
        }
        if packed < 0 {
            return format!("packed={} (negative / invalid)", packed);
        }
        match unpack_host_texture_packed(packed) {
            Some((asset_id, kind)) => format!("packed={} ({:?} id={})", packed, kind, asset_id),
            None => format!("packed={} (unpack failed)", packed),
        }
    }

    let store = &registry.material_property_store;
    let mut warnings: HashMap<(String, String), ShaderHudAccum> = HashMap::new();

    for batch in draw_batches {
        for draw in &batch.draws {
            let request = shader_request_label(registry, draw.shader_key.host_shader_asset_id);
            let lookup = MaterialPropertyLookupIds {
                material_asset_id: draw.material_id,
                mesh_property_block_slot0: draw.mesh_renderer_property_block_slot0_id,
            };

            match draw.pipeline_variant {
                PipelineVariant::NativeUiUnlit { .. }
                | PipelineVariant::NativeUiUnlitStencil { .. } => {
                    let ids = &render_config.ui_unlit_property_ids;
                    let (uniform, main_packed, mask_packed) =
                        ui_unlit_material_uniform(store, lookup, ids);

                    if ids.main_tex < 0 {
                        accumulate_shader_hud_line(
                            &mut warnings,
                            request.clone(),
                            format!(
                                "_MainTex property id unmapped (material={} block={:?})",
                                draw.material_id, draw.mesh_renderer_property_block_slot0_id
                            ),
                            draw.mesh_asset_id,
                        );
                    } else if main_packed == 0 {
                        accumulate_shader_hud_line(
                            &mut warnings,
                            request.clone(),
                            format!(
                                "_MainTex missing on material/property block (material={} block={:?} pid={} {})",
                                draw.material_id,
                                draw.mesh_renderer_property_block_slot0_id,
                                ids.main_tex,
                                packed_texture_label(main_packed)
                            ),
                            draw.mesh_asset_id,
                        );
                    } else if let Some(texture_id) = texture2d_asset_id_from_packed(main_packed) {
                        if registry.get_texture(texture_id).is_none() {
                            accumulate_shader_hud_line(
                                &mut warnings,
                                request.clone(),
                                format!(
                                    "_MainTex asset {} missing from registry (material={} block={:?} pid={} {})",
                                    texture_id,
                                    draw.material_id,
                                    draw.mesh_renderer_property_block_slot0_id,
                                    ids.main_tex,
                                    packed_texture_label(main_packed)
                                ),
                                draw.mesh_asset_id,
                            );
                        } else if !gpu.texture2d_gpu.contains_key(&texture_id) {
                            accumulate_shader_hud_line(
                                &mut warnings,
                                request.clone(),
                                format!(
                                    "_MainTex asset {} not GPU-resident (material={} block={:?} pid={} {})",
                                    texture_id,
                                    draw.material_id,
                                    draw.mesh_renderer_property_block_slot0_id,
                                    ids.main_tex,
                                    packed_texture_label(main_packed)
                                ),
                                draw.mesh_asset_id,
                            );
                        }
                    } else if main_packed > 0 {
                        accumulate_shader_hud_line(
                            &mut warnings,
                            request.clone(),
                            format!(
                                "_MainTex is not a Texture2D (material={} block={:?} pid={} {})",
                                draw.material_id,
                                draw.mesh_renderer_property_block_slot0_id,
                                ids.main_tex,
                                packed_texture_label(main_packed)
                            ),
                            draw.mesh_asset_id,
                        );
                    }

                    let mask_requested =
                        (uniform.flags & (32 | 64)) != 0 || ids.mask_tex >= 0 || mask_packed != 0;
                    if mask_requested {
                        if ids.mask_tex < 0 {
                            accumulate_shader_hud_line(
                                &mut warnings,
                                request.clone(),
                                format!(
                                    "_MaskTex property id unmapped (material={} block={:?})",
                                    draw.material_id, draw.mesh_renderer_property_block_slot0_id
                                ),
                                draw.mesh_asset_id,
                            );
                        } else if let Some(texture_id) = texture2d_asset_id_from_packed(mask_packed)
                        {
                            if registry.get_texture(texture_id).is_none() {
                                accumulate_shader_hud_line(
                                    &mut warnings,
                                    request.clone(),
                                    format!(
                                        "_MaskTex asset {} missing from registry (material={} block={:?} pid={} {})",
                                        texture_id,
                                        draw.material_id,
                                        draw.mesh_renderer_property_block_slot0_id,
                                        ids.mask_tex,
                                        packed_texture_label(mask_packed)
                                    ),
                                    draw.mesh_asset_id,
                                );
                            } else if !gpu.texture2d_gpu.contains_key(&texture_id) {
                                accumulate_shader_hud_line(
                                    &mut warnings,
                                    request.clone(),
                                    format!(
                                        "_MaskTex asset {} not GPU-resident (material={} block={:?} pid={} {})",
                                        texture_id,
                                        draw.material_id,
                                        draw.mesh_renderer_property_block_slot0_id,
                                        ids.mask_tex,
                                        packed_texture_label(mask_packed)
                                    ),
                                    draw.mesh_asset_id,
                                );
                            }
                        } else if mask_packed > 0 {
                            accumulate_shader_hud_line(
                                &mut warnings,
                                request.clone(),
                                format!(
                                    "_MaskTex is not a Texture2D (material={} block={:?} pid={} {})",
                                    draw.material_id,
                                    draw.mesh_renderer_property_block_slot0_id,
                                    ids.mask_tex,
                                    packed_texture_label(mask_packed)
                                ),
                                draw.mesh_asset_id,
                            );
                        }
                    }

                    let unsupported_state_checks = [
                        ("_ZWrite", float_value(store, lookup, ids.zwrite), 1.0f32),
                        ("_Cull", float_value(store, lookup, ids.cull), 2.0f32),
                        (
                            "_ColorMask",
                            float_value(store, lookup, ids.color_mask),
                            15.0f32,
                        ),
                        (
                            "_StencilComp",
                            float_value(store, lookup, ids.stencil_comp),
                            8.0f32,
                        ),
                        (
                            "_Stencil",
                            float_value(store, lookup, ids.stencil_ref),
                            0.0f32,
                        ),
                        (
                            "_StencilOp",
                            float_value(store, lookup, ids.stencil_op),
                            0.0f32,
                        ),
                        (
                            "_StencilWriteMask",
                            float_value(store, lookup, ids.stencil_write_mask),
                            255.0f32,
                        ),
                        (
                            "_StencilReadMask",
                            float_value(store, lookup, ids.stencil_read_mask),
                            255.0f32,
                        ),
                        (
                            "_OffsetFactor",
                            float_value(store, lookup, ids.offset_factor),
                            0.0f32,
                        ),
                        (
                            "_OffsetUnits",
                            float_value(store, lookup, ids.offset_units),
                            0.0f32,
                        ),
                    ];
                    for (label, value, unity_default) in unsupported_state_checks {
                        if let Some(v) = value
                            && (v - unity_default).abs() > 0.01
                        {
                            accumulate_shader_hud_line(
                                &mut warnings,
                                request.clone(),
                                format!(
                                    "{}={} is set but current UI pipeline does not honor per-material state yet",
                                    label, v
                                ),
                                draw.mesh_asset_id,
                            );
                        }
                    }
                }
                PipelineVariant::NativeUiTextUnlit { .. }
                | PipelineVariant::NativeUiTextUnlitStencil { .. } => {
                    let ids = &render_config.ui_text_unlit_property_ids;
                    let (_uniform, atlas_packed) =
                        ui_text_unlit_material_uniform(store, lookup, ids);

                    if ids.font_atlas < 0 {
                        accumulate_shader_hud_line(
                            &mut warnings,
                            request.clone(),
                            format!(
                                "_FontAtlas property id unmapped (material={} block={:?})",
                                draw.material_id, draw.mesh_renderer_property_block_slot0_id
                            ),
                            draw.mesh_asset_id,
                        );
                    } else if atlas_packed == 0 {
                        accumulate_shader_hud_line(
                            &mut warnings,
                            request.clone(),
                            format!(
                                "_FontAtlas missing on material/property block (material={} block={:?} pid={} {})",
                                draw.material_id,
                                draw.mesh_renderer_property_block_slot0_id,
                                ids.font_atlas,
                                packed_texture_label(atlas_packed)
                            ),
                            draw.mesh_asset_id,
                        );
                    } else if let Some(texture_id) = texture2d_asset_id_from_packed(atlas_packed) {
                        if registry.get_texture(texture_id).is_none() {
                            accumulate_shader_hud_line(
                                &mut warnings,
                                request.clone(),
                                format!(
                                    "_FontAtlas asset {} missing from registry (material={} block={:?} pid={} {})",
                                    texture_id,
                                    draw.material_id,
                                    draw.mesh_renderer_property_block_slot0_id,
                                    ids.font_atlas,
                                    packed_texture_label(atlas_packed)
                                ),
                                draw.mesh_asset_id,
                            );
                        } else if !gpu.texture2d_gpu.contains_key(&texture_id) {
                            accumulate_shader_hud_line(
                                &mut warnings,
                                request.clone(),
                                format!(
                                    "_FontAtlas asset {} not GPU-resident (material={} block={:?} pid={} {})",
                                    texture_id,
                                    draw.material_id,
                                    draw.mesh_renderer_property_block_slot0_id,
                                    ids.font_atlas,
                                    packed_texture_label(atlas_packed)
                                ),
                                draw.mesh_asset_id,
                            );
                        }
                    } else if atlas_packed > 0 {
                        accumulate_shader_hud_line(
                            &mut warnings,
                            request.clone(),
                            format!(
                                "_FontAtlas is not a Texture2D (material={} block={:?} pid={} {})",
                                draw.material_id,
                                draw.mesh_renderer_property_block_slot0_id,
                                ids.font_atlas,
                                packed_texture_label(atlas_packed)
                            ),
                            draw.mesh_asset_id,
                        );
                    }

                    let outline_requested = has_float_flag(store, lookup, ids.outline);
                    if outline_requested && ids.outline_color < 0 {
                        accumulate_shader_hud_line(
                            &mut warnings,
                            request.clone(),
                            "_OutlineColor property id unmapped while OUTLINE is enabled"
                                .to_string(),
                            draw.mesh_asset_id,
                        );
                    }
                }
                _ => {}
            }
        }
    }

    let mut warning_lines: Vec<_> = warnings
        .into_iter()
        .map(|((request, message), accum)| {
            (
                accum.draws,
                format!(
                    "draws {:>4}  meshes {:>4}  |  {}  |  {}",
                    accum.draws,
                    accum.meshes.len(),
                    request,
                    message
                ),
            )
        })
        .collect();
    warning_lines.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    warning_lines
        .into_iter()
        .take(8)
        .map(|(_, line)| line)
        .collect()
}
