//! Label strings and draw-row predicates for debug HUD tables.

use crate::materials::{MaterialBlendMode, RasterPipelineKind};
use crate::render_graph::WorldMeshDrawStateRow;

pub(super) fn device_type_label(kind: wgpu::DeviceType) -> &'static str {
    match kind {
        wgpu::DeviceType::Other => "other / unknown",
        wgpu::DeviceType::IntegratedGpu => "integrated GPU",
        wgpu::DeviceType::DiscreteGpu => "discrete GPU",
        wgpu::DeviceType::VirtualGpu => "virtual GPU",
        wgpu::DeviceType::Cpu => "software / CPU",
    }
}

pub(super) fn pipeline_label(pipeline: &RasterPipelineKind) -> String {
    match pipeline {
        RasterPipelineKind::EmbeddedStem(stem) => stem.to_string(),
        RasterPipelineKind::DebugWorldNormals => "debug_world_normals".to_string(),
    }
}

pub(super) fn draw_state_is_uiish(row: &WorldMeshDrawStateRow) -> bool {
    row.is_overlay
        || row.alpha_blended
        || matches!(
            &row.pipeline,
            RasterPipelineKind::EmbeddedStem(stem)
                if stem.starts_with("ui_")
                    || stem.contains("text")
                    || stem.contains("overlay")
        )
}

pub(super) fn draw_state_has_override(row: &WorldMeshDrawStateRow) -> bool {
    row.depth_write.is_some()
        || row.depth_compare.is_some()
        || row.depth_offset.is_some()
        || row.color_mask.is_some()
        || row.stencil_enabled
}

pub(super) fn blend_mode_label(mode: MaterialBlendMode) -> String {
    match mode {
        MaterialBlendMode::StemDefault => "stem".to_string(),
        MaterialBlendMode::Opaque => "opaque".to_string(),
        MaterialBlendMode::UnityBlend { src, dst } => format!("unity {src}/{dst}"),
    }
}

pub(super) fn color_mask_label(mask: Option<u8>) -> String {
    let Some(mask) = mask else {
        return "pass".to_string();
    };
    let mut out = String::new();
    if mask & 8 != 0 {
        out.push('R');
    }
    if mask & 4 != 0 {
        out.push('G');
    }
    if mask & 2 != 0 {
        out.push('B');
    }
    if mask & 1 != 0 {
        out.push('A');
    }
    if out.is_empty() {
        "none".to_string()
    } else {
        out
    }
}

pub(super) fn stencil_label(row: &WorldMeshDrawStateRow) -> String {
    if !row.stencil_enabled {
        return "off".to_string();
    }
    format!(
        "ref={} cmp={} pass={} read=0x{:02x} write=0x{:02x}",
        row.stencil_reference,
        row.stencil_compare,
        row.stencil_pass_op,
        row.stencil_read_mask,
        row.stencil_write_mask
    )
}

pub(super) fn ztest_label(value: Option<u8>) -> &'static str {
    match value {
        Some(0) => "off",
        Some(1) => "never",
        Some(2) => "less",
        Some(3) => "equal",
        Some(4) => "lequal",
        Some(5) => "greater",
        Some(6) => "not-equal",
        Some(7) => "gequal",
        Some(8) => "always",
        Some(_) => "invalid",
        None => "pass",
    }
}

pub(super) fn offset_label(value: Option<(u32, i32)>) -> String {
    match value {
        Some((factor_bits, units)) => format!("{:.3}/{}", f32::from_bits(factor_bits), units),
        None => "pass".to_string(),
    }
}
