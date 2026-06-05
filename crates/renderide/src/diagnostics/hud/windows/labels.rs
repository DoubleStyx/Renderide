//! Label strings and draw-row predicates for debug HUD tables.

use crate::materials::{MaterialBlendMode, MaterialDepthCompareOverride, RasterPipelineKind};
use crate::world_mesh::{TransparentMaterialClass, WorldMeshDrawStateRow};

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
        RasterPipelineKind::Null => "null".to_string(),
    }
}

pub(super) fn draw_state_is_uiish(row: &WorldMeshDrawStateRow) -> bool {
    row.is_overlay
        || row.alpha_blended
        || row.transparent_class.is_transparent()
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

/// Returns the compact HUD label for a transparent material class.
pub(super) fn transparent_class_label(class: TransparentMaterialClass) -> &'static str {
    class.label()
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

pub(super) fn ztest_label(value: Option<MaterialDepthCompareOverride>) -> &'static str {
    match value {
        Some(MaterialDepthCompareOverride::HostValue(0)) => "off",
        Some(MaterialDepthCompareOverride::HostValue(1)) => "never",
        Some(MaterialDepthCompareOverride::HostValue(2)) => "less",
        Some(MaterialDepthCompareOverride::HostValue(3)) => "equal",
        Some(MaterialDepthCompareOverride::HostValue(4)) => "lequal",
        Some(MaterialDepthCompareOverride::HostValue(5)) => "greater",
        Some(MaterialDepthCompareOverride::HostValue(6)) => "not-equal",
        Some(MaterialDepthCompareOverride::HostValue(7)) => "gequal",
        Some(MaterialDepthCompareOverride::HostValue(8)) => "always",
        Some(MaterialDepthCompareOverride::HostValue(_)) => "invalid",
        None => "pass",
    }
}

pub(super) fn offset_label(value: Option<(u32, i32)>) -> String {
    match value {
        Some((factor_bits, units)) => format!("{:.3}/{}", f32::from_bits(factor_bits), units),
        None => "pass".to_string(),
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for compact HUD labels.

    use super::*;

    /// Host `_ZTest` values keep their user-facing labels.
    #[test]
    fn ztest_label_keeps_host_value_labels() {
        assert_eq!(
            ztest_label(Some(MaterialDepthCompareOverride::HostValue(6))),
            "not-equal"
        );
        assert_eq!(
            ztest_label(Some(MaterialDepthCompareOverride::HostValue(8))),
            "always"
        );
    }

    /// Missing pass overrides keep their fallback label.
    #[test]
    fn ztest_label_shows_missing_override_as_pass() {
        assert_eq!(ztest_label(None), "pass");
    }
}
