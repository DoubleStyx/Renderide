//! Frame-accumulating counters for [`super::collect::apply_native_ui_pipeline_variant`] and PBR UI-vert fallback.
//!
//! Enabled with [`crate::config::RenderConfig::native_ui_routing_metrics`]; read via
//! [`NativeUiRoutingFrameMetrics::snapshot_and_reset`] when building [`crate::diagnostics::LiveFrameDiagnostics`].

use std::sync::atomic::{AtomicU64, Ordering};

static SKIP_NATIVE_UI_OFF: AtomicU64 = AtomicU64::new(0);
static SKIP_SHADER_DEBUG_LEGACY: AtomicU64 = AtomicU64::new(0);
static SKIP_SKINNED: AtomicU64 = AtomicU64::new(0);
static SKIP_BAD_MATERIAL_BLOCK: AtomicU64 = AtomicU64::new(0);
static SKIP_NO_SURFACE: AtomicU64 = AtomicU64::new(0);
static SKIP_STENCIL_WORLD: AtomicU64 = AtomicU64::new(0);
static SKIP_STENCIL_PIPELINES_OFF: AtomicU64 = AtomicU64::new(0);
static SKIP_NO_HOST_SHADER: AtomicU64 = AtomicU64::new(0);
static SKIP_MESH_NO_UV0: AtomicU64 = AtomicU64::new(0);
static SKIP_UNRECOGNIZED_SHADER: AtomicU64 = AtomicU64::new(0);
static ROUTED_UI_UNLIT: AtomicU64 = AtomicU64::new(0);
static ROUTED_UI_UNLIT_STENCIL: AtomicU64 = AtomicU64::new(0);
static ROUTED_UI_TEXT: AtomicU64 = AtomicU64::new(0);
static ROUTED_UI_TEXT_STENCIL: AtomicU64 = AtomicU64::new(0);
static PBR_UIVERT_FALLBACK: AtomicU64 = AtomicU64::new(0);

/// Per-frame native UI routing totals (sum over all drawable resolutions in the last frame).
#[derive(Clone, Copy, Debug, Default)]
pub struct NativeUiRoutingFrameMetrics {
    pub skip_native_ui_wgsl_off: u64,
    pub skip_shader_debug_force_legacy: u64,
    pub skip_skinned: u64,
    pub skip_bad_material_block: u64,
    pub skip_no_surface: u64,
    pub skip_stencil_on_world_mesh: u64,
    pub skip_stencil_pipelines_disabled: u64,
    pub skip_no_host_shader: u64,
    pub skip_mesh_no_uv0: u64,
    pub skip_unrecognized_shader: u64,
    pub routed_ui_unlit: u64,
    pub routed_ui_unlit_stencil: u64,
    pub routed_ui_text_unlit: u64,
    pub routed_ui_text_unlit_stencil: u64,
    pub pbr_uivert_fallback: u64,
}

impl NativeUiRoutingFrameMetrics {
    /// Reads all counters and clears them for the next frame.
    pub fn snapshot_and_reset() -> Self {
        Self {
            skip_native_ui_wgsl_off: SKIP_NATIVE_UI_OFF.swap(0, Ordering::Relaxed),
            skip_shader_debug_force_legacy: SKIP_SHADER_DEBUG_LEGACY.swap(0, Ordering::Relaxed),
            skip_skinned: SKIP_SKINNED.swap(0, Ordering::Relaxed),
            skip_bad_material_block: SKIP_BAD_MATERIAL_BLOCK.swap(0, Ordering::Relaxed),
            skip_no_surface: SKIP_NO_SURFACE.swap(0, Ordering::Relaxed),
            skip_stencil_on_world_mesh: SKIP_STENCIL_WORLD.swap(0, Ordering::Relaxed),
            skip_stencil_pipelines_disabled: SKIP_STENCIL_PIPELINES_OFF.swap(0, Ordering::Relaxed),
            skip_no_host_shader: SKIP_NO_HOST_SHADER.swap(0, Ordering::Relaxed),
            skip_mesh_no_uv0: SKIP_MESH_NO_UV0.swap(0, Ordering::Relaxed),
            skip_unrecognized_shader: SKIP_UNRECOGNIZED_SHADER.swap(0, Ordering::Relaxed),
            routed_ui_unlit: ROUTED_UI_UNLIT.swap(0, Ordering::Relaxed),
            routed_ui_unlit_stencil: ROUTED_UI_UNLIT_STENCIL.swap(0, Ordering::Relaxed),
            routed_ui_text_unlit: ROUTED_UI_TEXT.swap(0, Ordering::Relaxed),
            routed_ui_text_unlit_stencil: ROUTED_UI_TEXT_STENCIL.swap(0, Ordering::Relaxed),
            pbr_uivert_fallback: PBR_UIVERT_FALLBACK.swap(0, Ordering::Relaxed),
        }
    }

    /// Sum of all native UI pipeline selections (non-stencil + stencil).
    pub fn routed_native_total(self) -> u64 {
        self.routed_ui_unlit
            + self.routed_ui_unlit_stencil
            + self.routed_ui_text_unlit
            + self.routed_ui_text_unlit_stencil
    }

    /// Sum of skip counters (approximate drawable-level misses for the strangler branch).
    pub fn skips_total(self) -> u64 {
        self.skip_native_ui_wgsl_off
            + self.skip_shader_debug_force_legacy
            + self.skip_skinned
            + self.skip_bad_material_block
            + self.skip_no_surface
            + self.skip_stencil_on_world_mesh
            + self.skip_stencil_pipelines_disabled
            + self.skip_no_host_shader
            + self.skip_mesh_no_uv0
            + self.skip_unrecognized_shader
    }
}

/// Records one native UI routing skip when metrics are enabled.
pub(crate) fn record_native_ui_skip(metrics: bool, kind: NativeUiSkipKind) {
    if !metrics {
        return;
    }
    let ctr = match kind {
        NativeUiSkipKind::NativeUiWgslOff => &SKIP_NATIVE_UI_OFF,
        NativeUiSkipKind::ShaderDebugForceLegacy => &SKIP_SHADER_DEBUG_LEGACY,
        NativeUiSkipKind::Skinned => &SKIP_SKINNED,
        NativeUiSkipKind::BadMaterialBlock => &SKIP_BAD_MATERIAL_BLOCK,
        NativeUiSkipKind::NoSurface => &SKIP_NO_SURFACE,
        NativeUiSkipKind::StencilOnWorldMesh => &SKIP_STENCIL_WORLD,
        NativeUiSkipKind::StencilPipelinesOff => &SKIP_STENCIL_PIPELINES_OFF,
        NativeUiSkipKind::NoHostShader => &SKIP_NO_HOST_SHADER,
        NativeUiSkipKind::MeshNoUv0 => &SKIP_MESH_NO_UV0,
        NativeUiSkipKind::UnrecognizedShader => &SKIP_UNRECOGNIZED_SHADER,
    };
    ctr.fetch_add(1, Ordering::Relaxed);
}

/// Records a successful route to a native UI pipeline variant.
pub(crate) fn record_native_ui_routed(metrics: bool, family: NativeUiRoutedFamily) {
    if !metrics {
        return;
    }
    let ctr = match family {
        NativeUiRoutedFamily::UiUnlit => &ROUTED_UI_UNLIT,
        NativeUiRoutedFamily::UiUnlitStencil => &ROUTED_UI_UNLIT_STENCIL,
        NativeUiRoutedFamily::UiTextUnlit => &ROUTED_UI_TEXT,
        NativeUiRoutedFamily::UiTextUnlitStencil => &ROUTED_UI_TEXT_STENCIL,
    };
    ctr.fetch_add(1, Ordering::Relaxed);
}

/// Records that [`super::collect::apply_ui_mesh_pbr_fallback_for_non_native_shader`] chose [`crate::gpu::PipelineVariant::Pbr`].
pub(crate) fn record_pbr_uivert_fallback(metrics: bool) {
    if metrics {
        PBR_UIVERT_FALLBACK.fetch_add(1, Ordering::Relaxed);
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum NativeUiSkipKind {
    NativeUiWgslOff,
    ShaderDebugForceLegacy,
    Skinned,
    BadMaterialBlock,
    NoSurface,
    StencilOnWorldMesh,
    StencilPipelinesOff,
    NoHostShader,
    MeshNoUv0,
    UnrecognizedShader,
}

#[derive(Clone, Copy, Debug)]
#[allow(clippy::enum_variant_names)]
pub(crate) enum NativeUiRoutedFamily {
    UiUnlit,
    UiUnlitStencil,
    UiTextUnlit,
    UiTextUnlitStencil,
}
