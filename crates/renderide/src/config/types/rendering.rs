//! Rendering toggles, MSAA, vsync, scene-color format, recording parallelism. Persisted as `[rendering]`.

use serde::de::{self, Deserializer, Visitor};
use serde::{Deserialize, Serialize};
use wgpu::TextureFormat;

/// Rendering toggles and scalars. Persisted as `[rendering]`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RenderingSettings {
    /// Swapchain vsync mode ([`VsyncMode::Off`] / [`VsyncMode::On`] / [`VsyncMode::Auto`]);
    /// applied live without restart through [`crate::gpu::GpuContext::set_present_mode`]. Old
    /// `vsync = true/false` and `vsync = "adaptive"` configs still load (a custom deserializer
    /// maps the booleans to [`VsyncMode::On`] / [`VsyncMode::Off`] and the historical `"adaptive"`
    /// token to [`VsyncMode::Auto`]).
    pub vsync: VsyncMode,
    /// Wall-clock budget per frame for cooperative mesh/texture integration ([`crate::runtime::RendererRuntime::run_asset_integration`]), in milliseconds.
    #[serde(rename = "asset_integration_budget_ms")]
    pub asset_integration_budget_ms: u32,
    /// Upper bound for [`crate::shared::RendererInitResult::max_texture_size`] sent to the host.
    /// `0` means use the GPU’s [`wgpu::Limits::max_texture_dimension_2d`] (after device creation).
    /// Non-zero values are clamped to the GPU maximum.
    #[serde(rename = "reported_max_texture_size")]
    pub reported_max_texture_size: u32,
    /// When `true`, host [`crate::shared::SetRenderTextureFormat`] assets allocate **HDR** color
    /// (`Rgba16Float`, Unity `ARGBHalf` parity). When `false` (default), **`Rgba8Unorm`** is used to
    /// reduce VRAM for typical LDR render targets (mirrors, cameras, UI).
    #[serde(rename = "render_texture_hdr_color")]
    pub render_texture_hdr_color: bool,
    /// When non-zero, logs a **warning** when combined resident Texture2D + render-texture bytes exceed
    /// this many mebibytes (best-effort accounting).
    #[serde(rename = "texture_vram_budget_mib")]
    pub texture_vram_budget_mib: u32,
    /// Multisample anti-aliasing for the main window forward path (clustered forward). Effective sample
    /// count is clamped to the GPU’s supported maximum for the swapchain format. VR and offscreen host
    /// render textures stay at 1× until extended separately.
    pub msaa: MsaaSampleCount,
    /// Format for the **scene-color** HDR target the forward pass renders into before
    /// [`crate::render_graph::passes::SceneColorComposePass`] writes the displayable target.
    ///
    /// This is intermediate precision/range (e.g. [`SceneColorFormat::Rgba16Float`]), not the OS
    /// swapchain HDR mode.
    #[serde(rename = "scene_color_format")]
    pub scene_color_format: SceneColorFormat,
    /// Whether to record per-view encoders in parallel using rayon.
    ///
    /// [`RecordParallelism::PerViewParallel`] (default) records views on rayon worker threads
    /// for a CPU-side speedup on multi-view workloads (stereo VR, secondary-camera RTs).
    /// [`RecordParallelism::Serial`] records views sequentially on the main thread, which can
    /// simplify debugging but leaves throughput on the table on multi-view scenes. Requires
    /// all per-view pass nodes to be `Send` (enforced by trait bounds).
    #[serde(rename = "record_parallelism", default)]
    pub record_parallelism: RecordParallelism,
    /// Clustered-light assignment backend used before the world forward pass.
    ///
    /// [`ClusterAssignmentMode::GpuScan`] preserves the original compute path that scans every
    /// light for every froxel. [`ClusterAssignmentMode::CpuFroxel`] uses a light-centric CPU
    /// assignment path where it is safe for the shared cluster buffers.
    /// [`ClusterAssignmentMode::Auto`] keeps the compute path for ordinary scenes and enables the
    /// CPU path for the first stereo view when the light count is high enough to justify it.
    #[serde(rename = "cluster_assignment", default)]
    pub cluster_assignment: ClusterAssignmentMode,
}

impl Default for RenderingSettings {
    fn default() -> Self {
        Self {
            vsync: VsyncMode::default(),
            asset_integration_budget_ms: 3,
            reported_max_texture_size: 0,
            render_texture_hdr_color: false,
            texture_vram_budget_mib: 0,
            msaa: MsaaSampleCount::default(),
            scene_color_format: SceneColorFormat::default(),
            record_parallelism: RecordParallelism::default(),
            cluster_assignment: ClusterAssignmentMode::default(),
        }
    }
}

/// Swapchain vsync mode persisted in `config.toml` as `[rendering] vsync`.
///
/// Three values matching what desktop and VR titles typically expose: **Off** (tearing, lowest
/// latency), **On** (no tearing, low latency — prefers `Mailbox` over `Fifo`), **Auto** (vsync
/// when the renderer hits the deadline, tear instead of stutter when it misses — `FifoRelaxed`).
/// Defaults to [`Self::Off`].
///
/// Resolution to a [`wgpu::PresentMode`] happens in [`VsyncMode::resolve_present_mode`], which
/// probes the surface's actual capabilities rather than trusting wgpu's `Auto*` shortcuts (those
/// always pick `Fifo` for vsync-on, leaving no-tearing behind a deeper compositor queue than
/// necessary).
///
/// A custom [`Deserialize`] also accepts the historical `vsync = true / false` booleans and the
/// pre-rename `vsync = "adaptive"` token so older `config.toml` files keep loading without
/// manual migration.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VsyncMode {
    /// No vsync. Lowest latency, may tear; CPU/GPU run uncapped. Resolves to `Immediate` when
    /// the surface advertises it, otherwise falls through `Mailbox` and finally `Fifo`.
    #[default]
    Off,
    /// Vsync without tearing. Resolves to `Mailbox` when the surface advertises it (low-latency
    /// no-tear presentation), otherwise falls back to `Fifo`. Prefer this over the deprecated
    /// `wgpu::PresentMode::AutoVsync` mapping which always picks `Fifo`.
    On,
    /// Adaptive vsync. Resolves to `FifoRelaxed` when supported (vsync until a frame misses its
    /// deadline, then tears once instead of waiting another full vblank), otherwise falls back
    /// to `Fifo`. Renamed from the historical `Adaptive`; the custom [`Deserialize`] still
    /// accepts `vsync = "adaptive"` (and `"fifo_relaxed"` / `"relaxed"`) for backward compat.
    Auto,
}

impl VsyncMode {
    /// All variants for ImGui pickers and config round-trips.
    pub const ALL: [Self; 3] = [Self::Off, Self::On, Self::Auto];

    /// Short label for the renderer config window.
    pub fn label(self) -> &'static str {
        match self {
            Self::Off => "Off",
            Self::On => "On",
            Self::Auto => "Auto",
        }
    }

    /// Resolves this mode to a [`wgpu::PresentMode`] that the surface actually supports, using
    /// explicit low-latency preference chains rather than wgpu's lazy `Auto*` shortcuts.
    ///
    /// Each variant walks an ordered preference list and picks the first entry present in
    /// `supported` ([`wgpu::SurfaceCapabilities::present_modes`]). [`wgpu::PresentMode::Fifo`] is
    /// required to be supported by every conformant surface ([wgpu spec][1]), so the chain
    /// always terminates.
    ///
    /// | Variant            | Preference order                            | Behavior                                                          |
    /// | ------------------ | ------------------------------------------- | ----------------------------------------------------------------- |
    /// | [`Self::Off`]      | `Immediate` → `Mailbox` → `Fifo`            | Lowest latency; tears                                             |
    /// | [`Self::On`]       | `Mailbox` → `Fifo`                          | No-tear vsync without the FIFO queue depth                        |
    /// | [`Self::Auto`]     | `FifoRelaxed` → `Fifo`                      | Vsync until a frame misses; then tear once instead of half-rate   |
    ///
    /// Unlike `wgpu::PresentMode::AutoVsync` (which always resolves to plain `Fifo`) the [`Self::On`]
    /// arm probes for `Mailbox` first, which avoids the extra queueing on desktop backends that
    /// expose it while retaining a mandatory `Fifo` fallback.
    ///
    /// [1]: https://www.w3.org/TR/webgpu/#dom-gpupresentmode-fifo
    pub fn resolve_present_mode(self, supported: &[wgpu::PresentMode]) -> wgpu::PresentMode {
        use wgpu::PresentMode::*;
        match self {
            Self::Off => first_supported_present_mode(&[Immediate, Mailbox, Fifo], supported),
            Self::On => first_supported_present_mode(&[Mailbox, Fifo], supported),
            Self::Auto => first_supported_present_mode(&[FifoRelaxed, Fifo], supported),
        }
    }
}

/// Walks `preferred` in order and returns the first variant present in `supported`, falling back
/// to [`wgpu::PresentMode::Fifo`] when nothing matches.
///
/// `Fifo` is the unconditional fallback because every conformant surface advertises it; see
/// [`VsyncMode::resolve_present_mode`] for the per-mode preference chains that route through here.
fn first_supported_present_mode(
    preferred: &[wgpu::PresentMode],
    supported: &[wgpu::PresentMode],
) -> wgpu::PresentMode {
    preferred
        .iter()
        .copied()
        .find(|m| supported.contains(m))
        .unwrap_or(wgpu::PresentMode::Fifo)
}

impl<'de> Deserialize<'de> for VsyncMode {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct VsyncModeVisitor;

        impl Visitor<'_> for VsyncModeVisitor {
            type Value = VsyncMode;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a vsync mode (`off` / `on` / `auto`) or a boolean")
            }

            fn visit_bool<E: de::Error>(self, v: bool) -> Result<Self::Value, E> {
                Ok(if v { VsyncMode::On } else { VsyncMode::Off })
            }

            fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
                match v.trim().to_ascii_lowercase().as_str() {
                    "off" | "false" | "0" | "no" | "none" => Ok(VsyncMode::Off),
                    "on" | "true" | "1" | "yes" | "vsync" | "fifo" => Ok(VsyncMode::On),
                    "auto" | "adaptive" | "fifo_relaxed" | "fiforelaxed" | "relaxed" => {
                        Ok(VsyncMode::Auto)
                    }
                    other => Err(E::custom(format!(
                        "unknown vsync mode `{other}`; expected `off`, `on`, or `auto`"
                    ))),
                }
            }

            fn visit_string<E: de::Error>(self, v: String) -> Result<Self::Value, E> {
                self.visit_str(&v)
            }
        }

        deserializer.deserialize_any(VsyncModeVisitor)
    }
}

/// Controls whether per-view encoder recording uses rayon for parallelism.
///
/// The default [`RecordParallelism::PerViewParallel`] records per-view encoders on rayon
/// workers for CPU-side speedup on stereo / multi-camera scenes. Switch to
/// [`RecordParallelism::Serial`] only for debugging or when isolating regressions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecordParallelism {
    /// Record each per-view encoder sequentially on the main thread. Safe and debuggable.
    Serial,
    /// Record each per-view encoder on a rayon worker thread. Requires all per-view pass nodes
    /// to be `Send` (enforced at compile time by the trait bound on [`crate::render_graph::PassNode`]).
    #[default]
    PerViewParallel,
}

/// Clustered-light assignment backend for the per-view light list pass.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClusterAssignmentMode {
    /// Select the safest fast path for the current view. Currently uses CPU froxel assignment for
    /// the first stereo view only when the scene has many lights, otherwise falls back to GPU scan.
    #[default]
    Auto,
    /// Preserve the original compute shader path: one compute thread per froxel scans all lights.
    GpuScan,
    /// Use CPU light-centric froxel assignment where the shared cluster-buffer ordering permits it.
    CpuFroxel,
}

impl ClusterAssignmentMode {
    /// All variants for ImGui pickers and config round-trips.
    pub const ALL: [Self; 3] = [Self::Auto, Self::GpuScan, Self::CpuFroxel];

    /// Short label for the renderer config window.
    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "Auto",
            Self::GpuScan => "GPU scan",
            Self::CpuFroxel => "CPU froxel",
        }
    }
}

/// Intermediate scene color format for the forward pass (pre-compose, pre-post-processing).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SceneColorFormat {
    /// `rgba16float`: wide dynamic range and alpha (default HDR scene target).
    #[default]
    Rgba16Float,
    /// `rg11b10float`: lower bandwidth; no distinct alpha channel (avoid with premultiplied transparency).
    Rg11b10Float,
    /// `rgba8unorm`: LDR scene color (debug / parity).
    Rgba8Unorm,
}

impl SceneColorFormat {
    /// All variants for config UI and persistence.
    pub const ALL: [Self; 3] = [Self::Rgba16Float, Self::Rg11b10Float, Self::Rgba8Unorm];

    /// [`wgpu::TextureFormat`] for graph transients and forward color attachments.
    pub fn wgpu_format(self) -> TextureFormat {
        match self {
            Self::Rgba16Float => TextureFormat::Rgba16Float,
            Self::Rg11b10Float => TextureFormat::Rg11b10Ufloat,
            Self::Rgba8Unorm => TextureFormat::Rgba8Unorm,
        }
    }

    /// Short label for the renderer config window.
    pub fn label(self) -> &'static str {
        match self {
            Self::Rgba16Float => "RGBA16Float (HDR scene)",
            Self::Rg11b10Float => "RG11B10Float (packed HDR)",
            Self::Rgba8Unorm => "RGBA8 UNORM (LDR scene)",
        }
    }
}

/// MSAA sample count for the main desktop swapchain forward path ([`RenderingSettings::msaa`]).
///
/// Tiers stop at **8×**; higher modes are not exposed (and are rarely supported for common surface
/// formats on desktop GPUs).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MsaaSampleCount {
    /// No multisampling (`sample_count` 1).
    #[default]
    Off,
    /// 2× MSAA.
    X2,
    /// 4× MSAA.
    X4,
    /// 8× MSAA (largest tier in settings; the GPU may still cap lower).
    #[serde(alias = "x16")]
    X8,
}

impl MsaaSampleCount {
    /// All variants for ImGui lists and config round-trips.
    pub const ALL: [Self; 4] = [Self::Off, Self::X2, Self::X4, Self::X8];

    /// Requested [`wgpu::RenderPipeline`] / attachment sample count (`1` = off).
    pub fn as_count(self) -> u32 {
        match self {
            Self::Off => 1,
            Self::X2 => 2,
            Self::X4 => 4,
            Self::X8 => 8,
        }
    }

    /// Stable string for TOML / UI (`off`, `x2`, …).
    pub fn as_persist_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::X2 => "x2",
            Self::X4 => "x4",
            Self::X8 => "x8",
        }
    }

    /// Parses case-insensitive persisted or UI token.
    ///
    /// Older **`x16` / `16` / `16x`** tokens map to [`Self::X8`] so configs predating the 8× cap
    /// continue to load.
    pub fn from_persist_str(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "off" | "1" | "1x" | "none" => Some(Self::Off),
            "x2" | "2" | "2x" => Some(Self::X2),
            "x4" | "4" | "4x" => Some(Self::X4),
            "x8" | "8" | "8x" => Some(Self::X8),
            "x16" | "16" | "16x" => Some(Self::X8),
            _ => None,
        }
    }

    /// Short label for developer UI.
    pub fn label(self) -> &'static str {
        match self {
            Self::Off => "1× (off)",
            Self::X2 => "2×",
            Self::X4 => "4×",
            Self::X8 => "8×",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ClusterAssignmentMode, MsaaSampleCount, SceneColorFormat};
    use crate::config::types::RendererSettings;
    use wgpu::TextureFormat;

    #[test]
    fn msaa_sample_count_from_persist_str_aliases_and_counts() {
        assert_eq!(
            MsaaSampleCount::from_persist_str("off"),
            Some(MsaaSampleCount::Off)
        );
        assert_eq!(
            MsaaSampleCount::from_persist_str("1x"),
            Some(MsaaSampleCount::Off)
        );
        assert_eq!(
            MsaaSampleCount::from_persist_str("X2"),
            Some(MsaaSampleCount::X2)
        );
        assert_eq!(
            MsaaSampleCount::from_persist_str("4"),
            Some(MsaaSampleCount::X4)
        );
        assert_eq!(
            MsaaSampleCount::from_persist_str("x16"),
            Some(MsaaSampleCount::X8)
        );
        assert_eq!(
            MsaaSampleCount::from_persist_str("16x"),
            Some(MsaaSampleCount::X8)
        );
        assert_eq!(MsaaSampleCount::from_persist_str("bogus"), None);

        assert_eq!(MsaaSampleCount::Off.as_count(), 1);
        assert_eq!(MsaaSampleCount::X2.as_count(), 2);
        assert_eq!(MsaaSampleCount::X4.as_count(), 4);
        assert_eq!(MsaaSampleCount::X8.as_count(), 8);
        assert_eq!(MsaaSampleCount::X8.as_persist_str(), "x8");
    }

    #[test]
    fn msaa_all_variants_persist_str_round_trip() {
        for v in MsaaSampleCount::ALL {
            let s = v.as_persist_str();
            assert_eq!(
                MsaaSampleCount::from_persist_str(s),
                Some(v),
                "round-trip failed for {s}"
            );
        }
    }

    #[test]
    fn msaa_toml_round_trip_all_variants() {
        for v in MsaaSampleCount::ALL {
            let mut s = RendererSettings::default();
            s.rendering.msaa = v;
            let toml = toml::to_string(&s).expect("serialize");
            let back: RendererSettings = toml::from_str(&toml).expect("deserialize");
            assert_eq!(back.rendering.msaa, v);
        }
    }

    #[test]
    fn msaa_labels_are_non_empty() {
        for v in MsaaSampleCount::ALL {
            assert!(!v.label().is_empty());
        }
    }

    #[test]
    fn scene_color_format_wgpu_mapping() {
        assert_eq!(
            SceneColorFormat::Rgba16Float.wgpu_format(),
            TextureFormat::Rgba16Float
        );
        assert_eq!(
            SceneColorFormat::Rg11b10Float.wgpu_format(),
            TextureFormat::Rg11b10Ufloat
        );
        assert_eq!(
            SceneColorFormat::Rgba8Unorm.wgpu_format(),
            TextureFormat::Rgba8Unorm
        );
    }

    #[test]
    fn scene_color_format_all_covers_every_variant() {
        for v in SceneColorFormat::ALL {
            let _ = v.wgpu_format();
            assert!(!v.label().is_empty());
        }
    }

    #[test]
    fn scene_color_format_toml_roundtrip() {
        let mut s = RendererSettings::default();
        s.rendering.scene_color_format = SceneColorFormat::Rg11b10Float;
        let toml = toml::to_string(&s).expect("serialize");
        let back: RendererSettings = toml::from_str(&toml).expect("deserialize");
        assert_eq!(
            back.rendering.scene_color_format,
            SceneColorFormat::Rg11b10Float
        );
    }

    #[test]
    fn cluster_assignment_toml_roundtrip() {
        for mode in ClusterAssignmentMode::ALL {
            let mut s = RendererSettings::default();
            s.rendering.cluster_assignment = mode;
            let toml = toml::to_string(&s).expect("serialize");
            let back: RendererSettings = toml::from_str(&toml).expect("deserialize");
            assert_eq!(back.rendering.cluster_assignment, mode);
        }
    }
}

#[cfg(test)]
mod vsync_resolution_tests {
    use super::VsyncMode;
    use crate::config::types::RendererSettings;
    use wgpu::PresentMode;

    #[test]
    fn off_prefers_immediate_when_supported() {
        let supported = [
            PresentMode::Immediate,
            PresentMode::Mailbox,
            PresentMode::Fifo,
        ];
        assert_eq!(
            VsyncMode::Off.resolve_present_mode(&supported),
            PresentMode::Immediate
        );
    }

    #[test]
    fn modes_choose_preferred_modes_when_everything_is_supported() {
        let supported = [
            PresentMode::Immediate,
            PresentMode::Mailbox,
            PresentMode::FifoRelaxed,
            PresentMode::Fifo,
        ];

        assert_eq!(
            VsyncMode::Off.resolve_present_mode(&supported),
            PresentMode::Immediate
        );
        assert_eq!(
            VsyncMode::On.resolve_present_mode(&supported),
            PresentMode::Mailbox
        );
        assert_eq!(
            VsyncMode::Auto.resolve_present_mode(&supported),
            PresentMode::FifoRelaxed
        );
    }

    #[test]
    fn off_falls_through_to_mailbox_then_fifo() {
        let mailbox_only = [PresentMode::Mailbox, PresentMode::Fifo];
        assert_eq!(
            VsyncMode::Off.resolve_present_mode(&mailbox_only),
            PresentMode::Mailbox
        );
        let fifo_only = [PresentMode::Fifo];
        assert_eq!(
            VsyncMode::Off.resolve_present_mode(&fifo_only),
            PresentMode::Fifo
        );
    }

    #[test]
    fn on_prefers_mailbox_over_fifo() {
        let supported = [PresentMode::Mailbox, PresentMode::Fifo];
        assert_eq!(
            VsyncMode::On.resolve_present_mode(&supported),
            PresentMode::Mailbox
        );
    }

    #[test]
    fn on_falls_back_to_fifo_when_mailbox_missing() {
        // Models a Vulkan adapter that exposes only `Fifo` + `FifoRelaxed` (no Mailbox/Immediate).
        let no_mailbox = [PresentMode::Fifo, PresentMode::FifoRelaxed];
        assert_eq!(
            VsyncMode::On.resolve_present_mode(&no_mailbox),
            PresentMode::Fifo
        );
    }

    #[test]
    fn auto_prefers_fifo_relaxed_when_supported() {
        let supported = [PresentMode::Fifo, PresentMode::FifoRelaxed];
        assert_eq!(
            VsyncMode::Auto.resolve_present_mode(&supported),
            PresentMode::FifoRelaxed
        );
    }

    #[test]
    fn auto_falls_back_to_fifo_when_relaxed_missing() {
        let fifo_only = [PresentMode::Fifo];
        assert_eq!(
            VsyncMode::Auto.resolve_present_mode(&fifo_only),
            PresentMode::Fifo
        );
    }

    #[test]
    fn empty_supported_list_falls_back_to_fifo() {
        // `Fifo` is required to be supported by every conformant surface, so the helper still
        // returns it even if the caller passes an empty (or stripped) capability list.
        for mode in VsyncMode::ALL {
            assert_eq!(
                mode.resolve_present_mode(&[]),
                PresentMode::Fifo,
                "mode {mode:?} must terminate at Fifo when nothing is advertised"
            );
        }
    }

    #[test]
    fn legacy_adaptive_token_loads_as_auto() {
        let toml = "[rendering]\nvsync = \"adaptive\"\n";
        let parsed: RendererSettings = toml::from_str(toml).expect("legacy adaptive token");
        assert_eq!(parsed.rendering.vsync, VsyncMode::Auto);
    }

    #[test]
    fn legacy_relaxed_aliases_load_as_auto() {
        for token in ["fifo_relaxed", "fiforelaxed", "relaxed"] {
            let toml = format!("[rendering]\nvsync = \"{token}\"\n");
            let parsed: RendererSettings = toml::from_str(&toml).expect("relaxed alias");
            assert_eq!(
                parsed.rendering.vsync,
                VsyncMode::Auto,
                "token `{token}` must map to Auto"
            );
        }
    }

    #[test]
    fn auto_serializes_as_snake_case() {
        let mut s = RendererSettings::default();
        s.rendering.vsync = VsyncMode::Auto;
        let toml = toml::to_string(&s).expect("serialize");
        let back: RendererSettings = toml::from_str(&toml).expect("deserialize");
        assert_eq!(back.rendering.vsync, VsyncMode::Auto);
        assert!(
            toml.contains("vsync = \"auto\""),
            "expected snake_case `auto` in serialized TOML, got: {toml}"
        );
    }
}
