//! Startup-selected world-mesh path for isolating command and geometry corruption.

use std::sync::LazyLock;

/// Process environment override for the world-mesh diagnostic path.
pub(crate) const WORLD_MESH_PATH_ENV: &str = "RENDERIDE_WORLD_MESH_PATH";

/// World-mesh command and geometry source selected for the lifetime of this renderer process.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum WorldMeshRenderPath {
    /// Production path: GPU culling writes indirect commands that draw from the shared arena.
    #[default]
    Gpu,
    /// Experimental: `Gpu`, plus CPU-built indirect commands for the groups GPU cull leaves behind.
    ///
    /// Under `Gpu` alone the CPU indirect buffer is not provisioned, so every group outside a GPU
    /// run drops to a per-mesh draw even when it is arena-resident and static. Traces put that at
    /// ~82 groups per subpass. CPU runs here are clamped to the next GPU run boundary, otherwise a
    /// run would swallow groups whose visibility the GPU owns and draw them unculled.
    GpuHybrid,
    /// Diagnostic path: CPU-built indirect commands draw from the same shared arena.
    CpuIndirect,
    /// Diagnostic path: direct indexed draws bind mesh-local slices of the shared arena.
    ArenaDirect,
    /// Diagnostic baseline: direct indexed draws bind retained per-mesh source buffers.
    ///
    /// Select this before process startup. Sources released by an earlier process cannot be
    /// reconstructed in-place.
    DedicatedDirect,
}

impl WorldMeshRenderPath {
    /// Stable environment/log spelling.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Gpu => "gpu",
            Self::GpuHybrid => "gpu_hybrid",
            Self::CpuIndirect => "cpu_indirect",
            Self::ArenaDirect => "arena_direct",
            Self::DedicatedDirect => "dedicated_direct",
        }
    }

    /// Whether the world compute pass may generate raster commands.
    pub(crate) const fn uses_gpu_generated_commands(self) -> bool {
        matches!(self, Self::Gpu | Self::GpuHybrid)
    }

    /// Whether raster passes consume indexed-indirect commands.
    pub(crate) const fn uses_indirect_draws(self) -> bool {
        matches!(self, Self::Gpu | Self::GpuHybrid | Self::CpuIndirect)
    }

    /// Whether CPU-built indirect runs fill the gaps between GPU-culled runs.
    pub(crate) const fn fills_indirect_gaps_on_cpu(self) -> bool {
        matches!(self, Self::GpuHybrid)
    }

    /// Whether raster passes and population expose the shared geometry arena.
    pub(crate) const fn uses_geometry_arena(self) -> bool {
        !matches!(self, Self::DedicatedDirect)
    }

    /// Whether committed arena residency may release duplicate per-mesh source buffers.
    pub(crate) const fn releases_dedicated_sources(self) -> bool {
        !matches!(self, Self::DedicatedDirect)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct WorldMeshRenderPathSelection {
    path: WorldMeshRenderPath,
    raw: Option<String>,
    recognized: bool,
}

impl WorldMeshRenderPathSelection {
    fn from_raw(raw: Option<&str>) -> Self {
        let Some(raw) = raw else {
            return Self {
                path: WorldMeshRenderPath::Gpu,
                raw: None,
                recognized: true,
            };
        };
        let normalized = raw.trim().to_ascii_lowercase();
        let path = match normalized.as_str() {
            "" | "gpu" => WorldMeshRenderPath::Gpu,
            "gpu_hybrid" => WorldMeshRenderPath::GpuHybrid,
            "cpu_indirect" => WorldMeshRenderPath::CpuIndirect,
            "arena_direct" => WorldMeshRenderPath::ArenaDirect,
            "dedicated_direct" => WorldMeshRenderPath::DedicatedDirect,
            _ => {
                return Self {
                    path: WorldMeshRenderPath::Gpu,
                    raw: Some(raw.to_owned()),
                    recognized: false,
                };
            }
        };
        Self {
            path,
            raw: Some(raw.to_owned()),
            recognized: true,
        }
    }
}

fn selected_world_mesh_render_path() -> &'static WorldMeshRenderPathSelection {
    static SELECTED: LazyLock<WorldMeshRenderPathSelection> = LazyLock::new(|| {
        let raw = std::env::var(WORLD_MESH_PATH_ENV).ok();
        WorldMeshRenderPathSelection::from_raw(raw.as_deref())
    });
    &SELECTED
}

/// Returns the process-lifetime world-mesh path.
#[inline]
pub(crate) fn world_mesh_render_path() -> WorldMeshRenderPath {
    selected_world_mesh_render_path().path
}

/// Emits the startup banner for the selected path and reports invalid overrides.
pub(crate) fn log_world_mesh_render_path() {
    let selection = selected_world_mesh_render_path();
    if !selection.recognized {
        logger::warn!(
            "Invalid {WORLD_MESH_PATH_ENV}={:?}; falling back to gpu",
            selection.raw.as_deref().unwrap_or_default()
        );
    }
    let path = selection.path;
    let source = selection.raw.as_ref().map_or("default", |_| "environment");
    logger::info!(
        "World-mesh render path: mode={} source={} gpu_commands={} indirect_draws={} \
         geometry_arena={} release_dedicated_sources={}",
        path.as_str(),
        source,
        path.uses_gpu_generated_commands(),
        path.uses_indirect_draws(),
        path.uses_geometry_arena(),
        path.releases_dedicated_sources(),
    );
    if path == WorldMeshRenderPath::DedicatedDirect {
        logger::info!(
            "World-mesh dedicated_direct diagnostic requires a fresh renderer process so every \
             per-mesh source buffer is retained from upload"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{WorldMeshRenderPath, WorldMeshRenderPathSelection};

    fn parse(raw: Option<&str>) -> WorldMeshRenderPathSelection {
        WorldMeshRenderPathSelection::from_raw(raw)
    }

    #[test]
    fn missing_or_empty_override_selects_gpu() {
        assert_eq!(parse(None).path, WorldMeshRenderPath::Gpu);
        assert_eq!(parse(Some("")).path, WorldMeshRenderPath::Gpu);
        assert_eq!(parse(Some("  ")).path, WorldMeshRenderPath::Gpu);
    }

    #[test]
    fn hybrid_only_differs_from_gpu_by_the_cpu_gap_fill() {
        let gpu = WorldMeshRenderPath::Gpu;
        let hybrid = WorldMeshRenderPath::GpuHybrid;

        assert!(!gpu.fills_indirect_gaps_on_cpu());
        assert!(hybrid.fills_indirect_gaps_on_cpu());
        assert_eq!(
            hybrid.uses_gpu_generated_commands(),
            gpu.uses_gpu_generated_commands()
        );
        assert_eq!(hybrid.uses_indirect_draws(), gpu.uses_indirect_draws());
        assert_eq!(hybrid.uses_geometry_arena(), gpu.uses_geometry_arena());
        assert_eq!(
            hybrid.releases_dedicated_sources(),
            gpu.releases_dedicated_sources()
        );
    }

    #[test]
    fn no_other_path_fills_gaps_on_cpu() {
        for path in [
            WorldMeshRenderPath::Gpu,
            WorldMeshRenderPath::CpuIndirect,
            WorldMeshRenderPath::ArenaDirect,
            WorldMeshRenderPath::DedicatedDirect,
        ] {
            assert!(!path.fills_indirect_gaps_on_cpu(), "{}", path.as_str());
        }
    }

    #[test]
    fn every_documented_path_parses_case_insensitively() {
        assert_eq!(parse(Some("gpu")).path, WorldMeshRenderPath::Gpu);
        assert_eq!(
            parse(Some(" GPU_HYBRID ")).path,
            WorldMeshRenderPath::GpuHybrid
        );
        assert_eq!(
            parse(Some(" CPU_INDIRECT ")).path,
            WorldMeshRenderPath::CpuIndirect
        );
        assert_eq!(
            parse(Some("arena_direct")).path,
            WorldMeshRenderPath::ArenaDirect
        );
        assert_eq!(
            parse(Some("Dedicated_Direct")).path,
            WorldMeshRenderPath::DedicatedDirect
        );
    }

    #[test]
    fn invalid_override_falls_back_to_gpu_and_is_reportable() {
        let selected = parse(Some("unsafe-mystery-mode"));
        assert_eq!(selected.path, WorldMeshRenderPath::Gpu);
        assert!(!selected.recognized);
        assert_eq!(selected.raw.as_deref(), Some("unsafe-mystery-mode"));
    }

    #[test]
    fn path_capabilities_form_the_expected_diagnostic_ladder() {
        let gpu = WorldMeshRenderPath::Gpu;
        assert!(gpu.uses_gpu_generated_commands());
        assert!(gpu.uses_indirect_draws());
        assert!(gpu.uses_geometry_arena());
        assert!(gpu.releases_dedicated_sources());

        let cpu_indirect = WorldMeshRenderPath::CpuIndirect;
        assert!(!cpu_indirect.uses_gpu_generated_commands());
        assert!(cpu_indirect.uses_indirect_draws());
        assert!(cpu_indirect.uses_geometry_arena());
        assert!(cpu_indirect.releases_dedicated_sources());

        let arena_direct = WorldMeshRenderPath::ArenaDirect;
        assert!(!arena_direct.uses_gpu_generated_commands());
        assert!(!arena_direct.uses_indirect_draws());
        assert!(arena_direct.uses_geometry_arena());
        assert!(arena_direct.releases_dedicated_sources());

        let dedicated_direct = WorldMeshRenderPath::DedicatedDirect;
        assert!(!dedicated_direct.uses_gpu_generated_commands());
        assert!(!dedicated_direct.uses_indirect_draws());
        assert!(!dedicated_direct.uses_geometry_arena());
        assert!(!dedicated_direct.releases_dedicated_sources());
    }
}
