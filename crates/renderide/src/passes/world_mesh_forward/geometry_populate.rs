//! Geometry arena population.
//!
//! Copies visible and shadow-casting static mesh vertex streams into the shared geometry
//! mega-buffer before shadow-atlas or per-view command buffers are recorded. The asset list is
//! deduplicated during frame preparation, avoiding cross-view residency and buffer-growth ordering
//! races.

use crate::assets::mesh::GpuMesh;
use crate::gpu_pools::MeshPool;
use crate::gpu_pools::geometry_arena::{
    ArenaStream, GeometryArena, GeometryArenaCapacityHint, MeshStreamSources,
};
use crate::render_graph::context::EncoderPassCtx;
use crate::render_graph::error::{RenderPassError, SetupError};
use crate::render_graph::pass::{EncoderPass, PassBuilder, PassPhase};

/// Frame-global pass that copies static geometry ahead of shadow and per-view indirect draws.
pub(crate) struct GeometryArenaPopulatePass;

impl GeometryArenaPopulatePass {
    /// Creates the population pass.
    pub(crate) const fn new() -> Self {
        Self
    }
}

impl EncoderPass for GeometryArenaPopulatePass {
    fn name(&self) -> &str {
        "GeometryArenaPopulate"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        b.encoder();
        b.cull_exempt();
        b.never_parallel();
        Ok(())
    }

    fn should_record(&self, ctx: &EncoderPassCtx<'_, '_, '_>) -> Result<bool, RenderPassError> {
        if !ctx.frame.systems.frame_resources.has_frame_gpu() {
            return Ok(false);
        }
        if !ctx
            .frame
            .systems
            .frame_resources
            .geometry_arena_populate_plan()
            .mesh_asset_ids
            .is_empty()
        {
            return Ok(true);
        }
        let Some(arena) = ctx.frame.systems.frame_resources.shared_geometry_arena() else {
            return Ok(false);
        };
        let mesh_pool_generation = ctx
            .frame
            .systems
            .asset_resources
            .mesh_pool()
            .mutation_generation();
        let guard = arena.read();
        Ok(guard.as_ref().is_some_and(|arena| {
            arena.reclamation_requested()
                || arena.needs_mesh_pool_synchronization(mesh_pool_generation)
        }))
    }

    fn record(&self, ctx: &mut EncoderPassCtx<'_, '_, '_>) -> Result<(), RenderPassError> {
        profiling::scope!("world_mesh_forward::geometry_arena_populate");
        let frame = &ctx.frame;
        let Some(gpu_limits) = frame.view.gpu_limits.as_deref() else {
            return Ok(());
        };
        // The arena is also the canonical source for direct static draws, including downlevel
        // devices without indirect-first-instance support. Indirect consumers apply their own
        // feature gates when selecting a recording path.
        let Some(arena_arc) = frame.systems.frame_resources.shared_geometry_arena() else {
            return Ok(());
        };
        let mesh_pool = frame.systems.asset_resources.mesh_pool();
        let populate_plan = frame.systems.frame_resources.geometry_arena_populate_plan();
        let mut profile = crate::profiling::WorldMeshGeometryArenaProfileSample {
            input_draws: populate_plan.input_draws,
            unique_meshes: populate_plan.mesh_asset_ids.len(),
            deformed_draws: populate_plan.deformed_draws,
            ..Default::default()
        };

        let mut guard = arena_arc.write();
        let mut initial_mesh_order = None;
        if guard.is_none() {
            let mut asset_ids = populate_plan.mesh_asset_ids.to_vec();
            // Wide/sparse stream layouts must receive low base-vertex offsets. Otherwise one rare
            // 64-byte stream appearing late forces a mostly empty buffer spanning every earlier
            // vertex and can dominate arena VRAM.
            asset_ids.sort_unstable_by(|left, right| {
                geometry_stream_layout_weight(mesh_pool, *right)
                    .cmp(&geometry_stream_layout_weight(mesh_pool, *left))
                    .then_with(|| left.cmp(right))
            });
            let capacity_hint = geometry_arena_capacity_hint(mesh_pool, &asset_ids);
            let Some(arena) = GeometryArena::new_with_capacity_hint(
                ctx.device,
                gpu_limits.max_buffer_size(),
                &capacity_hint,
            ) else {
                // The device refused the core arenas. Leave the store uninitialized so every mesh
                // draws from its dedicated buffers, and retry when the plan next changes.
                profile.allocation_failures = profile.allocation_failures.saturating_add(1);
                crate::profiling::plot_world_mesh_geometry_arena(profile);
                return Ok(());
            };
            *guard = Some(arena);
            initial_mesh_order = Some(asset_ids);
        }
        let arena = guard
            .as_mut()
            .expect("geometry arena was initialized above");
        arena.synchronize_mesh_pool(mesh_pool);
        let reclaim = arena.reclaim_high_water_at_frame_boundary(ctx.device, ctx.encoder);
        profile.compaction_evaluated = reclaim.evaluated;
        profile.compaction_reclaimed_bytes = reclaim.reclaimed_bytes;
        profile.compaction_copy_bytes = reclaim.copy_bytes;

        let mut optional: Vec<(ArenaStream, &wgpu::Buffer)> =
            Vec::with_capacity(ArenaStream::ALL.len());
        let mesh_asset_ids = initial_mesh_order
            .as_deref()
            .unwrap_or(populate_plan.mesh_asset_ids);
        for &mesh_asset_id in mesh_asset_ids {
            let Some(mesh) = mesh_pool.get(mesh_asset_id) else {
                profile.missing_meshes = profile.missing_meshes.saturating_add(1);
                continue;
            };
            if mesh.dynamic_geometry {
                if crate::particles::is_generated_particle_mesh_asset_id(mesh_asset_id) {
                    profile.generated_meshes = profile.generated_meshes.saturating_add(1);
                } else {
                    profile.dynamic_meshes = profile.dynamic_meshes.saturating_add(1);
                }
                continue;
            }
            // Only meshes explicitly admitted by upload-time validation may lose per-mesh vertex
            // fetch bounds in the shared arena. Dedicated static fallbacks include malformed
            // index payloads that would otherwise read a neighboring allocation.
            if !mesh.uses_shared_static_geometry() {
                continue;
            }
            optional.clear();
            for stream in ArenaStream::ALL {
                push_stream(&mut optional, stream, mesh_stream(mesh, stream));
            }
            let already_resident = arena.mesh(mesh_asset_id).is_some();
            if already_resident {
                profile.already_resident_meshes = profile.already_resident_meshes.saturating_add(1);
            }
            if let (Some(position), Some(index)) = (
                mesh.positions_buffer.as_deref(),
                mesh.index_buffer.as_deref(),
            ) {
                // Revalidate source identity and declared counts while dedicated sources exist.
                // `ensure_mesh` replaces a stale allocation before copying.
                let allocation = arena.ensure_mesh(
                    ctx.device,
                    ctx.encoder,
                    mesh_asset_id,
                    &MeshStreamSources {
                        position,
                        index,
                        narrow: mesh.index_format == wgpu::IndexFormat::Uint16,
                        vertex_count: mesh.vertex_count,
                        index_count: mesh.index_count,
                        optional: &optional,
                    },
                );
                if let Some(allocation) = allocation {
                    profile.ready_meshes = profile.ready_meshes.saturating_add(1);
                    if shared_sources_can_be_released(mesh, allocation) {
                        arena.mark_source_release_pending(mesh_asset_id);
                    }
                } else {
                    profile.allocation_failures = profile.allocation_failures.saturating_add(1);
                }
                continue;
            }
            if already_resident {
                // Shared-static meshes release their dedicated position/index handles after the
                // arena copy reaches submit retention. Their committed core remains authoritative,
                // while newly demanded optional streams may still be materialized independently.
                let allocation = arena.ensure_optional_streams(
                    ctx.device,
                    ctx.encoder,
                    mesh_asset_id,
                    &optional,
                );
                if let Some(allocation) = allocation {
                    profile.ready_meshes = profile.ready_meshes.saturating_add(1);
                    if shared_sources_can_be_released(mesh, allocation) {
                        arena.mark_source_release_pending(mesh_asset_id);
                    }
                } else {
                    profile.allocation_failures = profile.allocation_failures.saturating_add(1);
                }
                continue;
            }
            if mesh.positions_buffer.is_none() {
                profile.missing_position_meshes = profile.missing_position_meshes.saturating_add(1);
            } else {
                profile.allocation_failures = profile.allocation_failures.saturating_add(1);
            }
        }
        profile.allocated_bytes = arena.allocated_bytes();
        profile.resident_allocation_bytes = arena.resident_allocation_bytes();
        crate::profiling::plot_world_mesh_geometry_arena(profile);
        Ok(())
    }

    fn phase(&self) -> PassPhase {
        PassPhase::FrameGlobal
    }
}

fn geometry_stream_layout_weight(mesh_pool: &MeshPool, mesh_asset_id: i32) -> u64 {
    let Some(mesh) = mesh_pool.get(mesh_asset_id) else {
        return 0;
    };
    if !mesh.uses_shared_static_geometry() {
        return 0;
    }
    ArenaStream::ALL
        .into_iter()
        .filter(|&stream| mesh_stream(mesh, stream).is_some())
        .map(ArenaStream::stride)
        .sum()
}

fn shared_sources_can_be_released(
    mesh: &GpuMesh,
    allocation: crate::gpu_pools::geometry_arena::GeometryAllocation,
) -> bool {
    if !mesh.uses_shared_static_geometry() {
        return false;
    }
    let dedicated_streams = mesh.dedicated_derived_stream_mask();
    let has_duplicate_sources = mesh.index_buffer.is_some() || !dedicated_streams.is_empty();
    has_duplicate_sources && allocation.derived_stream_mask().contains(dedicated_streams)
}

fn geometry_arena_capacity_hint(
    mesh_pool: &MeshPool,
    mesh_asset_ids: &[i32],
) -> GeometryArenaCapacityHint {
    let mut hint = GeometryArenaCapacityHint::default();
    for &mesh_asset_id in mesh_asset_ids {
        let Some(mesh) = mesh_pool.get(mesh_asset_id) else {
            continue;
        };
        if mesh.dynamic_geometry
            || !mesh.uses_shared_static_geometry()
            || mesh.positions_buffer.is_none()
        {
            continue;
        }
        hint.include_mesh(
            mesh.vertex_count,
            mesh.index_count,
            mesh.index_format == wgpu::IndexFormat::Uint16,
            ArenaStream::ALL
                .into_iter()
                .filter(|&stream| mesh_stream(mesh, stream).is_some()),
        );
    }
    hint
}

fn mesh_stream(mesh: &GpuMesh, stream: ArenaStream) -> Option<&wgpu::Buffer> {
    match stream {
        ArenaStream::Normal => mesh.normals_buffer.as_deref(),
        ArenaStream::Uv0 => mesh.uv0_buffer.as_deref(),
        ArenaStream::Color => mesh.color_buffer.as_deref(),
        ArenaStream::Tangent => mesh.tangent_buffer.as_deref(),
        ArenaStream::RawTangent => mesh.raw_tangent_buffer.as_deref(),
        ArenaStream::Uv1 => mesh.uv1_buffer.as_deref(),
        ArenaStream::Uv2 => mesh.uv2_buffer.as_deref(),
        ArenaStream::Uv3 => mesh.uv3_buffer.as_deref(),
        ArenaStream::WideLow => mesh.wide_low_uv_buffer.as_deref(),
        ArenaStream::WideHigh => mesh.wide_high_uv_buffer.as_deref(),
    }
}

fn push_stream<'a>(
    out: &mut Vec<(ArenaStream, &'a wgpu::Buffer)>,
    stream: ArenaStream,
    buffer: Option<&'a wgpu::Buffer>,
) {
    if let Some(buffer) = buffer {
        out.push((stream, buffer));
    }
}
