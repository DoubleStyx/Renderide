//! Host writeback for skinned mesh realtime bounds rows requested during frame submit.

use glam::{Mat4, Vec3};

use crate::gpu_pools::MeshPool;
use crate::ipc::SharedMemoryAccessor;
use crate::scene::{RenderSpaceId, SceneCoordinator};
use crate::shared::packing_extras::SKINNED_MESH_REALTIME_BOUNDS_UPDATE_HOST_ROW_BYTES;
use crate::shared::{FrameSubmitData, RenderBoundingBox, SkinnedMeshRealtimeBoundsUpdate};
use crate::world_mesh::culling::world_aabb_from_local_bounds;

/// Small finite half-extent used when the host needs an availability answer but the renderer lacks
/// enough state to compute a mesh-derived bound for this frame.
const FALLBACK_BOUNDS_EXTENT: f32 = 0.001;

/// Per-frame counters for realtime skinned bounds writeback.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct SkinnedRealtimeBoundsReport {
    /// Non-sentinel rows the renderer attempted to answer.
    rows_requested: u64,
    /// Non-sentinel rows written with finite bounds.
    rows_answered: u64,
    /// Rows answered with the small finite fallback bound.
    fallback_rows: u64,
    /// Rows whose renderable index did not resolve to a skinned renderer.
    missing_renderers: u64,
    /// Rows whose skinned renderer had neither host-posed bounds nor a resident mesh fallback.
    missing_meshes: u64,
    /// Rows whose skinned renderer had no usable world transform.
    missing_transforms: u64,
    /// Rows whose mesh or transform data produced non-finite bounds.
    invalid_bounds: u64,
    /// Shared-memory buffers that could not be opened, decoded, mutated, or flushed.
    shared_memory_failures: u64,
}

impl SkinnedRealtimeBoundsReport {
    /// Returns `true` when the report has any row activity or failure to expose.
    fn has_activity(self) -> bool {
        self.rows_requested > 0 || self.shared_memory_failures > 0
    }

    /// Returns `true` when at least one row needed a degraded answer or writeback failed.
    fn has_anomaly(self) -> bool {
        self.fallback_rows > 0 || self.shared_memory_failures > 0
    }

    /// Logs a concise frame-level summary, keeping all-success frames at trace level.
    pub(super) fn log(self, frame_index: i32) {
        if !self.has_activity() {
            return;
        }
        if self.has_anomaly() {
            logger::debug!(
                "skinned realtime bounds frame_index={} requested={} answered={} fallback={} missing_renderers={} missing_meshes={} missing_transforms={} invalid_bounds={} shared_memory_failures={}",
                frame_index,
                self.rows_requested,
                self.rows_answered,
                self.fallback_rows,
                self.missing_renderers,
                self.missing_meshes,
                self.missing_transforms,
                self.invalid_bounds,
                self.shared_memory_failures
            );
        } else {
            logger::trace!(
                "skinned realtime bounds frame_index={} requested={} answered={}",
                frame_index,
                self.rows_requested,
                self.rows_answered
            );
        }
    }
}

/// Writes realtime global bounds for all skinned mesh rows requested by a host frame submit.
pub(super) fn answer_skinned_realtime_bounds(
    shm: &mut SharedMemoryAccessor,
    scene: &SceneCoordinator,
    mesh_pool: &MeshPool,
    data: &FrameSubmitData,
) -> SkinnedRealtimeBoundsReport {
    profiling::scope!("scene::answer_skinned_realtime_bounds");
    let mut report = SkinnedRealtimeBoundsReport::default();
    for space_update in &data.render_spaces {
        let Some(update) = space_update.skinned_mesh_renderers_update.as_ref() else {
            continue;
        };
        if update.realtime_bounds_updates.length <= 0 {
            continue;
        }

        let space_id = RenderSpaceId(space_update.id);
        let ctx = format!(
            "skinned realtime_bounds_updates scene_id={}",
            space_update.id
        );
        let result = shm
            .access_mut_memory_packable_rows_until_with_max::<SkinnedMeshRealtimeBoundsUpdate, _>(
                &update.realtime_bounds_updates,
                SKINNED_MESH_REALTIME_BOUNDS_UPDATE_HOST_ROW_BYTES,
                SharedMemoryAccessor::MAX_ACCESS_COPY_BYTES,
                Some(&ctx),
                |row| {
                    if row.renderable_index < 0 {
                        return true;
                    }
                    report.rows_requested = report.rows_requested.saturating_add(1);
                    let resolved = resolve_row_bounds(
                        scene,
                        mesh_pool,
                        space_id,
                        row.renderable_index as usize,
                    );
                    row.computed_global_bounds = resolved.bounds;
                    report.rows_answered = report.rows_answered.saturating_add(1);
                    match resolved.fallback {
                        None => {}
                        Some(BoundsFallbackReason::MissingRenderer) => {
                            report.fallback_rows = report.fallback_rows.saturating_add(1);
                            report.missing_renderers = report.missing_renderers.saturating_add(1);
                        }
                        Some(BoundsFallbackReason::MissingMesh) => {
                            report.fallback_rows = report.fallback_rows.saturating_add(1);
                            report.missing_meshes = report.missing_meshes.saturating_add(1);
                        }
                        Some(BoundsFallbackReason::MissingTransform) => {
                            report.fallback_rows = report.fallback_rows.saturating_add(1);
                            report.missing_transforms = report.missing_transforms.saturating_add(1);
                        }
                        Some(BoundsFallbackReason::InvalidBounds) => {
                            report.fallback_rows = report.fallback_rows.saturating_add(1);
                            report.invalid_bounds = report.invalid_bounds.saturating_add(1);
                        }
                    }
                    false
                },
            );
        if let Err(err) = result {
            report.shared_memory_failures = report.shared_memory_failures.saturating_add(1);
            logger::warn!(
                "failed to answer skinned realtime bounds for scene_id={}: {}",
                space_update.id,
                err
            );
        }
    }
    report
}

/// Result of resolving one realtime bounds request.
#[derive(Clone, Copy, Debug)]
struct ResolvedRealtimeBounds {
    /// Finite bounds to write back into the host row.
    bounds: RenderBoundingBox,
    /// Degraded-resolution reason, if a fallback was required.
    fallback: Option<BoundsFallbackReason>,
}

/// Reason a realtime bounds row used a finite fallback bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BoundsFallbackReason {
    /// The row referenced a skinned renderable index outside the scene table.
    MissingRenderer,
    /// The renderer had neither host-posed bounds nor a resident mesh fallback.
    MissingMesh,
    /// The renderer had no usable world matrix for its transform.
    MissingTransform,
    /// The mesh bounds or transform produced non-finite world bounds.
    InvalidBounds,
}

/// Resolves one realtime bounds row to mesh-derived global bounds or a finite fallback.
fn resolve_row_bounds(
    scene: &SceneCoordinator,
    mesh_pool: &MeshPool,
    space_id: RenderSpaceId,
    renderable_index: usize,
) -> ResolvedRealtimeBounds {
    let Some(renderer) = scene
        .space(space_id)
        .and_then(|space| space.skinned_mesh_renderers().get(renderable_index))
    else {
        return fallback_bounds(None, BoundsFallbackReason::MissingRenderer);
    };

    let root_node_id = renderer
        .root_bone_transform_id
        .filter(|&id| id >= 0)
        .unwrap_or(renderer.base.node_id);
    let world_matrix = if root_node_id >= 0 {
        scene.world_matrix(space_id, root_node_id as usize)
    } else {
        None
    };
    let Some(world_matrix) = world_matrix else {
        return fallback_bounds(None, BoundsFallbackReason::MissingTransform);
    };
    let object_bounds = renderer.posed_object_bounds.as_ref().or_else(|| {
        mesh_pool
            .get(renderer.base.mesh_asset_id)
            .map(|mesh| &mesh.bounds)
    });
    let Some(object_bounds) = object_bounds else {
        return fallback_bounds(Some(world_matrix), BoundsFallbackReason::MissingMesh);
    };
    let Some(bounds) = global_bounds_from_local_bounds(object_bounds, world_matrix) else {
        return fallback_bounds(Some(world_matrix), BoundsFallbackReason::InvalidBounds);
    };
    ResolvedRealtimeBounds {
        bounds,
        fallback: None,
    }
}

/// Transforms mesh-local center/extents bounds into host-facing global center/extents bounds.
fn global_bounds_from_local_bounds(
    local_bounds: &RenderBoundingBox,
    world_matrix: Mat4,
) -> Option<RenderBoundingBox> {
    let (world_min, world_max) = world_aabb_from_local_bounds(local_bounds, world_matrix)?;
    let center = (world_min + world_max) * 0.5;
    let extents = (world_max - world_min).abs() * 0.5;
    if !vec3_finite(center) || !vec3_finite(extents) {
        return None;
    }
    Some(RenderBoundingBox { center, extents })
}

/// Builds a small valid bound near the renderer transform when precise bounds are unavailable.
fn fallback_bounds(
    world_matrix: Option<Mat4>,
    reason: BoundsFallbackReason,
) -> ResolvedRealtimeBounds {
    let center = world_matrix
        .map(|matrix| matrix.transform_point3(Vec3::ZERO))
        .filter(|value| vec3_finite(*value))
        .unwrap_or(Vec3::ZERO);
    ResolvedRealtimeBounds {
        bounds: RenderBoundingBox {
            center,
            extents: Vec3::splat(FALLBACK_BOUNDS_EXTENT),
        },
        fallback: Some(reason),
    }
}

/// Returns whether all vector components are finite.
fn vec3_finite(value: Vec3) -> bool {
    value.x.is_finite() && value.y.is_finite() && value.z.is_finite()
}

#[cfg(test)]
mod tests {
    use glam::{Mat4, Vec3};

    use crate::gpu_pools::MeshPool;
    use crate::ipc::SharedMemoryAccessor;
    use crate::scene::SceneCoordinator;
    use crate::shared::memory_packable::MemoryPackable;
    use crate::shared::memory_packer::MemoryPacker;
    use crate::shared::{
        FrameSubmitData, RenderBoundingBox, RenderSpaceUpdate, SkinnedMeshRealtimeBoundsUpdate,
        SkinnedMeshRenderablesUpdate,
    };

    use super::{
        BoundsFallbackReason, FALLBACK_BOUNDS_EXTENT,
        SKINNED_MESH_REALTIME_BOUNDS_UPDATE_HOST_ROW_BYTES, answer_skinned_realtime_bounds,
        fallback_bounds, global_bounds_from_local_bounds,
    };

    fn unique_prefix(label: &str) -> String {
        format!(
            "renderide_test_skinned_bounds_{label}_{}",
            std::process::id()
        )
    }

    fn encode_realtime_bounds_rows(rows: &mut [SkinnedMeshRealtimeBoundsUpdate]) -> Vec<u8> {
        let mut bytes = vec![0u8; rows.len() * SKINNED_MESH_REALTIME_BOUNDS_UPDATE_HOST_ROW_BYTES];
        for (row, chunk) in rows
            .iter_mut()
            .zip(bytes.chunks_exact_mut(SKINNED_MESH_REALTIME_BOUNDS_UPDATE_HOST_ROW_BYTES))
        {
            let mut packer = MemoryPacker::new(chunk);
            row.pack(&mut packer);
            assert_eq!(packer.remaining_len(), 0, "test row must fill host stride");
        }
        bytes
    }

    #[test]
    fn global_bounds_from_local_bounds_applies_translation_and_scale() {
        let local = RenderBoundingBox {
            center: Vec3::new(1.0, 2.0, 3.0),
            extents: Vec3::new(0.5, 1.0, 2.0),
        };
        let world = Mat4::from_scale_rotation_translation(
            Vec3::new(2.0, 3.0, 4.0),
            glam::Quat::IDENTITY,
            Vec3::new(10.0, -5.0, 1.0),
        );

        let bounds = global_bounds_from_local_bounds(&local, world).expect("finite bounds");

        assert_vec3_near(bounds.center, Vec3::new(12.0, 1.0, 13.0));
        assert_vec3_near(bounds.extents, Vec3::new(1.0, 3.0, 8.0));
    }

    #[test]
    fn global_bounds_from_local_bounds_rejects_non_finite_input() {
        let local = RenderBoundingBox {
            center: Vec3::new(f32::NAN, 0.0, 0.0),
            extents: Vec3::ONE,
        };

        assert!(global_bounds_from_local_bounds(&local, Mat4::IDENTITY).is_none());
    }

    #[test]
    fn fallback_bounds_uses_transform_translation_when_finite() {
        let world = Mat4::from_translation(Vec3::new(5.0, -2.0, 7.0));

        let resolved = fallback_bounds(Some(world), BoundsFallbackReason::MissingMesh);

        assert_vec3_near(resolved.bounds.center, Vec3::new(5.0, -2.0, 7.0));
        assert_vec3_near(resolved.bounds.extents, Vec3::splat(FALLBACK_BOUNDS_EXTENT));
        assert_eq!(resolved.fallback, Some(BoundsFallbackReason::MissingMesh));
    }

    #[test]
    fn answer_skinned_realtime_bounds_writes_missing_renderer_fallback() {
        let prefix = unique_prefix("writeback_missing_renderer");
        let mut rows = [
            SkinnedMeshRealtimeBoundsUpdate {
                renderable_index: 0,
                computed_global_bounds: RenderBoundingBox {
                    center: Vec3::splat(f32::NAN),
                    extents: Vec3::splat(f32::NAN),
                },
            },
            SkinnedMeshRealtimeBoundsUpdate {
                renderable_index: -1,
                computed_global_bounds: RenderBoundingBox {
                    center: Vec3::new(8.0, 9.0, 10.0),
                    extents: Vec3::ONE,
                },
            },
        ];
        let bytes = encode_realtime_bounds_rows(&mut rows);
        let cfg = renderide_shared::SharedMemoryWriterConfig {
            prefix: prefix.clone(),
            destroy_on_drop: true,
            ..renderide_shared::SharedMemoryWriterConfig::default()
        };
        let mut writer =
            renderide_shared::SharedMemoryWriter::open(cfg, 21, bytes.len()).expect("open writer");
        writer.write_at(0, &bytes).expect("write rows");
        writer.flush();
        let descriptor = writer.descriptor_for(0, bytes.len() as i32);
        let mut data = FrameSubmitData::default();
        data.render_spaces.push(RenderSpaceUpdate {
            id: 7,
            skinned_mesh_renderers_update: Some(SkinnedMeshRenderablesUpdate {
                realtime_bounds_updates: descriptor,
                ..Default::default()
            }),
            ..Default::default()
        });
        let mut shm = SharedMemoryAccessor::new(prefix);
        let scene = SceneCoordinator::new();
        let mesh_pool = MeshPool::default_pool();

        let report = answer_skinned_realtime_bounds(&mut shm, &scene, &mesh_pool, &data);
        let readback = shm
            .access_copy_memory_packable_rows::<SkinnedMeshRealtimeBoundsUpdate>(
                &descriptor,
                SKINNED_MESH_REALTIME_BOUNDS_UPDATE_HOST_ROW_BYTES,
                Some("realtime_bounds"),
            )
            .expect("read rows");

        assert_eq!(report.rows_requested, 1);
        assert_eq!(report.rows_answered, 1);
        assert_eq!(report.fallback_rows, 1);
        assert_eq!(report.missing_renderers, 1);
        assert_vec3_near(readback[0].computed_global_bounds.center, Vec3::ZERO);
        assert_vec3_near(
            readback[0].computed_global_bounds.extents,
            Vec3::splat(FALLBACK_BOUNDS_EXTENT),
        );
        assert_vec3_near(
            readback[1].computed_global_bounds.center,
            Vec3::new(8.0, 9.0, 10.0),
        );
    }

    fn assert_vec3_near(actual: Vec3, expected: Vec3) {
        let diff = (actual - expected).abs();
        assert!(
            diff.max_element() <= 1e-5,
            "actual={actual:?} expected={expected:?}"
        );
    }
}
