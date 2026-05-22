//! Host render spaces and dense transform arenas.
//!
//! ## Dense indices
//!
//! The host assigns each transform a dense index `i` in `0..nodes.len()`. After growth and
//! **swap-with-last** removals, index `i` still refers to `nodes[i]` and `node_parents[i]`.
//!
//! ## Removal order
//!
//! [`TransformsUpdate::removals`](crate::shared::TransformsUpdate) is a shared-memory array of
//! `i32` indices, read in buffer order until a negative terminator (typically `-1`). Removals are
//! **not** sorted: order matches the host batch and defines which element is swapped into which slot.
//!
//! ## World matrices
//!
//! Cached [`WorldTransformCache::world_matrices`](WorldTransformCache) are the full hierarchy
//! result per node (parent chain). Use [`SceneCoordinator::world_matrix`] for meshes, lights, and
//! bones. The render-space root transform applies to the
//! **view**, not to object
//! matrices; only use [`SceneCoordinator::world_matrix_including_space_root`] when a host contract
//! explicitly requires that composite.
//!
//! ## IPC
//!
//! Transform and mesh batches require a live [`crate::ipc::SharedMemoryAccessor`]. Frame payloads
//! that list [`RenderSpaceUpdate`](crate::shared::RenderSpaceUpdate) without shared memory are
//! skipped by the runtime until init provides a prefix.
//!
//! ## Mesh renderables
//!
//! Static and skinned mesh renderers use dense
//! `renderable_index` <-> `Vec` index, with removals in buffer order (swap-with-last).
//!
//! ## Lights
//!
//! [`LightCache`](lights::LightCache) merges [`FrameSubmitData`](crate::shared::FrameSubmitData) light
//! batches and [`LightsBufferRendererSubmission`](crate::shared::LightsBufferRendererSubmission) payloads;
//! [`SceneCoordinator::resolve_lights_for_render_context_into`] produces [`ResolvedLight`] rows
//! for GPU packing in the backend.
//!
//! ## Layout
//!
//! - **`coordinator/`** -- [`SceneCoordinator`] registry and [`FrameSubmitData`] orchestration; world-matrix helpers for render context / overlays live alongside in `queries`.
//! - **IPC apply** -- [`camera`], [`transforms`], [`meshes`], [`lights`].
//! - **`overrides/`** -- host transform/material override mirror (`types`, `space_impl`, `apply`).
//!
//! ## Reflection probes
//!
//! [`RenderSpaceUpdate::reflection_probe_sh2_taks`](crate::shared::RenderSpaceUpdate) is completed
//! by the backend SH2 service after scene state has been applied, so host rows are never left in
//! [`ComputeResult::Scheduled`](crate::shared::ComputeResult).

mod blit_to_display;
mod camera;
mod coordinator;
mod dense_update;
mod error;
mod ids;
mod layer;
pub mod lights;
mod lod_groups;
mod math;
mod meshes;
mod overrides;
mod pose;
mod reflection_probe;
mod render_space;
mod transforms;
mod world;

pub use camera::CameraRenderableEntry;
pub use coordinator::{
    RenderWorldMaterialOverrideDirty, RenderWorldRendererDirty, RenderWorldRendererKind,
    RenderWorldTransformDirty, SceneApplyReport, SceneCacheFlushReport, SceneCoordinator,
};
pub use ids::RenderSpaceId;
pub use lights::{ResolvedLight, light_contributes, light_has_negative_contribution};
pub(crate) use lod_groups::{LodEntry, LodRendererKind, LodRendererRef};
pub use math::render_transform_to_matrix;
pub use meshes::types::{
    MeshMaterialSlot, MeshRendererInstanceId, SkinnedMeshRenderer, StaticMeshRenderer,
};
pub use overrides::MeshRendererOverrideTarget;
pub(crate) use reflection_probe::changed_probe_completion;
pub use reflection_probe::{
    DrainedReflectionProbeRenderChanges, ReflectionProbeEntry,
    ReflectionProbeOnChangesRenderRequest, reflection_probe_skybox_only,
    reflection_probe_use_box_projection,
};
pub use render_space::RenderSpaceView;
