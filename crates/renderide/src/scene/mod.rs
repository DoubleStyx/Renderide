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
//! bones. [`RenderSpaceState::root_transform`](render_space::RenderSpaceState) applies to the
//! **view** ([`RenderSpaceState::view_transform`](render_space::RenderSpaceState)), not to object
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
//! [`RenderSpaceState::static_mesh_renderers`](RenderSpaceState::static_mesh_renderers) and
//! [`RenderSpaceState::skinned_mesh_renderers`](RenderSpaceState::skinned_mesh_renderers) use dense
//! `renderable_index` ↔ `Vec` index, with removals in buffer order (swap-with-last).
//!
//! ## Lights
//!
//! [`LightCache`](lights::LightCache) merges [`FrameSubmitData`](crate::shared::FrameSubmitData) light
//! batches and [`LightsBufferRendererSubmission`](crate::shared::LightsBufferRendererSubmission) payloads;
//! [`SceneCoordinator::resolve_lights_world`](SceneCoordinator::resolve_lights_world) produces
//! [`ResolvedLight`](ResolvedLight) for [`GpuLight`](crate::backend::GpuLight) packing in the backend.
//!
//! ## Layout
//!
//! - **`coordinator/`** — [`SceneCoordinator`] registry and [`FrameSubmitData`] orchestration; world-matrix helpers for render context / overlays live alongside in `world_queries`.
//! - **IPC apply** — [`camera_apply`], [`transforms_apply`], [`mesh_apply`], [`lights`].
//! - **`render_overrides/`** — host transform/material override mirror (`types`, `space_impl`, `apply`).
//!
//! ## Reflection probes
//!
//! [`RenderSpaceUpdate::reflection_probe_sh2_taks`](crate::shared::RenderSpaceUpdate) is completed in
//! shared memory by marking each task [`ComputeResult::Failed`](crate::shared::ComputeResult) until
//! SH2 extraction is implemented (module `reflection_probe_sh2`).

mod camera_apply;
mod coordinator;
mod error;
mod ids;
pub mod lights;
mod math;
mod mesh_apply;
mod mesh_material_row;
mod mesh_renderable;
mod pose;
mod reflection_probe_sh2;
mod render_overrides;
mod render_space;
mod transforms_apply;
mod world;

pub use camera_apply::CameraRenderableEntry;
pub use coordinator::SceneCoordinator;
pub use error::SceneError;
pub use ids::{RenderSpaceId, TransformIndex};
pub use lights::{light_casts_shadows, light_contributes, CachedLight, LightCache, ResolvedLight};
pub use math::render_transform_to_matrix;
pub use mesh_renderable::{MeshMaterialSlot, SkinnedMeshRenderer, StaticMeshRenderer};
pub use render_space::RenderSpaceState;
pub use transforms_apply::TransformRemovalEvent;
pub use world::WorldTransformCache;
