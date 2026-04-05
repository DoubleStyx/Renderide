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
//! Cached [`WorldTransformCache::world_matrices`](WorldTransformCache) are **space-local** (parent
//! chain only). To include the render-space root TRS, use
//! [`SceneCoordinator::world_matrix_with_root`].
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

mod coordinator;
mod error;
mod ids;
pub mod lights;
mod math;
mod mesh_apply;
mod mesh_material_row;
mod mesh_renderable;
mod pose;
mod render_space;
mod transforms_apply;
mod world;

pub use coordinator::SceneCoordinator;
pub use error::SceneError;
pub use ids::{RenderSpaceId, TransformIndex};
pub use lights::{light_casts_shadows, CachedLight, LightCache, ResolvedLight};
pub use mesh_renderable::{MeshMaterialSlot, SkinnedMeshRenderer, StaticMeshRenderer};
pub use render_space::RenderSpaceState;
pub use transforms_apply::TransformRemovalEvent;
pub use world::WorldTransformCache;
