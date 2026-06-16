//! Per-frame GPU bind groups, per-view light staging, shared cluster buffers, and per-view
//! per-draw instance resources.
//!
//! [`FrameResourceManager`] owns the fallback `@group(0)` frame resources
//! ([`crate::backend::frame_gpu::FrameGpuResources`]), the empty `@group(1)` fallback
//! ([`crate::backend::frame_gpu::EmptyMaterialBindGroup`]), per-view frame/light bind resources
//! ([`PerViewFrameState`]), a `@group(2)` per-draw instance storage slab per render view, and the
//! CPU-side packed light buffers used by [`crate::passes::ClusteredLightPass`] and the forward
//! pass.
//!
//! Cluster buffers are shared and grow before graph recording so every planned viewport has
//! enough dynamic index storage for its current light pack. Per-view state is keyed by
//! [`crate::camera::ViewId`] and created lazily on first use; retired explicitly when a secondary
//! RT camera is destroyed.
//!
//! Per-draw resources follow the same ownership model: one grow-on-demand slab per
//! [`crate::camera::ViewId`], created lazily so no view can exhaust another view's per-draw
//! capacity.

mod cluster_layout;
mod graph_frame_resources;
mod lights;
mod manager;
mod per_view;
mod per_view_state;
mod shadows;
mod view_desc;

#[cfg(test)]
mod tests;

pub use manager::FrameResourceManager;

pub(crate) use shadows::{ShadowCasterSet, ShadowFramePlan, ShadowRenderView};
pub(crate) use view_desc::{FrameLightCullDesc, FrameLightViewDesc};

// Re-exports kept for intra-doc links in sibling modules
// (e.g. `frame_gpu.rs`, `cluster_gpu.rs` reference these types by path).
#[expect(unused_imports, reason = "supports rustdoc intra-doc links")]
pub(crate) use per_view_state::{PerViewFrameState, PerViewPerDrawScratch};
