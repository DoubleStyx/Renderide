//! Persistent ping-pong resource registry for render-graph history slots.
//!
//! A **history slot** is a pair of GPU resources (textures or buffers) that the graph's
//! [`crate::render_graph::resources::ImportSource::PingPong`] and
//! [`crate::render_graph::resources::BufferImportSource::PingPong`] reference by
//! [`crate::render_graph::resources::HistorySlotId`]. The previous frame writes slot index
//! `current()`; the next frame swaps and writes the other half. This structure keeps both halves
//! alive across frames so the read of last frame's data is just a lookup, not a re-copy.
//!
//! Hi-Z registers view-scoped texture history here while keeping CPU snapshots and readback policy
//! on [`crate::occlusion::OcclusionSystem`]. Future TAA, SSR, or cached-shadow systems can declare
//! their persistent resources through the same owner instead of hand-rolling ping-pong pairs.

mod buffer;
mod registry;
mod texture;

pub use crate::history_texture::HistoryTextureMipViews;
pub use registry::{HistoryRegistry, HistoryRegistryError, HistoryResourceScope};
pub use texture::TextureHistorySpec;
