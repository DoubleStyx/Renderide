//! Frame-time expansion of scene renderables into [`FramePreparedDraw`] entries.
//!
//! Walks scene renderable sources in deterministic order, performs frame-scope filters, and
//! emits one entry per `(renderer, material slot)` pair.

#[cfg(test)]
mod chunking;
mod context;
mod material_keys;
mod mesh_particles;
mod render_buffers;
mod renderers;

#[cfg(test)]
pub(in crate::world_mesh::draw_prep) use chunking::expand_space_into_aggressive;
pub(in crate::world_mesh::draw_prep) use material_keys::{
    empty_material_key_signature, populate_runs_and_material_keys,
};
pub(in crate::world_mesh::draw_prep) use render_buffers::expand_render_buffer_renderers_into;
#[cfg(test)]
pub(in crate::world_mesh::draw_prep) use renderers::{estimated_draw_count, expand_space_into};
pub(in crate::world_mesh::draw_prep) use renderers::{
    expand_skinned_renderer_into, expand_static_renderer_into,
};
