//! Helpers for graph-managed world-mesh forward passes (prepare, per-draw packing, MSAA depth).

mod camera;
mod color_snapshot;
mod frame_uniforms;
mod material_resolve;
mod msaa_depth;
mod prepare;
mod raster_recording;
mod slab;

pub(super) use color_snapshot::encode_world_mesh_forward_color_snapshot;
pub(super) use msaa_depth::{
    encode_msaa_depth_resolve_after_clear_only, encode_world_mesh_forward_depth_snapshot,
    resolve_forward_msaa_views,
};
pub(super) use prepare::prepare_world_mesh_forward_frame;
pub(super) use raster_recording::{
    record_world_mesh_forward_intersection_graph_raster,
    record_world_mesh_forward_opaque_graph_raster,
    record_world_mesh_forward_transparent_graph_raster, stencil_load_ops,
};
