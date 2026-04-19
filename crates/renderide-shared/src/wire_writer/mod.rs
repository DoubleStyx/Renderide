//! Host-side wire encoders that round-trip with the renderer's parsers.
//!
//! These helpers produce byte buffers for the shared-memory regions referenced by the
//! `RendererCommand` payloads (`MeshUploadData.buffer`, `TransformsUpdate.pose_updates`,
//! `MeshRenderablesUpdate.{additions, mesh_states, mesh_materials_and_property_blocks}`, ...).
//!
//! Every encoder has a round-trip unit test that decodes its output via the renderer-side parser
//! (which lives in the `renderide` crate). When [`renderide_shared`](crate)'s tests fail we have
//! caught a drift between the host-side writer and the renderer-side reader **before** the full
//! integration test ever runs.

pub mod mesh_layout;
pub mod mesh_renderers;
pub mod render_space;
pub mod transforms;

pub use mesh_layout::{write_mesh_payload, MeshLayoutInput, MeshPayload};
pub use mesh_renderers::{encode_additions, encode_mesh_states, encode_packed_material_ids};
pub use render_space::{
    SphereSceneInputs, SphereSceneSharedMemoryLayout, SphereSceneSharedMemoryRegions,
};
pub use transforms::{encode_transform_pose_updates, TransformPoseRow};
