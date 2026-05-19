//! Shared shader ABI contracts for `@group(0)` bind groups and packed GPU rows.
//!
//! - [`lights`] -- [`GpuLight`] row + per-frame buffer cap.
//! - [`reflection_probes`] -- [`GpuReflectionProbeMetadata`] row + probe metadata constants.
//! - [`shadows`] -- realtime shadow-map metadata rows and texture limits.
//! - [`cluster_params`] -- clustered-light compute slab sizing constants.
//! - [`bind_group`] -- `@group(0)` BindGroupLayout used by every material pipeline.

mod bind_group;
mod cluster_params;
mod lights;
mod reflection_probes;
mod shadows;

pub use bind_group::{
    empty_material_bind_group_layout, frame_bind_group_layout, frame_bind_group_layout_entries,
};
pub use cluster_params::{CLUSTER_LIGHT_RANGE_WORDS, CLUSTER_PARAMS_UNIFORM_SIZE};
pub use lights::{GpuLight, MAX_LIGHTS};
pub use reflection_probes::{
    GpuReflectionProbeMetadata, REFLECTION_PROBE_ATLAS_FORMAT,
    REFLECTION_PROBE_METADATA_BOX_PROJECTION, REFLECTION_PROBE_METADATA_SH2_SOURCE_LOCAL,
};
pub use shadows::{
    GpuShadowLight, GpuShadowView, MAX_SHADOW_VIEWS, SHADOW_ARRAY_LAYERS, SHADOW_DEPTH_FORMAT,
};
