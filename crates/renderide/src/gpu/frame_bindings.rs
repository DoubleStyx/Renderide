//! Shared shader ABI contracts for `@group(0)` bind groups and packed GPU rows.
//!
//! - [`lights`] -- [`GpuLight`] row + per-frame buffer cap.
//! - [`reflection_probes`] -- [`GpuReflectionProbeMetadata`] row + probe metadata constants.
//! - [`cluster_params`] -- clustered-light compute slab sizing constants.
//! - [`bind_group`] -- `@group(0)` BindGroupLayout used by every material pipeline.

mod bind_group;
mod cluster_params;
mod lights;
mod reflection_probes;

pub use bind_group::{
    empty_material_bind_group_layout, frame_bind_group_layout, frame_bind_group_layout_entries,
};
pub use cluster_params::{CLUSTER_LIGHT_RANGE_WORDS, CLUSTER_PARAMS_UNIFORM_SIZE};
pub use lights::{
    GpuLight, GpuShadowView, LIGHT_COOKIE_KIND_DIRECTIONAL_2D, LIGHT_COOKIE_KIND_NONE,
    LIGHT_COOKIE_KIND_POINT_CUBE, LIGHT_COOKIE_KIND_SPOT_2D, LIGHT_COOKIE_WRAP_MODE_CLAMP,
    LIGHT_COOKIE_WRAP_MODE_MASK, LIGHT_COOKIE_WRAP_MODE_MIRROR, LIGHT_COOKIE_WRAP_MODE_MIRROR_ONCE,
    LIGHT_COOKIE_WRAP_MODE_REPEAT, LIGHT_COOKIE_WRAP_U_SHIFT, LIGHT_COOKIE_WRAP_V_SHIFT,
    MAX_LIGHTS, MAX_SHADOW_VIEWS, SHADOW_VIEW_KIND_DIRECTIONAL, SHADOW_VIEW_KIND_POINT,
    SHADOW_VIEW_KIND_SPOT,
};
pub use reflection_probes::{
    GpuReflectionProbeMetadata, REFLECTION_PROBE_ATLAS_FORMAT,
    REFLECTION_PROBE_METADATA_BOX_PROJECTION, REFLECTION_PROBE_METADATA_SH2_SOURCE_LOCAL,
};
