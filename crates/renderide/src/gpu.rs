//! GPU device, adapter, swapchain, frame uniforms, profiling, and VR mirror blit.
//!
//! Top-level layout:
//! - [`context`] -- [`GpuContext`] (instance, surface, device, swapchain) and construction.
//! - [`adapter`] -- adapter selection, device creation, feature negotiation, MSAA probing.
//! - [`limits`] -- [`GpuLimits`] capability snapshot and bounds helpers.
//! - [`depth`] -- reverse-Z conventions, depth-stencil format choice, and [`OutputDepthMode`].
//! - [`frame_globals`] -- WGSL-matched per-frame uniform structs.
//! - [`frame_bindings`] -- shader ABI: `@group(0)` BGL, light rows, reflection-probe rows.
//! - [`profiling`] -- frame-bracket GPU timestamps and CPU/GPU wall-clock timing.
//! - [`sync`] -- Vulkan queue serialisation and mapped-buffer health.
//! - [`driver_thread`] -- dedicated submit/present worker.
//! - [`present`], [`display_blit`], [`vr_mirror`], [`msaa_depth_resolve`] -- presentation passes.
//! - [`bind_layout`] -- reusable [`wgpu::BindGroupLayoutEntry`] factories.
//! - [`instance_setup`] -- renderer-policy clamps applied at instance/device creation.
//!
//! `blit_kit` (private) holds helpers shared by [`display_blit`] and [`vr_mirror`].

mod adapter;
mod blit_kit;
mod context;
mod instance_setup;
mod submission_state;
mod sync;
mod vr_mirror;

pub(crate) mod bind_layout;
pub(crate) mod depth;
pub(crate) mod display_blit;
pub(crate) mod driver_thread;
pub(crate) mod frame_bindings;
pub(crate) mod frame_globals;
pub(crate) mod limits;
pub(crate) mod msaa_depth_resolve;
pub(crate) mod present;
pub(crate) mod profiling;

// --- Cross-layer re-exports (renderide-internal contract; not part of any external API) ---
pub(crate) use context::{GpuContext, GpuError};
pub(crate) use depth::{
    MAIN_FORWARD_DEPTH_CLEAR, MAIN_FORWARD_DEPTH_COMPARE, OutputDepthMode,
    main_forward_depth_stencil_format,
};
pub(crate) use display_blit::DisplayBlitResources;
pub(crate) use frame_bindings::{
    CLUSTER_LIGHT_RANGE_WORDS, CLUSTER_PARAMS_UNIFORM_SIZE, GpuLight, GpuReflectionProbeMetadata,
    LIGHT_COOKIE_KIND_NONE, LIGHT_COOKIE_KIND_POINT_CUBE, LIGHT_COOKIE_KIND_SPOT_2D, MAX_LIGHTS,
    REFLECTION_PROBE_ATLAS_FORMAT, REFLECTION_PROBE_METADATA_BOX_PROJECTION,
    REFLECTION_PROBE_METADATA_SH2_SOURCE_LOCAL, empty_material_bind_group_layout,
    frame_bind_group_layout, frame_bind_group_layout_entries,
};
pub(crate) use instance_setup::{RENDERER_MAX_TEXTURE_DIMENSION_2D, instance_flags_for_gpu_init};
pub(crate) use limits::{CUBEMAP_ARRAY_LAYERS, GpuLimits};
pub(crate) use msaa_depth_resolve::{
    MsaaDepthResolveMonoTargets, MsaaDepthResolveResources, MsaaDepthResolveStereoTargets,
};
pub(crate) use vr_mirror::{VR_MIRROR_EYE_LAYER, VrMirrorBlitResources};

// --- Legacy submodule-path re-exports (preserve external `crate::gpu::<x>::*` paths) ---
//
// External code references `crate::gpu::frame_cpu_gpu_timing::*` and
// `crate::gpu::GpuQueueAccessGate`; both physically live under newer parent modules now.
pub(crate) use profiling::frame_bracket;
pub(crate) use profiling::frame_cpu_gpu_timing;
pub(crate) use sync::mapped_buffer_health::GpuMappedBufferHealth;
pub(crate) use sync::queue_access_gate::{GpuQueueAccessGate, GpuQueueAccessMode};
