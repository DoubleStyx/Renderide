//! GPU state, pipelines, and mesh rendering.

pub mod accel;
pub mod cluster_buffer;
pub mod frame_scheduler;
pub mod mesh;
pub mod pipeline;
pub mod registry;
pub mod rtao_textures;
pub mod state;

pub use accel::{
    AccelCache, RayTracingState, build_blas_for_mesh, build_tlas, needs_scene_ray_tracing_accel,
    remove_blas, shadow_cast_mode_in_scene_tlas, update_tlas,
};
pub use frame_scheduler::GpuFrameScheduler;
pub use mesh::{GpuMeshBuffers, compute_vertex_stride_from_mesh, create_mesh_buffers};
pub use pipeline::{MAX_INSTANCE_RUN, RenderPipeline, UniformData};
pub use registry::{PipelineKey, PipelineManager, PipelineRegistry, PipelineVariant};
pub use state::{
    GpuState, clamp_surface_extent, ensure_depth_texture, init_gpu, reconfigure_surface_for_window,
};
