//! Host-driven lights: CPU cache from [`FrameSubmitData`](crate::shared::FrameSubmitData) and light buffer submissions.
//!
//! Scene lights are **logical** state (poses, types, shadow params). GPU storage buffer allocation
//! and [`crate::backend::light_gpu::GpuLight`] packing live in the backend.

mod apply;
mod cache;
mod types;

pub use apply::{apply_light_renderables_update, apply_lights_buffer_renderers_update};
pub use cache::LightCache;
pub use types::{light_casts_shadows, CachedLight, ResolvedLight};
