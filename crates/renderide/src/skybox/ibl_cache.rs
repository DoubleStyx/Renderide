//! Unified IBL bake cache for specular reflection sources.
//!
//! Owns one in-flight bake job tracker, mip-0 producer pipelines for constant colors and cubemaps,
//! one source-pyramid downsample pipeline, a seam stitch pipeline, and one GGX convolve pipeline.
//! For each new active reflection source the cache:
//!
//! 1. Allocates a source Rgba16Float cubemap and a filtered output cubemap with full mip chains.
//! 2. Records a mip-0 producer compute pass that converts the source into a scratch cube.
//! 3. Stitches scratch mip 0 into the source cube and copies it into filtered output mip 0.
//! 4. Records downsample passes that build and stitch the source radiance mip pyramid.
//! 5. Records one GGX/cosine convolve pass per filtered mip in `1..N`, then stitches each mip.
//! 6. Submits the encoder through [`crate::backend::gpu_jobs::GpuSubmitJobTracker`] and parks the
//!    cube in `pending` until the submit-completion callback promotes it to `completed`.
//!
//! The completed prefiltered cube is reused by reflection probes so every source type reaches
//! shader sampling through a single GGX-prefiltered cube.

mod bind_groups;
mod cache;
mod convolver;
mod encode;
mod errors;
mod key;
mod mip_loop;
mod pipeline;
mod pipeline_store;
mod resources;
mod sampler;
#[cfg(test)]
mod topology;

pub(crate) use cache::SkyboxIblCache;
pub(crate) use convolver::SkyboxIblConvolver;
pub(crate) use key::{SkyboxIblKey, build_key, clamp_face_size, mip_extent, mip_levels_for_edge};
