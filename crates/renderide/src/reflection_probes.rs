//! Nonblocking GPU SH2 projection for reflection-probe host tasks.

mod cubemap_assets;
mod projection_pipeline;
mod readback_jobs;
mod sh2_math;
mod sh2_system;
mod source_resolution;
pub(crate) mod specular;
mod task_rows;

use sh2_math::constant_color_sh2;
#[cfg(test)]
use sh2_system::SkyParamMode;
use sh2_system::{
    CubemapResidency, CubemapSourceMaterialIdentity, DEFAULT_SAMPLE_SIZE, GpuSh2Source,
    MAX_PENDING_JOB_AGE_FRAMES, SH2_OUTPUT_BYTES, Sh2ProjectParams, Sh2SourceKey,
};

pub(crate) use cubemap_assets::{ReflectionProbeCubemapAsset, ReflectionProbeCubemapAssets};
pub(crate) use sh2_system::ReflectionProbeSh2System;
