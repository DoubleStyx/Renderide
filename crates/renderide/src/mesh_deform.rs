//! Mesh skinning / blendshape scatter compute preprocess, sparse buffer checks, and per-draw
//! uniform packing for `@group(2)` in the world mesh forward pass.

mod blendshape_bind_chunks;
mod mesh_preprocess;
mod per_draw_uniforms;
mod range_alloc;
mod scratch;
mod skin_cache;
mod skinning_palette;
mod wgsl_mat3x3;

pub use blendshape_bind_chunks::{
    BLENDSHAPE_SPARSE_MIN_BUFFER_BYTES, blendshape_sparse_buffers_fit_device,
    plan_blendshape_scatter_chunks,
};
pub use mesh_preprocess::MeshPreprocessPipelines;
pub use per_draw_uniforms::{
    INITIAL_PER_DRAW_UNIFORM_SLOTS, PER_DRAW_UNIFORM_STRIDE, PaddedPerDrawUniforms,
};
pub use range_alloc::Range;
pub use scratch::{
    BlendshapeBindGroupKey, MeshDeformScratch, SkinningBindGroupKey, advance_slab_cursor,
    buffer_identity,
};
pub use skin_cache::{
    DeformSignature, EntryNeed, GpuSkinCache, SkinCacheEntry, SkinCacheKey, SkinCacheRendererKind,
};
pub use skinning_palette::{
    SkinningPaletteParams, write_skinning_palette_bytes, write_skinning_palette_bytes_serial,
};
