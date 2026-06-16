//! Material resolution and batch-key caching for world-mesh draw prep.

mod cache;
mod key;
mod keys;
mod resolve;
mod slot;
mod transparent;

pub use cache::FrameMaterialBatchCache;
pub use key::{MaterialDrawBatchKey, compute_batch_key_hash};
pub use transparent::TransparentMaterialClass;

pub(crate) use resolve::{
    BILLBOARD_RENDER_BUFFER_BIT, MaterialResolveCtx, apply_render_buffer_mesh_pipeline_override,
    batch_key_for_slot_cached,
};
pub(crate) use slot::normalized_material_slot;
