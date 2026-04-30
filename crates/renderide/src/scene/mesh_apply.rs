//! Static and skinned mesh renderable updates from shared memory.
//!
//! Split into three submodules: [`static_meshes`] handles the static-renderer dense table,
//! [`skinned_meshes`] orchestrates the skinned dense table plus its bone / blendshape /
//! bounds sub-applies, and [`fixups`] holds the transform-removal id sweeps that both paths
//! call before applying their dense updates. Module-private warn-once dedup state lives in
//! [`diagnostics`].

mod diagnostics;
pub(crate) mod fixups;
pub(crate) mod skinned_meshes;
pub(crate) mod static_meshes;

pub use skinned_meshes::ExtractedSkinnedMeshRenderablesUpdate;
pub use static_meshes::ExtractedMeshRenderablesUpdate;

pub(crate) use fixups::fixup_static_meshes_for_transform_removals;
pub(crate) use skinned_meshes::{
    apply_skinned_mesh_renderables_update_extracted, extract_skinned_mesh_renderables_update,
};
pub(crate) use static_meshes::{
    apply_mesh_renderables_update_extracted, extract_mesh_renderables_update,
};
