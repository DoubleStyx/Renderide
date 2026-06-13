//! Phase A: serial shared-memory pre-extract of one render space's update payloads.
//!
//! Every per-space helper takes `&mut SharedMemoryAccessor`, so reads stay serial here while the
//! owned payload bundles produced by [`extract_render_space_update`] feed the parallel Phase B
//! apply in [`super::mutate`].

use crate::ipc::SharedMemoryAccessor;
use crate::scene::blit_to_display::{ExtractedBlitToDisplayUpdate, extract_blit_to_display_update};
use crate::scene::camera::{ExtractedCameraRenderablesUpdate, extract_camera_renderables_update};
use crate::scene::camera_portal::{
    ExtractedCameraPortalRenderablesUpdate, extract_camera_portal_renderables_update,
};
use crate::scene::error::SceneError;
use crate::scene::ids::RenderSpaceId;
use crate::scene::layer::{ExtractedLayerUpdate, extract_layer_update};
use crate::scene::lod_groups::{
    ExtractedLodGroupRenderablesUpdate, extract_lod_group_renderables_update,
};
use crate::scene::meshes::{
    ExtractedMeshRenderablesUpdate, ExtractedSkinnedMeshRenderablesUpdate,
    extract_mesh_renderables_update, extract_skinned_mesh_renderables_update,
};
use crate::scene::overrides::{
    ExtractedRenderMaterialOverridesUpdate, ExtractedRenderTransformOverridesUpdate,
    extract_render_material_overrides_update, extract_render_transform_overrides_update,
};
use crate::scene::reflection_probe::{
    ExtractedReflectionProbeRenderablesUpdate, extract_reflection_probe_renderables_update,
};
use crate::scene::render_buffers::{
    ExtractedBillboardRenderBufferUpdate, ExtractedMeshRenderBufferUpdate,
    ExtractedTrailRendererUpdate, extract_billboard_render_buffer_update,
    extract_mesh_render_buffer_update, extract_trail_renderer_update,
};
use crate::scene::transforms::{ExtractedTransformsUpdate, extract_transforms_update};
use crate::shared::RenderSpaceUpdate;

macro_rules! extract_optional_render_space_update {
    ($shm:expr, $update:expr, $field:ident, $scope:literal, $extract:path $(, $extra:expr)* $(,)?) => {{
        match ($update).$field.as_ref() {
            Some(payload) => {
                profiling::scope!($scope);
                Some($extract($shm, payload, $($extra,)* ($update).id)?)
            }
            None => None,
        }
    }};
}

/// Owned per-space payload bundle: every shared-memory buffer referenced by one
/// [`RenderSpaceUpdate`] pre-read into [`Vec`]s, ready for parallel apply.
///
/// Each `Option<...>` field mirrors the corresponding `Option<...>` on [`RenderSpaceUpdate`] and is
/// `None` when the host omitted that update kind for this tick.
pub(in crate::scene::coordinator) struct ExtractedRenderSpaceUpdate {
    /// Render space identity for this chunk (mirrors [`RenderSpaceUpdate::id`]).
    pub space_id: RenderSpaceId,
    /// Camera-renderable update payload.
    pub cameras: Option<ExtractedCameraRenderablesUpdate>,
    /// Camera-portal renderable update payload.
    pub camera_portals: Option<ExtractedCameraPortalRenderablesUpdate>,
    /// Reflection-probe renderable update payload.
    pub reflection_probes: Option<ExtractedReflectionProbeRenderablesUpdate>,
    /// Dense transform-table update payload.
    pub transforms: Option<ExtractedTransformsUpdate>,
    /// Static mesh-renderable update payload.
    pub meshes: Option<ExtractedMeshRenderablesUpdate>,
    /// Skinned mesh-renderable update payload (state, bones, blendshapes).
    pub skinned_meshes: Option<ExtractedSkinnedMeshRenderablesUpdate>,
    /// Layer-assignment update payload.
    pub layers: Option<ExtractedLayerUpdate>,
    /// LOD group update payload.
    pub lod_groups: Option<ExtractedLodGroupRenderablesUpdate>,
    /// Render-context transform-override update payload.
    pub transform_overrides: Option<ExtractedRenderTransformOverridesUpdate>,
    /// Render-context material-override update payload.
    pub material_overrides: Option<ExtractedRenderMaterialOverridesUpdate>,
    /// `BlitToDisplay` renderables update payload.
    pub blit_to_displays: Option<ExtractedBlitToDisplayUpdate>,
    /// PhotonDust billboard renderer update payload.
    pub billboard_render_buffers: Option<ExtractedBillboardRenderBufferUpdate>,
    /// PhotonDust mesh-particle renderer update payload.
    pub mesh_render_buffers: Option<ExtractedMeshRenderBufferUpdate>,
    /// PhotonDust trail renderer update payload.
    pub trail_render_buffers: Option<ExtractedTrailRendererUpdate>,
}

/// Extracted renderer payloads tied directly to scene geometry and visibility.
struct ExtractedGeometryRenderSpaceUpdates {
    /// Camera renderer update payload.
    cameras: Option<ExtractedCameraRenderablesUpdate>,
    /// Camera-portal renderer update payload.
    camera_portals: Option<ExtractedCameraPortalRenderablesUpdate>,
    /// Reflection-probe renderer update payload.
    reflection_probes: Option<ExtractedReflectionProbeRenderablesUpdate>,
    /// Transform update payload.
    transforms: Option<ExtractedTransformsUpdate>,
    /// Static mesh renderer update payload.
    meshes: Option<ExtractedMeshRenderablesUpdate>,
    /// Skinned mesh renderer update payload.
    skinned_meshes: Option<ExtractedSkinnedMeshRenderablesUpdate>,
    /// Layer update payload.
    layers: Option<ExtractedLayerUpdate>,
    /// LOD-group renderer update payload.
    lod_groups: Option<ExtractedLodGroupRenderablesUpdate>,
}

/// Extracted renderer payloads for render-context state and generated particle renderers.
struct ExtractedContextRenderSpaceUpdates {
    /// Render-context transform-override update payload.
    transform_overrides: Option<ExtractedRenderTransformOverridesUpdate>,
    /// Render-context material-override update payload.
    material_overrides: Option<ExtractedRenderMaterialOverridesUpdate>,
    /// `BlitToDisplay` renderables update payload.
    blit_to_displays: Option<ExtractedBlitToDisplayUpdate>,
    /// PhotonDust billboard renderer update payload.
    billboard_render_buffers: Option<ExtractedBillboardRenderBufferUpdate>,
    /// PhotonDust mesh-particle renderer update payload.
    mesh_render_buffers: Option<ExtractedMeshRenderBufferUpdate>,
    /// PhotonDust trail renderer update payload.
    trail_render_buffers: Option<ExtractedTrailRendererUpdate>,
}

/// Reads every shared-memory buffer referenced by `update` into owned vectors.
///
/// Light updates are intentionally **not** extracted here: their apply step mutates the shared
/// [`crate::scene::LightCache`] and is handled in a separate serial pass (see
/// [`super::light_updates_view`]).
pub(in crate::scene::coordinator) fn extract_render_space_update(
    shm: &mut SharedMemoryAccessor,
    update: &RenderSpaceUpdate,
    frame_index: i32,
) -> Result<ExtractedRenderSpaceUpdate, SceneError> {
    profiling::scope!("scene::extract_render_space");
    let space_id = RenderSpaceId(update.id);
    let geometry = extract_geometry_render_space_updates(shm, update, frame_index)?;
    let context = extract_context_render_space_updates(shm, update)?;
    Ok(ExtractedRenderSpaceUpdate {
        space_id,
        cameras: geometry.cameras,
        camera_portals: geometry.camera_portals,
        reflection_probes: geometry.reflection_probes,
        transforms: geometry.transforms,
        meshes: geometry.meshes,
        skinned_meshes: geometry.skinned_meshes,
        layers: geometry.layers,
        lod_groups: geometry.lod_groups,
        transform_overrides: context.transform_overrides,
        material_overrides: context.material_overrides,
        blit_to_displays: context.blit_to_displays,
        billboard_render_buffers: context.billboard_render_buffers,
        mesh_render_buffers: context.mesh_render_buffers,
        trail_render_buffers: context.trail_render_buffers,
    })
}

/// Extracts scene-geometry update payloads referenced by `update`.
fn extract_geometry_render_space_updates(
    shm: &mut SharedMemoryAccessor,
    update: &RenderSpaceUpdate,
    frame_index: i32,
) -> Result<ExtractedGeometryRenderSpaceUpdates, SceneError> {
    Ok(ExtractedGeometryRenderSpaceUpdates {
        cameras: extract_optional_render_space_update!(
            shm,
            update,
            cameras_update,
            "scene::extract_render_space::cameras",
            extract_camera_renderables_update,
        ),
        camera_portals: extract_optional_render_space_update!(
            shm,
            update,
            camera_portals_update,
            "scene::extract_render_space::camera_portals",
            extract_camera_portal_renderables_update,
        ),
        reflection_probes: extract_optional_render_space_update!(
            shm,
            update,
            reflection_probes_update,
            "scene::extract_render_space::reflection_probes",
            extract_reflection_probe_renderables_update,
        ),
        transforms: extract_optional_render_space_update!(
            shm,
            update,
            transforms_update,
            "scene::extract_render_space::transforms",
            extract_transforms_update,
            frame_index,
        ),
        meshes: extract_optional_render_space_update!(
            shm,
            update,
            mesh_renderers_update,
            "scene::extract_render_space::meshes",
            extract_mesh_renderables_update,
        ),
        skinned_meshes: extract_optional_render_space_update!(
            shm,
            update,
            skinned_mesh_renderers_update,
            "scene::extract_render_space::skinned_meshes",
            extract_skinned_mesh_renderables_update,
        ),
        layers: extract_optional_render_space_update!(
            shm,
            update,
            layers_update,
            "scene::extract_render_space::layers",
            extract_layer_update,
        ),
        lod_groups: extract_optional_render_space_update!(
            shm,
            update,
            lod_group_update,
            "scene::extract_render_space::lod_groups",
            extract_lod_group_renderables_update,
        ),
    })
}

/// Extracts render-context and generated-particle update payloads referenced by `update`.
fn extract_context_render_space_updates(
    shm: &mut SharedMemoryAccessor,
    update: &RenderSpaceUpdate,
) -> Result<ExtractedContextRenderSpaceUpdates, SceneError> {
    Ok(ExtractedContextRenderSpaceUpdates {
        transform_overrides: extract_optional_render_space_update!(
            shm,
            update,
            render_transform_overrides_update,
            "scene::extract_render_space::transform_overrides",
            extract_render_transform_overrides_update,
        ),
        material_overrides: extract_optional_render_space_update!(
            shm,
            update,
            render_material_overrides_update,
            "scene::extract_render_space::material_overrides",
            extract_render_material_overrides_update,
        ),
        blit_to_displays: extract_optional_render_space_update!(
            shm,
            update,
            blit_to_displays_update,
            "scene::extract_render_space::blit_to_displays",
            extract_blit_to_display_update,
        ),
        billboard_render_buffers: extract_optional_render_space_update!(
            shm,
            update,
            billboard_buffers_update,
            "scene::extract_render_space::billboard_render_buffers",
            extract_billboard_render_buffer_update,
        ),
        mesh_render_buffers: extract_optional_render_space_update!(
            shm,
            update,
            mesh_render_buffers_update,
            "scene::extract_render_space::mesh_render_buffers",
            extract_mesh_render_buffer_update,
        ),
        trail_render_buffers: extract_optional_render_space_update!(
            shm,
            update,
            trail_renderers_update,
            "scene::extract_render_space::trail_renderers",
            extract_trail_renderer_update,
        ),
    })
}
