//! Resolves the active main render space's skybox material into a unified IBL bake source.
//!
//! Returns an [`SkyboxIblSource`] that the [`crate::skybox::SkyboxIblCache`] consumes to schedule
//! a GGX-prefiltered cubemap bake. Three source variants are produced today:
//! - [`SkyboxIblSource::Analytic`] for procedural / gradient skyboxes.
//! - [`SkyboxIblSource::Cubemap`] for Projection360 `_MainCube` (or `_MainTex` cubemap fallback).
//! - [`SkyboxIblSource::Equirect`] for Projection360 `_MainTex` Texture2D.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::assets::asset_transfer_queue::AssetTransferQueue;
use crate::assets::texture::HostTextureAssetKind;
use crate::backend::material_property_reader::{float4_property, texture_property};
use crate::materials::MaterialSystem;
use crate::materials::host_data::{
    MaterialPropertyLookupIds, MaterialPropertyStore, PropertyIdRegistry,
};
use crate::scene::SceneCoordinator;
use crate::skybox::params::{
    DEFAULT_MAIN_TEX_ST, PROJECTION360_DEFAULT_FOV, SkyboxEvaluatorParams, gradient_sky_params,
    procedural_sky_params,
};

/// Active skybox source to be baked into a GGX-prefiltered cubemap.
///
/// `Analytic` is boxed because [`AnalyticIblSource`] embeds the full
/// [`SkyboxEvaluatorParams`] (~=1.1 KiB of gradient arrays), an order of magnitude larger than the
/// other variants.
pub(crate) enum SkyboxIblSource {
    /// Analytic procedural / gradient skybox evaluator.
    Analytic(Box<AnalyticIblSource>),
    /// Resident host-uploaded cubemap (`Projection360 _MainCube`).
    Cubemap(CubemapIblSource),
    /// Resident host-uploaded equirect Texture2D (`Projection360 _MainTex`).
    Equirect(EquirectIblSource),
}

/// Analytic skybox material identity and evaluator parameters.
pub(crate) struct AnalyticIblSource {
    /// Active skybox material asset id.
    pub material_asset_id: i32,
    /// Material property generation; invalidates the bake when material props change.
    pub material_generation: u64,
    /// Stable hash of the shader route stem ("gradient" / "procedural" variants).
    pub route_hash: u64,
    /// Packed evaluator parameters for the analytic mip-0 producer.
    pub params: SkyboxEvaluatorParams,
}

/// Resident cubemap source identity and GPU handle.
pub(crate) struct CubemapIblSource {
    /// Source cubemap asset id.
    pub asset_id: i32,
    /// Resident cubemap face edge in texels (mip 0).
    pub face_size: u32,
    /// Resident mip count of the source cubemap.
    pub mip_levels_resident: u32,
    /// Whether sampling needs V-axis storage compensation.
    pub storage_v_inverted: bool,
    /// Cube-dimension texture view used as the bake input.
    pub view: Arc<wgpu::TextureView>,
}

/// Resident equirect Texture2D source identity and GPU handle.
pub(crate) struct EquirectIblSource {
    /// Source Texture2D asset id.
    pub asset_id: i32,
    /// Resident mip count of the source texture.
    pub mip_levels_resident: u32,
    /// Whether sampling needs V-axis storage compensation.
    pub storage_v_inverted: bool,
    /// 2D texture view used as the bake input.
    pub view: Arc<wgpu::TextureView>,
    /// Projection360 `_FOV` parameters.
    pub equirect_fov: [f32; 4],
    /// Projection360 `_MainTex_ST` parameters.
    pub equirect_st: [f32; 4],
}

/// Resolves the active main render space's skybox material into an IBL bake source.
pub(crate) fn resolve_active_main_skybox_ibl_source(
    scene: &SceneCoordinator,
    materials: &MaterialSystem,
    assets: &AssetTransferQueue,
) -> Option<SkyboxIblSource> {
    let material_asset_id = active_main_skybox_material_asset_id(scene)?;
    let store = materials.material_property_store();
    let shader_asset_id = store.shader_asset_for_material(material_asset_id)?;
    let route_name = shader_route_name(materials, shader_asset_id)?;
    let route_lower = route_name.to_ascii_lowercase();
    let lookup = MaterialPropertyLookupIds {
        material_asset_id,
        mesh_property_block_slot0: None,
    };

    if route_lower.contains("projection360") {
        return resolve_projection360_source(
            store,
            materials.property_id_registry(),
            assets,
            lookup,
        );
    }
    if route_lower.contains("gradient") {
        let params = gradient_sky_params(store, materials.property_id_registry(), lookup);
        return Some(SkyboxIblSource::Analytic(Box::new(AnalyticIblSource {
            material_asset_id,
            material_generation: store.material_generation(material_asset_id),
            route_hash: hash_route_name(&route_name),
            params,
        })));
    }
    if route_lower.contains("procedural") {
        let params = procedural_sky_params(store, materials.property_id_registry(), lookup);
        return Some(SkyboxIblSource::Analytic(Box::new(AnalyticIblSource {
            material_asset_id,
            material_generation: store.material_generation(material_asset_id),
            route_hash: hash_route_name(&route_name),
            params,
        })));
    }
    logger::trace!(
        "skybox specular: unsupported active skybox route '{route_name}' for material {material_asset_id}"
    );
    None
}

/// Returns the skybox material id from the active non-overlay render space.
fn active_main_skybox_material_asset_id(scene: &SceneCoordinator) -> Option<i32> {
    let material_asset_id = scene.active_main_space()?.skybox_material_asset_id;
    (material_asset_id >= 0).then_some(material_asset_id)
}

/// Returns a shader route name or stem for a shader asset id.
fn shader_route_name(materials: &MaterialSystem, shader_asset_id: i32) -> Option<String> {
    let registry = materials.material_registry()?;
    registry
        .stem_for_shader_asset(shader_asset_id)
        .map(str::to_string)
}

/// Hashes a shader route name into a stable cache discriminator.
pub(crate) fn hash_route_name(route: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    route.hash(&mut hasher);
    hasher.finish()
}

/// Resolves the primary Projection360 source from a skybox material.
fn resolve_projection360_source(
    store: &MaterialPropertyStore,
    registry: &PropertyIdRegistry,
    assets: &AssetTransferQueue,
    lookup: MaterialPropertyLookupIds,
) -> Option<SkyboxIblSource> {
    let main_cube = texture_property(store, registry, lookup, "_MainCube")
        .or_else(|| texture_property(store, registry, lookup, "_Cube"));
    if let Some((asset_id, HostTextureAssetKind::Cubemap)) = main_cube {
        return resolve_projection360_cubemap_source(assets, asset_id);
    }
    if let Some((asset_id, kind)) = main_cube {
        logger::trace!(
            "skybox specular: Projection360 _MainCube asset {asset_id} has unsupported kind {kind:?}"
        );
    }

    match texture_property(store, registry, lookup, "_MainTex")
        .or_else(|| texture_property(store, registry, lookup, "_Tex"))
    {
        Some((asset_id, HostTextureAssetKind::Texture2D)) => {
            resolve_projection360_equirect_source(store, registry, assets, lookup, asset_id)
        }
        Some((asset_id, HostTextureAssetKind::Cubemap)) => {
            resolve_projection360_cubemap_source(assets, asset_id)
        }
        Some((asset_id, kind)) => {
            logger::trace!(
                "skybox specular: Projection360 _MainTex asset {asset_id} has unsupported kind {kind:?}"
            );
            None
        }
        None => {
            logger::trace!("skybox specular: Projection360 skybox has no _MainCube or _MainTex");
            None
        }
    }
}

/// Resolves a resident Projection360 cubemap source.
fn resolve_projection360_cubemap_source(
    assets: &AssetTransferQueue,
    asset_id: i32,
) -> Option<SkyboxIblSource> {
    let Some(cubemap) = assets.cubemap_pool().get(asset_id) else {
        logger::trace!("skybox specular: cubemap asset {asset_id} is not allocated yet");
        return None;
    };
    if cubemap.mip_levels_resident == 0 {
        logger::trace!("skybox specular: cubemap asset {asset_id} has no resident mips");
        return None;
    }
    Some(SkyboxIblSource::Cubemap(CubemapIblSource {
        asset_id,
        face_size: cubemap.size,
        mip_levels_resident: cubemap.mip_levels_resident,
        storage_v_inverted: cubemap.storage_v_inverted,
        view: cubemap.view.clone(),
    }))
}

/// Resolves a resident Projection360 equirectangular Texture2D source.
fn resolve_projection360_equirect_source(
    store: &MaterialPropertyStore,
    registry: &PropertyIdRegistry,
    assets: &AssetTransferQueue,
    lookup: MaterialPropertyLookupIds,
    asset_id: i32,
) -> Option<SkyboxIblSource> {
    let Some(texture) = assets.texture_pool().get(asset_id) else {
        logger::trace!("skybox specular: equirect Texture2D asset {asset_id} is not allocated yet");
        return None;
    };
    if texture.mip_levels_resident == 0 {
        logger::trace!("skybox specular: equirect Texture2D asset {asset_id} has no resident mips");
        return None;
    }
    Some(SkyboxIblSource::Equirect(EquirectIblSource {
        asset_id,
        mip_levels_resident: texture.mip_levels_resident,
        storage_v_inverted: texture.storage_v_inverted,
        view: texture.view.clone(),
        equirect_fov: float4_property(store, registry, lookup, "_FOV", PROJECTION360_DEFAULT_FOV),
        equirect_st: float4_property(store, registry, lookup, "_MainTex_ST", DEFAULT_MAIN_TEX_ST),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::ipc::SharedMemoryAccessor;
    use crate::materials::host_data::MaterialPropertyValue;
    use crate::scene::SceneCoordinator;
    use crate::shared::{FrameSubmitData, RenderSpaceUpdate};

    /// Packs a host texture id using the same high-bit asset-kind tag as the shared host packer.
    fn pack_host_texture(asset_id: i32, kind: HostTextureAssetKind) -> i32 {
        let type_bits = 3u32;
        let pack_type_shift = 32u32 - type_bits;
        ((asset_id as u32) | ((kind as u32) << pack_type_shift)) as i32
    }

    /// Creates a lookup and empty material property store for resolver tests.
    fn store_and_lookup(
        material_asset_id: i32,
    ) -> (MaterialPropertyStore, MaterialPropertyLookupIds) {
        (
            MaterialPropertyStore::new(),
            MaterialPropertyLookupIds {
                material_asset_id,
                mesh_property_block_slot0: None,
            },
        )
    }

    #[test]
    fn projection360_prefers_main_cube_over_main_tex() {
        let registry = PropertyIdRegistry::new();
        let (mut store, lookup) = store_and_lookup(7);
        store.set_material(
            lookup.material_asset_id,
            registry.intern("_MainTex"),
            MaterialPropertyValue::Texture(pack_host_texture(11, HostTextureAssetKind::Cubemap)),
        );
        store.set_material(
            lookup.material_asset_id,
            registry.intern("_MainCube"),
            MaterialPropertyValue::Texture(pack_host_texture(42, HostTextureAssetKind::Cubemap)),
        );

        assert_eq!(
            resolve_projection360_source_kind(&store, &registry, lookup),
            Some((42, HostTextureAssetKind::Cubemap))
        );
    }

    #[test]
    fn projection360_accepts_cubemap_main_tex_fallback() {
        let registry = PropertyIdRegistry::new();
        let (mut store, lookup) = store_and_lookup(8);
        store.set_material(
            lookup.material_asset_id,
            registry.intern("_MainTex"),
            MaterialPropertyValue::Texture(pack_host_texture(13, HostTextureAssetKind::Cubemap)),
        );

        assert_eq!(
            resolve_projection360_source_kind(&store, &registry, lookup),
            Some((13, HostTextureAssetKind::Cubemap))
        );
    }

    #[test]
    fn projection360_accepts_texture2d_main_tex() {
        let registry = PropertyIdRegistry::new();
        let (mut store, lookup) = store_and_lookup(9);
        store.set_material(
            lookup.material_asset_id,
            registry.intern("_MainTex"),
            MaterialPropertyValue::Texture(pack_host_texture(15, HostTextureAssetKind::Texture2D)),
        );

        assert_eq!(
            resolve_projection360_source_kind(&store, &registry, lookup),
            Some((15, HostTextureAssetKind::Texture2D))
        );
    }

    #[test]
    fn projection360_rejects_unsupported_main_cube_without_main_tex() {
        let registry = PropertyIdRegistry::new();
        let (mut store, lookup) = store_and_lookup(9);
        store.set_material(
            lookup.material_asset_id,
            registry.intern("_MainCube"),
            MaterialPropertyValue::Texture(pack_host_texture(15, HostTextureAssetKind::Texture2D)),
        );

        assert_eq!(
            resolve_projection360_source_kind(&store, &registry, lookup),
            None
        );
    }

    /// Resolves only the property-level source kind for unit tests that do not allocate GPU assets.
    fn resolve_projection360_source_kind(
        store: &MaterialPropertyStore,
        registry: &PropertyIdRegistry,
        lookup: MaterialPropertyLookupIds,
    ) -> Option<(i32, HostTextureAssetKind)> {
        let main_cube = texture_property(store, registry, lookup, "_MainCube")
            .or_else(|| texture_property(store, registry, lookup, "_Cube"));
        if let Some((asset_id, HostTextureAssetKind::Cubemap)) = main_cube {
            return Some((asset_id, HostTextureAssetKind::Cubemap));
        }
        let main_tex = texture_property(store, registry, lookup, "_MainTex")
            .or_else(|| texture_property(store, registry, lookup, "_Tex"));
        match main_tex {
            Some((
                asset_id,
                kind @ (HostTextureAssetKind::Texture2D | HostTextureAssetKind::Cubemap),
            )) => Some((asset_id, kind)),
            _ => None,
        }
    }

    #[test]
    fn active_main_skybox_uses_lowest_active_non_overlay_space() {
        let mut scene = SceneCoordinator::new();
        let mut shm = SharedMemoryAccessor::new(String::new());
        scene
            .apply_frame_submit(
                &mut shm,
                &FrameSubmitData {
                    render_spaces: vec![
                        RenderSpaceUpdate {
                            id: 10,
                            is_active: true,
                            is_overlay: false,
                            skybox_material_asset_id: 100,
                            ..RenderSpaceUpdate::default()
                        },
                        RenderSpaceUpdate {
                            id: 2,
                            is_active: true,
                            is_overlay: false,
                            skybox_material_asset_id: 200,
                            ..RenderSpaceUpdate::default()
                        },
                        RenderSpaceUpdate {
                            id: 1,
                            is_active: true,
                            is_overlay: true,
                            skybox_material_asset_id: 300,
                            ..RenderSpaceUpdate::default()
                        },
                    ],
                    ..FrameSubmitData::default()
                },
            )
            .expect("empty render-space headers should apply without shared buffers");

        assert_eq!(active_main_skybox_material_asset_id(&scene), Some(200));
    }
}
