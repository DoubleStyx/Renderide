//! Source-key resolution for reflection-probe SH2 projection tasks.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use glam::Vec4;

use super::task_rows::TaskHeader;
use super::{
    constant_color_sh2, storage_v_inverted_flag, GpuSh2Source, Projection360EquirectKey,
    Sh2ProjectParams, Sh2SourceKey, SkyParamMode, DEFAULT_MAIN_TEX_ST, DEFAULT_SAMPLE_SIZE,
    PROJECTION360_DEFAULT_FOV,
};
use crate::assets::material::{
    MaterialPropertyLookupIds, MaterialPropertyStore, MaterialPropertyValue, PropertyIdRegistry,
};
use crate::assets::texture::{unpack_host_texture_packed, HostTextureAssetKind};
use crate::scene::{reflection_probe_skybox_only, RenderSpaceId, SceneCoordinator};
use crate::shared::{ReflectionProbeClear, ReflectionProbeType, RenderSH2};

/// Either a synchronous CPU result or a GPU source to project.
pub(super) enum Sh2ResolvedSource {
    /// CPU-computed SH2.
    Cpu(Box<RenderSH2>),
    /// GPU-computed SH2 source.
    Gpu(GpuSh2Source),
    /// Source is expected to become available later.
    Postpone,
}

/// Resolves a host task into a cache key and source payload.
pub(super) fn resolve_task_source(
    scene: &SceneCoordinator,
    materials: &crate::backend::MaterialSystem,
    assets: &crate::backend::AssetTransferQueue,
    render_space_id: i32,
    task: TaskHeader,
) -> Option<(Sh2SourceKey, Sh2ResolvedSource)> {
    if task.renderable_index < 0 || task.reflection_probe_renderable_index < 0 {
        return None;
    }
    let space = scene.space(RenderSpaceId(render_space_id))?;
    let probe = space
        .reflection_probes
        .get(task.reflection_probe_renderable_index as usize)?;
    let state = probe.state;
    if state.clear_flags == ReflectionProbeClear::Color {
        let color = state.background_color * state.intensity.max(0.0);
        let key = Sh2SourceKey::ConstantColor {
            render_space_id,
            color_bits: vec4_bits(color),
        };
        return Some((
            key,
            Sh2ResolvedSource::Cpu(Box::new(constant_color_sh2(color.truncate()))),
        ));
    }

    if state.r#type == ReflectionProbeType::Baked {
        if state.cubemap_asset_id < 0 {
            return None;
        }
        let Some(cubemap) = assets.cubemap_pool.get_texture(state.cubemap_asset_id) else {
            return Some((
                Sh2SourceKey::Cubemap {
                    render_space_id,
                    asset_id: state.cubemap_asset_id,
                    size: 0,
                    resident_mips: 0,
                    sample_size: DEFAULT_SAMPLE_SIZE,
                    material_generation: 0,
                },
                Sh2ResolvedSource::Postpone,
            ));
        };
        if cubemap.mip_levels_resident == 0 {
            return Some((
                Sh2SourceKey::Cubemap {
                    render_space_id,
                    asset_id: state.cubemap_asset_id,
                    size: cubemap.size,
                    resident_mips: 0,
                    sample_size: DEFAULT_SAMPLE_SIZE,
                    material_generation: 0,
                },
                Sh2ResolvedSource::Postpone,
            ));
        }
        let key = Sh2SourceKey::Cubemap {
            render_space_id,
            asset_id: state.cubemap_asset_id,
            size: cubemap.size,
            resident_mips: cubemap.mip_levels_resident,
            sample_size: DEFAULT_SAMPLE_SIZE,
            material_generation: 0,
        };
        return Some((
            key,
            Sh2ResolvedSource::Gpu(GpuSh2Source::Cubemap {
                asset_id: state.cubemap_asset_id,
            }),
        ));
    }

    if !reflection_probe_skybox_only(state.flags) {
        return None;
    }
    resolve_skybox_source(
        render_space_id,
        space.skybox_material_asset_id,
        materials,
        assets,
    )
}

/// Resolves an active skybox material into a source payload.
fn resolve_skybox_source(
    render_space_id: i32,
    material_asset_id: i32,
    materials: &crate::backend::MaterialSystem,
    assets: &crate::backend::AssetTransferQueue,
) -> Option<(Sh2SourceKey, Sh2ResolvedSource)> {
    if material_asset_id < 0 {
        return None;
    }
    let store = materials.material_property_store();
    let generation = store.material_generation(material_asset_id);
    let shader_asset_id = store.shader_asset_for_material(material_asset_id)?;
    let route_name = shader_route_name(materials, shader_asset_id);
    let route_hash = hash_route_name(route_name.as_deref().unwrap_or(""));
    let lookup = MaterialPropertyLookupIds {
        material_asset_id,
        mesh_property_block_slot0: None,
    };
    let registry = materials.property_id_registry();

    if route_name
        .as_deref()
        .is_some_and(|name| name.to_ascii_lowercase().contains("projection360"))
    {
        return resolve_projection360_source(
            render_space_id,
            store,
            registry,
            assets,
            lookup,
            generation,
        );
    }
    if route_name
        .as_deref()
        .is_some_and(|name| name.to_ascii_lowercase().contains("gradient"))
    {
        let params = gradient_sky_params(store, registry, lookup);
        let key = Sh2SourceKey::SkyParams {
            render_space_id,
            material_asset_id,
            material_generation: generation,
            sample_size: DEFAULT_SAMPLE_SIZE,
            route_hash,
        };
        return Some((
            key,
            Sh2ResolvedSource::Gpu(GpuSh2Source::SkyParams {
                params: Box::new(params),
            }),
        ));
    }
    if route_name
        .as_deref()
        .is_some_and(|name| name.to_ascii_lowercase().contains("procedural"))
    {
        let params = procedural_sky_params(store, registry, lookup);
        let key = Sh2SourceKey::SkyParams {
            render_space_id,
            material_asset_id,
            material_generation: generation,
            sample_size: DEFAULT_SAMPLE_SIZE,
            route_hash,
        };
        return Some((
            key,
            Sh2ResolvedSource::Gpu(GpuSh2Source::SkyParams {
                params: Box::new(params),
            }),
        ));
    }
    None
}

/// Returns a shader route name or stem for a shader asset id.
fn shader_route_name(
    materials: &crate::backend::MaterialSystem,
    shader_asset_id: i32,
) -> Option<String> {
    let registry = materials.material_registry()?;
    if let Some(stem) = registry.stem_for_shader_asset(shader_asset_id) {
        return Some(stem.to_string());
    }
    registry
        .shader_routes_for_hud()
        .into_iter()
        .find(|(id, _, _)| *id == shader_asset_id)
        .and_then(|(_, _, name)| name)
}

/// Resolves a `Projection360` material to a texture-backed source.
fn resolve_projection360_source(
    render_space_id: i32,
    store: &MaterialPropertyStore,
    registry: &PropertyIdRegistry,
    assets: &crate::backend::AssetTransferQueue,
    lookup: MaterialPropertyLookupIds,
    generation: u64,
) -> Option<(Sh2SourceKey, Sh2ResolvedSource)> {
    let main_cube = property_texture(store, registry, lookup, "_MainCube")
        .or_else(|| property_texture(store, registry, lookup, "_Cube"));
    if let Some((asset_id, HostTextureAssetKind::Cubemap)) = main_cube {
        return Some(resolve_projection360_cubemap_source(
            render_space_id,
            assets,
            asset_id,
            generation,
        ));
    }

    let main_tex = property_texture(store, registry, lookup, "_MainTex")
        .or_else(|| property_texture(store, registry, lookup, "_Tex"));
    match main_tex {
        Some((asset_id, HostTextureAssetKind::Texture2D)) => {
            Some(resolve_projection360_texture2d_source(
                render_space_id,
                store,
                registry,
                assets,
                lookup,
                asset_id,
                generation,
            ))
        }
        Some((asset_id, HostTextureAssetKind::Cubemap)) => Some(
            resolve_projection360_cubemap_source(render_space_id, assets, asset_id, generation),
        ),
        _ => None,
    }
}

/// Resolves a `Projection360` cubemap binding into an SH2 source.
fn resolve_projection360_cubemap_source(
    render_space_id: i32,
    assets: &crate::backend::AssetTransferQueue,
    asset_id: i32,
    generation: u64,
) -> (Sh2SourceKey, Sh2ResolvedSource) {
    let Some(cubemap) = assets.cubemap_pool.get_texture(asset_id) else {
        return (
            Sh2SourceKey::Cubemap {
                render_space_id,
                asset_id,
                size: 0,
                resident_mips: 0,
                sample_size: DEFAULT_SAMPLE_SIZE,
                material_generation: generation,
            },
            Sh2ResolvedSource::Postpone,
        );
    };
    let key = Sh2SourceKey::Cubemap {
        render_space_id,
        asset_id,
        size: cubemap.size,
        resident_mips: cubemap.mip_levels_resident,
        sample_size: DEFAULT_SAMPLE_SIZE,
        material_generation: generation,
    };
    if cubemap.mip_levels_resident == 0 {
        return (key, Sh2ResolvedSource::Postpone);
    }
    (
        key,
        Sh2ResolvedSource::Gpu(GpuSh2Source::Cubemap { asset_id }),
    )
}

/// Resolves a `Projection360` equirectangular 2D binding into an SH2 source.
fn resolve_projection360_texture2d_source(
    render_space_id: i32,
    store: &MaterialPropertyStore,
    registry: &PropertyIdRegistry,
    assets: &crate::backend::AssetTransferQueue,
    lookup: MaterialPropertyLookupIds,
    asset_id: i32,
    generation: u64,
) -> (Sh2SourceKey, Sh2ResolvedSource) {
    let mut params = projection360_equirect_params(store, registry, lookup, false);
    let Some(tex) = assets.texture_pool.get_texture(asset_id) else {
        return (
            projection360_equirect_source_key(
                render_space_id,
                asset_id,
                0,
                0,
                0,
                generation,
                &params,
            ),
            Sh2ResolvedSource::Postpone,
        );
    };
    params = projection360_equirect_params(store, registry, lookup, tex.storage_v_inverted);
    let key = projection360_equirect_source_key(
        render_space_id,
        asset_id,
        tex.width,
        tex.height,
        tex.mip_levels_resident,
        generation,
        &params,
    );
    if tex.mip_levels_resident == 0 {
        return (key, Sh2ResolvedSource::Postpone);
    }
    (
        key,
        Sh2ResolvedSource::Gpu(GpuSh2Source::EquirectTexture2D {
            asset_id,
            params: Box::new(params),
        }),
    )
}

/// Builds an equirectangular source key from texture residency and Projection360 parameters.
fn projection360_equirect_source_key(
    render_space_id: i32,
    asset_id: i32,
    width: u32,
    height: u32,
    resident_mips: u32,
    generation: u64,
    params: &Sh2ProjectParams,
) -> Sh2SourceKey {
    Sh2SourceKey::EquirectTexture2D {
        render_space_id,
        asset_id,
        width,
        height,
        resident_mips,
        sample_size: DEFAULT_SAMPLE_SIZE,
        material_generation: generation,
        projection: Projection360EquirectKey::from_params(params),
    }
}

/// Reads a packed texture property by host name.
fn property_texture(
    store: &MaterialPropertyStore,
    registry: &PropertyIdRegistry,
    lookup: MaterialPropertyLookupIds,
    name: &str,
) -> Option<(i32, HostTextureAssetKind)> {
    let pid = registry.intern(name);
    match store.get_merged(lookup, pid) {
        Some(MaterialPropertyValue::Texture(packed)) => unpack_host_texture_packed(*packed),
        _ => None,
    }
}

/// Builds the `Projection360` equirectangular sampling payload shared with the compute shader.
fn projection360_equirect_params(
    store: &MaterialPropertyStore,
    registry: &PropertyIdRegistry,
    lookup: MaterialPropertyLookupIds,
    storage_v_inverted: bool,
) -> Sh2ProjectParams {
    let mut params = Sh2ProjectParams::empty(SkyParamMode::Procedural);
    params.color0 = property_float4(store, registry, lookup, "_FOV", PROJECTION360_DEFAULT_FOV);
    params.color1 = property_float4(store, registry, lookup, "_MainTex_ST", DEFAULT_MAIN_TEX_ST);
    params.scalars = [storage_v_inverted_flag(storage_v_inverted), 0.0, 0.0, 0.0];
    params
}

/// Builds parameter payload for a procedural sky material.
fn procedural_sky_params(
    store: &MaterialPropertyStore,
    registry: &PropertyIdRegistry,
    lookup: MaterialPropertyLookupIds,
) -> Sh2ProjectParams {
    let mut params = Sh2ProjectParams::empty(SkyParamMode::Procedural);
    params.color0 = property_float4(store, registry, lookup, "_SkyTint", [0.5, 0.5, 0.5, 1.0]);
    params.color1 = property_float4(
        store,
        registry,
        lookup,
        "_GroundColor",
        [0.35, 0.35, 0.35, 1.0],
    );
    params.direction = property_float4(
        store,
        registry,
        lookup,
        "_SunDirection",
        [0.0, 1.0, 0.0, 0.0],
    );
    let exposure = property_float(store, registry, lookup, "_Exposure", 1.0);
    let sun_size = property_float(store, registry, lookup, "_SunSize", 0.04);
    params.scalars = [exposure, sun_size, 0.0, 0.0];
    params.gradient_color0[0] =
        property_float4(store, registry, lookup, "_SunColor", [1.0, 0.95, 0.85, 1.0]);
    params
}

/// Builds parameter payload for a gradient sky material.
fn gradient_sky_params(
    store: &MaterialPropertyStore,
    registry: &PropertyIdRegistry,
    lookup: MaterialPropertyLookupIds,
) -> Sh2ProjectParams {
    let mut params = Sh2ProjectParams::empty(SkyParamMode::Gradient);
    params.color0 = property_float4(store, registry, lookup, "_BaseColor", [0.0, 0.0, 0.0, 1.0]);
    params.dirs_spread = array16_from_property(store, registry, lookup, "_DirsSpread");
    params.gradient_color0 = array16_from_property(store, registry, lookup, "_Color0");
    params.gradient_color1 = array16_from_property(store, registry, lookup, "_Color1");
    params.gradient_params = array16_from_property(store, registry, lookup, "_Params");
    params.gradient_count = property_float(store, registry, lookup, "_Gradients", 0.0)
        .round()
        .clamp(0.0, 16.0) as u32;
    if params.gradient_count == 0 {
        params.gradient_count = params
            .dirs_spread
            .iter()
            .position(|v| v.iter().all(|c| c.abs() < 1e-6))
            .unwrap_or(16) as u32;
    }
    params
}

/// Reads a float material property.
fn property_float(
    store: &MaterialPropertyStore,
    registry: &PropertyIdRegistry,
    lookup: MaterialPropertyLookupIds,
    name: &str,
    fallback: f32,
) -> f32 {
    let pid = registry.intern(name);
    match store.get_merged(lookup, pid) {
        Some(MaterialPropertyValue::Float(v)) => *v,
        _ => fallback,
    }
}

/// Reads a float4 material property.
fn property_float4(
    store: &MaterialPropertyStore,
    registry: &PropertyIdRegistry,
    lookup: MaterialPropertyLookupIds,
    name: &str,
    fallback: [f32; 4],
) -> [f32; 4] {
    let pid = registry.intern(name);
    match store.get_merged(lookup, pid) {
        Some(MaterialPropertyValue::Float4(v)) => *v,
        _ => fallback,
    }
}

/// Reads up to sixteen float4 rows from a material property.
fn array16_from_property(
    store: &MaterialPropertyStore,
    registry: &PropertyIdRegistry,
    lookup: MaterialPropertyLookupIds,
    name: &str,
) -> [[f32; 4]; 16] {
    let pid = registry.intern(name);
    let mut out = [[0.0; 4]; 16];
    if let Some(MaterialPropertyValue::Float4Array(values)) = store.get_merged(lookup, pid) {
        for (dst, src) in out.iter_mut().zip(values.iter()) {
            *dst = *src;
        }
    }
    out
}

/// Bit pattern for a `Vec4`.
fn vec4_bits(v: Vec4) -> [u32; 4] {
    [v.x.to_bits(), v.y.to_bits(), v.z.to_bits(), v.w.to_bits()]
}

/// Hashes a route name into a stable source-key discriminator.
fn hash_route_name(route: &str) -> u64 {
    let mut h = DefaultHasher::new();
    route.hash(&mut h);
    h.finish()
}
