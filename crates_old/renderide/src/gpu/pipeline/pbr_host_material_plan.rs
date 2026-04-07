//! Host material data bound into the stock non-skinned PBR shader via the uniform ring.
//!
//! [`super::uniforms::Uniforms`] / WGSL `UniformsSlot` (`uniform_ring.wgsl`) carry optional
//! `host_base_color` and `host_metallic_roughness`, populated at draw time from
//! [`crate::assets::MaterialPropertyStore`] in [`crate::render::pass::mesh_draw::fill_pbr_host_uniform_extras`].
//!
//! Property names are mapped through [`crate::assets::apply_froox_material_property_name_to_pbr_host_config`]
//! (`_Color`, `_Metallic`, `_Glossiness`). Albedo textures and full Standard shader parity remain
//! future work; this module name is retained as the planning anchor for that expansion.

/// Documents the host→PBR uniform hook; all GPU state lives on [`super::uniforms::Uniforms`] slots today.
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuPbrHostMaterialPlan;
