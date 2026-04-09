//! Parse composed WGSL with naga and derive [`wgpu::BindGroupLayoutEntry`] lists for `@group(1)` and
//! `@group(2)`, and a [`ReflectedRasterLayout::layout_fingerprint`] for tests and diagnostics.
//!
//! Validates `@group(0)` against [`crate::backend::frame_gpu::FrameGpuResources`] buffer sizes and
//! optional scene-depth snapshot bindings.

use std::collections::{BTreeMap, HashMap};
use std::hash::{Hash, Hasher};
use std::num::NonZeroU64;

use naga::front::wgsl::parse_str;
use naga::proc::Layouter;
use naga::valid::{Capabilities, ValidationFlags, Validator};
use naga::{
    AddressSpace, ArraySize, Binding, ImageClass, ImageDimension, ScalarKind, ShaderStage,
    StorageAccess, TypeInner, VectorSize,
};
use thiserror::Error;

use crate::backend::GpuLight;
use crate::gpu::frame_globals::FrameGpuUniforms;
use crate::gpu::PER_DRAW_UNIFORM_STRIDE;

/// Scalar shape of a named uniform struct member (for CPU packing from host properties).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReflectedUniformScalarKind {
    /// Single `f32`.
    F32,
    /// `vec4<f32>` (or equivalent 16-byte float vector).
    Vec4,
    /// Single `u32` (e.g. shader `flags`).
    U32,
    /// Not mapped automatically (padding or unsupported type).
    Unsupported,
}

/// Byte layout of one field inside a `@group(1)` `var<uniform>` struct (from naga struct member offsets).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReflectedUniformField {
    /// Byte offset within the uniform block (WGSL struct layout).
    pub offset: u32,
    /// Size in bytes (`Layouter` type size).
    pub size: u32,
    /// Host packing strategy for this member.
    pub kind: ReflectedUniformScalarKind,
}

/// Uniform block at `@group(1)` (typically `@binding(0)`) used for material constants.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReflectedMaterialUniformBlock {
    /// WGSL binding index for this uniform buffer (expected `0` for current materials).
    pub binding: u32,
    /// Total uniform block size in bytes (including tail padding).
    pub total_size: u32,
    /// Struct member name → layout (only members with names; excludes padding-only slots if unnamed).
    pub fields: HashMap<String, ReflectedUniformField>,
}

/// Result of [`reflect_raster_material_wgsl`].
#[derive(Debug)]
pub struct ReflectedRasterLayout {
    /// Stable hash of material + per-draw bind group layout shapes (tests, diagnostics, future cache versioning).
    pub layout_fingerprint: u64,
    /// `@group(1)` entries sorted by binding index.
    pub material_entries: Vec<wgpu::BindGroupLayoutEntry>,
    /// `@group(2)` entries sorted by binding index.
    pub per_draw_entries: Vec<wgpu::BindGroupLayoutEntry>,
    /// First `var<uniform>` in `@group(1)` with a struct body, if any (for CPU packing without hand-written `#[repr(C)]` structs).
    pub material_uniform: Option<ReflectedMaterialUniformBlock>,
    /// `@group(1)` `@binding` → WGSL global identifier (matches Unity host property names where applicable).
    pub material_group1_names: HashMap<u32, String>,
    /// Highest `@location` index on `vs_main` vertex inputs (excluding builtins); `>= 2` implies a UV stream at `location(2)`.
    pub vs_max_vertex_location: Option<u32>,
}

/// Errors from [`reflect_raster_material_wgsl`].
#[derive(Debug, Error)]
pub enum ReflectError {
    #[error("WGSL parse: {0}")]
    Parse(String),
    #[error("WGSL validate: {0}")]
    Validate(String),
    #[error("layout computation: {0}")]
    Layout(String),
    #[error("group(0) must have uniform binding 0 size {expected_frame}, storage binding 1 stride {expected_light}, bindings 2–3 u32 stride {expected_cluster_u32}; got b0={got0:?} b1={got1:?} b2={got2:?} b3={got3:?}")]
    FrameGroupMismatch {
        expected_frame: u32,
        expected_light: u32,
        expected_cluster_u32: u32,
        got0: Option<u32>,
        got1: Option<u32>,
        got2: Option<u32>,
        got3: Option<u32>,
    },
    #[error("unsupported global resource at group {group} binding {binding}: {reason}")]
    UnsupportedBinding {
        group: u32,
        binding: u32,
        reason: String,
    },
    #[error("invalid bind group index {0} (only 0, 1, 2 are allowed for raster materials)")]
    InvalidBindGroup(u32),
}

/// Parses and validates WGSL, checks frame globals, and builds layout entries for groups 1 and 2.
pub fn reflect_raster_material_wgsl(source: &str) -> Result<ReflectedRasterLayout, ReflectError> {
    let module = parse_str(source).map_err(|e| ReflectError::Parse(e.to_string()))?;
    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
    validator
        .subgroup_stages(naga::valid::ShaderStages::all())
        .subgroup_operations(naga::valid::SubgroupOperationSet::all());
    validator
        .validate(&module)
        .map_err(|e| ReflectError::Validate(e.to_string()))?;

    let mut layouter = Layouter::default();
    layouter
        .update(module.to_ctx())
        .map_err(|e| ReflectError::Layout(e.to_string()))?;

    validate_frame_group0(&module, &layouter)?;

    let mut g1: BTreeMap<u32, wgpu::BindGroupLayoutEntry> = BTreeMap::new();
    let mut g2: BTreeMap<u32, wgpu::BindGroupLayoutEntry> = BTreeMap::new();

    for (_, gv) in module.global_variables.iter() {
        let Some(rb) = gv.binding else {
            continue;
        };
        if rb.group > 2 {
            return Err(ReflectError::InvalidBindGroup(rb.group));
        }
        if rb.group == 0 {
            continue;
        }
        let entry = global_to_layout_entry(&module, &layouter, gv, rb.group, rb.binding)?;
        match rb.group {
            1 => {
                g1.insert(rb.binding, entry);
            }
            2 => {
                g2.insert(rb.binding, entry);
            }
            _ => {}
        }
    }

    let material_entries: Vec<_> = g1.into_values().collect();
    let per_draw_entries: Vec<_> = g2.into_values().collect();

    let material_uniform = reflect_first_group1_uniform_struct(&module, &layouter);
    let material_group1_names = reflect_group1_global_binding_names(&module);
    let vs_max_vertex_location = reflect_vs_main_max_vertex_location(&module);

    let layout_fingerprint = fingerprint_layout(
        &material_entries,
        &per_draw_entries,
        vs_max_vertex_location,
        &material_group1_names,
    );

    Ok(ReflectedRasterLayout {
        layout_fingerprint,
        material_entries,
        per_draw_entries,
        material_uniform,
        material_group1_names,
        vs_max_vertex_location,
    })
}

/// Returns true when `vs_main` uses vertex `@location` 2 or higher (UV0 stream).
pub fn reflect_vertex_shader_needs_uv0_stream(wgsl_source: &str) -> bool {
    reflect_raster_material_wgsl(wgsl_source)
        .ok()
        .and_then(|r| r.vs_max_vertex_location)
        .map(|m| m >= 2)
        .unwrap_or(false)
}

/// Returns true when `vs_main` uses vertex `@location` 3 or higher (extra color stream).
pub fn reflect_vertex_shader_needs_color_stream(wgsl_source: &str) -> bool {
    reflect_raster_material_wgsl(wgsl_source)
        .ok()
        .and_then(|r| r.vs_max_vertex_location)
        .map(|m| m >= 3)
        .unwrap_or(false)
}

fn reflect_group1_global_binding_names(module: &naga::Module) -> HashMap<u32, String> {
    let mut out = HashMap::new();
    for (_, gv) in module.global_variables.iter() {
        let Some(rb) = gv.binding else {
            continue;
        };
        if rb.group != 1 {
            continue;
        }
        let Some(name) = gv.name.as_deref() else {
            continue;
        };
        out.insert(rb.binding, name.to_string());
    }
    out
}

fn reflect_vs_main_max_vertex_location(module: &naga::Module) -> Option<u32> {
    let ep = module
        .entry_points
        .iter()
        .find(|e| e.stage == ShaderStage::Vertex && e.name == "vs_main")?;
    let func = &ep.function;
    let mut max: Option<u32> = None;
    for arg in &func.arguments {
        if let Some(Binding::Location { location, .. }) = arg.binding {
            max = Some(max.map_or(location, |m| m.max(location)));
        }
    }
    max
}

fn uniform_member_kind(
    module: &naga::Module,
    ty: naga::Handle<naga::Type>,
) -> ReflectedUniformScalarKind {
    match &module.types[ty].inner {
        TypeInner::Scalar(sc) => match sc.kind {
            ScalarKind::Float => ReflectedUniformScalarKind::F32,
            ScalarKind::Uint => ReflectedUniformScalarKind::U32,
            ScalarKind::Sint => ReflectedUniformScalarKind::Unsupported,
            ScalarKind::Bool => ReflectedUniformScalarKind::Unsupported,
            ScalarKind::AbstractInt | ScalarKind::AbstractFloat => {
                ReflectedUniformScalarKind::Unsupported
            }
        },
        TypeInner::Vector { size, scalar } => {
            if *size == VectorSize::Quad && scalar.kind == ScalarKind::Float {
                ReflectedUniformScalarKind::Vec4
            } else {
                ReflectedUniformScalarKind::Unsupported
            }
        }
        _ => ReflectedUniformScalarKind::Unsupported,
    }
}

/// Finds the first `@group(1)` `var<uniform>` with a struct type and records member offsets/sizes.
fn reflect_first_group1_uniform_struct(
    module: &naga::Module,
    layouter: &Layouter,
) -> Option<ReflectedMaterialUniformBlock> {
    for (_, gv) in module.global_variables.iter() {
        let Some(rb) = gv.binding else {
            continue;
        };
        if rb.group != 1 {
            continue;
        }
        let (space, data_ty) = resource_data_ty(module, gv);
        if space != AddressSpace::Uniform {
            continue;
        }
        let inner = &module.types[data_ty].inner;
        let TypeInner::Struct { members, .. } = inner else {
            continue;
        };
        let mut fields = HashMap::new();
        for m in members.iter() {
            let Some(name) = m.name.as_deref() else {
                continue;
            };
            let size = layouter[m.ty].size;
            let kind = uniform_member_kind(module, m.ty);
            fields.insert(
                name.to_string(),
                ReflectedUniformField {
                    offset: m.offset,
                    size,
                    kind,
                },
            );
        }
        let total_size = layouter[data_ty].size;
        return Some(ReflectedMaterialUniformBlock {
            binding: rb.binding,
            total_size,
            fields,
        });
    }
    None
}

/// Resolves the address space and data type for a global resource (WGSL may use a plain struct/array
/// type with [`naga::GlobalVariable::space`], or a [`TypeInner::Pointer`] wrapper).
fn resource_data_ty(
    module: &naga::Module,
    gv: &naga::GlobalVariable,
) -> (AddressSpace, naga::Handle<naga::Type>) {
    match &module.types[gv.ty].inner {
        TypeInner::Pointer { base, space } => (*space, *base),
        _ => (gv.space, gv.ty),
    }
}

/// Stride of one element in a runtime-sized storage array (e.g. `GpuLight`, `u32`, `atomic<u32>`).
fn storage_array_element_stride(
    module: &naga::Module,
    layouter: &Layouter,
    data_ty: naga::Handle<naga::Type>,
    binding: u32,
) -> Result<u32, ReflectError> {
    match &module.types[data_ty].inner {
        TypeInner::Array { base: el, size, .. } => {
            let el_stride = layouter[*el].to_stride();
            match size {
                ArraySize::Pending(_) => Err(ReflectError::UnsupportedBinding {
                    group: 0,
                    binding,
                    reason: "pending array size".into(),
                }),
                ArraySize::Constant(_) | ArraySize::Dynamic => Ok(el_stride),
            }
        }
        _ => Err(ReflectError::UnsupportedBinding {
            group: 0,
            binding,
            reason: "expected runtime-sized array".into(),
        }),
    }
}

fn validate_frame_group0(module: &naga::Module, layouter: &Layouter) -> Result<(), ReflectError> {
    let expected_frame = std::mem::size_of::<FrameGpuUniforms>() as u32;
    let expected_light = std::mem::size_of::<GpuLight>() as u32;
    let expected_cluster_u32 = std::mem::size_of::<u32>() as u32;

    let mut b0_size: Option<u32> = None;
    let mut b1_stride: Option<u32> = None;
    let mut b2_stride: Option<u32> = None;
    let mut b3_stride: Option<u32> = None;

    for (_, gv) in module.global_variables.iter() {
        let Some(rb) = gv.binding else {
            continue;
        };
        if rb.group != 0 {
            continue;
        }
        if rb.binding > 5 {
            return Err(ReflectError::UnsupportedBinding {
                group: 0,
                binding: rb.binding,
                reason: "only bindings 0..=5 are supported for raster frame globals".into(),
            });
        }
        let (space, data_ty) = resource_data_ty(module, gv);
        match (rb.binding, space) {
            (4, AddressSpace::Handle) => {
                validate_frame_depth_texture_binding(module, data_ty, false, rb.binding)?;
            }
            (5, AddressSpace::Handle) => {
                validate_frame_depth_texture_binding(module, data_ty, true, rb.binding)?;
            }
            (_, AddressSpace::Uniform) => {
                if rb.binding == 0 {
                    b0_size = Some(layouter[data_ty].size);
                }
            }
            (_, AddressSpace::Storage { .. }) => {
                let stride = storage_array_element_stride(module, layouter, data_ty, rb.binding)?;
                match rb.binding {
                    1 => b1_stride = Some(stride),
                    2 => b2_stride = Some(stride),
                    3 => b3_stride = Some(stride),
                    _ => {}
                }
            }
            _ => {}
        }
    }

    if b0_size == Some(expected_frame)
        && b1_stride == Some(expected_light)
        && b2_stride == Some(expected_cluster_u32)
        && b3_stride == Some(expected_cluster_u32)
    {
        Ok(())
    } else {
        Err(ReflectError::FrameGroupMismatch {
            expected_frame,
            expected_light,
            expected_cluster_u32,
            got0: b0_size,
            got1: b1_stride,
            got2: b2_stride,
            got3: b3_stride,
        })
    }
}

fn validate_frame_depth_texture_binding(
    module: &naga::Module,
    data_ty: naga::Handle<naga::Type>,
    arrayed: bool,
    binding: u32,
) -> Result<(), ReflectError> {
    match &module.types[data_ty].inner {
        TypeInner::Image {
            dim,
            arrayed: got_arrayed,
            class: ImageClass::Depth { multi },
        } if *dim == ImageDimension::D2 && *got_arrayed == arrayed && !*multi => Ok(()),
        TypeInner::Image { .. } => Err(ReflectError::UnsupportedBinding {
            group: 0,
            binding,
            reason: if arrayed {
                "expected texture_depth_2d_array".into()
            } else {
                "expected texture_depth_2d".into()
            },
        }),
        _ => Err(ReflectError::UnsupportedBinding {
            group: 0,
            binding,
            reason: "expected depth texture handle".into(),
        }),
    }
}

fn global_to_layout_entry(
    module: &naga::Module,
    layouter: &Layouter,
    gv: &naga::GlobalVariable,
    group: u32,
    binding: u32,
) -> Result<wgpu::BindGroupLayoutEntry, ReflectError> {
    let visibility = wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT;
    let (space, data_ty) = resource_data_ty(module, gv);

    match space {
        AddressSpace::Uniform => {
            let size = layouter[data_ty].size;
            let min_binding_size = NonZeroU64::new(u64::from(size)).ok_or_else(|| {
                ReflectError::UnsupportedBinding {
                    group,
                    binding,
                    reason: "zero-sized uniform".into(),
                }
            })?;
            Ok(wgpu::BindGroupLayoutEntry {
                binding,
                visibility,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: group == 2,
                    min_binding_size: Some(min_binding_size),
                },
                count: None,
            })
        }
        AddressSpace::Storage { access } => {
            let read_only = !access.contains(StorageAccess::STORE);
            let base_ty = &module.types[data_ty];
            let (min_binding_size, buf_ty) =
                match &base_ty.inner {
                    TypeInner::Array {
                        base: elem, size, ..
                    } => {
                        let stride = layouter[*elem].to_stride();
                        let min =
                            match size {
                                ArraySize::Dynamic => NonZeroU64::new(u64::from(stride))
                                    .ok_or_else(|| ReflectError::UnsupportedBinding {
                                        group,
                                        binding,
                                        reason: "zero stride storage array".into(),
                                    })?,
                                ArraySize::Constant(n) => {
                                    let n = n.get();
                                    let total = stride.saturating_mul(n);
                                    NonZeroU64::new(u64::from(total)).ok_or_else(|| {
                                        ReflectError::UnsupportedBinding {
                                            group,
                                            binding,
                                            reason: "zero-sized storage array".into(),
                                        }
                                    })?
                                }
                                ArraySize::Pending(_) => {
                                    return Err(ReflectError::UnsupportedBinding {
                                        group,
                                        binding,
                                        reason: "pending array size".into(),
                                    });
                                }
                            };
                        (min, wgpu::BufferBindingType::Storage { read_only })
                    }
                    _ => {
                        let size = layouter[data_ty].size;
                        let min = NonZeroU64::new(u64::from(size)).ok_or_else(|| {
                            ReflectError::UnsupportedBinding {
                                group,
                                binding,
                                reason: "zero-sized storage buffer".into(),
                            }
                        })?;
                        (min, wgpu::BufferBindingType::Storage { read_only })
                    }
                };
            Ok(wgpu::BindGroupLayoutEntry {
                binding,
                visibility,
                ty: wgpu::BindingType::Buffer {
                    ty: buf_ty,
                    has_dynamic_offset: false,
                    min_binding_size: Some(min_binding_size),
                },
                count: None,
            })
        }
        AddressSpace::Handle => {
            let inner = &module.types[data_ty];
            match &inner.inner {
                TypeInner::Image {
                    dim,
                    arrayed,
                    class,
                } => {
                    if *arrayed {
                        return Err(ReflectError::UnsupportedBinding {
                            group,
                            binding,
                            reason: "arrayed images not supported yet".into(),
                        });
                    }
                    if *dim != ImageDimension::D2 {
                        return Err(ReflectError::UnsupportedBinding {
                            group,
                            binding,
                            reason: "only 2D textures supported".into(),
                        });
                    }
                    let sample_type = match class {
                        ImageClass::Sampled { kind, multi } => {
                            if *multi {
                                return Err(ReflectError::UnsupportedBinding {
                                    group,
                                    binding,
                                    reason: "multisampled textures not supported yet".into(),
                                });
                            }
                            match kind {
                                ScalarKind::Float => {
                                    wgpu::TextureSampleType::Float { filterable: true }
                                }
                                ScalarKind::Sint => wgpu::TextureSampleType::Sint,
                                ScalarKind::Uint => wgpu::TextureSampleType::Uint,
                                ScalarKind::Bool => {
                                    return Err(ReflectError::UnsupportedBinding {
                                        group,
                                        binding,
                                        reason: "bool texture sample".into(),
                                    });
                                }
                                ScalarKind::AbstractInt | ScalarKind::AbstractFloat => {
                                    return Err(ReflectError::UnsupportedBinding {
                                        group,
                                        binding,
                                        reason: "abstract texture sample".into(),
                                    });
                                }
                            }
                        }
                        ImageClass::Depth { multi } => {
                            if *multi {
                                return Err(ReflectError::UnsupportedBinding {
                                    group,
                                    binding,
                                    reason: "multisampled depth not supported yet".into(),
                                });
                            }
                            wgpu::TextureSampleType::Depth
                        }
                        ImageClass::Storage { .. } | ImageClass::External => {
                            return Err(ReflectError::UnsupportedBinding {
                                group,
                                binding,
                                reason: "storage/external images not supported yet".into(),
                            });
                        }
                    };
                    Ok(wgpu::BindGroupLayoutEntry {
                        binding,
                        visibility,
                        ty: wgpu::BindingType::Texture {
                            sample_type,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    })
                }
                TypeInner::Sampler { comparison } => Ok(wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility,
                    ty: if *comparison {
                        wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison)
                    } else {
                        wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)
                    },
                    count: None,
                }),
                _ => Err(ReflectError::UnsupportedBinding {
                    group,
                    binding,
                    reason: "unsupported handle type".into(),
                }),
            }
        }
        _ => Err(ReflectError::UnsupportedBinding {
            group,
            binding,
            reason: "unsupported address space for global resource".into(),
        }),
    }
}

fn fingerprint_layout(
    material: &[wgpu::BindGroupLayoutEntry],
    per_draw: &[wgpu::BindGroupLayoutEntry],
    vs_max_vertex_location: Option<u32>,
    group1_names: &HashMap<u32, String>,
) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hash;
    let mut h = DefaultHasher::new();
    1u8.hash(&mut h);
    hash_entries(material, &mut h);
    2u8.hash(&mut h);
    hash_entries(per_draw, &mut h);
    3u8.hash(&mut h);
    vs_max_vertex_location.hash(&mut h);
    let mut keys: Vec<u32> = group1_names.keys().copied().collect();
    keys.sort_unstable();
    for k in keys {
        k.hash(&mut h);
        group1_names[&k].hash(&mut h);
    }
    h.finish()
}

fn hash_entries(entries: &[wgpu::BindGroupLayoutEntry], h: &mut impl Hasher) {
    entries.len().hash(h);
    for e in entries {
        e.binding.hash(h);
        hash_binding_type(&e.ty, h);
    }
}

fn hash_binding_type(ty: &wgpu::BindingType, h: &mut impl Hasher) {
    match ty {
        wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset,
            min_binding_size,
        } => {
            0u8.hash(h);
            std::mem::discriminant(ty).hash(h);
            has_dynamic_offset.hash(h);
            min_binding_size.map(|n| n.get()).hash(h);
        }
        wgpu::BindingType::Texture {
            sample_type,
            view_dimension,
            multisampled,
        } => {
            1u8.hash(h);
            std::mem::discriminant(sample_type).hash(h);
            std::mem::discriminant(view_dimension).hash(h);
            multisampled.hash(h);
        }
        wgpu::BindingType::Sampler(ty) => {
            2u8.hash(h);
            std::mem::discriminant(ty).hash(h);
        }
        wgpu::BindingType::StorageTexture { .. } => {
            3u8.hash(h);
        }
        wgpu::BindingType::AccelerationStructure { .. } => {
            6u8.hash(h);
        }
        wgpu::BindingType::ExternalTexture => {
            7u8.hash(h);
        }
    }
}

/// Validates that `@group(2)` matches the dynamic per-draw uniform slab (single binding, 256-byte stride).
pub fn validate_per_draw_group2(
    entries: &[wgpu::BindGroupLayoutEntry],
) -> Result<(), ReflectError> {
    if entries.len() != 1 {
        return Err(ReflectError::UnsupportedBinding {
            group: 2,
            binding: 0,
            reason: format!(
                "expected exactly one per-draw binding, got {}",
                entries.len()
            ),
        });
    }
    let e = &entries[0];
    if e.binding != 0 {
        return Err(ReflectError::UnsupportedBinding {
            group: 2,
            binding: e.binding,
            reason: "per-draw binding must be @binding(0)".into(),
        });
    }
    match e.ty {
        wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: true,
            min_binding_size: Some(n),
        } if n.get() == PER_DRAW_UNIFORM_STRIDE as u64 => Ok(()),
        _ => Err(ReflectError::UnsupportedBinding {
            group: 2,
            binding: 0,
            reason: "expected var<uniform> dynamic offset min_binding_size 256".into(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflect_debug_world_normals_default_embedded() {
        let wgsl = crate::embedded_shaders::embedded_target_wgsl("debug_world_normals_default")
            .expect("stem");
        let r = reflect_raster_material_wgsl(wgsl).expect("reflect");
        assert!(r.material_entries.is_empty());
        validate_per_draw_group2(&r.per_draw_entries).expect("per_draw");
        assert_ne!(r.layout_fingerprint, 0);
        assert_eq!(
            r.vs_max_vertex_location,
            Some(1),
            "debug normals: position + normal only"
        );
    }

    /// Every composed `shaders/target/*.wgsl` must declare the full [`FrameGpuResources`] `@group(0)`
    /// contract; naga-oil strips unused imports, so a material that omits cluster buffer references
    /// can fail at runtime during pipeline creation unless this test catches it.
    #[test]
    fn reflect_all_embedded_material_targets_match_frame_group0() {
        for stem in crate::embedded_shaders::COMPILED_MATERIAL_STEMS {
            let wgsl = crate::embedded_shaders::embedded_target_wgsl(stem)
                .unwrap_or_else(|| panic!("embedded_target_wgsl missing for {stem}"));
            reflect_raster_material_wgsl(wgsl).unwrap_or_else(|e| {
                panic!("reflect_raster_material_wgsl failed for {stem}: {e}");
            });
        }
    }
}
