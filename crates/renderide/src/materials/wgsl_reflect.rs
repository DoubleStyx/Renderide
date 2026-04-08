//! Parse composed WGSL with naga and derive [`wgpu::BindGroupLayoutEntry`] lists for `@group(1)` and
//! `@group(2)`, and a [`ReflectedRasterLayout::layout_fingerprint`] for pipeline cache keys.
//!
//! Validates `@group(0)` against [`crate::backend::frame_gpu::FrameGpuResources`] buffer sizes.

use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};
use std::num::NonZeroU64;

use naga::front::wgsl::parse_str;
use naga::proc::Layouter;
use naga::valid::{Capabilities, ValidationFlags, Validator};
use naga::{
    AddressSpace, ArraySize, ImageClass, ImageDimension, ScalarKind, StorageAccess, TypeInner,
};
use thiserror::Error;

use crate::backend::GpuLight;
use crate::gpu::frame_globals::FrameGpuUniforms;
use crate::gpu::PER_DRAW_UNIFORM_STRIDE;

/// Result of [`reflect_raster_material_wgsl`].
#[derive(Debug)]
pub struct ReflectedRasterLayout {
    /// Stable hash of material + per-draw bind group layout shapes (for [`super::MaterialPipelineCacheKey`]).
    pub layout_fingerprint: u64,
    /// `@group(1)` entries sorted by binding index.
    pub material_entries: Vec<wgpu::BindGroupLayoutEntry>,
    /// `@group(2)` entries sorted by binding index.
    pub per_draw_entries: Vec<wgpu::BindGroupLayoutEntry>,
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
    #[error("group(0) must have uniform binding 0 size {expected_frame} and storage binding 1 element stride {expected_light}; got binding0 size {got0:?} binding1 stride {got1:?}")]
    FrameGroupMismatch {
        expected_frame: u32,
        expected_light: u32,
        got0: Option<u32>,
        got1: Option<u32>,
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

    let layout_fingerprint = fingerprint_layout(&material_entries, &per_draw_entries);

    Ok(ReflectedRasterLayout {
        layout_fingerprint,
        material_entries,
        per_draw_entries,
    })
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

fn validate_frame_group0(module: &naga::Module, layouter: &Layouter) -> Result<(), ReflectError> {
    let expected_frame = std::mem::size_of::<FrameGpuUniforms>() as u32;
    let expected_light = std::mem::size_of::<GpuLight>() as u32;

    let mut b0_size: Option<u32> = None;
    let mut b1_stride: Option<u32> = None;

    for (_, gv) in module.global_variables.iter() {
        let Some(rb) = gv.binding else {
            continue;
        };
        if rb.group != 0 {
            continue;
        }
        let (space, data_ty) = resource_data_ty(module, gv);
        match space {
            AddressSpace::Uniform => {
                if rb.binding == 0 {
                    b0_size = Some(layouter[data_ty].size);
                }
            }
            AddressSpace::Storage { .. } => {
                if rb.binding == 1 {
                    let stride = match &module.types[data_ty].inner {
                        TypeInner::Array { base: el, size, .. } => {
                            let el_stride = layouter[*el].to_stride();
                            match size {
                                ArraySize::Pending(_) => {
                                    return Err(ReflectError::UnsupportedBinding {
                                        group: 0,
                                        binding: 1,
                                        reason: "pending array size".into(),
                                    });
                                }
                                ArraySize::Constant(_) | ArraySize::Dynamic => el_stride,
                            }
                        }
                        _ => {
                            return Err(ReflectError::UnsupportedBinding {
                                group: 0,
                                binding: 1,
                                reason: "expected runtime-sized array".into(),
                            });
                        }
                    };
                    b1_stride = Some(stride);
                }
            }
            _ => {}
        }
    }

    if b0_size == Some(expected_frame) && b1_stride == Some(expected_light) {
        Ok(())
    } else {
        Err(ReflectError::FrameGroupMismatch {
            expected_frame,
            expected_light,
            got0: b0_size,
            got1: b1_stride,
        })
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
) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    let mut h = DefaultHasher::new();
    1u8.hash(&mut h);
    hash_entries(material, &mut h);
    2u8.hash(&mut h);
    hash_entries(per_draw, &mut h);
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
    }

    #[test]
    fn reflect_world_unlit_default_embedded() {
        let wgsl =
            crate::embedded_shaders::embedded_target_wgsl("world_unlit_default").expect("stem");
        let r = reflect_raster_material_wgsl(wgsl).expect("reflect");
        assert_eq!(r.material_entries.len(), 3);
        validate_per_draw_group2(&r.per_draw_entries).expect("per_draw");
        assert_ne!(r.layout_fingerprint, 0);
    }
}
