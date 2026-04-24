//! Converts naga global variables at `@group(1)` / `@group(2)` into `wgpu::BindGroupLayoutEntry`.

use std::num::NonZeroU64;

use naga::proc::Layouter;
use naga::{
    AddressSpace, ArraySize, ImageClass, ImageDimension, Module, ScalarKind, StorageAccess,
    TypeInner,
};

use super::resource::resource_data_ty;
use super::types::ReflectError;

/// Uniform buffer slot at `@group(1)` / dynamic `@group(2)`.
fn layout_entry_for_uniform(
    layouter: &Layouter,
    data_ty: naga::Handle<naga::Type>,
    group: u32,
    binding: u32,
    visibility: wgpu::ShaderStages,
) -> Result<wgpu::BindGroupLayoutEntry, ReflectError> {
    let size = layouter[data_ty].size;
    let min_binding_size =
        NonZeroU64::new(u64::from(size)).ok_or_else(|| ReflectError::UnsupportedBinding {
            group,
            binding,
            reason: "zero-sized uniform".into(),
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

/// Read-only or read-write storage buffer (including unsized arrays) for material or per-draw data.
fn layout_entry_for_storage(
    module: &Module,
    layouter: &Layouter,
    data_ty: naga::Handle<naga::Type>,
    access: StorageAccess,
    group: u32,
    binding: u32,
    visibility: wgpu::ShaderStages,
) -> Result<wgpu::BindGroupLayoutEntry, ReflectError> {
    let read_only = !access.contains(StorageAccess::STORE);
    let base_ty = &module.types[data_ty];
    let (min_binding_size, buf_ty) = match &base_ty.inner {
        TypeInner::Array {
            base: elem, size, ..
        } => {
            let stride = layouter[*elem].to_stride();
            let min = match size {
                ArraySize::Dynamic => NonZeroU64::new(u64::from(stride)).ok_or_else(|| {
                    ReflectError::UnsupportedBinding {
                        group,
                        binding,
                        reason: "zero stride storage array".into(),
                    }
                })?,
                ArraySize::Constant(n) => {
                    let n = n.get();
                    if group == 2 {
                        // Per-draw slab: `set_bind_group` dynamic offset selects one row; the shader still
                        // indexes `instances[instance_index]` within the bound window (base instance or row 0).
                        NonZeroU64::new(u64::from(stride)).ok_or_else(|| {
                            ReflectError::UnsupportedBinding {
                                group,
                                binding,
                                reason: "zero stride per-draw storage array".into(),
                            }
                        })?
                    } else {
                        let total = stride.saturating_mul(n);
                        NonZeroU64::new(u64::from(total)).ok_or_else(|| {
                            ReflectError::UnsupportedBinding {
                                group,
                                binding,
                                reason: "zero-sized storage array".into(),
                            }
                        })?
                    }
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
            // `@group(2)` per-draw slab: dynamic offset selects a row; base-instance path uses offset 0.
            has_dynamic_offset: group == 2,
            min_binding_size: Some(min_binding_size),
        },
        count: None,
    })
}

/// Maps a WGSL `texture_*` [`ImageClass`] to the corresponding [`wgpu::TextureSampleType`].
fn texture_sample_type_for_image_class(
    class: &ImageClass,
    group: u32,
    binding: u32,
) -> Result<wgpu::TextureSampleType, ReflectError> {
    match class {
        ImageClass::Sampled { kind, multi } => {
            if *multi {
                return Err(ReflectError::UnsupportedBinding {
                    group,
                    binding,
                    reason: "multisampled textures not supported yet".into(),
                });
            }
            match kind {
                ScalarKind::Float => Ok(wgpu::TextureSampleType::Float { filterable: true }),
                ScalarKind::Sint => Ok(wgpu::TextureSampleType::Sint),
                ScalarKind::Uint => Ok(wgpu::TextureSampleType::Uint),
                ScalarKind::Bool => Err(ReflectError::UnsupportedBinding {
                    group,
                    binding,
                    reason: "bool texture sample".into(),
                }),
                ScalarKind::AbstractInt | ScalarKind::AbstractFloat => {
                    Err(ReflectError::UnsupportedBinding {
                        group,
                        binding,
                        reason: "abstract texture sample".into(),
                    })
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
            Ok(wgpu::TextureSampleType::Depth)
        }
        ImageClass::Storage { .. } | ImageClass::External => {
            Err(ReflectError::UnsupportedBinding {
                group,
                binding,
                reason: "storage/external images not supported yet".into(),
            })
        }
    }
}

/// Layout entry for a sampled WGSL texture binding.
fn layout_entry_for_sampled_image(
    dim: &ImageDimension,
    arrayed: bool,
    class: &ImageClass,
    group: u32,
    binding: u32,
    visibility: wgpu::ShaderStages,
) -> Result<wgpu::BindGroupLayoutEntry, ReflectError> {
    if arrayed {
        return Err(ReflectError::UnsupportedBinding {
            group,
            binding,
            reason: "arrayed images not supported yet".into(),
        });
    }
    let view_dimension = match *dim {
        ImageDimension::D2 => wgpu::TextureViewDimension::D2,
        ImageDimension::D3 => wgpu::TextureViewDimension::D3,
        ImageDimension::Cube => wgpu::TextureViewDimension::Cube,
        ImageDimension::D1 => {
            return Err(ReflectError::UnsupportedBinding {
                group,
                binding,
                reason: "only 2D, 3D, and cube textures supported".into(),
            });
        }
    };
    let sample_type = texture_sample_type_for_image_class(class, group, binding)?;
    Ok(wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Texture {
            sample_type,
            view_dimension,
            multisampled: false,
        },
        count: None,
    })
}

/// Layout entry for a WGSL `sampler` / `sampler_comparison` binding.
fn layout_entry_for_sampler_handle(
    comparison: bool,
    binding: u32,
    visibility: wgpu::ShaderStages,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: if comparison {
            wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison)
        } else {
            wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)
        },
        count: None,
    }
}

/// Texture or sampler handle for sampled bindings.
fn layout_entry_for_handle(
    module: &Module,
    data_ty: naga::Handle<naga::Type>,
    group: u32,
    binding: u32,
    visibility: wgpu::ShaderStages,
) -> Result<wgpu::BindGroupLayoutEntry, ReflectError> {
    let inner = &module.types[data_ty];
    match &inner.inner {
        TypeInner::Image {
            dim,
            arrayed,
            class,
        } => layout_entry_for_sampled_image(dim, *arrayed, class, group, binding, visibility),
        TypeInner::Sampler { comparison } => Ok(layout_entry_for_sampler_handle(
            *comparison,
            binding,
            visibility,
        )),
        _ => Err(ReflectError::UnsupportedBinding {
            group,
            binding,
            reason: "unsupported handle type".into(),
        }),
    }
}

pub(super) fn global_to_layout_entry(
    module: &Module,
    layouter: &Layouter,
    gv: &naga::GlobalVariable,
    group: u32,
    binding: u32,
) -> Result<wgpu::BindGroupLayoutEntry, ReflectError> {
    let visibility = wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT;
    let (space, data_ty) = resource_data_ty(module, gv);

    match space {
        AddressSpace::Uniform => {
            layout_entry_for_uniform(layouter, data_ty, group, binding, visibility)
        }
        AddressSpace::Storage { access } => layout_entry_for_storage(
            module, layouter, data_ty, access, group, binding, visibility,
        ),
        AddressSpace::Handle => {
            layout_entry_for_handle(module, data_ty, group, binding, visibility)
        }
        _ => Err(ReflectError::UnsupportedBinding {
            group,
            binding,
            reason: "unsupported address space for global resource".into(),
        }),
    }
}
