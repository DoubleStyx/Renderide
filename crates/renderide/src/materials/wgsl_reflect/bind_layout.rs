//! Converts naga global variables at `@group(1)` / `@group(2)` into `wgpu::BindGroupLayoutEntry`.

use std::num::NonZeroU64;

use naga::proc::Layouter;
use naga::{
    AddressSpace, ArraySize, ImageClass, ImageDimension, Module, ScalarKind, StorageAccess,
    TypeInner,
};

use super::resource::resource_data_ty;
use super::types::ReflectError;

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
                    // Per-draw `@group(2)` uses the full storage slab; draw slots are selected with
                    // `@builtin(instance_index)` and `draw_indexed` instance ranges, not dynamic offsets.
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
