//! Validates `@group(0)` against frame globals and optional depth snapshot handles.

use naga::proc::Layouter;
use naga::{AddressSpace, ImageClass, ImageDimension, Module, ScalarKind, TypeInner};

use crate::backend::GpuLight;
use crate::gpu::frame_globals::FrameGpuUniforms;

use super::resource::{resource_data_ty, storage_array_element_stride};
use super::types::ReflectError;

/// Snapshot textures declared by the reflected material through frame-global bindings.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct FrameSnapshotUsage {
    /// Whether the material declares `scene_depth` or `scene_depth_array`.
    pub depth: bool,
    /// Whether the material declares `scene_color` or `scene_color_array`.
    pub color: bool,
}

/// Reflects scene snapshot texture use from the material's live group-0 bindings.
pub(super) fn reflect_frame_snapshot_usage(module: &Module) -> FrameSnapshotUsage {
    let mut usage = FrameSnapshotUsage::default();
    for (_, gv) in module.global_variables.iter() {
        let Some(rb) = gv.binding else {
            continue;
        };
        if rb.group != 0 {
            continue;
        }
        match rb.binding {
            4 | 5 => usage.depth = true,
            6 | 7 => usage.color = true,
            _ => {}
        }
    }
    usage
}

/// Validates group-0 frame-global bindings against the renderer's fixed bind-group layout.
pub(super) fn validate_frame_group0(
    module: &Module,
    layouter: &Layouter,
) -> Result<(), ReflectError> {
    let expected_frame = size_of::<FrameGpuUniforms>() as u32;
    let expected_light = size_of::<GpuLight>() as u32;
    let expected_cluster_u32 = size_of::<u32>() as u32;

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
        if rb.binding > 11 {
            return Err(ReflectError::UnsupportedBinding {
                group: 0,
                binding: rb.binding,
                reason: "only bindings 0..=11 are supported for raster frame globals".into(),
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
            (6, AddressSpace::Handle) => {
                validate_frame_color_texture_binding(module, data_ty, false, rb.binding)?;
            }
            (7, AddressSpace::Handle) => {
                validate_frame_color_texture_binding(module, data_ty, true, rb.binding)?;
            }
            (8, AddressSpace::Handle) => {
                validate_frame_color_sampler_binding(module, data_ty, rb.binding)?;
            }
            (9, AddressSpace::Handle) => {
                validate_frame_skybox_specular_cube_binding(module, data_ty, rb.binding)?;
            }
            (10, AddressSpace::Handle) => {
                validate_frame_color_sampler_binding(module, data_ty, rb.binding)?;
            }
            (11, AddressSpace::Handle) => {
                validate_frame_color_texture_binding(module, data_ty, false, rb.binding)?;
            }
            (0, AddressSpace::Uniform) => {
                b0_size = Some(layouter[data_ty].size);
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

fn validate_frame_skybox_specular_cube_binding(
    module: &Module,
    data_ty: naga::Handle<naga::Type>,
    binding: u32,
) -> Result<(), ReflectError> {
    match &module.types[data_ty].inner {
        TypeInner::Image {
            dim: ImageDimension::Cube,
            arrayed: false,
            class:
                ImageClass::Sampled {
                    kind: ScalarKind::Float,
                    multi: false,
                },
        } => Ok(()),
        TypeInner::Image { .. } => Err(ReflectError::UnsupportedBinding {
            group: 0,
            binding,
            reason: "expected texture_cube<f32>".into(),
        }),
        _ => Err(ReflectError::UnsupportedBinding {
            group: 0,
            binding,
            reason: "expected sampled cube texture handle".into(),
        }),
    }
}

fn validate_frame_depth_texture_binding(
    module: &Module,
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

fn validate_frame_color_texture_binding(
    module: &Module,
    data_ty: naga::Handle<naga::Type>,
    arrayed: bool,
    binding: u32,
) -> Result<(), ReflectError> {
    match &module.types[data_ty].inner {
        TypeInner::Image {
            dim,
            arrayed: got_arrayed,
            class:
                ImageClass::Sampled {
                    kind: ScalarKind::Float,
                    multi,
                },
        } if *dim == ImageDimension::D2 && *got_arrayed == arrayed && !*multi => Ok(()),
        TypeInner::Image { .. } => Err(ReflectError::UnsupportedBinding {
            group: 0,
            binding,
            reason: if arrayed {
                "expected texture_2d_array<f32>".into()
            } else {
                "expected texture_2d<f32>".into()
            },
        }),
        _ => Err(ReflectError::UnsupportedBinding {
            group: 0,
            binding,
            reason: "expected sampled float texture handle".into(),
        }),
    }
}

fn validate_frame_color_sampler_binding(
    module: &Module,
    data_ty: naga::Handle<naga::Type>,
    binding: u32,
) -> Result<(), ReflectError> {
    match &module.types[data_ty].inner {
        TypeInner::Sampler { comparison: false } => Ok(()),
        _ => Err(ReflectError::UnsupportedBinding {
            group: 0,
            binding,
            reason: "expected filtering sampler".into(),
        }),
    }
}
