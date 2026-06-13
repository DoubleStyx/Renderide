//! Validates `@group(0)` against frame globals and optional depth snapshot handles.

use naga::proc::Layouter;
use naga::{
    AddressSpace, ImageClass, ImageDimension, Module, ResourceBinding, ScalarKind, TypeInner,
};

use crate::gpu::frame_globals::FrameGpuUniforms;
use crate::gpu::{GpuLight, GpuReflectionProbeMetadata, GpuShadowView};

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
    let expected = FrameGroup0Expected::new();
    let mut seen = FrameGroup0Seen::default();

    for (_, gv) in module.global_variables.iter() {
        let Some(rb) = gv.binding else {
            continue;
        };
        if rb.group != 0 {
            continue;
        }
        let (space, data_ty) = resource_data_ty(module, gv);
        record_frame_group0_binding(module, layouter, rb, space, data_ty, &mut seen)?;
    }

    let probe_stride_matches = seen
        .b12_stride
        .is_none_or(|stride| stride == expected.probe);
    let shadow_stride_matches = seen
        .b16_stride
        .is_none_or(|stride| stride == expected.shadow_view);
    if seen.b0_size == Some(expected.frame)
        && seen.b1_stride == Some(expected.light)
        && seen.b2_stride == Some(expected.cluster_range)
        && seen.b3_stride == Some(expected.cluster_index)
        && probe_stride_matches
        && shadow_stride_matches
    {
        Ok(())
    } else {
        Err(ReflectError::FrameGroupMismatch {
            expected_frame: expected.frame,
            expected_light: expected.light,
            expected_cluster_range: expected.cluster_range,
            expected_cluster_index: expected.cluster_index,
            expected_probe: expected.probe,
            got0: seen.b0_size,
            got1: seen.b1_stride,
            got2: seen.b2_stride,
            got3: seen.b3_stride,
            got12: seen.b12_stride,
        })
    }
}

struct FrameGroup0Expected {
    frame: u32,
    light: u32,
    cluster_range: u32,
    cluster_index: u32,
    probe: u32,
    shadow_view: u32,
}

impl FrameGroup0Expected {
    fn new() -> Self {
        Self {
            frame: size_of::<FrameGpuUniforms>() as u32,
            light: size_of::<GpuLight>() as u32,
            cluster_range: size_of::<[u32; 2]>() as u32,
            cluster_index: size_of::<u32>() as u32,
            probe: size_of::<GpuReflectionProbeMetadata>() as u32,
            shadow_view: size_of::<GpuShadowView>() as u32,
        }
    }
}

#[derive(Default)]
struct FrameGroup0Seen {
    b0_size: Option<u32>,
    b1_stride: Option<u32>,
    b2_stride: Option<u32>,
    b3_stride: Option<u32>,
    b12_stride: Option<u32>,
    b16_stride: Option<u32>,
}

fn record_frame_group0_binding(
    module: &Module,
    layouter: &Layouter,
    rb: ResourceBinding,
    space: AddressSpace,
    data_ty: naga::Handle<naga::Type>,
    seen: &mut FrameGroup0Seen,
) -> Result<(), ReflectError> {
    if rb.binding > 18 {
        return Err(ReflectError::UnsupportedBinding {
            group: 0,
            binding: rb.binding,
            reason: "only bindings 0..=18 are supported for raster frame globals".into(),
        });
    }
    match (rb.binding, space) {
        (4, AddressSpace::Handle) => {
            validate_frame_depth_texture_binding(module, data_ty, false, rb.binding)?;
        }
        (5 | 17, AddressSpace::Handle) => {
            validate_frame_depth_texture_binding(module, data_ty, true, rb.binding)?;
        }
        (6 | 11, AddressSpace::Handle) => {
            validate_frame_color_texture_binding(module, data_ty, false, rb.binding)?;
        }
        (7 | 13 | 14, AddressSpace::Handle) => {
            validate_frame_color_texture_binding(module, data_ty, true, rb.binding)?;
        }
        (8 | 10 | 15, AddressSpace::Handle) => {
            validate_frame_color_sampler_binding(module, data_ty, rb.binding)?;
        }
        (9, AddressSpace::Handle) => {
            validate_frame_reflection_probe_array_binding(module, data_ty, rb.binding)?;
        }
        (18, AddressSpace::Handle) => {
            validate_frame_comparison_sampler_binding(module, data_ty, rb.binding)?;
        }
        (0, AddressSpace::Uniform) => seen.b0_size = Some(layouter[data_ty].size),
        (_, AddressSpace::Storage { .. }) => {
            record_frame_group0_storage_stride(module, layouter, data_ty, rb.binding, seen)?;
        }
        _ => {}
    }
    Ok(())
}

fn record_frame_group0_storage_stride(
    module: &Module,
    layouter: &Layouter,
    data_ty: naga::Handle<naga::Type>,
    binding: u32,
    seen: &mut FrameGroup0Seen,
) -> Result<(), ReflectError> {
    let stride = storage_array_element_stride(module, layouter, data_ty, binding)?;
    match binding {
        1 => seen.b1_stride = Some(stride),
        2 => seen.b2_stride = Some(stride),
        3 => seen.b3_stride = Some(stride),
        12 => seen.b12_stride = Some(stride),
        16 => seen.b16_stride = Some(stride),
        _ => {}
    }
    Ok(())
}

fn validate_frame_comparison_sampler_binding(
    module: &Module,
    data_ty: naga::Handle<naga::Type>,
    binding: u32,
) -> Result<(), ReflectError> {
    match &module.types[data_ty].inner {
        TypeInner::Sampler { comparison: true } => Ok(()),
        _ => Err(ReflectError::UnsupportedBinding {
            group: 0,
            binding,
            reason: "expected comparison sampler".into(),
        }),
    }
}

fn validate_frame_reflection_probe_array_binding(
    module: &Module,
    data_ty: naga::Handle<naga::Type>,
    binding: u32,
) -> Result<(), ReflectError> {
    match &module.types[data_ty].inner {
        TypeInner::Image {
            dim: ImageDimension::D2,
            arrayed: true,
            class:
                ImageClass::Sampled {
                    kind: ScalarKind::Float,
                    multi: false,
                },
        } => Ok(()),
        TypeInner::Image { .. } => Err(ReflectError::UnsupportedBinding {
            group: 0,
            binding,
            reason: "expected texture_2d_array<f32>".into(),
        }),
        _ => Err(ReflectError::UnsupportedBinding {
            group: 0,
            binding,
            reason: "expected sampled 2D-array texture handle".into(),
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
