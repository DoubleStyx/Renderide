//! Validates `@group(0)` against frame globals and optional depth snapshot handles.

use naga::proc::Layouter;
use naga::{AddressSpace, ImageClass, ImageDimension, Module, ScalarKind, TypeInner};

use crate::gpu::frame_globals::FrameGpuUniforms;
use crate::gpu::{GpuLight, GpuReflectionProbeMetadata, GpuShadowLight, GpuShadowView};

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

/// Expected frame-global storage sizes and strides.
#[derive(Clone, Copy, Debug)]
struct FrameGroup0Expected {
    /// Byte size of the frame uniform.
    frame: u32,
    /// Storage stride for one light row.
    light: u32,
    /// Storage stride for one cluster range row.
    cluster_range: u32,
    /// Storage stride for one cluster light-index row.
    cluster_index: u32,
    /// Storage stride for one reflection-probe metadata row.
    probe: u32,
    /// Storage stride for one per-light shadow metadata row.
    shadow_light: u32,
    /// Storage stride for one shadow-view metadata row.
    shadow_view: u32,
}

impl FrameGroup0Expected {
    /// Builds the renderer's fixed frame-global layout expectations.
    fn renderer_layout() -> Self {
        Self {
            frame: size_of::<FrameGpuUniforms>() as u32,
            light: size_of::<GpuLight>() as u32,
            cluster_range: size_of::<[u32; 2]>() as u32,
            cluster_index: size_of::<u32>() as u32,
            probe: size_of::<GpuReflectionProbeMetadata>() as u32,
            shadow_light: size_of::<GpuShadowLight>() as u32,
            shadow_view: size_of::<GpuShadowView>() as u32,
        }
    }
}

/// Reflected frame-global uniform sizes and storage strides.
#[derive(Clone, Copy, Debug, Default)]
struct FrameGroup0Observed {
    /// Reflected byte size for binding 0.
    b0_size: Option<u32>,
    /// Reflected stride for binding 1.
    b1_stride: Option<u32>,
    /// Reflected stride for binding 2.
    b2_stride: Option<u32>,
    /// Reflected stride for binding 3.
    b3_stride: Option<u32>,
    /// Reflected stride for binding 12.
    b12_stride: Option<u32>,
    /// Reflected stride for binding 15.
    b15_stride: Option<u32>,
    /// Reflected stride for binding 16.
    b16_stride: Option<u32>,
}

impl FrameGroup0Observed {
    /// Records one storage-buffer element stride.
    fn set_storage_stride(&mut self, binding: u32, stride: u32) {
        match binding {
            1 => self.b1_stride = Some(stride),
            2 => self.b2_stride = Some(stride),
            3 => self.b3_stride = Some(stride),
            12 => self.b12_stride = Some(stride),
            15 => self.b15_stride = Some(stride),
            16 => self.b16_stride = Some(stride),
            _ => {}
        }
    }

    /// Validates reflected values against the expected renderer layout.
    fn validate(self, expected: FrameGroup0Expected) -> Result<(), ReflectError> {
        let probe_stride_matches = self
            .b12_stride
            .is_none_or(|stride| stride == expected.probe);
        let shadow_light_stride_matches = self
            .b15_stride
            .is_none_or(|stride| stride == expected.shadow_light);
        let shadow_view_stride_matches = self
            .b16_stride
            .is_none_or(|stride| stride == expected.shadow_view);
        if self.b0_size == Some(expected.frame)
            && self.b1_stride == Some(expected.light)
            && self.b2_stride == Some(expected.cluster_range)
            && self.b3_stride == Some(expected.cluster_index)
            && probe_stride_matches
            && shadow_light_stride_matches
            && shadow_view_stride_matches
        {
            return Ok(());
        }
        Err(ReflectError::FrameGroupMismatch {
            expected_frame: expected.frame,
            expected_light: expected.light,
            expected_cluster_range: expected.cluster_range,
            expected_cluster_index: expected.cluster_index,
            expected_probe: expected.probe,
            expected_shadow_light: expected.shadow_light,
            expected_shadow_view: expected.shadow_view,
            got0: self.b0_size,
            got1: self.b1_stride,
            got2: self.b2_stride,
            got3: self.b3_stride,
            got12: self.b12_stride,
            got15: self.b15_stride,
            got16: self.b16_stride,
        })
    }
}

/// Validates group-0 frame-global bindings against the renderer's fixed bind-group layout.
pub(super) fn validate_frame_group0(
    module: &Module,
    layouter: &Layouter,
) -> Result<(), ReflectError> {
    collect_frame_group0_observed(module, layouter)?
        .validate(FrameGroup0Expected::renderer_layout())
}

/// Collects reflected sizes and validates handle binding types for frame globals.
fn collect_frame_group0_observed(
    module: &Module,
    layouter: &Layouter,
) -> Result<FrameGroup0Observed, ReflectError> {
    let mut observed = FrameGroup0Observed::default();
    for (_, gv) in module.global_variables.iter() {
        let Some(rb) = gv.binding else {
            continue;
        };
        if rb.group != 0 {
            continue;
        }
        if rb.binding > 16 {
            return Err(ReflectError::UnsupportedBinding {
                group: 0,
                binding: rb.binding,
                reason: "only bindings 0..=16 are supported for raster frame globals".into(),
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
                validate_frame_reflection_probe_array_binding(module, data_ty, rb.binding)?;
            }
            (10, AddressSpace::Handle) => {
                validate_frame_color_sampler_binding(module, data_ty, rb.binding)?;
            }
            (11, AddressSpace::Handle) => {
                validate_frame_color_texture_binding(module, data_ty, false, rb.binding)?;
            }
            (13, AddressSpace::Handle) => {
                validate_frame_depth_texture_binding(module, data_ty, true, rb.binding)?;
            }
            (14, AddressSpace::Handle) => {
                validate_frame_comparison_sampler_binding(module, data_ty, rb.binding)?;
            }
            (0, AddressSpace::Uniform) => {
                observed.b0_size = Some(layouter[data_ty].size);
            }
            (_, AddressSpace::Storage { .. }) => {
                let stride = storage_array_element_stride(module, layouter, data_ty, rb.binding)?;
                observed.set_storage_stride(rb.binding, stride);
            }
            _ => {}
        }
    }
    Ok(observed)
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
