//! Prepared skybox/background draw for the world-mesh forward opaque pass.

use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::{Arc, OnceLock};

use bytemuck::{Pod, Zeroable};
use parking_lot::Mutex;

use crate::backend::{EmbeddedTexturePools, FrameGpuResources};
use crate::camera::ViewId;
use crate::camera::view_matrix_for_world_mesh_render_space;
use crate::embedded_shaders;
use crate::materials::host_data::MaterialPropertyLookupIds;
use crate::render_graph::frame_params::{
    FrameRenderParams, PreparedClearColorSkybox, PreparedMaterialSkybox, PreparedSkybox,
    WorldMeshForwardPipelineState,
};
use crate::render_graph::frame_upload_batch::FrameUploadBatch;
use crate::shared::CameraClearMode;

/// Minimum binding size for [`SkyboxViewUniforms`].
const SKYBOX_VIEW_UNIFORM_SIZE: u64 = size_of::<SkyboxViewUniforms>() as u64;

/// Skybox material family supported by the dedicated background draw.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum SkyboxFamily {
    /// Froox `Projection360Material`.
    Projection360,
    /// Froox `GradientSkyMaterial`.
    Gradient,
    /// Froox `ProceduralSkyMaterial`.
    Procedural,
}

impl SkyboxFamily {
    /// Resolves the supported family from an embedded material stem.
    fn from_stem(stem: &str) -> Option<Self> {
        let base = stem
            .strip_suffix("_default")
            .or_else(|| stem.strip_suffix("_multiview"))
            .unwrap_or(stem);
        match base.to_ascii_lowercase().as_str() {
            "projection360" => Some(Self::Projection360),
            "gradientskybox" => Some(Self::Gradient),
            "proceduralskybox" | "proceduralsky" => Some(Self::Procedural),
            _ => None,
        }
    }

    /// Embedded backend shader target for this family and view permutation.
    fn shader_target(self, multiview: bool) -> &'static str {
        match (self, multiview) {
            (Self::Projection360, false) => "skybox_projection360_default",
            (Self::Projection360, true) => "skybox_projection360_multiview",
            (Self::Gradient, false) => "skybox_gradientskybox_default",
            (Self::Gradient, true) => "skybox_gradientskybox_multiview",
            (Self::Procedural, false) => "skybox_proceduralskybox_default",
            (Self::Procedural, true) => "skybox_proceduralskybox_multiview",
        }
    }
}

/// Render-target state that must match the containing skybox render pass.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct SkyboxPipelineTarget {
    /// HDR scene-color format.
    color_format: wgpu::TextureFormat,
    /// Depth-stencil attachment format used by the containing world pass.
    depth_stencil_format: Option<wgpu::TextureFormat>,
    /// Raster sample count.
    sample_count: u32,
    /// Whether the target uses stereo multiview.
    multiview: bool,
}

impl SkyboxPipelineTarget {
    /// Builds the target descriptor from the prepared world-mesh forward pipeline state.
    fn from_forward_state(pipeline_state: &WorldMeshForwardPipelineState) -> Self {
        Self {
            color_format: pipeline_state.pass_desc.surface_format,
            depth_stencil_format: pipeline_state.pass_desc.depth_stencil_format,
            sample_count: pipeline_state.pass_desc.sample_count,
            multiview: pipeline_state.use_multiview,
        }
    }
}

/// Cached material skybox pipeline key.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct SkyboxPipelineKey {
    /// Supported sky material family.
    family: SkyboxFamily,
    /// Render-target state required by wgpu pipeline/pass compatibility.
    target: SkyboxPipelineTarget,
}

/// Cached solid-color background pipeline key.
type ClearPipelineKey = SkyboxPipelineTarget;

/// Per-view cached uniform buffer and bind group.
struct SkyboxViewBinding {
    /// Uniform buffer updated during the prepare callback.
    buffer: wgpu::Buffer,
    /// Bind group for the uniform buffer.
    bind_group: Arc<wgpu::BindGroup>,
}

/// Draw-local skybox uniforms consumed by `@group(2)` material skybox shaders.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct SkyboxViewUniforms {
    /// View-to-world X basis for the left eye or mono view.
    view_x_left: [f32; 4],
    /// View-to-world Y basis for the left eye or mono view.
    view_y_left: [f32; 4],
    /// View-to-world Z basis for the left eye or mono view.
    view_z_left: [f32; 4],
    /// View-to-world X basis for the right eye.
    view_x_right: [f32; 4],
    /// View-to-world Y basis for the right eye.
    view_y_right: [f32; 4],
    /// View-to-world Z basis for the right eye.
    view_z_right: [f32; 4],
    /// Background color for `CameraClearMode::Color`.
    clear_color: [f32; 4],
    /// Reserved padding.
    _pad: [f32; 4],
}

impl SkyboxViewUniforms {
    /// Builds view bases and clear color for the current view.
    fn from_frame(frame: &FrameRenderParams<'_>) -> Self {
        let (left, right) = skybox_world_to_view_pair(frame);
        let (lx, ly, lz) = view_to_world_basis(left);
        let (rx, ry, rz) = view_to_world_basis(right);
        Self {
            view_x_left: lx,
            view_y_left: ly,
            view_z_left: lz,
            view_x_right: rx,
            view_y_right: ry,
            view_z_right: rz,
            clear_color: frame.view.clear.color.to_array(),
            _pad: [0.0; 4],
        }
    }
}

/// Persistent skybox caches owned by the forward prepare pass.
pub(super) struct SkyboxRenderer {
    view_layout: OnceLock<wgpu::BindGroupLayout>,
    material_pipelines: Mutex<HashMap<SkyboxPipelineKey, Arc<wgpu::RenderPipeline>>>,
    clear_pipelines: Mutex<HashMap<ClearPipelineKey, Arc<wgpu::RenderPipeline>>>,
    view_bindings: Mutex<HashMap<ViewId, SkyboxViewBinding>>,
}

impl std::fmt::Debug for SkyboxRenderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SkyboxRenderer").finish_non_exhaustive()
    }
}

impl Default for SkyboxRenderer {
    fn default() -> Self {
        Self {
            view_layout: OnceLock::new(),
            material_pipelines: Mutex::new(HashMap::new()),
            clear_pipelines: Mutex::new(HashMap::new()),
            view_bindings: Mutex::new(HashMap::new()),
        }
    }
}

impl SkyboxRenderer {
    /// Removes draw-local uniform bindings for views that are no longer active.
    pub(super) fn release_view_resources(&self, retired_views: &[ViewId]) {
        if retired_views.is_empty() {
            return;
        }
        let mut bindings = self.view_bindings.lock();
        for view_id in retired_views {
            bindings.remove(view_id);
        }
    }

    /// Prepares the background draw for this view, if any.
    pub(super) fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        upload_batch: &FrameUploadBatch,
        frame: &FrameRenderParams<'_>,
        pipeline_state: &WorldMeshForwardPipelineState,
    ) -> Option<PreparedSkybox> {
        match frame.view.clear.mode {
            CameraClearMode::Skybox => {
                self.prepare_material_skybox(device, queue, upload_batch, frame, pipeline_state)
            }
            CameraClearMode::Color => {
                self.prepare_clear_color(device, upload_batch, frame, pipeline_state)
            }
            CameraClearMode::Depth | CameraClearMode::Nothing => None,
        }
    }

    /// Resolves the active render-space skybox material into a prepared draw.
    fn prepare_material_skybox(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        upload_batch: &FrameUploadBatch,
        frame: &FrameRenderParams<'_>,
        pipeline_state: &WorldMeshForwardPipelineState,
    ) -> Option<PreparedSkybox> {
        let material_asset_id = frame
            .shared
            .scene
            .active_main_space()?
            .skybox_material_asset_id;
        if material_asset_id < 0 {
            return None;
        }

        let materials = frame.shared.materials;
        let store = materials.material_property_store();
        let shader_asset_id = store.shader_asset_for_material(material_asset_id)?;
        let registry = materials.material_registry()?;
        let stem = skybox_stem_for_shader_asset(registry, shader_asset_id)?;
        let family = SkyboxFamily::from_stem(stem.as_str())?;
        let embedded_bind = materials.embedded_material_bind()?;
        let pools = EmbeddedTexturePools {
            texture: &frame.shared.asset_transfers.texture_pool,
            texture3d: &frame.shared.asset_transfers.texture3d_pool,
            cubemap: &frame.shared.asset_transfers.cubemap_pool,
            render_texture: &frame.shared.asset_transfers.render_texture_pool,
            video_texture: &frame.shared.asset_transfers.video_texture_pool,
        };
        let lookup = MaterialPropertyLookupIds {
            material_asset_id,
            mesh_property_block_slot0: None,
        };
        let material_bind_group = embedded_bind
            .embedded_material_bind_group(
                stem.as_str(),
                queue,
                store,
                &pools,
                lookup,
                frame.view.offscreen_write_render_texture_asset_id,
            )
            .ok()?;
        let material_layout = embedded_bind
            .embedded_material_bind_group_layout(stem.as_str())
            .ok()?;
        let view_bind_group = self.view_bind_group(device, upload_batch, frame);
        let target = SkyboxPipelineTarget::from_forward_state(pipeline_state);
        let pipeline = self.material_pipeline(device, &material_layout, family, target)?;
        Some(PreparedSkybox::Material(PreparedMaterialSkybox {
            pipeline,
            material_bind_group,
            view_bind_group,
        }))
    }

    /// Builds a prepared fullscreen draw for `CameraClearMode::Color`.
    fn prepare_clear_color(
        &self,
        device: &wgpu::Device,
        upload_batch: &FrameUploadBatch,
        frame: &FrameRenderParams<'_>,
        pipeline_state: &WorldMeshForwardPipelineState,
    ) -> Option<PreparedSkybox> {
        let view_bind_group = self.view_bind_group(device, upload_batch, frame);
        let target = SkyboxPipelineTarget::from_forward_state(pipeline_state);
        let pipeline = self.clear_pipeline(device, target)?;
        Some(PreparedSkybox::ClearColor(PreparedClearColorSkybox {
            pipeline,
            view_bind_group,
        }))
    }

    /// Returns the cached draw-local skybox view bind-group layout.
    fn view_layout(&self, device: &wgpu::Device) -> &wgpu::BindGroupLayout {
        self.view_layout.get_or_init(|| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("skybox_view"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(SKYBOX_VIEW_UNIFORM_SIZE),
                    },
                    count: None,
                }],
            })
        })
    }

    /// Updates and returns the per-view skybox uniform bind group.
    fn view_bind_group(
        &self,
        device: &wgpu::Device,
        upload_batch: &FrameUploadBatch,
        frame: &FrameRenderParams<'_>,
    ) -> Arc<wgpu::BindGroup> {
        let view_id = frame.view.view_id;
        let uniforms = SkyboxViewUniforms::from_frame(frame);
        let (buffer, bind_group) = {
            let mut bindings = self.view_bindings.lock();
            let entry = bindings.entry(view_id).or_insert_with(|| {
                let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("skybox_view_uniform"),
                    size: SKYBOX_VIEW_UNIFORM_SIZE,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let bind_group = Arc::new(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("skybox_view"),
                    layout: self.view_layout(device),
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffer.as_entire_binding(),
                    }],
                }));
                SkyboxViewBinding { buffer, bind_group }
            });
            let resolved = (entry.buffer.clone(), Arc::clone(&entry.bind_group));
            drop(bindings);
            resolved
        };
        upload_batch.write_buffer(&buffer, 0, bytemuck::bytes_of(&uniforms));
        bind_group
    }

    /// Returns a cached material skybox pipeline for the view target state.
    fn material_pipeline(
        &self,
        device: &wgpu::Device,
        material_layout: &wgpu::BindGroupLayout,
        family: SkyboxFamily,
        target: SkyboxPipelineTarget,
    ) -> Option<Arc<wgpu::RenderPipeline>> {
        let key = SkyboxPipelineKey { family, target };
        {
            let guard = self.material_pipelines.lock();
            if let Some(pipeline) = guard.get(&key) {
                return Some(Arc::clone(pipeline));
            }
        }

        let shader_target = family.shader_target(target.multiview);
        let source = embedded_shaders::embedded_target_wgsl(shader_target)?;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(shader_target),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let frame_layout = FrameGpuResources::bind_group_layout(device);
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(shader_target),
            bind_group_layouts: &[
                Some(&frame_layout),
                Some(material_layout),
                Some(self.view_layout(device)),
            ],
            immediate_size: 0,
        });
        let pipeline = Arc::new(create_skybox_pipeline(
            device,
            shader_target,
            &shader,
            &layout,
            target,
        ));
        let mut guard = self.material_pipelines.lock();
        if let Some(existing) = guard.get(&key) {
            return Some(Arc::clone(existing));
        }
        guard.insert(key, Arc::clone(&pipeline));
        drop(guard);
        Some(pipeline)
    }

    /// Returns a cached solid-color background pipeline for the view target state.
    fn clear_pipeline(
        &self,
        device: &wgpu::Device,
        target: SkyboxPipelineTarget,
    ) -> Option<Arc<wgpu::RenderPipeline>> {
        let key = target;
        {
            let guard = self.clear_pipelines.lock();
            if let Some(pipeline) = guard.get(&key) {
                return Some(Arc::clone(pipeline));
            }
        }

        let shader_target = "skybox_solid_color";
        let source = embedded_shaders::embedded_target_wgsl(shader_target)?;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(shader_target),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(shader_target),
            bind_group_layouts: &[Some(self.view_layout(device))],
            immediate_size: 0,
        });
        let pipeline = Arc::new(create_skybox_pipeline(
            device,
            shader_target,
            &shader,
            &layout,
            key,
        ));
        let mut guard = self.clear_pipelines.lock();
        if let Some(existing) = guard.get(&key) {
            return Some(Arc::clone(existing));
        }
        guard.insert(key, Arc::clone(&pipeline));
        drop(guard);
        Some(pipeline)
    }
}

/// Records a prepared skybox/background draw before world meshes.
pub(super) fn record_prepared_skybox(
    rpass: &mut wgpu::RenderPass<'_>,
    frame: &FrameRenderParams<'_>,
    prepared: &PreparedSkybox,
) -> bool {
    profiling::scope!("world_mesh_forward::skybox_record");
    match prepared {
        PreparedSkybox::Material(skybox) => {
            let Some(frame_bg) = frame
                .shared
                .frame_resources
                .per_view_frame(frame.view.view_id)
                .map(|s| s.frame_bind_group.clone())
            else {
                return false;
            };
            rpass.set_pipeline(skybox.pipeline.as_ref());
            rpass.set_bind_group(0, frame_bg.as_ref(), &[]);
            rpass.set_bind_group(1, skybox.material_bind_group.as_ref(), &[]);
            rpass.set_bind_group(2, skybox.view_bind_group.as_ref(), &[]);
            rpass.draw(0..3, 0..1);
            true
        }
        PreparedSkybox::ClearColor(clear) => {
            rpass.set_pipeline(clear.pipeline.as_ref());
            rpass.set_bind_group(0, clear.view_bind_group.as_ref(), &[]);
            rpass.draw(0..3, 0..1);
            true
        }
    }
}

/// Creates a fullscreen skybox/background render pipeline compatible with the world pass.
fn create_skybox_pipeline(
    device: &wgpu::Device,
    label: &str,
    shader: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    target: SkyboxPipelineTarget,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: target.color_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: target
            .depth_stencil_format
            .map(|format| wgpu::DepthStencilState {
                format,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
        multisample: wgpu::MultisampleState {
            count: target.sample_count.max(1),
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview_mask: target
            .multiview
            .then(|| std::num::NonZeroU32::new(3))
            .flatten(),
        cache: None,
    })
}

/// Resolves a host shader asset id into the embedded skybox material stem.
fn skybox_stem_for_shader_asset(
    registry: &crate::materials::MaterialRegistry,
    shader_asset_id: i32,
) -> Option<String> {
    registry
        .stem_for_shader_asset(shader_asset_id)
        .map(str::to_string)
}

/// Finds the world-to-view matrices used for skybox ray reconstruction.
fn skybox_world_to_view_pair(frame: &FrameRenderParams<'_>) -> (glam::Mat4, glam::Mat4) {
    let hc = frame.view.host_camera;
    if let (true, Some(stereo)) = (hc.vr_active, hc.stereo) {
        return stereo.view_only;
    }
    let view = hc.explicit_world_to_view.unwrap_or_else(|| {
        frame
            .shared
            .scene
            .active_main_space()
            .map_or(glam::Mat4::IDENTITY, |space| {
                view_matrix_for_world_mesh_render_space(frame.shared.scene, space)
            })
    });
    (view, view)
}

/// Converts a world-to-view matrix into packed view-to-world basis vectors.
fn view_to_world_basis(world_to_view: glam::Mat4) -> ([f32; 4], [f32; 4], [f32; 4]) {
    let view_to_world = world_to_view.inverse();
    (
        view_to_world.x_axis.truncate().extend(0.0).to_array(),
        view_to_world.y_axis.truncate().extend(0.0).to_array(),
        view_to_world.z_axis.truncate().extend(0.0).to_array(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipelines::{SHADER_PERM_MULTIVIEW_STEREO, ShaderPermutation};

    #[test]
    fn skybox_family_resolves_supported_stems() {
        assert_eq!(
            SkyboxFamily::from_stem("projection360_default"),
            Some(SkyboxFamily::Projection360)
        );
        assert_eq!(
            SkyboxFamily::from_stem("gradientskybox_default"),
            Some(SkyboxFamily::Gradient)
        );
        assert_eq!(
            SkyboxFamily::from_stem("proceduralskybox_multiview"),
            Some(SkyboxFamily::Procedural)
        );
        assert_eq!(SkyboxFamily::from_stem("pbsmetallic_default"), None);
    }

    #[test]
    fn skybox_view_uniforms_are_16_byte_aligned() {
        assert_eq!(size_of::<SkyboxViewUniforms>() % 16, 0);
        assert_eq!(SKYBOX_VIEW_UNIFORM_SIZE, 128);
    }

    #[test]
    fn material_skybox_uses_multiview_shader_targets() {
        assert_eq!(
            SkyboxFamily::Gradient.shader_target(false),
            "skybox_gradientskybox_default"
        );
        assert_eq!(
            SkyboxFamily::Gradient.shader_target(true),
            "skybox_gradientskybox_multiview"
        );
    }

    #[test]
    fn multiview_permutation_constant_stays_distinct() {
        assert_ne!(ShaderPermutation(0), SHADER_PERM_MULTIVIEW_STEREO);
    }
}
