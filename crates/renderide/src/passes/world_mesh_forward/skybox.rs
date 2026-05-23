//! Prepared skybox/background draw for the world-mesh forward opaque pass.

mod pipeline;

use std::num::NonZeroU64;
use std::sync::{Arc, OnceLock};

use bytemuck::{Pod, Zeroable};
use hashbrown::HashMap;
use parking_lot::Mutex;
use wgpu::util::DeviceExt;

use super::WorldMeshForwardPipelineState;
use super::raster_recording::frame_bind_group_for_view;
use crate::assets::resolve::assets_search_candidates;
use crate::camera::{CameraProjectionKind, ViewId, world_to_view_pair_for_skybox};
use crate::embedded_shaders;
use crate::gpu::frame_bind_group_layout;
use crate::graph_inputs::GraphPassFrame;
use crate::materials::host_data::MaterialPropertyLookupIds;
use crate::materials::{EmbeddedMaterialBindShader, EmbeddedTexturePools};
use crate::render_graph::blackboard::Blackboard;
use crate::render_graph::frame_upload_batch::GraphUploadSink;
use crate::shared::CameraClearMode;
use crate::skybox::{PreparedClearColorSkybox, PreparedMaterialSkybox, PreparedSkybox};

use pipeline::{
    ClearPipelineKey, SkyboxDepthState, SkyboxFamily, SkyboxPipelineKey, SkyboxPipelineTarget,
    create_skybox_pipeline,
};

/// Minimum binding size for [`SkyboxViewUniforms`].
const SKYBOX_VIEW_UNIFORM_SIZE: u64 = size_of::<SkyboxViewUniforms>() as u64;

/// Per-view cached uniform buffer and bind group.
struct SkyboxViewBinding {
    /// Uniform buffer updated during backend world-mesh frame planning.
    buffer: wgpu::Buffer,
    /// Bind group for the uniform buffer.
    bind_group: Arc<wgpu::BindGroup>,
}

/// Draw-local skybox uniforms consumed by `@group(2)` material skybox shaders.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct SkyboxViewUniforms {
    /// View-to-world basis for the left eye or mono view.
    view_left: [f32; 16],
    /// View-to-world basis for the right eye.
    view_right: [f32; 16],
    /// World-to-view basis for the left eye or mono view.
    world_left: [f32; 16],
    /// World-to-view basis for the right eye.
    world_right: [f32; 16],
    /// Background color for `CameraClearMode::Color`.
    clear_color: [f32; 4],
    /// `.x`: ndc Y sign passed to the fragment shader (1.0 normal, -1.0 for offscreen-RT views).
    /// Offscreen-RT views pre-multiply a clip-space Y flip into the world view-projection so the
    /// render-texture lands V=0 bottom. The skybox is a fullscreen pass whose vertex Y flip is a
    /// rasterization no-op, so we flip the ndc.y the fragment receives instead -- that inverts the
    /// computed view ray, which is what actually changes which sky direction is sampled per
    /// framebuffer row. `.y` is the left/mono orthographic flag, `.z` is the right-eye
    /// orthographic flag, and `.w` is reserved padding.
    ndc_y_sign_pad: [f32; 4],
}

impl SkyboxViewUniforms {
    /// Builds view bases and clear color for the current view.
    fn from_frame(frame: &GraphPassFrame<'_>) -> Self {
        let (left, right) = skybox_world_to_view_pair(frame);
        let ndc_y_sign = if frame.view.offscreen_write_render_texture_asset_id.is_some() {
            -1.0
        } else {
            1.0
        };
        let ortho_flag = projection_kind_orthographic_flag(frame.view.host_camera.projection_kind);
        Self {
            view_left: left.inverse().to_cols_array(),
            view_right: right.inverse().to_cols_array(),
            world_left: left.to_cols_array(),
            world_right: right.to_cols_array(),
            clear_color: frame.view.clear.color.to_array(),
            ndc_y_sign_pad: [ndc_y_sign, ortho_flag, ortho_flag, 0.0],
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct SkyboxMeshBufferKey {
    family: SkyboxFamily,
    device: wgpu::Device,
}

/// Persistent skybox caches owned by backend world-mesh frame planning.
pub(crate) struct SkyboxRenderer {
    view_layout: OnceLock<wgpu::BindGroupLayout>,
    material_pipelines: Mutex<HashMap<SkyboxPipelineKey, Arc<wgpu::RenderPipeline>>>,
    clear_pipelines: Mutex<HashMap<ClearPipelineKey, Arc<wgpu::RenderPipeline>>>,
    view_bindings: Mutex<HashMap<ViewId, SkyboxViewBinding>>,
    skybox_mesh_buffers: Mutex<HashMap<SkyboxMeshBufferKey, wgpu::Buffer>>,
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
            skybox_mesh_buffers: Mutex::new(HashMap::new()),
        }
    }
}

impl SkyboxRenderer {
    /// Removes draw-local uniform bindings for views that are no longer active.
    pub(crate) fn release_view_resources(&self, retired_views: &[ViewId]) {
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
        uploads: GraphUploadSink<'_>,
        frame: &GraphPassFrame<'_>,
        pipeline_state: &WorldMeshForwardPipelineState,
    ) -> Option<PreparedSkybox> {
        match frame.view.clear.mode {
            CameraClearMode::Skybox => self
                .prepare_material_skybox(device, uploads, frame, pipeline_state)
                .or_else(|| self.prepare_clear_color(device, uploads, frame, pipeline_state)),
            CameraClearMode::Color => {
                self.prepare_clear_color(device, uploads, frame, pipeline_state)
            }
            CameraClearMode::Depth | CameraClearMode::Nothing => None,
        }
    }

    /// Resolves the active render-space skybox material into a prepared draw.
    fn prepare_material_skybox(
        &self,
        device: &wgpu::Device,
        uploads: GraphUploadSink<'_>,
        frame: &GraphPassFrame<'_>,
        pipeline_state: &WorldMeshForwardPipelineState,
    ) -> Option<PreparedSkybox> {
        let material_asset_id = frame
            .shared
            .scene
            .active_main_space()?
            .skybox_material_asset_id();
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
            texture: frame.shared.asset_resources.texture_pool(),
            texture3d: frame.shared.asset_resources.texture3d_pool(),
            cubemap: frame.shared.asset_resources.cubemap_pool(),
            render_texture: frame.shared.asset_resources.render_texture_pool(),
            video_texture: frame.shared.asset_resources.video_texture_pool(),
        };
        let lookup = MaterialPropertyLookupIds {
            material_asset_id,
            mesh_property_block_slot0: None,
            mesh_renderer_property_block_id: None,
        };
        let shader_variant_bits = registry.variant_bits_for_shader_asset(shader_asset_id);
        let material_bind = embedded_bind
            .embedded_material_bind_group_with_cache_key(
                EmbeddedMaterialBindShader {
                    stem: stem.as_str(),
                    shader_variant_bits,
                },
                uploads,
                store,
                &pools,
                lookup,
                frame.view.offscreen_write_render_texture_asset_id,
            )
            .ok()
            .map(|(_, group)| group)?;
        let material_layout = embedded_bind
            .embedded_material_bind_group_layout(stem.as_str())
            .ok()?;
        let view_bind_group = self.view_bind_group(device, uploads, frame);
        let target = SkyboxPipelineTarget::from_forward_state(pipeline_state);
        let pipeline = self.material_pipeline(device, &material_layout, family, target)?;
        let skybox_mesh_buffer = self.skybox_mesh_buffer(device, family);
        Some(PreparedSkybox::Material(PreparedMaterialSkybox {
            pipeline,
            material_bind_group: material_bind.bind_group,
            material_uniform_dynamic_offset: material_bind.uniform_dynamic_offset,
            view_bind_group,
            skybox_mesh_buffer,
        }))
    }

    /// Builds a prepared fullscreen draw for `CameraClearMode::Color`.
    fn prepare_clear_color(
        &self,
        device: &wgpu::Device,
        uploads: GraphUploadSink<'_>,
        frame: &GraphPassFrame<'_>,
        pipeline_state: &WorldMeshForwardPipelineState,
    ) -> Option<PreparedSkybox> {
        let view_bind_group = self.view_bind_group(device, uploads, frame);
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
        uploads: GraphUploadSink<'_>,
        frame: &GraphPassFrame<'_>,
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
                crate::profiling::note_resource_churn!(Buffer, "passes::skybox_view_uniform");
                let bind_group = Arc::new(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("skybox_view"),
                    layout: self.view_layout(device),
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffer.as_entire_binding(),
                    }],
                }));
                crate::profiling::note_resource_churn!(BindGroup, "passes::skybox_view_bind_group");
                SkyboxViewBinding { buffer, bind_group }
            });
            let resolved = (entry.buffer.clone(), Arc::clone(&entry.bind_group));
            drop(bindings);
            resolved
        };
        uploads.write_buffer(&buffer, 0, bytemuck::bytes_of(&uniforms));
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
        let frame_layout = frame_bind_group_layout(device);
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
            family,
            target,
            SkyboxDepthState::fixed_background(),
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

        let family = SkyboxFamily::Clear;
        let shader_target = family.shader_target(false);
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
            family,
            key,
            SkyboxDepthState::fixed_background(),
        ));
        let mut guard = self.clear_pipelines.lock();
        if let Some(existing) = guard.get(&key) {
            return Some(Arc::clone(existing));
        }
        guard.insert(key, Arc::clone(&pipeline));
        drop(guard);
        Some(pipeline)
    }

    fn skybox_mesh_buffer(&self, device: &wgpu::Device, family: SkyboxFamily) -> wgpu::Buffer {
        let key = SkyboxMeshBufferKey {
            device: device.clone(),
            family,
        };
        {
            if let Some(buffer) = self.skybox_mesh_buffers.lock().get(&key) {
                return buffer.clone();
            }
        }

        let buffer = skybox_sphere_buffer(device, family.draw_vertex_count());
        let mut guard = self.skybox_mesh_buffers.lock();
        guard.insert(key, buffer.clone());
        buffer
    }
}

/// Records a prepared skybox/background draw after opaque world meshes.
pub(super) fn record_prepared_skybox(
    rpass: &mut wgpu::RenderPass<'_>,
    frame: &GraphPassFrame<'_>,
    blackboard: &Blackboard,
    prepared: &PreparedSkybox,
) -> bool {
    profiling::scope!("world_mesh_forward::skybox_record");
    match prepared {
        PreparedSkybox::Material(skybox) => {
            let Some(frame_bg) = frame_bind_group_for_view(frame, blackboard) else {
                return false;
            };
            rpass.set_pipeline(skybox.pipeline.as_ref());
            rpass.set_bind_group(0, frame_bg.as_ref(), &[]);
            if let Some(offset) = skybox.material_uniform_dynamic_offset {
                rpass.set_bind_group(1, skybox.material_bind_group.as_ref(), &[offset]);
            } else {
                rpass.set_bind_group(1, skybox.material_bind_group.as_ref(), &[]);
            }
            rpass.set_bind_group(2, skybox.view_bind_group.as_ref(), &[]);
            let vertex_count = skybox.skybox_mesh_buffer.size() / 12;
            rpass.set_vertex_buffer(0, skybox.skybox_mesh_buffer.slice(0..));
            rpass.draw(0..vertex_count as u32, 0..1);
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
fn skybox_world_to_view_pair(frame: &GraphPassFrame<'_>) -> (glam::Mat4, glam::Mat4) {
    world_to_view_pair_for_skybox(frame.shared.scene, &frame.view.host_camera)
}

fn skybox_sphere_buffer(device: &wgpu::Device, vertex_count: u32) -> wgpu::Buffer {
    let vertices = if vertex_count == 0 {
        skybox_sphere_vertices()
    } else {
        &vec![[0.0; 3]; vertex_count as usize]
    };
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Skybox_mesh"),
        contents: bytemuck::cast_slice(vertices),
        usage: wgpu::BufferUsages::VERTEX,
    })
}

static SKYBOX_SPHERE_VERTICES: OnceLock<Vec<[f32; 3]>> = OnceLock::new();

fn skybox_sphere_vertices() -> &'static Vec<[f32; 3]> {
    SKYBOX_SPHERE_VERTICES.get_or_init(parse_skybox_sphere)
}

fn parse_skybox_sphere() -> Vec<[f32; 3]> {
    let asset_folders = assets_search_candidates();
    let Some(skybox_path) = asset_folders
        .iter()
        .map(|p| p.join("models/skybox.glb"))
        .find(|p| p.is_file())
    else {
        logger::error!("Could not locate skybox model");
        return vec![[0.0; 3]; 3];
    };
    let (doc, buffers, _) = match gltf::import(skybox_path.as_path()) {
        Ok(result) => result,
        Err(err) => {
            logger::error!("Could not parse {}: {err}", skybox_path.display());
            return vec![[0.0; 3]; 3];
        }
    };
    let mut vertices = Vec::new();
    for mesh in doc.meshes() {
        logger::debug!(
            "Skybox: Reading mesh {} {}",
            mesh.index(),
            mesh.name().unwrap_or("")
        );
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|p| Some(&buffers[p.index()]));
            let positions = reader
                .read_positions()
                .map(|pi| pi.collect())
                .unwrap_or(Vec::new());
            if let Some(indices) = reader.read_indices() {
                for index in indices.into_u32() {
                    if let Some(&position) = positions.get(index as usize) {
                        vertices.push(position);
                    }
                }
            }
        }
    }
    logger::info!(
        "Skybox mesh parsed successfully, {} vertices in mesh",
        vertices.len()
    );
    vertices
}

fn projection_kind_orthographic_flag(kind: CameraProjectionKind) -> f32 {
    if kind == CameraProjectionKind::Orthographic {
        1.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::materials::{SHADER_PERM_MULTIVIEW_STEREO, ShaderPermutation};

    #[test]
    fn skybox_view_uniforms_are_16_byte_aligned() {
        assert_eq!(size_of::<SkyboxViewUniforms>() % 16, 0);
    }

    #[test]
    fn multiview_permutation_constant_stays_distinct() {
        assert_ne!(ShaderPermutation(0), SHADER_PERM_MULTIVIEW_STEREO);
    }
}
