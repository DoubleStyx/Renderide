//! Offscreen passes for scene cameras targeting host [`crate::resources::GpuRenderTexture`] assets.

use std::sync::Arc;

use rayon::prelude::*;

use crate::assets::material::MaterialDictionary;
use crate::backend::OcclusionSystem;
use crate::gpu::GpuContext;
use crate::materials::{MaterialRouter, RasterPipelineKind};
use crate::pipelines::ShaderPermutation;
use crate::render_graph::{
    build_world_mesh_cull_proj_params, camera_state_enabled, collect_and_sort_world_mesh_draws,
    collect_and_sort_world_mesh_draws_with_parallelism, draw_filter_from_camera_entry,
    host_camera_frame_for_render_texture, CameraTransformDrawFilter, DrawCollectionContext,
    ExternalOffscreenTargets, FrameView, FrameViewTarget, GraphExecuteError, HostCameraFrame,
    OcclusionViewId, OutputDepthMode, WorldMeshCullInput, WorldMeshDrawCollectParallelism,
    WorldMeshDrawCollection,
};
use crate::scene::{RenderSpaceId, SceneCoordinator};

use super::RendererRuntime;
use winit::window::Window;

/// Resolved secondary camera target and host frame data (one entry per RT draw).
struct SecondaryRtPrepared {
    host_camera: HostCameraFrame,
    filter: CameraTransformDrawFilter,
    rt_id: i32,
    color_view: Arc<wgpu::TextureView>,
    depth_texture: Arc<wgpu::Texture>,
    depth_view: Arc<wgpu::TextureView>,
    viewport: (u32, u32),
    color_format: wgpu::TextureFormat,
}

/// Secondary offscreen [`FrameView`]s with prefetched draws, then the main swapchain view.
fn build_desktop_multi_view_frame_list<'a>(
    prepared: &'a [SecondaryRtPrepared],
    secondary_prefetched: Vec<WorldMeshDrawCollection>,
    hc: HostCameraFrame,
    main_collection: WorldMeshDrawCollection,
) -> Vec<FrameView<'a>> {
    let mut views: Vec<FrameView<'a>> = Vec::new();
    for (prep, collection) in prepared.iter().zip(secondary_prefetched.into_iter()) {
        let ext = ExternalOffscreenTargets {
            render_texture_asset_id: prep.rt_id,
            color_view: prep.color_view.as_ref(),
            depth_texture: prep.depth_texture.as_ref(),
            depth_view: prep.depth_view.as_ref(),
            extent_px: prep.viewport,
            color_format: prep.color_format,
        };
        views.push(FrameView {
            host_camera: prep.host_camera,
            target: FrameViewTarget::OffscreenRt(ext),
            draw_filter: Some(prep.filter.clone()),
            prefetched_world_mesh_draws: Some(collection),
        });
    }

    views.push(FrameView {
        host_camera: hc,
        target: FrameViewTarget::Swapchain,
        draw_filter: None,
        prefetched_world_mesh_draws: Some(main_collection),
    });

    views
}

impl RendererRuntime {
    /// Renders secondary cameras to host render textures before the main swapchain pass.
    ///
    /// Prefer [`Self::render_all_views`] for desktop; this VR path batches all secondary cameras
    /// into one [`RenderBackend::execute_multi_view_frame`](crate::backend::RenderBackend::execute_multi_view_frame)
    /// call, skipping redundant mesh deform and Hi-Z readback (already done earlier in the tick).
    pub fn render_secondary_cameras_to_render_textures(
        &mut self,
        gpu: &mut GpuContext,
        window: &Window,
    ) -> Result<(), GraphExecuteError> {
        self.backend
            .frame_resources
            .prepare_lights_from_scene(&self.scene);
        self.sync_debug_hud_diagnostics_from_settings();

        let prepared = self.collect_secondary_rt_prepared();
        if prepared.is_empty() {
            return Ok(());
        }

        let render_context = self.scene.active_main_render_context();
        let scene_ref: &SceneCoordinator = &self.scene;
        let property_store = self.backend.material_property_store();
        let mesh_pool = self.backend.mesh_pool();
        let fallback_router = MaterialRouter::new(RasterPipelineKind::DebugWorldNormals);
        let router_ref = self
            .backend
            .materials
            .material_registry()
            .map(|r| &r.router)
            .unwrap_or(&fallback_router);

        let occlusion_ref: &OcclusionSystem = &self.backend.occlusion;
        let inner_parallelism = if prepared.len() > 1 {
            WorldMeshDrawCollectParallelism::SerialInnerForNestedBatch
        } else {
            WorldMeshDrawCollectParallelism::Full
        };

        // Hi-Z snapshot reads must stay serial: [`OcclusionSystem`] is not `Sync` (wgpu readback state).
        let cull_snapshots: Vec<Option<SecondaryCullSnapshot>> = prepared
            .iter()
            .map(|prep| secondary_cull_snapshot(scene_ref, occlusion_ref, prep))
            .collect();

        let prefetched: Vec<crate::render_graph::WorldMeshDrawCollection> = prepared
            .par_iter()
            .zip(cull_snapshots.par_iter())
            .map(|(prep, snap)| {
                let dict = MaterialDictionary::new(property_store);
                let culling = snap.as_ref().map(|s| WorldMeshCullInput {
                    proj: s.proj,
                    host_camera: &prep.host_camera,
                    hi_z: s.hi_z.clone(),
                    hi_z_temporal: s.hi_z_temporal.clone(),
                });
                collect_and_sort_world_mesh_draws_with_parallelism(
                    &DrawCollectionContext {
                        scene: scene_ref,
                        mesh_pool,
                        material_dict: &dict,
                        material_router: router_ref,
                        shader_perm: ShaderPermutation(0),
                        render_context,
                        head_output_transform: prep.host_camera.head_output_transform,
                        culling: culling.as_ref(),
                        transform_filter: Some(&prep.filter),
                    },
                    inner_parallelism,
                )
            })
            .collect();

        let mut views: Vec<FrameView<'_>> = Vec::with_capacity(prepared.len());
        for (prep, collection) in prepared.iter().zip(prefetched.into_iter()) {
            let ext = ExternalOffscreenTargets {
                render_texture_asset_id: prep.rt_id,
                color_view: prep.color_view.as_ref(),
                depth_texture: prep.depth_texture.as_ref(),
                depth_view: prep.depth_view.as_ref(),
                extent_px: prep.viewport,
                color_format: prep.color_format,
            };
            views.push(FrameView {
                host_camera: prep.host_camera,
                target: FrameViewTarget::OffscreenRt(ext),
                draw_filter: Some(prep.filter.clone()),
                prefetched_world_mesh_draws: Some(collection),
            });
        }
        if !views.is_empty() {
            let requested_msaa = self
                .settings
                .read()
                .map(|s| s.rendering.msaa.as_count())
                .unwrap_or(1);
            gpu.set_swapchain_msaa_requested(requested_msaa);
            self.backend
                .execute_multi_view_frame(gpu, window, scene_ref, views, true)?;
        }
        Ok(())
    }

    /// Renders all views for this tick (secondary RTs + main swapchain) in one unified pass.
    pub fn render_all_views(
        &mut self,
        gpu: &mut GpuContext,
        window: &Window,
    ) -> Result<(), GraphExecuteError> {
        self.backend
            .frame_resources
            .prepare_lights_from_scene(&self.scene);
        self.sync_debug_hud_diagnostics_from_settings();

        let prepared = self.collect_secondary_rt_prepared();
        let render_context = self.scene.active_main_render_context();
        let scene_ref: &SceneCoordinator = &self.scene;
        let property_store = self.backend.material_property_store();
        let mesh_pool = self.backend.mesh_pool();
        let fallback_router = MaterialRouter::new(RasterPipelineKind::DebugWorldNormals);
        let router_ref = self
            .backend
            .materials
            .material_registry()
            .map(|r| &r.router)
            .unwrap_or(&fallback_router);

        let occlusion_ref: &OcclusionSystem = &self.backend.occlusion;
        let inner_parallelism = if prepared.len() > 1 {
            WorldMeshDrawCollectParallelism::SerialInnerForNestedBatch
        } else {
            WorldMeshDrawCollectParallelism::Full
        };

        let cull_snapshots: Vec<Option<SecondaryCullSnapshot>> = prepared
            .iter()
            .map(|prep| secondary_cull_snapshot(scene_ref, occlusion_ref, prep))
            .collect();

        let secondary_prefetched: Vec<crate::render_graph::WorldMeshDrawCollection> = prepared
            .par_iter()
            .zip(cull_snapshots.par_iter())
            .map(|(prep, snap)| {
                let dict = MaterialDictionary::new(property_store);
                let culling = snap.as_ref().map(|s| WorldMeshCullInput {
                    proj: s.proj,
                    host_camera: &prep.host_camera,
                    hi_z: s.hi_z.clone(),
                    hi_z_temporal: s.hi_z_temporal.clone(),
                });
                collect_and_sort_world_mesh_draws_with_parallelism(
                    &DrawCollectionContext {
                        scene: scene_ref,
                        mesh_pool,
                        material_dict: &dict,
                        material_router: router_ref,
                        shader_perm: ShaderPermutation(0),
                        render_context,
                        head_output_transform: prep.host_camera.head_output_transform,
                        culling: culling.as_ref(),
                        transform_filter: Some(&prep.filter),
                    },
                    inner_parallelism,
                )
            })
            .collect();

        let hc = self.host_camera;
        let dict = MaterialDictionary::new(property_store);
        let culling_main = if hc.suppress_occlusion_temporal {
            None
        } else {
            let cull_proj =
                build_world_mesh_cull_proj_params(scene_ref, gpu.surface_extent_px(), &hc);
            let main_snap = main_cull_snapshot(self);
            Some(WorldMeshCullInput {
                proj: cull_proj,
                host_camera: &hc,
                hi_z: main_snap.hi_z,
                hi_z_temporal: main_snap.hi_z_temporal,
            })
        };
        let culling_main_ref = culling_main.as_ref();
        let main_collection = collect_and_sort_world_mesh_draws(&DrawCollectionContext {
            scene: scene_ref,
            mesh_pool,
            material_dict: &dict,
            material_router: router_ref,
            shader_perm: ShaderPermutation(0),
            render_context,
            head_output_transform: hc.head_output_transform,
            culling: culling_main_ref,
            transform_filter: None,
        });

        let views = build_desktop_multi_view_frame_list(
            &prepared,
            secondary_prefetched,
            hc,
            main_collection,
        );

        let requested_msaa = self
            .settings
            .read()
            .map(|s| s.rendering.msaa.as_count())
            .unwrap_or(1);
        gpu.set_swapchain_msaa_requested(requested_msaa);
        self.backend
            .execute_multi_view_frame(gpu, window, scene_ref, views, true)
    }

    fn collect_secondary_rt_prepared(&mut self) -> Vec<SecondaryRtPrepared> {
        let mut tasks: Vec<(RenderSpaceId, f32, usize)> = Vec::new();
        for sid in self.scene.render_space_ids() {
            let Some(space) = self.scene.space(sid) else {
                continue;
            };
            if !space.is_active {
                continue;
            }
            for (idx, cam) in space.cameras.iter().enumerate() {
                if !camera_state_enabled(cam.state.flags) {
                    continue;
                }
                if cam.state.render_texture_asset_id < 0 {
                    continue;
                }
                tasks.push((sid, cam.state.depth, idx));
            }
        }
        tasks.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut prepared: Vec<SecondaryRtPrepared> = Vec::new();
        for (sid, _, cam_idx) in tasks {
            let Some(space) = self.scene.space(sid) else {
                continue;
            };
            let Some(entry) = space.cameras.get(cam_idx) else {
                continue;
            };
            if !camera_state_enabled(entry.state.flags) {
                continue;
            }
            let rt_id = entry.state.render_texture_asset_id;
            let (color_view, depth_texture, depth_view, viewport, color_format) = {
                let Some(rt) = self.backend.render_texture_pool().get(rt_id) else {
                    logger::trace!(
                        "secondary camera: render texture asset {rt_id} not resident; skipping"
                    );
                    continue;
                };
                let Some(dt) = rt.depth_texture.clone() else {
                    logger::warn!("secondary camera: render texture {rt_id} missing depth");
                    continue;
                };
                let Some(dv) = rt.depth_view.clone() else {
                    logger::warn!("secondary camera: render texture {rt_id} missing depth view");
                    continue;
                };
                (
                    rt.color_view.clone(),
                    dt,
                    dv,
                    (rt.width, rt.height),
                    rt.wgpu_color_format,
                )
            };
            let Some(world_m) = self.scene.world_matrix(sid, entry.transform_id as usize) else {
                continue;
            };
            let hc = host_camera_frame_for_render_texture(
                &self.host_camera,
                &entry.state,
                viewport,
                world_m,
                &self.scene,
            );
            let filter = draw_filter_from_camera_entry(entry);
            prepared.push(SecondaryRtPrepared {
                host_camera: hc,
                filter,
                rt_id,
                color_view,
                depth_texture,
                depth_view,
                viewport,
                color_format,
            });
        }
        prepared
    }
}

struct SecondaryCullSnapshot {
    proj: crate::render_graph::WorldMeshCullProjParams,
    hi_z: Option<crate::render_graph::HiZCullData>,
    hi_z_temporal: Option<crate::render_graph::HiZTemporalState>,
}

/// Builds frustum + Hi-Z cull inputs for one secondary RT.
///
/// Callers keep this **serial** per prepared camera: [`OcclusionSystem`] is not [`Sync`] (wgpu
/// readback state), so it cannot be shared across rayon worker threads.
fn secondary_cull_snapshot(
    scene: &SceneCoordinator,
    occlusion: &OcclusionSystem,
    prep: &SecondaryRtPrepared,
) -> Option<SecondaryCullSnapshot> {
    if prep.host_camera.suppress_occlusion_temporal {
        return None;
    }
    let proj = build_world_mesh_cull_proj_params(scene, prep.viewport, &prep.host_camera);
    let view_id = OcclusionViewId::OffscreenRenderTexture(prep.rt_id);
    let depth_mode = OutputDepthMode::DesktopSingle;
    Some(SecondaryCullSnapshot {
        proj,
        hi_z: occlusion.hi_z_cull_data(depth_mode, view_id),
        hi_z_temporal: occlusion.hi_z_temporal_snapshot(view_id),
    })
}

struct MainCullSnap {
    hi_z: Option<crate::render_graph::HiZCullData>,
    hi_z_temporal: Option<crate::render_graph::HiZTemporalState>,
}

fn main_cull_snapshot(runtime: &RendererRuntime) -> MainCullSnap {
    let depth_mode = OutputDepthMode::DesktopSingle;
    MainCullSnap {
        hi_z: runtime
            .backend
            .occlusion
            .hi_z_cull_data(depth_mode, OcclusionViewId::Main),
        hi_z_temporal: runtime
            .backend
            .occlusion
            .hi_z_temporal_snapshot(OcclusionViewId::Main),
    }
}
