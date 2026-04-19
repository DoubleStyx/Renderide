//! Offscreen passes for scene cameras targeting host [`crate::resources::GpuRenderTexture`] assets.

use std::sync::Arc;

use rayon::prelude::*;

use crate::assets::material::MaterialDictionary;
use crate::backend::OcclusionSystem;
use crate::gpu::GpuContext;
use crate::materials::{MaterialPipelinePropertyIds, MaterialRouter, RasterPipelineKind};
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

/// Cheap-clone snapshot of [`crate::gpu::PrimaryOffscreenTargets`] used by the headless render path
/// to satisfy the borrow checker: clones are cheap (`wgpu::Texture` and `wgpu::TextureView` are
/// internally `Arc`-backed) and let the substitution borrow from a stack-local instead of a
/// long-lived `&mut gpu`. Without this split, holding `&gpu.primary_offscreen` while passing
/// `&mut gpu` to the next call is a borrow error.
struct HeadlessOffscreenSnapshot {
    color_view: wgpu::TextureView,
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    extent_px: (u32, u32),
    color_format: wgpu::TextureFormat,
}

impl HeadlessOffscreenSnapshot {
    /// Lazily allocates the headless primary targets if needed and snapshots cheap clones of
    /// their handles. Returns [`None`] when `gpu` is windowed.
    fn from_gpu(gpu: &mut GpuContext) -> Option<Self> {
        let targets = gpu.primary_offscreen_targets()?;
        Some(Self {
            color_view: targets.color_view.clone(),
            depth_texture: targets.depth_texture.clone(),
            depth_view: targets.depth_view.clone(),
            extent_px: targets.extent_px,
            color_format: targets.color_format,
        })
    }

    /// Replaces every [`FrameViewTarget::Swapchain`] in `views` with an
    /// [`FrameViewTarget::OffscreenRt`] backed by this snapshot's owned handles. The borrowed
    /// references are valid for as long as `&self` outlives `views`, which is enforced by the
    /// `'a` lifetime.
    fn substitute_swapchain_views<'a>(&'a self, views: &mut [FrameView<'a>]) {
        for view in views.iter_mut() {
            if matches!(view.target, FrameViewTarget::Swapchain) {
                view.target = FrameViewTarget::OffscreenRt(ExternalOffscreenTargets {
                    render_texture_asset_id: -1,
                    color_view: &self.color_view,
                    depth_texture: &self.depth_texture,
                    depth_view: &self.depth_view,
                    extent_px: self.extent_px,
                    color_format: self.color_format,
                });
            }
        }
    }
}

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
    for (prep, collection) in prepared.iter().zip(secondary_prefetched) {
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
    ) -> Result<(), GraphExecuteError> {
        self.backend
            .frame_resources
            .prepare_lights_from_scene(&self.scene);
        self.sync_debug_hud_diagnostics_from_settings();

        let prepared = self.collect_secondary_rt_prepared();
        if prepared.is_empty() {
            return Ok(());
        }

        let requested_msaa = self
            .settings
            .read()
            .map(|s| s.rendering.msaa.as_count())
            .unwrap_or(1);
        let prev_msaa = gpu.swapchain_msaa_effective();
        gpu.set_swapchain_msaa_requested(requested_msaa);
        self.transient_evict_stale_msaa_tiers_if_changed(prev_msaa, gpu.swapchain_msaa_effective());

        let render_context = self.scene.active_main_render_context();
        let scene_ref: &SceneCoordinator = &self.scene;
        let property_store = self.backend.material_property_store();
        let pipeline_property_ids =
            MaterialPipelinePropertyIds::new(self.backend.property_id_registry());
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
                        pipeline_property_ids: &pipeline_property_ids,
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
        for (prep, collection) in prepared.iter().zip(prefetched) {
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
            self.backend
                .execute_multi_view_frame(gpu, scene_ref, &mut views, true)?;
        }
        Ok(())
    }

    /// Renders all views for this tick (secondary RTs + main camera) in one unified pass.
    ///
    /// In headless mode (`gpu.is_headless()`), the main `Swapchain` view is transparently
    /// substituted for an `OffscreenRt` view backed by [`GpuContext::primary_offscreen_targets`]
    /// before submission. The render graph stack itself stays oblivious to mode.
    pub fn render_all_views(&mut self, gpu: &mut GpuContext) -> Result<(), GraphExecuteError> {
        self.backend
            .frame_resources
            .prepare_lights_from_scene(&self.scene);
        self.sync_debug_hud_diagnostics_from_settings();

        let prepared = self.collect_secondary_rt_prepared();

        let requested_msaa = self
            .settings
            .read()
            .map(|s| s.rendering.msaa.as_count())
            .unwrap_or(1);
        let prev_msaa = gpu.swapchain_msaa_effective();
        gpu.set_swapchain_msaa_requested(requested_msaa);
        self.transient_evict_stale_msaa_tiers_if_changed(prev_msaa, gpu.swapchain_msaa_effective());

        let render_context = self.scene.active_main_render_context();
        let scene_ref: &SceneCoordinator = &self.scene;
        let property_store = self.backend.material_property_store();
        let pipeline_property_ids =
            MaterialPipelinePropertyIds::new(self.backend.property_id_registry());
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
                        pipeline_property_ids: &pipeline_property_ids,
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
            pipeline_property_ids: &pipeline_property_ids,
            shader_perm: ShaderPermutation(0),
            render_context,
            head_output_transform: hc.head_output_transform,
            culling: culling_main_ref,
            transform_filter: None,
        });

        // Headless substitution: snapshot persistent offscreen handles BEFORE building views so
        // we can borrow from a local instead of a long-lived `&mut gpu` (which would conflict
        // with the `&mut gpu` we hand to `execute_multi_view_frame`).
        let headless_snapshot = if gpu.is_headless() {
            HeadlessOffscreenSnapshot::from_gpu(gpu)
        } else {
            None
        };

        let mut views = build_desktop_multi_view_frame_list(
            &prepared,
            secondary_prefetched,
            hc,
            main_collection,
        );

        if let Some(snapshot) = headless_snapshot.as_ref() {
            snapshot.substitute_swapchain_views(&mut views);
        }

        self.backend
            .execute_multi_view_frame(gpu, scene_ref, &mut views, true)
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
            let mut hc = host_camera_frame_for_render_texture(
                &self.host_camera,
                &entry.state,
                viewport,
                world_m,
                &self.scene,
            );
            let filter = draw_filter_from_camera_entry(entry);
            // Selective secondary cameras (dashboards, in-world UI panels, mirrors on specific
            // subtrees) render tens of draws, not thousands. Hi-Z snapshots + occlusion temporal
            // cost a per-camera readback path with negligible payoff at that scale — skip them.
            if !entry.selective_transform_ids.is_empty() {
                hc.suppress_occlusion_temporal = true;
            }
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
