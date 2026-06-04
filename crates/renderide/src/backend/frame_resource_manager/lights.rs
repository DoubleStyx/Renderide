//! Light preparation and per-view light access for [`FrameResourceManager`].

use rayon::prelude::*;

use crate::camera::ViewId;
use crate::cpu_parallelism::{
    LIGHT_WORK_PARALLEL_MIN_LIGHTS, admit_coarse_space_items, current_reference_worker_count,
    record_parallel_admission,
};
use crate::render_graph::GraphAssetResources;
use crate::scene::{
    RenderSpaceId, ResolvedLight, SceneCoordinator, light_contributes,
    light_has_negative_contribution,
};

use super::super::light_gpu::{
    GpuLight, MAX_LIGHTS, gpu_light_from_resolved_with_cookie,
    order_lights_for_clustered_shading_in_place,
};
use super::manager::FrameResourceManager;
use super::per_view_state::PreparedViewLights;
use super::view_desc::FrameLightViewDesc;

/// Per-view light packs assigned to one Rayon worker.
const LIGHT_VIEW_PREP_PARALLEL_CHUNK_VIEWS: usize = 1;
/// View count required before per-view light preparation fans out.
const LIGHT_VIEW_PREP_PARALLEL_MIN_VIEWS: usize = LIGHT_VIEW_PREP_PARALLEL_CHUNK_VIEWS * 2;
/// Candidate light rows required before per-view light preparation fans out.
const LIGHT_VIEW_PREP_PARALLEL_MIN_LIGHTS: usize = 32;

struct PreparedViewLightPacket {
    view_id: ViewId,
    render_context: crate::shared::RenderingContext,
    render_space_filter: Option<RenderSpaceId>,
    render_shadows: bool,
    resolved_len: usize,
    signed_scene_color_required: bool,
    resolved: Vec<ResolvedLight>,
}

impl FrameResourceManager {
    /// Packed GPU lights from the last [`Self::prepare_lights_from_scene`] call.
    pub fn frame_lights(&self) -> &[GpuLight] {
        &self.light_scratch
    }

    /// Packed GPU lights for `view_id`, falling back to the last default frame pack.
    pub fn frame_lights_for_view(&self, view_id: ViewId) -> &[GpuLight] {
        self.per_view_lights
            .get(view_id)
            .map_or(self.light_scratch.as_slice(), |lights| {
                lights.lights.as_slice()
            })
    }

    /// Returns true when the current packed light set needs signed scene-color storage.
    pub fn signed_scene_color_required(&self) -> bool {
        self.signed_scene_color_required
    }

    /// Light count for the specified view's frame uniforms and shaders.
    pub fn frame_light_count_for_view_u32(&self, view_id: ViewId) -> u32 {
        self.frame_lights_for_view(view_id).len().min(MAX_LIGHTS) as u32
    }

    /// Fills the default main-view light scratch buffer from active render spaces.
    ///
    /// This compatibility entry point is used by unit tests and callers that do not have explicit
    /// view planning information. Normal graph rendering should call
    /// [`Self::prepare_lights_for_views`] so secondary cameras get render-context-aware light
    /// packs.
    #[cfg(test)]
    pub fn prepare_lights_from_scene(&mut self, scene: &SceneCoordinator) {
        self.prepare_lights_for_views(
            scene,
            [FrameLightViewDesc {
                view_id: ViewId::Main,
                render_context: scene.active_main_render_context(),
                render_space_filter: None,
                head_output_transform: glam::Mat4::IDENTITY,
                render_shadows: true,
            }],
            None,
        );
    }

    /// Fills per-view light scratch buffers from [`SceneCoordinator`].
    ///
    /// Inactive spaces are skipped so lights from a previously focused world do not persist into
    /// the next frame's shading. Views with a render-space filter only receive lights from that
    /// space. Non-contributing lights are filtered via [`light_contributes`] before clustered
    /// ordering, and each view's transforms are resolved with the same render context and
    /// head-output transform used by draw collection.
    pub(crate) fn prepare_lights_for_views<I>(
        &mut self,
        scene: &SceneCoordinator,
        views: I,
        asset_resources: Option<&dyn GraphAssetResources>,
    ) where
        I: IntoIterator<Item = FrameLightViewDesc>,
    {
        profiling::scope!("render::prepare_lights_for_views");
        let views = views.into_iter().collect::<Vec<_>>();
        self.light_scratch.clear();
        self.signed_scene_color_required = false;
        let packets = if should_parallelize_light_view_prep(scene, &views) {
            profiling::scope!("render::prepare_lights_for_views::parallel");
            views
                .par_iter()
                .with_min_len(LIGHT_VIEW_PREP_PARALLEL_CHUNK_VIEWS)
                .map(|&desc| prepare_lights_for_view_packet(scene, desc))
                .collect::<Vec<_>>()
        } else {
            profiling::scope!("render::prepare_lights_for_views::serial");
            views
                .iter()
                .map(|&desc| prepare_lights_for_view_packet(scene, desc))
                .collect::<Vec<_>>()
        };

        if let Some(fgpu) = self.frame_gpu() {
            fgpu.begin_light_cookie_frame();
        }
        let mut wrote_fallback = false;
        for packet in packets {
            self.commit_prepared_light_packet(packet, &mut wrote_fallback, asset_resources);
        }
        if self.signed_scene_color_required && !self.signed_scene_color_required_logged {
            logger::info!(
                "negative direct lights active: signed scene-color HDR will be used while negative lights are packed"
            );
            self.signed_scene_color_required_logged = true;
        }
    }

    fn commit_prepared_light_packet(
        &mut self,
        packet: PreparedViewLightPacket,
        wrote_fallback: &mut bool,
        asset_resources: Option<&dyn GraphAssetResources>,
    ) {
        if packet.resolved_len > MAX_LIGHTS && !self.lights_overflow_warned {
            logger::warn!(
                "scene contains {} contributing lights but the engine only uploads \
                 the first {MAX_LIGHTS} (MAX_LIGHTS); the remainder will be ignored for shading. \
                 This warning is only logged once per renderer instance.",
                packet.resolved_len
            );
            self.lights_overflow_warned = true;
        }

        self.signed_scene_color_required |= packet.signed_scene_color_required;
        let mut packed_lights = Vec::with_capacity(packet.resolved.len());
        for light in &packet.resolved {
            let cookie = self.frame_gpu().map_or(
                crate::backend::light_gpu::LightCookieBinding::NONE,
                |fgpu| fgpu.assign_light_cookie(light, asset_resources),
            );
            let mut gpu_light = gpu_light_from_resolved_with_cookie(light, cookie);
            if !packet.render_shadows {
                disable_gpu_light_shadows(&mut gpu_light);
            }
            packed_lights.push(gpu_light);
        }
        if !*wrote_fallback {
            self.light_scratch
                .extend_from_slice(packed_lights.as_slice());
            *wrote_fallback = true;
        }

        let light_count = packed_lights.len();
        let signed_scene_color_required = packet.signed_scene_color_required;
        let entry = self
            .per_view_lights
            .get_or_insert_with(packet.view_id, PreparedViewLights::default);
        entry.lights.clear();
        entry.lights.extend_from_slice(packed_lights.as_slice());
        entry.signed_scene_color_required = signed_scene_color_required;
        logger::trace!(
            "prepared lights for view {:?}: lights={} render_context={:?} render_space_filter={:?} render_shadows={}",
            packet.view_id,
            light_count,
            packet.render_context,
            packet.render_space_filter,
            packet.render_shadows
        );
    }
}

fn disable_gpu_light_shadows(light: &mut GpuLight) {
    light.shadow_type = 0;
    light.shadow_strength = 0.0;
    light.shadow_near_plane = 0.0;
    light.shadow_bias = 0.0;
    light.shadow_normal_bias = 0.0;
}

fn should_parallelize_light_view_prep(
    scene: &SceneCoordinator,
    views: &[FrameLightViewDesc],
) -> bool {
    views.len() >= LIGHT_VIEW_PREP_PARALLEL_MIN_VIEWS
        && views
            .iter()
            .map(|view| {
                scene.candidate_light_count_for_render_space_filter(view.render_space_filter)
            })
            .sum::<usize>()
            >= LIGHT_VIEW_PREP_PARALLEL_MIN_LIGHTS
}

fn prepare_lights_for_view_packet(
    scene: &SceneCoordinator,
    desc: FrameLightViewDesc,
) -> PreparedViewLightPacket {
    profiling::scope!("render::prepare_lights_for_view");
    let mut light_space_ids = Vec::new();
    collect_light_space_ids(scene, desc.render_space_filter, &mut light_space_ids);
    let mut resolved = Vec::new();
    resolve_lights_for_space_ids(scene, desc, &light_space_ids, &mut resolved);
    {
        profiling::scope!("render::prepare_lights::filter_contributors");
        resolved.retain(light_contributes);
    }
    order_lights_for_clustered_shading_in_place(&mut resolved);
    let resolved_len = resolved.len();
    let kept = resolved_len.min(MAX_LIGHTS);
    let signed_scene_color_required = resolved
        .iter()
        .take(kept)
        .any(light_has_negative_contribution);
    resolved.truncate(kept);
    PreparedViewLightPacket {
        view_id: desc.view_id,
        render_context: desc.render_context,
        render_space_filter: desc.render_space_filter,
        render_shadows: desc.render_shadows,
        resolved_len,
        signed_scene_color_required,
        resolved,
    }
}

fn collect_light_space_ids(
    scene: &SceneCoordinator,
    render_space_filter: Option<RenderSpaceId>,
    out: &mut Vec<RenderSpaceId>,
) {
    profiling::scope!("render::prepare_lights::collect_active_spaces");
    out.clear();
    if let Some(id) = render_space_filter {
        if scene.space(id).is_some_and(|space| space.is_active()) {
            out.push(id);
        }
        return;
    }
    out.extend(
        scene
            .render_space_ids()
            .filter(|id| scene.space(*id).is_some_and(|space| space.is_active())),
    );
}

fn resolve_lights_for_space_ids(
    scene: &SceneCoordinator,
    desc: FrameLightViewDesc,
    light_space_ids: &[RenderSpaceId],
    out: &mut Vec<ResolvedLight>,
) {
    if light_space_ids.is_empty() {
        return;
    }
    profiling::scope!("render::prepare_lights::resolve_spaces");
    let candidate_lights = light_space_ids
        .iter()
        .map(|&id| scene.candidate_light_count_for_render_space_filter(Some(id)))
        .sum::<usize>();
    let admission = if candidate_lights >= LIGHT_WORK_PARALLEL_MIN_LIGHTS {
        admit_coarse_space_items(light_space_ids.len(), current_reference_worker_count())
    } else {
        crate::cpu_parallelism::ParallelAdmission::Serial
    };
    record_parallel_admission(
        "light_space_resolve",
        candidate_lights,
        light_space_ids.len(),
        admission,
    );
    if let Some(chunk_size) = admission.chunk_size() {
        let packets = light_space_ids
            .par_iter()
            .with_min_len(chunk_size)
            .map(|&id| {
                let mut resolved = Vec::new();
                scene.resolve_lights_for_render_context_into(
                    id,
                    desc.render_context,
                    desc.head_output_transform,
                    &mut resolved,
                );
                resolved
            })
            .collect::<Vec<_>>();
        for mut packet in packets {
            out.append(&mut packet);
        }
    } else {
        for &id in light_space_ids {
            scene.resolve_lights_for_render_context_into(
                id,
                desc.render_context,
                desc.head_output_transform,
                out,
            );
        }
    }
}
