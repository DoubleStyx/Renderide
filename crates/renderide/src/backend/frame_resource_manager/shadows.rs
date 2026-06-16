//! Realtime shadow planning stored on [`FrameResourceManager`].

use std::sync::{Arc, Once};

use glam::{Mat4, Vec3};
use hashbrown::HashMap;

use crate::backend::HostShadowQuality;
use crate::camera::ViewId;
use crate::gpu::{
    GpuLimits, GpuShadowView, MAX_SHADOW_VIEWS, SHADOW_VIEW_KIND_DIRECTIONAL,
    SHADOW_VIEW_KIND_POINT, SHADOW_VIEW_KIND_SPOT,
};
use crate::materials::shadow_caster_policy_for_pipeline;
use crate::mesh_deform::SkinCacheKey;
use crate::render_phase::RenderPhaseSet;
use crate::shared::{LightType, ShadowCastMode};
use crate::world_mesh::culling::frustum::world_aabb_visible_in_homogeneous_clip;
use crate::world_mesh::{
    DrawGroup, InstancePlan, WorldMeshDrawItem, WorldMeshDrawPlan, WorldMeshPhase,
};

use super::super::shadow_atlas_format::select_shadow_atlas_format;
use super::manager::FrameResourceManager;

mod point_faces;

const POINT_FACE_COUNT: u32 = point_faces::POINT_FACE_COUNT;
const SHADOW_TYPE_NONE: u32 = 0;
static SHADOW_ATLAS_UNSUPPORTED_WARNING: Once = Once::new();

/// Shared shadow-caster draw packet for one source render view.
#[derive(Clone, Debug)]
pub(crate) struct ShadowCasterSet {
    /// Shadow-casting world-mesh draws shared by all shadow views for the source view.
    pub(crate) draws: Arc<[WorldMeshDrawItem]>,
    /// Instance grouping shared by every shadow map that renders [`Self::draws`].
    pub(crate) instance_plan: InstancePlan,
    /// First per-draw slab row reserved for this caster set.
    pub(crate) slab_slot_offset: usize,
}

/// Draw packet for one shadow atlas layer.
#[derive(Clone, Debug)]
pub(crate) struct ShadowRenderView {
    /// Atlas array layer rendered by this shadow view.
    pub(crate) layer: u32,
    /// Kind of light shadow view, matching `SHADOW_VIEW_KIND_*`.
    pub(crate) kind: u32,
    /// Full atlas-layer resolution in pixels.
    pub(crate) resolution: u32,
    /// Light-space projection-view matrix used for shadow caster vertices.
    pub(crate) view_proj: Mat4,
    /// World-space light position used by radial shadow casters.
    pub(crate) light_position: Vec3,
    /// Light range used by radial shadow casters.
    pub(crate) light_range: f32,
    /// Host-authored shadow bias used by radial shadow casters.
    pub(crate) shadow_bias: f32,
    /// Index of the shared caster set rendered by this shadow view.
    pub(crate) caster_set_index: usize,
    /// Shadow-caster groups that conservatively intersect this shadow view.
    visible_groups: RenderPhaseSet<WorldMeshPhase, DrawGroup>,
}

impl ShadowRenderView {
    /// Returns visible caster groups queued for `phase`.
    pub(crate) fn groups(&self, phase: WorldMeshPhase) -> &[DrawGroup] {
        self.visible_groups.phase(phase).items()
    }

    /// Creates a minimal shadow render view for unit tests outside this module.
    #[cfg(test)]
    pub(crate) fn for_tests(
        kind: u32,
        view_proj: Mat4,
        light_position: Vec3,
        light_range: f32,
        shadow_bias: f32,
    ) -> Self {
        Self {
            layer: 0,
            kind,
            resolution: 512,
            view_proj,
            light_position,
            light_range,
            shadow_bias,
            caster_set_index: 0,
            visible_groups: RenderPhaseSet::new(),
        }
    }
}

/// Read-only frame shadow plan consumed by the graph shadow pass.
#[derive(Clone, Debug, Default)]
pub(crate) struct ShadowFramePlan {
    /// Shared caster draw sets rendered by this frame's shadow views.
    pub(crate) caster_sets: Vec<ShadowCasterSet>,
    /// Shadow map views to render this frame.
    pub(crate) render_views: Vec<ShadowRenderView>,
    /// GPU metadata uploaded to the frame-global shadow-view storage buffer.
    pub(crate) metadata: Vec<GpuShadowView>,
    /// Requested atlas edge resolution.
    pub(crate) requested_resolution: u32,
    /// Requested atlas array-layer count.
    pub(crate) requested_layers: u32,
    /// Requested shadow per-draw slab rows across all atlas layers.
    pub(crate) requested_draw_slots: usize,
}

struct ShadowPlanningView<'a> {
    view_id: ViewId,
    draw_plan: &'a WorldMeshDrawPlan,
}

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
struct ShadowCasterGroupKey {
    mesh_asset_id: i32,
    first_index: u32,
    index_count: u32,
    front_face: crate::materials::RasterFrontFace,
    primitive_topology: crate::materials::RasterPrimitiveTopology,
    shadow_cast_mode: u8,
    cull_mode: Option<wgpu::Face>,
}

struct PendingShadowCasterGroup {
    representative_draw_idx: usize,
    members: Vec<usize>,
}

impl FrameResourceManager {
    /// Clears shadow assignments and stores an empty shadow frame plan.
    pub(crate) fn clear_shadow_frame(&mut self) {
        for lights in self.per_view_lights.values_mut() {
            for light in &mut lights.lights {
                clear_light_shadow_assignment(light);
            }
        }
        self.shadow_frame = ShadowFramePlan::default();
    }

    /// Plans shadow maps from packed per-view lights and sorted view draw plans.
    pub(crate) fn prepare_shadow_frame_for_views<'a, I>(
        &mut self,
        quality: HostShadowQuality,
        views: I,
    ) where
        I: IntoIterator<Item = (ViewId, &'a WorldMeshDrawPlan)>,
    {
        profiling::scope!("render::prepare_shadow_frame");
        self.clear_shadow_frame();
        if !shadow_atlas_rendering_supported(self.limits.as_deref()) {
            return;
        }
        let max_shadow_views = shadow_view_capacity(self.limits.as_deref());
        let mut plan = ShadowFramePlan {
            requested_resolution: 1,
            requested_layers: 1,
            ..ShadowFramePlan::default()
        };
        let planning_views = views
            .into_iter()
            .map(|(view_id, draw_plan)| ShadowPlanningView { view_id, draw_plan })
            .collect::<Vec<_>>();
        for view in planning_views {
            append_shadow_views_for_view(self, quality, view, &mut plan, max_shadow_views);
            if plan.metadata.len() >= max_shadow_views {
                break;
            }
        }
        plan.requested_layers = plan.render_views.len().max(1).min(u32::MAX as usize) as u32;
        refresh_shadow_metadata_atlas_rects(&mut plan);
        self.shadow_frame = plan;
    }

    /// Returns the current shadow frame plan.
    pub(crate) fn shadow_frame_plan(&self) -> &ShadowFramePlan {
        &self.shadow_frame
    }

    /// Shadow metadata rows uploaded before graph recording.
    pub(crate) fn frame_shadow_views(&self) -> &[GpuShadowView] {
        &self.shadow_frame.metadata
    }

    /// Atlas capacity requested by the current shadow frame.
    pub(crate) fn shadow_resource_request(&self) -> Option<(u32, u32)> {
        if self.shadow_frame.render_views.is_empty() {
            return None;
        }
        Some((
            self.shadow_frame.requested_resolution.max(1),
            self.shadow_frame.requested_layers.max(1),
        ))
    }

    /// Updates shadow metadata so it matches the atlas texture selected by frame-GPU sync.
    pub(crate) fn apply_shadow_atlas_resolution(&mut self, atlas_resolution: u32) {
        let plan = &mut self.shadow_frame;
        if plan.render_views.is_empty() {
            return;
        }
        let atlas_resolution = atlas_resolution.max(1);
        plan.requested_resolution = atlas_resolution;
        for (metadata, view) in plan.metadata.iter_mut().zip(plan.render_views.iter_mut()) {
            let old_resolution = view.resolution.max(1);
            let new_resolution = old_resolution.min(atlas_resolution).max(1);
            if new_resolution != old_resolution {
                view.resolution = new_resolution;
                if metadata.light_params[2].is_finite() {
                    metadata.light_params[2] *= old_resolution as f32 / new_resolution as f32;
                }
            }
            metadata.params[1] = 1.0 / new_resolution as f32;
        }
        refresh_shadow_metadata_atlas_rects(plan);
    }

    /// Per-draw slab slots required by all shadow map views.
    pub(crate) fn shadow_max_draw_slots(&self) -> usize {
        self.shadow_frame.requested_draw_slots
    }

    /// Deform keys required by this frame's shadow caster draws.
    pub(crate) fn shadow_mesh_deform_keys(&self) -> hashbrown::HashSet<SkinCacheKey> {
        let mut keys = hashbrown::HashSet::new();
        for caster_set in &self.shadow_frame.caster_sets {
            for item in caster_set.draws.iter() {
                if item.world_space_deformed || item.blendshape_deformed {
                    keys.insert(SkinCacheKey::from_draw_parts(
                        item.space_id,
                        item.skinned,
                        item.instance_id,
                    ));
                }
            }
        }
        keys
    }
}

fn shadow_atlas_rendering_supported(limits: Option<&GpuLimits>) -> bool {
    let Some(limits) = limits else {
        return true;
    };
    if select_shadow_atlas_format(limits).is_some() {
        return true;
    }
    SHADOW_ATLAS_UNSUPPORTED_WARNING.call_once(|| {
        logger::warn!(
            "sampled renderable depth shadow formats are unavailable; realtime shadow maps are disabled"
        );
    });
    false
}

fn append_shadow_views_for_view(
    manager: &mut FrameResourceManager,
    quality: HostShadowQuality,
    view: ShadowPlanningView<'_>,
    plan: &mut ShadowFramePlan,
    max_shadow_views: usize,
) {
    let Some(collection) = view.draw_plan.as_prefetched() else {
        return;
    };
    let shadow_draws = collection
        .items
        .iter()
        .filter(|item| {
            item.shadow_cast_mode != ShadowCastMode::Off
                && shadow_caster_policy_for_pipeline(&item.batch_key.pipeline).casts()
        })
        .cloned()
        .collect::<Arc<[WorldMeshDrawItem]>>();
    if shadow_draws.is_empty() {
        return;
    }
    let Some(lights) = manager.per_view_lights.get_mut(view.view_id) else {
        return;
    };
    let caster_set_index = plan.caster_sets.len();
    let supports_base_instance = manager
        .limits
        .as_deref()
        .is_none_or(|limits| limits.supports_base_instance);
    let instance_plan = build_shadow_caster_plan(&shadow_draws, supports_base_instance);
    let slab_slot_offset = plan.requested_draw_slots;
    plan.requested_draw_slots = plan.requested_draw_slots.saturating_add(shadow_draws.len());
    plan.caster_sets.push(ShadowCasterSet {
        draws: shadow_draws,
        instance_plan,
        slab_slot_offset,
    });

    let mut local_shadowed = 0u32;
    for light in &mut lights.lights {
        if light.shadow_type == SHADOW_TYPE_NONE || light.shadow_strength <= 0.0 {
            clear_light_shadow_assignment(light);
            continue;
        }
        let requested_count = shadow_view_count_for_light(light.light_type, quality);
        if requested_count == 0 {
            clear_light_shadow_assignment(light);
            continue;
        }
        if light.light_type != light_type_u32(LightType::Directional) {
            if local_shadowed >= quality.per_pixel_lights {
                clear_light_shadow_assignment(light);
                continue;
            }
            local_shadowed = local_shadowed.saturating_add(1);
        }
        let remaining = max_shadow_views.saturating_sub(plan.metadata.len());
        if remaining == 0 {
            clear_light_shadow_assignment(light);
            continue;
        }
        let view_count = requested_count.min(remaining as u32);
        let start = plan.metadata.len() as u32;
        let kind = shadow_kind_for_light(light.light_type);
        let light_position = Vec3::from_array(light.position);
        let light_range = light.range.max(0.001);
        let shadow_bias = light.shadow_bias.max(0.0);
        let resolution = shadow_resolution_for_light(light, quality, manager.limits.as_deref());
        plan.requested_resolution = plan.requested_resolution.max(resolution);
        for view_offset in 0..view_count {
            let layer = plan.render_views.len().min(u32::MAX as usize) as u32;
            let view_proj = shadow_projection_for_light(light, view_offset, view_count, quality);
            let visible_groups =
                visible_shadow_groups_for_view(view_proj, &plan.caster_sets[caster_set_index]);
            plan.metadata.push(gpu_shadow_view_for_light(
                light,
                view_proj,
                layer,
                resolution,
                view_offset,
                view_count,
                quality,
            ));
            plan.render_views.push(ShadowRenderView {
                layer,
                kind,
                resolution,
                view_proj,
                light_position,
                light_range,
                shadow_bias,
                caster_set_index,
                visible_groups,
            });
        }
        light.shadow_view_start = start;
        light.shadow_view_count = view_count;
        light.shadow_flags = 0;
    }
}

fn build_shadow_caster_plan(
    draws: &[WorldMeshDrawItem],
    supports_base_instance: bool,
) -> InstancePlan {
    profiling::scope!("render::prepare_shadow_frame::build_caster_plan");
    let mut plan = InstancePlan::new();
    if draws.is_empty() {
        return plan;
    }

    let mut group_index: HashMap<ShadowCasterGroupKey, usize> = HashMap::new();
    let mut pending_groups: Vec<PendingShadowCasterGroup> = Vec::new();
    for (draw_idx, item) in draws.iter().enumerate() {
        if shadow_draw_requires_singleton(item, supports_base_instance) {
            pending_groups.push(PendingShadowCasterGroup {
                representative_draw_idx: draw_idx,
                members: vec![draw_idx],
            });
            continue;
        }

        let key = shadow_caster_group_key(item);
        if let Some(&group_idx) = group_index.get(&key) {
            pending_groups[group_idx].members.push(draw_idx);
        } else {
            let group_idx = pending_groups.len();
            group_index.insert(key, group_idx);
            pending_groups.push(PendingShadowCasterGroup {
                representative_draw_idx: draw_idx,
                members: vec![draw_idx],
            });
        }
    }

    for group in pending_groups {
        let draw_group = append_shadow_draw_group(
            &mut plan.slab_layout,
            group.representative_draw_idx,
            &group.members,
        );
        plan.phase_mut(WorldMeshPhase::ForwardOpaque)
            .push(draw_group);
    }
    plan
}

fn shadow_draw_requires_singleton(item: &WorldMeshDrawItem, supports_base_instance: bool) -> bool {
    !supports_base_instance
        || item.skinned
        || item.world_space_deformed
        || item.blendshape_deformed
        || item.material_stack_order.is_some()
}

fn shadow_caster_group_key(item: &WorldMeshDrawItem) -> ShadowCasterGroupKey {
    ShadowCasterGroupKey {
        mesh_asset_id: item.mesh_asset_id,
        first_index: item.first_index,
        index_count: item.index_count,
        front_face: item.batch_key.front_face,
        primitive_topology: item.batch_key.primitive_topology,
        shadow_cast_mode: item.shadow_cast_mode as u8,
        cull_mode: if item.shadow_cast_mode == ShadowCastMode::DoubleSided {
            None
        } else {
            item.batch_key
                .render_state
                .resolved_cull_mode(Some(wgpu::Face::Back))
        },
    }
}

fn append_shadow_draw_group(
    slab_layout: &mut Vec<usize>,
    representative_draw_idx: usize,
    members: &[usize],
) -> DrawGroup {
    let first_instance = slab_layout.len() as u32;
    slab_layout.extend_from_slice(members);
    let count = members.len() as u32;
    DrawGroup {
        representative_draw_idx,
        instance_range: first_instance..first_instance + count,
        material_packet_idx: 0,
    }
}

fn visible_shadow_groups_for_view(
    view_proj: Mat4,
    caster_set: &ShadowCasterSet,
) -> RenderPhaseSet<WorldMeshPhase, DrawGroup> {
    profiling::scope!("render::prepare_shadow_frame::visible_groups");
    let mut visible = RenderPhaseSet::new();
    for phase in WorldMeshPhase::PRIMARY_FORWARD {
        for group in caster_set.instance_plan.phase(phase) {
            if shadow_group_visible_to_view(view_proj, caster_set, group) {
                visible.phase_mut(phase).push(group.clone());
            }
        }
    }
    visible
}

fn shadow_group_visible_to_view(
    view_proj: Mat4,
    caster_set: &ShadowCasterSet,
    group: &DrawGroup,
) -> bool {
    let start = group.instance_range.start as usize;
    let end = group.instance_range.end as usize;
    let Some(members) = caster_set.instance_plan.slab_layout.get(start..end) else {
        return true;
    };
    members.iter().any(|&draw_idx| {
        caster_set
            .draws
            .get(draw_idx)
            .is_none_or(|draw| shadow_draw_visible_to_view(view_proj, draw))
    })
}

fn shadow_draw_visible_to_view(view_proj: Mat4, draw: &WorldMeshDrawItem) -> bool {
    let Some((min, max)) = draw.world_aabb else {
        return true;
    };
    if !shadow_aabb_safe_for_cull(min, max) {
        return true;
    }
    world_aabb_visible_in_homogeneous_clip(view_proj, min, max)
}

fn shadow_aabb_safe_for_cull(min: Vec3, max: Vec3) -> bool {
    min.is_finite() && max.is_finite() && (max - min).cmpge(Vec3::ZERO).all()
}

fn shadow_tile_resolution(limits: Option<&GpuLimits>, requested: u32) -> u32 {
    let requested = requested.max(1);
    limits.map_or(requested, |limits| {
        requested.min(limits.max_texture_dimension_2d().max(1))
    })
}

fn shadow_resolution_for_light(
    light: &crate::gpu::GpuLight,
    quality: HostShadowQuality,
    limits: Option<&GpuLimits>,
) -> u32 {
    let requested = if light.shadow_map_resolution > 0 {
        light.shadow_map_resolution
    } else {
        quality_shadow_resolution_for_light(light.light_type, quality)
    };
    shadow_tile_resolution(limits, requested)
}

fn quality_shadow_resolution_for_light(light_type: u32, quality: HostShadowQuality) -> u32 {
    match light_type {
        x if x == light_type_u32(LightType::Directional) => {
            quality.tile_resolution_for_light_type(LightType::Directional)
        }
        x if x == light_type_u32(LightType::Spot) => {
            quality.tile_resolution_for_light_type(LightType::Spot)
        }
        x if x == light_type_u32(LightType::Point) => {
            quality.tile_resolution_for_light_type(LightType::Point)
        }
        _ => 1,
    }
}

fn shadow_view_capacity(limits: Option<&GpuLimits>) -> usize {
    let max_layers = limits.map_or(MAX_SHADOW_VIEWS, |limits| {
        limits.max_texture_array_layers().max(1) as usize
    });
    MAX_SHADOW_VIEWS.min(max_layers).max(1)
}

fn clear_light_shadow_assignment(light: &mut crate::gpu::GpuLight) {
    light.shadow_view_start = 0;
    light.shadow_view_count = 0;
    light.shadow_flags = 0;
}

fn shadow_view_count_for_light(light_type: u32, quality: HostShadowQuality) -> u32 {
    match light_type {
        x if x == light_type_u32(LightType::Directional) => quality.cascade_count.max(1),
        x if x == light_type_u32(LightType::Spot) => 1,
        x if x == light_type_u32(LightType::Point) => POINT_FACE_COUNT,
        _ => 0,
    }
}

fn gpu_shadow_view_for_light(
    light: &crate::gpu::GpuLight,
    view_proj: Mat4,
    layer: u32,
    resolution: u32,
    view_offset: u32,
    view_count: u32,
    quality: HostShadowQuality,
) -> GpuShadowView {
    let kind = shadow_kind_for_light(light.light_type);
    GpuShadowView {
        world_to_shadow: view_proj.to_cols_array_2d(),
        atlas_rect: [0.0, 0.0, 1.0, 1.0],
        params: [layer as f32, 1.0 / resolution.max(1) as f32, 0.0, 1.0],
        light_params: [
            kind as f32,
            view_offset as f32,
            shadow_normal_bias_world_units(
                light,
                kind,
                resolution,
                view_offset,
                view_count,
                quality,
            ),
            light.shadow_bias.max(0.0),
        ],
    }
}

fn refresh_shadow_metadata_atlas_rects(plan: &mut ShadowFramePlan) {
    let atlas_resolution = plan.requested_resolution.max(1);
    for (metadata, view) in plan.metadata.iter_mut().zip(plan.render_views.iter()) {
        metadata.atlas_rect = shadow_atlas_rect(view.resolution, atlas_resolution);
    }
}

fn shadow_atlas_rect(resolution: u32, atlas_resolution: u32) -> [f32; 4] {
    let scale = resolution.min(atlas_resolution).max(1) as f32 / atlas_resolution.max(1) as f32;
    [0.0, 0.0, scale, scale]
}

fn shadow_normal_bias_world_units(
    light: &crate::gpu::GpuLight,
    kind: u32,
    resolution: u32,
    view_offset: u32,
    view_count: u32,
    quality: HostShadowQuality,
) -> f32 {
    let bias = light.shadow_normal_bias.max(0.0);
    if bias <= 0.0 {
        return 0.0;
    }
    let texel_world_size = match kind {
        SHADOW_VIEW_KIND_DIRECTIONAL => {
            let cascade_scale = (view_offset + 1) as f32 / view_count.max(1) as f32;
            let extent = quality.shadow_distance.max(1.0) * cascade_scale;
            extent * 2.0 / resolution.max(1) as f32
        }
        SHADOW_VIEW_KIND_SPOT => {
            let cos_half = light.spot_cos_half_angle.clamp(0.001, 1.0);
            let sin_half = (1.0 - cos_half * cos_half).max(0.0).sqrt();
            let far = light.range.max(light.shadow_near_plane.max(0.001));
            far * (sin_half / cos_half) * 2.0 / resolution.max(1) as f32
        }
        SHADOW_VIEW_KIND_POINT => light.range.max(0.001) * 2.0 / resolution.max(1) as f32,
        _ => 0.0,
    };
    bias * texel_world_size
}

fn shadow_kind_for_light(light_type: u32) -> u32 {
    match light_type {
        x if x == light_type_u32(LightType::Directional) => SHADOW_VIEW_KIND_DIRECTIONAL,
        x if x == light_type_u32(LightType::Spot) => SHADOW_VIEW_KIND_SPOT,
        x if x == light_type_u32(LightType::Point) => SHADOW_VIEW_KIND_POINT,
        _ => 0,
    }
}

fn shadow_projection_for_light(
    light: &crate::gpu::GpuLight,
    view_offset: u32,
    view_count: u32,
    quality: HostShadowQuality,
) -> Mat4 {
    let position = Vec3::from_array(light.position);
    let direction = safe_dir(Vec3::from_array(light.direction), Vec3::NEG_Z);
    match light.light_type {
        x if x == light_type_u32(LightType::Directional) => {
            directional_shadow_projection(direction, view_offset, view_count, quality)
        }
        x if x == light_type_u32(LightType::Spot) => {
            spot_shadow_projection(light, position, direction)
        }
        x if x == light_type_u32(LightType::Point) => {
            point_shadow_projection(light, position, view_offset)
        }
        _ => Mat4::IDENTITY,
    }
}

fn directional_shadow_projection(
    direction: Vec3,
    view_offset: u32,
    view_count: u32,
    quality: HostShadowQuality,
) -> Mat4 {
    let distance = quality.shadow_distance.max(1.0);
    let cascade_scale = (view_offset + 1) as f32 / view_count.max(1) as f32;
    let extent = distance * cascade_scale;
    let eye = -direction * extent;
    let view = Mat4::look_at_rh(eye, Vec3::ZERO, light_up(direction));
    let proj = Mat4::orthographic_rh(-extent, extent, -extent, extent, 0.0, distance * 2.0);
    proj * view
}

fn spot_shadow_projection(light: &crate::gpu::GpuLight, position: Vec3, direction: Vec3) -> Mat4 {
    let near = light.shadow_near_plane.max(0.001);
    let far = light.range.max(near + 0.001);
    let half_angle = light.spot_cos_half_angle.clamp(0.0, 1.0).acos();
    let fovy = (half_angle * 2.0).clamp(0.01, std::f32::consts::PI - 0.01);
    let view = Mat4::look_at_rh(position, position + direction, light_up(direction));
    let proj = Mat4::perspective_rh(fovy, 1.0, near, far);
    proj * view
}

fn point_shadow_projection(light: &crate::gpu::GpuLight, position: Vec3, face: u32) -> Mat4 {
    let near = light.shadow_near_plane.max(0.001);
    let far = light.range.max(near + 0.001);
    let (direction, up) = point_faces::basis(face);
    let view = Mat4::look_at_rh(position, position + direction, up);
    let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, near, far);
    proj * view
}

fn light_up(direction: Vec3) -> Vec3 {
    if direction.y.abs() > 0.95 {
        Vec3::Z
    } else {
        Vec3::Y
    }
}

fn safe_dir(v: Vec3, fallback: Vec3) -> Vec3 {
    if v.is_finite() && v.length_squared() > 0.000_001 {
        v.normalize()
    } else {
        fallback
    }
}

fn light_type_u32(ty: LightType) -> u32 {
    match ty {
        LightType::Point => 0,
        LightType::Directional => 1,
        LightType::Spot => 2,
    }
}

#[cfg(test)]
mod ultra_tests;

#[cfg(test)]
mod tests;
