//! Realtime shadow planning stored on [`FrameResourceManager`].

use std::sync::Arc;

use glam::{Mat4, Vec3};

use crate::backend::HostShadowQuality;
use crate::camera::ViewId;
use crate::gpu::{
    GpuLimits, GpuShadowView, MAX_SHADOW_VIEWS, SHADOW_VIEW_KIND_DIRECTIONAL,
    SHADOW_VIEW_KIND_POINT, SHADOW_VIEW_KIND_SPOT,
};
use crate::materials::shadow_caster_policy_for_pipeline;
use crate::mesh_deform::SkinCacheKey;
use crate::shared::{LightType, ShadowCastMode};
use crate::world_mesh::{WorldMeshDrawItem, WorldMeshDrawPlan};

use super::manager::FrameResourceManager;

const POINT_FACE_COUNT: u32 = 6;
const SHADOW_TYPE_NONE: u32 = 0;

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
    /// World-space light position used by radial point-shadow casters.
    pub(crate) light_position: Vec3,
    /// Light range used by radial point-shadow casters.
    pub(crate) light_range: f32,
    /// Host-authored shadow bias used by radial point-shadow casters.
    pub(crate) shadow_bias: f32,
    /// First per-draw slab row reserved for this shadow view.
    pub(crate) slab_slot_offset: usize,
    /// Shadow-casting world-mesh draws for this shadow view.
    pub(crate) draws: Arc<[WorldMeshDrawItem]>,
}

/// Read-only frame shadow plan consumed by the graph shadow pass.
#[derive(Clone, Debug, Default)]
pub(crate) struct ShadowFramePlan {
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
        Some((
            self.shadow_frame.requested_resolution.max(1),
            self.shadow_frame.requested_layers.max(1),
        ))
    }

    /// Per-draw slab slots required by all shadow map views.
    pub(crate) fn shadow_max_draw_slots(&self) -> usize {
        self.shadow_frame.requested_draw_slots
    }

    /// Deform keys required by this frame's shadow caster draws.
    pub(crate) fn shadow_mesh_deform_keys(&self) -> hashbrown::HashSet<SkinCacheKey> {
        let mut keys = hashbrown::HashSet::new();
        for view in &self.shadow_frame.render_views {
            for item in view.draws.iter() {
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
                slab_slot_offset: plan.requested_draw_slots,
                draws: Arc::clone(&shadow_draws),
            });
            plan.requested_draw_slots =
                plan.requested_draw_slots.saturating_add(shadow_draws.len());
        }
        light.shadow_view_start = start;
        light.shadow_view_count = view_count;
        light.shadow_flags = 0;
    }
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
        quality.tile_resolution
    };
    shadow_tile_resolution(limits, requested)
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
    let (direction, up) = point_face_basis(face);
    let view = Mat4::look_at_rh(position, position + direction, up);
    let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, near, far);
    proj * view
}

fn point_face_basis(face: u32) -> (Vec3, Vec3) {
    match face % POINT_FACE_COUNT {
        0 => (Vec3::X, Vec3::Y),
        1 => (Vec3::NEG_X, Vec3::Y),
        2 => (Vec3::Y, Vec3::NEG_Z),
        3 => (Vec3::NEG_Y, Vec3::Z),
        4 => (Vec3::Z, Vec3::Y),
        _ => (Vec3::NEG_Z, Vec3::Y),
    }
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
mod tests {
    use std::sync::Arc;

    use hashbrown::HashMap;

    use crate::backend::HostShadowQuality;
    use crate::backend::frame_resource_manager::per_view_state::PreparedViewLights;
    use crate::camera::ViewId;
    use crate::gpu::{GpuLight, GpuLimits};
    use crate::materials::RasterPipelineKind;
    use crate::shared::{LightType, ShadowCastMode};
    use crate::world_mesh::draw_prep::WorldMeshDrawCollection;
    use crate::world_mesh::test_fixtures::{DummyDrawItemSpec, dummy_world_mesh_draw_item};
    use crate::world_mesh::{PrefetchedWorldMeshViewDraws, WorldMeshDrawItem, WorldMeshDrawPlan};
    use glam::Vec3;

    use super::{
        POINT_FACE_COUNT, light_type_u32, point_face_basis, point_shadow_projection,
        shadow_view_capacity, shadow_view_count_for_light,
    };

    fn limits(max_texture_dimension_2d: u32, max_texture_array_layers: u32) -> GpuLimits {
        GpuLimits::synthetic_for_tests(
            wgpu::Limits {
                max_texture_dimension_2d,
                max_texture_array_layers,
                ..Default::default()
            },
            wgpu::Features::empty(),
            HashMap::new(),
        )
    }

    fn shadowed_light(light_type: LightType) -> GpuLight {
        GpuLight {
            position: [3.0, 4.0, 5.0],
            light_type: light_type_u32(light_type),
            shadow_type: 1,
            shadow_strength: 1.0,
            shadow_near_plane: 0.05,
            shadow_bias: 0.25,
            range: 8.0,
            spot_cos_half_angle: 0.5,
            ..GpuLight::default()
        }
    }

    fn pbs_draw(node_id: i32, shadow_cast_mode: ShadowCastMode) -> WorldMeshDrawItem {
        let mut item = dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 1,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 1,
            node_id,
            slot_index: 0,
            collect_order: node_id.max(0) as usize,
            alpha_blended: false,
        });
        item.shadow_cast_mode = shadow_cast_mode;
        item.batch_key.pipeline =
            RasterPipelineKind::EmbeddedStem(Arc::from("pbsmetallic_default"));
        item
    }

    fn prefetched_plan(items: Vec<WorldMeshDrawItem>) -> WorldMeshDrawPlan {
        WorldMeshDrawPlan::Prefetched(Box::new(PrefetchedWorldMeshViewDraws::new(
            WorldMeshDrawCollection {
                draws_pre_cull: items.len(),
                items,
                draws_culled: 0,
                draws_hi_z_culled: 0,
                visibility: Default::default(),
                arrangement: Default::default(),
            },
            None,
        )))
    }

    #[test]
    fn point_lights_plan_six_shadow_views() {
        assert_eq!(
            shadow_view_count_for_light(
                light_type_u32(LightType::Point),
                HostShadowQuality::default()
            ),
            POINT_FACE_COUNT
        );
    }

    #[test]
    fn directional_lights_use_host_cascade_count() {
        let quality = HostShadowQuality {
            cascade_count: 2,
            ..HostShadowQuality::default()
        };
        assert_eq!(
            shadow_view_count_for_light(light_type_u32(LightType::Directional), quality),
            2
        );
    }

    #[test]
    fn clear_assignment_removes_shadow_view_link() {
        let mut light = GpuLight {
            shadow_view_start: 4,
            shadow_view_count: 2,
            shadow_flags: 7,
            ..GpuLight::default()
        };
        super::clear_light_shadow_assignment(&mut light);
        assert_eq!(light.shadow_view_start, 0);
        assert_eq!(light.shadow_view_count, 0);
        assert_eq!(light.shadow_flags, 0);
    }

    #[test]
    fn point_face_order_matches_shader_face_indices() {
        let expected = [
            (Vec3::X, Vec3::Y),
            (Vec3::NEG_X, Vec3::Y),
            (Vec3::Y, Vec3::NEG_Z),
            (Vec3::NEG_Y, Vec3::Z),
            (Vec3::Z, Vec3::Y),
            (Vec3::NEG_Z, Vec3::Y),
        ];
        for (face, expected_basis) in expected.into_iter().enumerate() {
            assert_eq!(point_face_basis(face as u32), expected_basis);
        }
    }

    #[test]
    fn point_shadow_faces_have_distinct_projection_matrices() {
        let light = GpuLight {
            range: 8.0,
            shadow_near_plane: 0.05,
            ..GpuLight::default()
        };
        let position = Vec3::new(1.0, 2.0, 3.0);
        let mut seen = Vec::new();
        for face in 0..POINT_FACE_COUNT {
            let matrix = point_shadow_projection(&light, position, face).to_cols_array();
            assert!(
                !seen.iter().any(|existing| existing == &matrix),
                "point shadow face {face} reused a previous projection"
            );
            seen.push(matrix);
        }
    }

    #[test]
    fn point_shadow_faces_reserve_disjoint_slab_ranges() {
        let mut manager = super::FrameResourceManager::new();
        manager
            .per_view_lights
            .get_or_insert_with(ViewId::Main, PreparedViewLights::default)
            .lights
            .push(GpuLight {
                position: [3.0, 4.0, 5.0],
                light_type: light_type_u32(LightType::Point),
                shadow_type: 1,
                shadow_strength: 1.0,
                shadow_near_plane: 0.05,
                shadow_bias: 0.25,
                range: 8.0,
                ..GpuLight::default()
            });

        let mut first = dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 1,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 1,
            node_id: 1,
            slot_index: 0,
            collect_order: 0,
            alpha_blended: false,
        });
        first.batch_key.pipeline =
            RasterPipelineKind::EmbeddedStem(Arc::from("pbsmetallic_default"));
        let mut second = first.clone();
        second.node_id = 2;
        second.collect_order = 1;

        let draw_plan = WorldMeshDrawPlan::Prefetched(Box::new(PrefetchedWorldMeshViewDraws::new(
            WorldMeshDrawCollection {
                items: vec![first, second],
                draws_pre_cull: 2,
                draws_culled: 0,
                draws_hi_z_culled: 0,
                visibility: Default::default(),
                arrangement: Default::default(),
            },
            None,
        )));
        manager.prepare_shadow_frame_for_views(
            HostShadowQuality::default(),
            [(ViewId::Main, &draw_plan)],
        );

        let plan = manager.shadow_frame_plan();
        assert_eq!(plan.render_views.len(), POINT_FACE_COUNT as usize);
        assert_eq!(plan.requested_draw_slots, POINT_FACE_COUNT as usize * 2);
        for (index, view) in plan.render_views.iter().enumerate() {
            assert_eq!(view.kind, crate::gpu::SHADOW_VIEW_KIND_POINT);
            assert_eq!(view.light_position, Vec3::new(3.0, 4.0, 5.0));
            assert_eq!(view.light_range, 8.0);
            assert_eq!(view.shadow_bias, 0.25);
            assert_eq!(view.slab_slot_offset, index * 2);
            assert_eq!(view.draws.len(), 2);
        }
        for (index, view) in plan.render_views.iter().enumerate() {
            let start = view.slab_slot_offset;
            let end = start + view.draws.len();
            for other in plan.render_views.iter().skip(index + 1) {
                let other_start = other.slab_slot_offset;
                let other_end = other_start + other.draws.len();
                assert!(end <= other_start || other_end <= start);
            }
        }
    }

    #[test]
    fn shadow_planning_excludes_shadow_cast_mode_off_draws() {
        let mut manager = super::FrameResourceManager::new();
        manager
            .per_view_lights
            .get_or_insert_with(ViewId::Main, PreparedViewLights::default)
            .lights
            .push(shadowed_light(LightType::Spot));
        let draw_plan = prefetched_plan(vec![
            pbs_draw(1, ShadowCastMode::Off),
            pbs_draw(2, ShadowCastMode::On),
            pbs_draw(3, ShadowCastMode::ShadowOnly),
        ]);

        manager.prepare_shadow_frame_for_views(
            HostShadowQuality::default(),
            [(ViewId::Main, &draw_plan)],
        );

        let plan = manager.shadow_frame_plan();
        assert_eq!(plan.render_views.len(), 1);
        let nodes = plan.render_views[0]
            .draws
            .iter()
            .map(|item| item.node_id)
            .collect::<Vec<_>>();
        assert_eq!(nodes, vec![2, 3]);
        assert_eq!(plan.requested_draw_slots, 2);
    }

    #[test]
    fn shadow_planning_uses_per_light_resolution_and_metadata_bias() {
        let mut manager = super::FrameResourceManager::new();
        let lights = &mut manager
            .per_view_lights
            .get_or_insert_with(ViewId::Main, PreparedViewLights::default)
            .lights;
        let mut low_resolution = shadowed_light(LightType::Spot);
        low_resolution.shadow_map_resolution = 512;
        low_resolution.shadow_normal_bias = 2.0;
        lights.push(low_resolution);
        lights.push(shadowed_light(LightType::Spot));
        let draw_plan = prefetched_plan(vec![pbs_draw(1, ShadowCastMode::On)]);

        manager.prepare_shadow_frame_for_views(
            HostShadowQuality::default(),
            [(ViewId::Main, &draw_plan)],
        );

        let plan = manager.shadow_frame_plan();
        assert_eq!(plan.render_views.len(), 2);
        assert_eq!(
            plan.requested_resolution,
            HostShadowQuality::default().tile_resolution
        );
        assert_eq!(plan.render_views[0].resolution, 512);
        assert_eq!(
            plan.render_views[1].resolution,
            HostShadowQuality::default().tile_resolution
        );
        assert_eq!(plan.metadata[0].params[1], 1.0 / 512.0);
        assert_eq!(plan.metadata[0].atlas_rect, [0.0, 0.0, 0.25, 0.25]);
        assert_eq!(plan.metadata[1].atlas_rect, [0.0, 0.0, 1.0, 1.0]);
        assert_eq!(plan.metadata[0].light_params[3], 0.25);
        assert!(plan.metadata[0].light_params[2] > 0.0);
    }

    #[test]
    fn shadow_planning_requests_only_custom_resolution_when_all_lights_override() {
        let mut manager = super::FrameResourceManager::new();
        let mut light = shadowed_light(LightType::Spot);
        light.shadow_map_resolution = 512;
        manager
            .per_view_lights
            .get_or_insert_with(ViewId::Main, PreparedViewLights::default)
            .lights
            .push(light);
        let draw_plan = prefetched_plan(vec![pbs_draw(1, ShadowCastMode::On)]);

        manager.prepare_shadow_frame_for_views(
            HostShadowQuality::default(),
            [(ViewId::Main, &draw_plan)],
        );

        let plan = manager.shadow_frame_plan();
        assert_eq!(plan.requested_resolution, 512);
        assert_eq!(plan.render_views[0].resolution, 512);
        assert_eq!(plan.metadata[0].atlas_rect, [0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn shadow_planning_clamps_to_gpu_atlas_capacity() {
        let mut manager = super::FrameResourceManager::new();
        manager.limits = Some(Arc::new(limits(1024, 2)));
        manager
            .per_view_lights
            .get_or_insert_with(ViewId::Main, PreparedViewLights::default)
            .lights
            .push(GpuLight {
                position: [3.0, 4.0, 5.0],
                light_type: light_type_u32(LightType::Point),
                shadow_type: 1,
                shadow_strength: 1.0,
                shadow_near_plane: 0.05,
                shadow_bias: 0.25,
                range: 8.0,
                ..GpuLight::default()
            });

        let mut item = dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 1,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 1,
            node_id: 1,
            slot_index: 0,
            collect_order: 0,
            alpha_blended: false,
        });
        item.batch_key.pipeline =
            RasterPipelineKind::EmbeddedStem(Arc::from("pbsmetallic_default"));
        let draw_plan = WorldMeshDrawPlan::Prefetched(Box::new(PrefetchedWorldMeshViewDraws::new(
            WorldMeshDrawCollection {
                items: vec![item],
                draws_pre_cull: 1,
                draws_culled: 0,
                draws_hi_z_culled: 0,
                visibility: Default::default(),
                arrangement: Default::default(),
            },
            None,
        )));

        manager.prepare_shadow_frame_for_views(
            HostShadowQuality::default(),
            [(ViewId::Main, &draw_plan)],
        );

        let plan = manager.shadow_frame_plan();
        assert_eq!(shadow_view_capacity(manager.limits.as_deref()), 2);
        assert_eq!(plan.requested_resolution, 1024);
        assert_eq!(plan.requested_layers, 2);
        assert_eq!(plan.render_views.len(), 2);
        assert_eq!(plan.metadata.len(), 2);
        assert_eq!(plan.requested_draw_slots, 2);
        for (index, view) in plan.render_views.iter().enumerate() {
            assert_eq!(view.layer, index as u32);
            assert_eq!(view.resolution, 1024);
            assert_eq!(plan.metadata[index].params[0], index as f32);
            assert_eq!(plan.metadata[index].params[1], 1.0 / 1024.0);
        }
        let light = &manager.per_view_lights.get(ViewId::Main).unwrap().lights[0];
        assert_eq!(light.shadow_view_start, 0);
        assert_eq!(light.shadow_view_count, 2);
    }
}
