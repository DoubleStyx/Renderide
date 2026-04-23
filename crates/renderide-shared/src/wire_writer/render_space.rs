//! High-level helper that bundles the four SHM-backed wire encoders into a single
//! [`crate::shared::RenderSpaceUpdate`] for the integration test scene (one sphere, one camera).
//!
//! Two complementary structs are provided:
//!
//! - [`SphereSceneSharedMemoryRegions`] holds the encoded byte chunks the host must write into a
//!   shared-memory buffer. The host calls [`SphereSceneSharedMemoryRegions::build`] once.
//! - [`SphereSceneSharedMemoryLayout`] holds the byte offsets where each chunk lives inside the
//!   host-allocated shared-memory buffer plus the buffer id and capacity for the resulting
//!   [`crate::shared::SharedMemoryBufferDescriptor`]s.
//!
//! The host writes each region at its assigned offset, then calls
//! [`build_sphere_render_space_update`] to assemble the final [`crate::shared::RenderSpaceUpdate`]
//! that embeds the descriptors.

use glam::{Quat, Vec3};

use crate::buffer::SharedMemoryBufferDescriptor;
use crate::shared::{
    MeshRenderablesUpdate, MeshRendererState, MotionVectorMode, RenderSH2, RenderSpaceUpdate,
    RenderTransform, ShadowCastMode, TransformsUpdate,
};
#[expect(unused_imports)] // keep `SharedMemoryWriterConfig` cross-reference rustdoc resolvable
use crate::SharedMemoryWriterConfig;

use super::{
    encode_additions, encode_mesh_states, encode_packed_material_ids,
    encode_transform_pose_updates, transforms::TransformPoseRow,
};

/// Inputs to the sphere scene builder.
///
/// `camera_world_pose` is the **camera in world space** (renderer derives the view matrix as
/// `inverse(root_transform)` per
/// [`crates/renderide/src/render_graph/cluster_frame.rs`](Renderide/crates/renderide/src/render_graph/cluster_frame.rs)).
/// To look at the origin from `(0,0,-3)` along `+Z` set
/// `camera_world_pose = RenderTransform { position: (0,0,-3), scale: (1,1,1), rotation: IDENTITY }`.
#[derive(Clone, Copy, Debug)]
pub struct SphereSceneInputs {
    /// Render-space id (must be `>= 0` and the smallest non-overlay active space for this to be
    /// the main desktop view). Use `1` to leave room for any host-default `0`.
    pub render_space_id: i32,
    /// World-space pose of the desktop camera (the renderer inverts this to get the view matrix).
    pub camera_world_pose: RenderTransform,
    /// Object pose for the sphere (transform index `0` in this scene).
    pub object_pose: RenderTransform,
    /// Mesh asset id the host has already uploaded via `MeshUploadData`.
    pub mesh_asset_id: i32,
    /// Material asset id bound to the sphere. The host does **not** call `SetShader` for this id,
    /// so the renderer's `MaterialRouter` falls back to `RasterPipelineKind::DebugWorldNormals`.
    pub material_asset_id: i32,
}

impl Default for SphereSceneInputs {
    fn default() -> Self {
        Self {
            render_space_id: 1,
            camera_world_pose: RenderTransform {
                position: Vec3::new(0.0, 0.0, -3.0),
                scale: Vec3::ONE,
                rotation: Quat::IDENTITY,
            },
            object_pose: RenderTransform {
                position: Vec3::ZERO,
                scale: Vec3::ONE,
                rotation: Quat::IDENTITY,
            },
            mesh_asset_id: 2,
            material_asset_id: 4,
        }
    }
}

/// Encoded byte chunks the host must write into the scene's shared-memory buffer.
#[derive(Clone, Debug)]
pub struct SphereSceneSharedMemoryRegions {
    /// Bytes for `TransformsUpdate.pose_updates`. One row: `transform_id=0, pose=object_pose`.
    pub pose_updates_bytes: Vec<u8>,
    /// Bytes for `MeshRenderablesUpdate.additions`. One entry: `node_id=0` then sentinel `-1`.
    pub additions_bytes: Vec<u8>,
    /// Bytes for `MeshRenderablesUpdate.mesh_states`. One row: `renderable_index=0, mesh_asset_id`,
    /// then sentinel `renderable_index=-1`.
    pub mesh_states_bytes: Vec<u8>,
    /// Bytes for `MeshRenderablesUpdate.mesh_materials_and_property_blocks`. One i32:
    /// `material_asset_id`.
    pub packed_material_ids_bytes: Vec<u8>,
}

impl SphereSceneSharedMemoryRegions {
    /// Builds all four encoded chunks from `inputs`.
    pub fn build(inputs: &SphereSceneInputs) -> Self {
        let pose_updates_bytes = encode_transform_pose_updates(&[TransformPoseRow {
            transform_id: 0,
            pose: inputs.object_pose,
        }]);
        let additions_bytes = encode_additions(&[0]);
        let mesh_states_bytes = encode_mesh_states(&[MeshRendererState {
            renderable_index: 0,
            mesh_asset_id: inputs.mesh_asset_id,
            material_count: 1,
            material_property_block_count: 0,
            sorting_order: 0,
            shadow_cast_mode: ShadowCastMode::Off,
            motion_vector_mode: MotionVectorMode::NoMotion,
            _padding: [0; 2],
        }]);
        let packed_material_ids_bytes = encode_packed_material_ids(&[inputs.material_asset_id]);
        Self {
            pose_updates_bytes,
            additions_bytes,
            mesh_states_bytes,
            packed_material_ids_bytes,
        }
    }

    /// Total bytes needed when each region is laid out back-to-back inside one SHM buffer (no
    /// padding between regions).
    pub fn total_bytes(&self) -> usize {
        self.pose_updates_bytes.len()
            + self.additions_bytes.len()
            + self.mesh_states_bytes.len()
            + self.packed_material_ids_bytes.len()
    }
}

/// Where each [`SphereSceneSharedMemoryRegions`] chunk lives inside the host-owned SHM buffer.
///
/// The host computes offsets back-to-back via [`Self::pack_back_to_back`] (or chooses its own
/// layout), then writes each chunk to the matching offset before calling
/// [`build_sphere_render_space_update`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SphereSceneSharedMemoryLayout {
    /// Cloudtoid `buffer_id` (matches what the host passed to
    /// [`crate::SharedMemoryWriterConfig::buffer_id`]).
    pub buffer_id: i32,
    /// Total capacity of the host SHM buffer (bytes).
    pub buffer_capacity: i32,
    /// Byte offset where `pose_updates_bytes` was written.
    pub pose_updates_offset: i32,
    /// Byte offset where `additions_bytes` was written.
    pub additions_offset: i32,
    /// Byte offset where `mesh_states_bytes` was written.
    pub mesh_states_offset: i32,
    /// Byte offset where `packed_material_ids_bytes` was written.
    pub packed_material_ids_offset: i32,
}

impl SphereSceneSharedMemoryLayout {
    /// Computes a back-to-back layout (no gaps) for `regions` inside a buffer of size
    /// `regions.total_bytes()`.
    pub fn pack_back_to_back(
        buffer_id: i32,
        buffer_capacity: i32,
        regions: &SphereSceneSharedMemoryRegions,
    ) -> Self {
        let pose_off = 0i32;
        let add_off = pose_off + regions.pose_updates_bytes.len() as i32;
        let states_off = add_off + regions.additions_bytes.len() as i32;
        let mats_off = states_off + regions.mesh_states_bytes.len() as i32;
        Self {
            buffer_id,
            buffer_capacity,
            pose_updates_offset: pose_off,
            additions_offset: add_off,
            mesh_states_offset: states_off,
            packed_material_ids_offset: mats_off,
        }
    }
}

/// Assembles the final [`RenderSpaceUpdate`] that ships with `FrameSubmitData.render_spaces` once
/// the host has written the four byte chunks into the scene's SHM buffer at `layout`'s offsets.
pub fn build_sphere_render_space_update(
    inputs: &SphereSceneInputs,
    regions: &SphereSceneSharedMemoryRegions,
    layout: &SphereSceneSharedMemoryLayout,
) -> RenderSpaceUpdate {
    let pose_desc = SharedMemoryBufferDescriptor {
        buffer_id: layout.buffer_id,
        buffer_capacity: layout.buffer_capacity,
        offset: layout.pose_updates_offset,
        length: regions.pose_updates_bytes.len() as i32,
    };
    let additions_desc = SharedMemoryBufferDescriptor {
        buffer_id: layout.buffer_id,
        buffer_capacity: layout.buffer_capacity,
        offset: layout.additions_offset,
        length: regions.additions_bytes.len() as i32,
    };
    let states_desc = SharedMemoryBufferDescriptor {
        buffer_id: layout.buffer_id,
        buffer_capacity: layout.buffer_capacity,
        offset: layout.mesh_states_offset,
        length: regions.mesh_states_bytes.len() as i32,
    };
    let mats_desc = SharedMemoryBufferDescriptor {
        buffer_id: layout.buffer_id,
        buffer_capacity: layout.buffer_capacity,
        offset: layout.packed_material_ids_offset,
        length: regions.packed_material_ids_bytes.len() as i32,
    };

    RenderSpaceUpdate {
        id: inputs.render_space_id,
        is_active: true,
        is_overlay: false,
        is_private: false,
        root_transform: inputs.camera_world_pose,
        view_position_is_external: false,
        override_view_position: false,
        skybox_material_asset_id: -1,
        ambient_light: RenderSH2::default(),
        overriden_view_transform: RenderTransform::default(),
        transforms_update: Some(TransformsUpdate {
            target_transform_count: 1,
            removals: SharedMemoryBufferDescriptor::default(),
            parent_updates: SharedMemoryBufferDescriptor::default(),
            pose_updates: pose_desc,
        }),
        mesh_renderers_update: Some(MeshRenderablesUpdate {
            mesh_states: states_desc,
            mesh_materials_and_property_blocks: mats_desc,
            removals: SharedMemoryBufferDescriptor::default(),
            additions: additions_desc,
        }),
        skinned_mesh_renderers_update: None,
        lights_update: None,
        cameras_update: None,
        camera_portals_update: None,
        reflection_probes_update: None,
        reflection_probe_sh2_taks: None,
        layers_update: None,
        billboard_buffers_update: None,
        mesh_render_buffers_update: None,
        trail_renderers_update: None,
        lights_buffer_renderers_update: None,
        render_transform_overrides_update: None,
        render_material_overrides_update: None,
        blit_to_displays_update: None,
        lod_group_update: None,
        gaussian_splat_renderers_update: None,
        reflection_probe_render_tasks: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn region_byte_lengths_match_expected() {
        let inputs = SphereSceneInputs::default();
        let regions = SphereSceneSharedMemoryRegions::build(&inputs);
        assert_eq!(regions.pose_updates_bytes.len(), 44);
        assert_eq!(regions.additions_bytes.len(), 8); // node_id=0 + sentinel
        assert_eq!(regions.mesh_states_bytes.len(), 48); // one row + sentinel
        assert_eq!(regions.packed_material_ids_bytes.len(), 4);
        assert_eq!(regions.total_bytes(), 44 + 8 + 48 + 4);
    }

    #[test]
    fn pack_back_to_back_offsets_increase_correctly() {
        let inputs = SphereSceneInputs::default();
        let regions = SphereSceneSharedMemoryRegions::build(&inputs);
        let layout = SphereSceneSharedMemoryLayout::pack_back_to_back(
            7,
            regions.total_bytes() as i32,
            &regions,
        );
        assert_eq!(layout.pose_updates_offset, 0);
        assert_eq!(layout.additions_offset, 44);
        assert_eq!(layout.mesh_states_offset, 52);
        assert_eq!(layout.packed_material_ids_offset, 100);
    }

    #[test]
    fn produced_render_space_update_has_expected_descriptors() {
        let inputs = SphereSceneInputs::default();
        let regions = SphereSceneSharedMemoryRegions::build(&inputs);
        let layout = SphereSceneSharedMemoryLayout::pack_back_to_back(
            13,
            regions.total_bytes() as i32,
            &regions,
        );
        let rs = build_sphere_render_space_update(&inputs, &regions, &layout);

        assert_eq!(rs.id, inputs.render_space_id);
        assert!(rs.is_active);
        assert!(!rs.is_overlay);

        let mr = rs.mesh_renderers_update.as_ref().expect("mesh update");
        assert_eq!(mr.additions.buffer_id, 13);
        assert_eq!(mr.additions.offset, 44);
        assert_eq!(mr.additions.length, 8);
        assert_eq!(mr.mesh_states.offset, 52);
        assert_eq!(mr.mesh_states.length, 48);
        assert_eq!(mr.mesh_materials_and_property_blocks.offset, 100);
        assert_eq!(mr.mesh_materials_and_property_blocks.length, 4);

        let tu = rs.transforms_update.as_ref().expect("transforms update");
        assert_eq!(tu.target_transform_count, 1);
        assert_eq!(tu.pose_updates.offset, 0);
        assert_eq!(tu.pose_updates.length, 44);
    }
}
