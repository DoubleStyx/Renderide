//! CPU bone palette matching [`super::passes::mesh_deform`] skinning dispatch for culling parity.

use glam::Mat4;
use rayon::prelude::*;

use crate::scene::{RenderSpaceId, SceneCoordinator};
use crate::shared::RenderingContext;

/// Bone count above which palette construction fans out across rayon.
const SKINNING_PALETTE_PARALLEL_MIN: usize = 64;

/// Bytes per column-major `mat4<f32>` slot in the GPU-facing palette buffer.
const PALETTE_BONE_BYTES: usize = 64;

/// Inputs for [`build_skinning_palette`].
pub struct SkinningPaletteParams<'a> {
    /// Scene graph and transforms for bone and SMR nodes.
    pub scene: &'a SceneCoordinator,
    /// Render space containing the skinned mesh.
    pub space_id: RenderSpaceId,
    /// Bind-pose inverse bind matrices from the mesh asset.
    pub skinning_bind_matrices: &'a [Mat4],
    /// Whether the mesh declares a skeleton rig.
    pub has_skeleton: bool,
    /// Per-bone transform indices (host order), or `-1` for bind-only.
    pub bone_transform_indices: &'a [i32],
    /// Skinned mesh renderer node id (`-1` when not applicable).
    pub smr_node_id: i32,
    /// Which rendering context (e.g. main vs mirror) to resolve transforms in.
    pub render_context: RenderingContext,
    /// Head/output matrix for VR / secondary views.
    pub head_output_transform: Mat4,
}

/// Builds the same `world_bone * skinning_bind_matrices[i]` palette as the skinning compute pass.
pub fn build_skinning_palette(params: SkinningPaletteParams<'_>) -> Option<Vec<Mat4>> {
    let SkinningPaletteParams {
        scene,
        space_id,
        skinning_bind_matrices,
        has_skeleton,
        bone_transform_indices,
        smr_node_id,
        render_context,
        head_output_transform,
    } = params;

    let bone_count = skinning_bind_matrices.len();
    if bone_count == 0 || !has_skeleton {
        return None;
    }

    let smr_world = (smr_node_id >= 0)
        .then(|| {
            scene.world_matrix_for_render_context(
                space_id,
                smr_node_id as usize,
                render_context,
                head_output_transform,
            )
        })
        .flatten()
        .unwrap_or(Mat4::IDENTITY);

    let bone_palette = |bi: usize, bind_mat: &Mat4| -> Mat4 {
        let tid = bone_transform_indices.get(bi).copied().unwrap_or(-1);
        if tid < 0 {
            smr_world
        } else {
            match scene.world_matrix_for_render_context(
                space_id,
                tid as usize,
                render_context,
                head_output_transform,
            ) {
                Some(world) => world * bind_mat,
                None => smr_world,
            }
        }
    };

    let out: Vec<Mat4> = if bone_count >= SKINNING_PALETTE_PARALLEL_MIN {
        skinning_bind_matrices
            .par_iter()
            .enumerate()
            .map(|(bi, bind_mat)| bone_palette(bi, bind_mat))
            .collect()
    } else {
        skinning_bind_matrices
            .iter()
            .enumerate()
            .map(|(bi, bind_mat)| bone_palette(bi, bind_mat))
            .collect()
    };
    Some(out)
}

/// Writes the same palette as [`build_skinning_palette`] directly into `out` as column-major
/// `mat4<f32>` bytes.
///
/// `out` is cleared before writing and retains its capacity between calls, which avoids the
/// per-dispatch matrix and byte-vector allocations in the mesh-deform hot path.
pub fn write_skinning_palette_bytes(
    params: SkinningPaletteParams<'_>,
    out: &mut Vec<u8>,
) -> Option<usize> {
    let SkinningPaletteParams {
        scene,
        space_id,
        skinning_bind_matrices,
        has_skeleton,
        bone_transform_indices,
        smr_node_id,
        render_context,
        head_output_transform,
    } = params;

    let bone_count = skinning_bind_matrices.len();
    if bone_count == 0 || !has_skeleton {
        return None;
    }
    let total_bytes = bone_count.saturating_mul(PALETTE_BONE_BYTES);
    out.clear();
    out.resize(total_bytes, 0);

    let smr_world = (smr_node_id >= 0)
        .then(|| {
            scene.world_matrix_for_render_context(
                space_id,
                smr_node_id as usize,
                render_context,
                head_output_transform,
            )
        })
        .flatten()
        .unwrap_or(Mat4::IDENTITY);

    let write_one = |slot: &mut [u8], bi: usize, bind_mat: &Mat4| {
        let tid = bone_transform_indices.get(bi).copied().unwrap_or(-1);
        let pal = if tid < 0 {
            smr_world
        } else {
            match scene.world_matrix_for_render_context(
                space_id,
                tid as usize,
                render_context,
                head_output_transform,
            ) {
                Some(world) => world * bind_mat,
                None => smr_world,
            }
        };
        slot.copy_from_slice(bytemuck::cast_slice(&pal.to_cols_array()));
    };

    if bone_count >= SKINNING_PALETTE_PARALLEL_MIN {
        out.par_chunks_exact_mut(PALETTE_BONE_BYTES)
            .zip(skinning_bind_matrices.par_iter().enumerate())
            .for_each(|(slot, (bi, bind_mat))| write_one(slot, bi, bind_mat));
    } else {
        for (slot, (bi, bind_mat)) in out
            .chunks_exact_mut(PALETTE_BONE_BYTES)
            .zip(skinning_bind_matrices.iter().enumerate())
        {
            write_one(slot, bi, bind_mat);
        }
    }
    Some(bone_count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::RenderSpaceId;
    use crate::shared::RenderTransform;
    use glam::{Quat, Vec3};

    fn identity_transform() -> RenderTransform {
        RenderTransform {
            position: Vec3::ZERO,
            scale: Vec3::ONE,
            rotation: Quat::IDENTITY,
        }
    }

    fn seed_scene_with_one_transform() -> (SceneCoordinator, RenderSpaceId) {
        let mut scene = SceneCoordinator::new();
        let id = RenderSpaceId(1);
        scene.test_seed_space_identity_worlds(id, vec![identity_transform()], vec![-1]);
        (scene, id)
    }

    fn translation_bind_matrices(count: usize) -> Vec<Mat4> {
        (0..count)
            .map(|i| Mat4::from_translation(Vec3::new(i as f32, 0.5 * i as f32, -(i as f32))))
            .collect()
    }

    #[test]
    fn build_palette_below_threshold_returns_world_times_bind() {
        let (scene, space_id) = seed_scene_with_one_transform();
        let binds = translation_bind_matrices(8);
        let bone_indices: Vec<i32> = vec![0; binds.len()];
        let out = build_skinning_palette(SkinningPaletteParams {
            scene: &scene,
            space_id,
            skinning_bind_matrices: &binds,
            has_skeleton: true,
            bone_transform_indices: &bone_indices,
            smr_node_id: 0,
            render_context: RenderingContext::UserView,
            head_output_transform: Mat4::IDENTITY,
        })
        .expect("palette");
        assert_eq!(out.len(), binds.len());
        for (got, want) in out.iter().zip(binds.iter()) {
            assert_eq!(got.to_cols_array(), want.to_cols_array());
        }
    }

    #[test]
    fn build_palette_parallel_path_matches_serial_for_large_bone_count() {
        let (scene, space_id) = seed_scene_with_one_transform();
        let bone_count = SKINNING_PALETTE_PARALLEL_MIN + 11;
        let binds = translation_bind_matrices(bone_count);
        let bone_indices: Vec<i32> = vec![0; binds.len()];
        let parallel = build_skinning_palette(SkinningPaletteParams {
            scene: &scene,
            space_id,
            skinning_bind_matrices: &binds,
            has_skeleton: true,
            bone_transform_indices: &bone_indices,
            smr_node_id: 0,
            render_context: RenderingContext::UserView,
            head_output_transform: Mat4::IDENTITY,
        })
        .expect("palette");
        assert_eq!(parallel.len(), bone_count);
        // Identity SMR world means each entry must equal its bind matrix.
        for (got, want) in parallel.iter().zip(binds.iter()) {
            assert_eq!(got.to_cols_array(), want.to_cols_array());
        }
    }

    #[test]
    fn write_palette_bytes_parallel_matches_build_palette() {
        let (scene, space_id) = seed_scene_with_one_transform();
        let bone_count = SKINNING_PALETTE_PARALLEL_MIN + 5;
        let binds = translation_bind_matrices(bone_count);
        let bone_indices: Vec<i32> = vec![0; binds.len()];
        let palette = build_skinning_palette(SkinningPaletteParams {
            scene: &scene,
            space_id,
            skinning_bind_matrices: &binds,
            has_skeleton: true,
            bone_transform_indices: &bone_indices,
            smr_node_id: 0,
            render_context: RenderingContext::UserView,
            head_output_transform: Mat4::IDENTITY,
        })
        .expect("palette");
        let mut bytes = Vec::new();
        let written = write_skinning_palette_bytes(
            SkinningPaletteParams {
                scene: &scene,
                space_id,
                skinning_bind_matrices: &binds,
                has_skeleton: true,
                bone_transform_indices: &bone_indices,
                smr_node_id: 0,
                render_context: RenderingContext::UserView,
                head_output_transform: Mat4::IDENTITY,
            },
            &mut bytes,
        )
        .expect("bytes");
        assert_eq!(written, bone_count);
        assert_eq!(bytes.len(), bone_count * PALETTE_BONE_BYTES);
        let mut expected: Vec<u8> = Vec::with_capacity(bone_count * PALETTE_BONE_BYTES);
        for m in &palette {
            expected.extend_from_slice(bytemuck::cast_slice::<f32, u8>(&m.to_cols_array()));
        }
        assert_eq!(bytes, expected);
    }
}
