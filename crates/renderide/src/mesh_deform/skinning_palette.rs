//! CPU bone palette matching [`super::passes::mesh_deform`] skinning dispatch for culling parity.

use glam::Mat4;
use rayon::prelude::*;

use crate::scene::{RenderSpaceId, SceneCoordinator};
use crate::shared::RenderingContext;

/// Bone count assigned to one palette construction worker chunk.
const SKINNING_PALETTE_PARALLEL_CHUNK_BONES: usize = 64;
/// Bone count above which palette construction fans out across rayon.
///
/// Per-bone work is one world lookup plus a Mat4 multiply, so two chunks are enough to amortize
/// worker dispatch once the palette reaches medium sizes.
const SKINNING_PALETTE_PARALLEL_MIN: usize = SKINNING_PALETTE_PARALLEL_CHUNK_BONES * 2;

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

/// Captures the inputs needed to resolve each bone's world matrix once and reuse the resolver
/// across rayon workers.
struct PaletteResolver<'a> {
    scene: &'a SceneCoordinator,
    space_id: RenderSpaceId,
    render_context: RenderingContext,
    head_output_transform: Mat4,
    smr_world: Mat4,
    bone_transform_indices: &'a [i32],
}

impl<'a> PaletteResolver<'a> {
    fn new(params: &'a SkinningPaletteParams<'a>) -> Self {
        let smr_world = (params.smr_node_id >= 0)
            .then(|| {
                params.scene.world_matrix_for_render_context(
                    params.space_id,
                    params.smr_node_id as usize,
                    params.render_context,
                    params.head_output_transform,
                )
            })
            .flatten()
            .unwrap_or(Mat4::IDENTITY);
        Self {
            scene: params.scene,
            space_id: params.space_id,
            render_context: params.render_context,
            head_output_transform: params.head_output_transform,
            smr_world,
            bone_transform_indices: params.bone_transform_indices,
        }
    }

    /// Resolves the `world_bone * bind_mat` palette entry for one bone, falling back to the SMR's
    /// world matrix when the bone's transform is missing or marked `-1` for bind-only.
    fn matrix(&self, bone_index: usize, bind_mat: &Mat4) -> Mat4 {
        let tid = self
            .bone_transform_indices
            .get(bone_index)
            .copied()
            .unwrap_or(-1);
        if tid < 0 {
            return self.smr_world;
        }
        match self.scene.world_matrix_for_render_context(
            self.space_id,
            tid as usize,
            self.render_context,
            self.head_output_transform,
        ) {
            Some(world) => world * bind_mat,
            None => self.smr_world,
        }
    }
}

/// Builds the same `world_bone * skinning_bind_matrices[i]` palette as the skinning compute pass.
#[cfg(test)]
pub fn build_skinning_palette(params: SkinningPaletteParams<'_>) -> Option<Vec<Mat4>> {
    let bone_count = params.skinning_bind_matrices.len();
    if bone_count == 0 || !params.has_skeleton {
        return None;
    }
    let resolver = PaletteResolver::new(&params);
    let out: Vec<Mat4> = if bone_count >= SKINNING_PALETTE_PARALLEL_MIN {
        params
            .skinning_bind_matrices
            .par_iter()
            .with_min_len(SKINNING_PALETTE_PARALLEL_CHUNK_BONES)
            .enumerate()
            .map(|(bi, bind_mat)| resolver.matrix(bi, bind_mat))
            .collect()
    } else {
        params
            .skinning_bind_matrices
            .iter()
            .enumerate()
            .map(|(bi, bind_mat)| resolver.matrix(bi, bind_mat))
            .collect()
    };
    Some(out)
}

/// Writes the skinning palette directly into `out` as column-major
/// `mat4<f32>` bytes.
///
/// `out` is cleared before writing and retains its capacity between calls, which avoids the
/// per-dispatch matrix and byte-vector allocations in the mesh-deform hot path.
pub fn write_skinning_palette_bytes(
    params: SkinningPaletteParams<'_>,
    out: &mut Vec<u8>,
) -> Option<usize> {
    write_skinning_palette_bytes_impl(params, out, true)
}

/// Writes the skinning palette without nested Rayon fan-out.
///
/// Item-level mesh-deform preplanning calls this from Rayon workers so a large batch of skinned
/// renderers does not recursively split every medium palette.
pub fn write_skinning_palette_bytes_serial(
    params: SkinningPaletteParams<'_>,
    out: &mut Vec<u8>,
) -> Option<usize> {
    write_skinning_palette_bytes_impl(params, out, false)
}

/// Shared palette writer with optional per-palette fan-out.
fn write_skinning_palette_bytes_impl(
    params: SkinningPaletteParams<'_>,
    out: &mut Vec<u8>,
    allow_parallel: bool,
) -> Option<usize> {
    let bone_count = params.skinning_bind_matrices.len();
    if bone_count == 0 || !params.has_skeleton {
        return None;
    }
    let total_bytes = bone_count.saturating_mul(PALETTE_BONE_BYTES);
    out.clear();
    out.reserve(total_bytes);
    let resolver = PaletteResolver::new(&params);

    if allow_parallel && bone_count >= SKINNING_PALETTE_PARALLEL_MIN {
        // SAFETY: `reserve(total_bytes)` above guarantees capacity for `total_bytes` bytes. The
        // par_chunks_exact_mut loop below fully overwrites every byte via `copy_from_slice`
        // before any read of `out`, so exposing the uninitialised range as `&mut [u8]` is sound.
        // `set_len` is the only way to materialise the parallel slot iterator without a wasted
        // O(N) zero-fill memset before the bone writes overwrite every byte.
        unsafe {
            out.set_len(total_bytes);
        }
        out.par_chunks_exact_mut(PALETTE_BONE_BYTES)
            .with_min_len(SKINNING_PALETTE_PARALLEL_CHUNK_BONES)
            .zip(
                params
                    .skinning_bind_matrices
                    .par_iter()
                    .with_min_len(SKINNING_PALETTE_PARALLEL_CHUNK_BONES)
                    .enumerate(),
            )
            .for_each(|(slot, (bi, bind_mat))| {
                let pal = resolver.matrix(bi, bind_mat);
                slot.copy_from_slice(bytemuck::cast_slice(&pal.to_cols_array()));
            });
    } else {
        for (bi, bind_mat) in params.skinning_bind_matrices.iter().enumerate() {
            let pal = resolver.matrix(bi, bind_mat);
            out.extend_from_slice(bytemuck::cast_slice(&pal.to_cols_array()));
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
        seed_scene_with_transforms(vec![identity_transform()], vec![-1])
    }

    fn seed_scene_with_transforms(
        nodes: Vec<RenderTransform>,
        parents: Vec<i32>,
    ) -> (SceneCoordinator, RenderSpaceId) {
        let mut scene = SceneCoordinator::new();
        let id = RenderSpaceId(1);
        scene.test_seed_space_identity_worlds(id, nodes, parents);
        (scene, id)
    }

    fn translation_bind_matrices(count: usize) -> Vec<Mat4> {
        (0..count)
            .map(|i| Mat4::from_translation(Vec3::new(i as f32, 0.5 * i as f32, -(i as f32))))
            .collect()
    }

    fn palette_for_one_identity_bound_bone(
        scene: &SceneCoordinator,
        space_id: RenderSpaceId,
        context: RenderingContext,
    ) -> Mat4 {
        build_skinning_palette(SkinningPaletteParams {
            scene,
            space_id,
            skinning_bind_matrices: &[Mat4::IDENTITY],
            has_skeleton: true,
            bone_transform_indices: &[0],
            smr_node_id: -1,
            render_context: context,
            head_output_transform: Mat4::IDENTITY,
        })
        .expect("palette")[0]
    }

    fn assert_palette_linear_columns(matrix: Mat4, expected_scale: Vec3) {
        assert!(
            matrix
                .x_axis
                .truncate()
                .abs_diff_eq(Vec3::new(expected_scale.x, 0.0, 0.0), 1e-6)
        );
        assert!(
            matrix
                .y_axis
                .truncate()
                .abs_diff_eq(Vec3::new(0.0, expected_scale.y, 0.0), 1e-6)
        );
        assert!(
            matrix
                .z_axis
                .truncate()
                .abs_diff_eq(Vec3::new(0.0, 0.0, expected_scale.z), 1e-6)
        );
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

    #[test]
    fn bone_palette_preserves_planar_zero_scale_bone() {
        let mut planar = identity_transform();
        planar.scale = Vec3::new(1.0, 0.0, 1.0);
        let (scene, space_id) = seed_scene_with_transforms(vec![planar], vec![-1]);

        let palette =
            palette_for_one_identity_bound_bone(&scene, space_id, RenderingContext::UserView);

        assert_palette_linear_columns(palette, planar.scale);
    }

    #[test]
    fn bone_palette_preserves_line_zero_scale_bone() {
        let mut line = identity_transform();
        line.scale = Vec3::new(1.0, 0.0, 0.0);
        let (scene, space_id) = seed_scene_with_transforms(vec![line], vec![-1]);

        let palette =
            palette_for_one_identity_bound_bone(&scene, space_id, RenderingContext::UserView);

        assert_palette_linear_columns(palette, line.scale);
    }

    #[test]
    fn bone_palette_preserves_point_zero_scale_bone() {
        let mut point = identity_transform();
        point.scale = Vec3::ZERO;
        let (scene, space_id) = seed_scene_with_transforms(vec![point], vec![-1]);

        let palette =
            palette_for_one_identity_bound_bone(&scene, space_id, RenderingContext::UserView);

        assert_palette_linear_columns(palette, point.scale);
    }

    #[test]
    fn bone_palette_zero_scale_override_is_context_local() {
        let (mut scene, space_id) = seed_scene_with_one_transform();
        scene.test_push_scale_render_transform_override(
            space_id,
            0,
            RenderingContext::UserView,
            Vec3::ZERO,
        );

        let user_palette =
            palette_for_one_identity_bound_bone(&scene, space_id, RenderingContext::UserView);
        let external_palette =
            palette_for_one_identity_bound_bone(&scene, space_id, RenderingContext::ExternalView);

        assert_palette_linear_columns(user_palette, Vec3::ZERO);
        assert_palette_linear_columns(external_palette, Vec3::ONE);
    }
}
