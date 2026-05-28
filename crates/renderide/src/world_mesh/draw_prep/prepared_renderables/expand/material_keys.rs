//! Material/property key collection for prepared renderable snapshots.

use hashbrown::HashSet;

use crate::world_mesh::draw_prep::collect::prepared::prepared_draws_share_renderer;

use super::super::{FramePreparedDraw, FramePreparedRun};

const MATERIAL_KEY_SIGNATURE_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const MATERIAL_KEY_SIGNATURE_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Signature for an empty prepared material live set.
#[inline]
pub(in crate::world_mesh::draw_prep) const fn empty_material_key_signature() -> u64 {
    MATERIAL_KEY_SIGNATURE_OFFSET
}

#[inline]
fn mix_material_key_signature(
    mut signature: u64,
    material_asset_id: i32,
    property_block_id: Option<i32>,
) -> u64 {
    let material_bits = material_asset_id as i64 as u64;
    let property_bits = property_block_id.map_or(0x9e37_79b9_7f4a_7c15, |id| id as i64 as u64);
    for part in [
        material_bits,
        property_bits,
        material_bits.rotate_left(17) ^ property_bits.rotate_right(11),
    ] {
        signature ^= part;
        signature = signature.wrapping_mul(MATERIAL_KEY_SIGNATURE_PRIME);
    }
    signature
}

/// Walks `draws` once and refreshes renderer-run ranges plus unique material/property keys.
///
/// Runs are detected post-build instead of plumbed through the parallel expansion so the
/// multi-space worker output can be merged with `Vec::append` without per-space offset adjustment.
///
/// Returns a deterministic signature of the first-seen unique material/property live set so
/// downstream caches can prove that an unchanged material generation also has unchanged
/// membership.
pub(in crate::world_mesh::draw_prep) fn populate_runs_and_material_keys(
    draws: &[FramePreparedDraw],
    runs: &mut Vec<FramePreparedRun>,
    material_property_keys: &mut Vec<(i32, Option<i32>)>,
    seen: &mut HashSet<(i32, Option<i32>)>,
) -> u64 {
    profiling::scope!("mesh::prepared_renderables::populate_run_starts");
    runs.clear();
    material_property_keys.clear();
    seen.clear();
    if draws.is_empty() {
        return empty_material_key_signature();
    }
    let mut signature = empty_material_key_signature();
    let mut run_start = 0usize;
    let mut prev = &draws[0];
    for (idx, d) in draws.iter().enumerate() {
        let key = (d.material_asset_id, d.property_block_id);
        if seen.insert(key) {
            material_property_keys.push(key);
            signature =
                mix_material_key_signature(signature, d.material_asset_id, d.property_block_id);
        }
        if idx > 0 && !prepared_draws_share_renderer(prev, d) {
            runs.push(FramePreparedRun {
                start: run_start as u32,
                end: idx as u32,
            });
            run_start = idx;
        }
        prev = d;
    }
    runs.push(FramePreparedRun {
        start: run_start as u32,
        end: draws.len() as u32,
    });
    signature ^ (material_property_keys.len() as u64)
}
