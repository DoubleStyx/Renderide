//! Reusable per-window scratch buffers used while building one [`super::InstancePlan`].

use hashbrown::HashMap;
use std::ops::Range;

use crate::world_mesh::draw_prep::WorldMeshDrawItem;

/// Within-window key for grouping draws that share `batch_key` (already adjacent after sort)
/// by mesh and submesh. Cheap to hash because `batch_key` is implicit (constant within the
/// caller's window).
#[derive(Hash, Eq, PartialEq, Clone, Copy)]
pub(super) struct MeshSubmeshKey {
    mesh_asset_id: i32,
    first_index: u32,
    index_count: u32,
}

/// Builds the grouping key for one draw item.
pub(super) fn mesh_submesh_key(item: &WorldMeshDrawItem) -> MeshSubmeshKey {
    MeshSubmeshKey {
        mesh_asset_id: item.mesh_asset_id,
        first_index: item.first_index,
        index_count: item.index_count,
    }
}

/// Reusable temporary storage for grouping one batch-key window.
#[derive(Default)]
pub(super) struct InstancePlanScratch {
    /// Map from mesh/submesh key to compact group index.
    window_groups: HashMap<MeshSubmeshKey, usize>,
    /// Member count per compact group.
    group_counts: Vec<usize>,
    /// Prefix-sum offsets into [`Self::group_members`].
    group_offsets: Vec<usize>,
    /// Mutable write cursors while filling [`Self::group_members`].
    group_write_offsets: Vec<usize>,
    /// Flat draw-index storage for every group in the current window.
    group_members: Vec<usize>,
    /// Representative sorted draw index per compact group.
    group_representative: Vec<usize>,
}

impl InstancePlanScratch {
    /// Rebuilds all scratch buffers for the supplied window.
    pub(super) fn rebuild(&mut self, draws: &[WorldMeshDrawItem], range: Range<usize>) {
        self.clear_window();
        self.count_groups(draws, range.clone());
        self.build_offsets();
        self.fill_members(draws, range);
    }

    /// Number of groups produced for the most recent window.
    pub(super) fn group_count(&self) -> usize {
        self.group_counts.len()
    }

    /// Member draw indices for the `idx`-th group of the most recent window.
    pub(super) fn group_members(&self, idx: usize) -> &[usize] {
        let start = self.group_offsets[idx];
        let end = self.group_offsets[idx + 1];
        &self.group_members[start..end]
    }

    /// Representative draw index for the `idx`-th group of the most recent window.
    pub(super) fn group_representative(&self, idx: usize) -> usize {
        self.group_representative[idx]
    }

    /// Clears previous-window scratch without releasing capacity.
    fn clear_window(&mut self) {
        self.window_groups.clear();
        self.group_counts.clear();
        self.group_offsets.clear();
        self.group_write_offsets.clear();
        self.group_members.clear();
        self.group_representative.clear();
    }

    /// Counts each mesh/submesh group in first-seen order.
    fn count_groups(&mut self, draws: &[WorldMeshDrawItem], range: Range<usize>) {
        for (offset, item) in draws[range.clone()].iter().enumerate() {
            let draw_idx = range.start + offset;
            let mk = mesh_submesh_key(item);
            if let Some(&group_idx) = self.window_groups.get(&mk) {
                self.group_counts[group_idx] += 1;
            } else {
                let group_idx = self.group_counts.len();
                self.window_groups.insert(mk, group_idx);
                self.group_representative.push(draw_idx);
                self.group_counts.push(1);
            }
        }
    }

    /// Builds prefix offsets and resets write cursors for the current group counts.
    fn build_offsets(&mut self) {
        self.group_offsets.reserve(self.group_counts.len() + 1);
        self.group_offsets.push(0);
        let mut next_offset = 0usize;
        for &count in &self.group_counts {
            next_offset += count;
            self.group_offsets.push(next_offset);
        }
        self.group_members.resize(next_offset, 0);
        self.group_write_offsets
            .extend_from_slice(&self.group_offsets[..self.group_counts.len()]);
    }

    /// Fills the flat member buffer using the offsets computed by [`Self::build_offsets`].
    fn fill_members(&mut self, draws: &[WorldMeshDrawItem], range: Range<usize>) {
        for (offset, item) in draws[range.clone()].iter().enumerate() {
            let Some(&group_idx) = self.window_groups.get(&mesh_submesh_key(item)) else {
                continue;
            };
            let write = self.group_write_offsets[group_idx];
            self.group_members[write] = range.start + offset;
            self.group_write_offsets[group_idx] += 1;
        }
    }
}
