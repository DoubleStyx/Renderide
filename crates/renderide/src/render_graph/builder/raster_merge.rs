//! Raster render-pass merge planning for compiled graph schedules.

use super::super::compiled::CompiledPassInfo;
use super::super::pass::{PassMergeHint, PassWorkloadFlags, RenderPassTemplate};
use super::super::schedule::{RenderPassMergeGroup, ScheduleStep};

/// Finds adjacent raster passes whose attachment templates are merge-compatible.
pub(super) fn plan_render_pass_merge_groups(
    steps: &[ScheduleStep],
    pass_info: &[CompiledPassInfo],
) -> Vec<RenderPassMergeGroup> {
    let mut groups = Vec::new();
    let mut start = 0usize;
    while start < steps.len() {
        let mut end = start + 1;
        while end < steps.len()
            && render_passes_are_merge_compatible(
                pass_info.get(steps[end - 1].pass_idx),
                pass_info.get(steps[end].pass_idx),
            )
        {
            end += 1;
        }
        if end - start > 1 {
            groups.push(RenderPassMergeGroup {
                start_step: start,
                end_step: end,
            });
        }
        start = end;
    }
    groups
}

/// Returns whether two compiled pass infos can share a merge group.
fn render_passes_are_merge_compatible(
    first: Option<&CompiledPassInfo>,
    second: Option<&CompiledPassInfo>,
) -> bool {
    let Some(first) = first else {
        return false;
    };
    let Some(second) = second else {
        return false;
    };
    if first
        .workload_flags
        .contains(PassWorkloadFlags::NEVER_MERGE)
        || second
            .workload_flags
            .contains(PassWorkloadFlags::NEVER_MERGE)
    {
        return false;
    }
    let Some(first_template) = &first.raster_template else {
        return false;
    };
    let Some(second_template) = &second.raster_template else {
        return false;
    };
    if !merge_hints_allow_group(first.merge_hint, second.merge_hint) {
        return false;
    }
    render_templates_are_merge_compatible(first_template, second_template)
}

/// Returns whether adjacent pass merge hints are compatible with one merge group.
fn merge_hints_allow_group(first: PassMergeHint, second: PassMergeHint) -> bool {
    first == second || first.attachment_reuse || second.attachment_reuse
}

/// Returns whether two raster templates target the same attachments.
fn render_templates_are_merge_compatible(
    first: &RenderPassTemplate,
    second: &RenderPassTemplate,
) -> bool {
    if first.multiview_mask != second.multiview_mask
        || first.color_attachments.len() != second.color_attachments.len()
    {
        return false;
    }
    if !first
        .color_attachments
        .iter()
        .zip(&second.color_attachments)
        .all(|(a, b)| a.target == b.target && a.resolve_to == b.resolve_to)
    {
        return false;
    }
    let first_depth = first
        .depth_stencil_attachment
        .as_ref()
        .map(|depth| depth.target);
    let second_depth = second
        .depth_stencil_attachment
        .as_ref()
        .map(|depth| depth.target);
    first_depth == second_depth
}
