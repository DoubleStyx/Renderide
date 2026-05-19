//! Topological ordering (Kahn) and import-root pass culling.

use hashbrown::HashMap;
use std::collections::HashSet;

use super::super::error::GraphBuildError;
use super::decl::SetupEntry;

pub(super) fn topo_sort(
    n: usize,
    edges: &HashSet<(usize, usize)>,
) -> Result<(Vec<usize>, Vec<usize>), GraphBuildError> {
    let mut in_degree = vec![0usize; n];
    let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(from, to) in edges {
        if from != to {
            neighbors[from].push(to);
            in_degree[to] += 1;
        }
    }
    for row in &mut neighbors {
        row.sort_unstable();
        row.dedup();
    }

    let mut current: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
    let mut sorted = Vec::with_capacity(n);
    let mut wave_by_node = vec![0usize; n];
    let mut wave_idx = 0usize;
    while !current.is_empty() {
        current.sort_unstable();
        let mut next = Vec::new();
        for node in current {
            wave_by_node[node] = wave_idx;
            sorted.push(node);
            for &neighbor in &neighbors[node] {
                in_degree[neighbor] = in_degree[neighbor].saturating_sub(1);
                if in_degree[neighbor] == 0 {
                    next.push(neighbor);
                }
            }
        }
        current = next;
        wave_idx += 1;
    }
    if sorted.len() != n {
        return Err(GraphBuildError::CycleDetected);
    }
    Ok((sorted, wave_by_node))
}

pub(super) fn retained_passes(
    n: usize,
    edges: &HashSet<(usize, usize)>,
    setups: &[SetupEntry],
) -> HashSet<usize> {
    let mut reverse: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(from, to) in edges {
        reverse[to].push(from);
    }

    let mut keep = HashSet::new();
    let mut stack = Vec::new();
    for (idx, entry) in setups.iter().enumerate() {
        let writes_import = entry
            .setup
            .accesses
            .iter()
            .any(|access| access.resource.is_imported() && access.writes());
        if writes_import || entry.setup.cull_exempt {
            keep.insert(idx);
            stack.push(idx);
        }
    }

    while let Some(idx) = stack.pop() {
        for &pred in &reverse[idx] {
            if keep.insert(pred) {
                stack.push(pred);
            }
        }
    }
    keep
}

pub(super) fn retained_ordinals(ordered: &[usize]) -> HashMap<usize, usize> {
    ordered
        .iter()
        .copied()
        .enumerate()
        .map(|(ordinal, original)| (original, ordinal))
        .collect()
}
