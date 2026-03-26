//! Frame counters for [`crate::assets::material_update_batch`] wire opcodes that are otherwise dropped
//! or only cursor-advanced.
//!
//! Enabled with [`crate::config::RenderConfig::material_batch_wire_metrics`]; read via
//! [`MaterialBatchWireFrameMetrics::snapshot_and_reset`] when building [`crate::diagnostics::LiveFrameDiagnostics`]
//! (wired from the main frame loop when diagnostics sampling runs).

use std::sync::atomic::{AtomicU64, Ordering};

static SET_FLOAT4X4: AtomicU64 = AtomicU64::new(0);
static SET_FLOAT_ARRAY: AtomicU64 = AtomicU64::new(0);
static SET_FLOAT4_ARRAY: AtomicU64 = AtomicU64::new(0);

/// Per-frame counts of material batch opcodes observed on the wire (last frame).
#[derive(Clone, Copy, Debug, Default)]
pub struct MaterialBatchWireFrameMetrics {
    /// `MaterialPropertyUpdateType::set_float4x4` records seen.
    pub set_float4x4: u64,
    /// `set_float_array` records seen.
    pub set_float_array: u64,
    /// `set_float4_array` records seen.
    pub set_float4_array: u64,
}

impl MaterialBatchWireFrameMetrics {
    /// Reads all counters and clears them for the next frame.
    pub fn snapshot_and_reset() -> Self {
        Self {
            set_float4x4: SET_FLOAT4X4.swap(0, Ordering::Relaxed),
            set_float_array: SET_FLOAT_ARRAY.swap(0, Ordering::Relaxed),
            set_float4_array: SET_FLOAT4_ARRAY.swap(0, Ordering::Relaxed),
        }
    }
}

/// Records one wire-metric event when [`crate::config::RenderConfig::material_batch_wire_metrics`] is on.
pub(crate) fn record_material_batch_wire(metrics: bool, kind: MaterialBatchWireKind) {
    if !metrics {
        return;
    }
    let ctr = match kind {
        MaterialBatchWireKind::SetFloat4x4 => &SET_FLOAT4X4,
        MaterialBatchWireKind::SetFloatArray => &SET_FLOAT_ARRAY,
        MaterialBatchWireKind::SetFloat4Array => &SET_FLOAT4_ARRAY,
    };
    ctr.fetch_add(1, Ordering::Relaxed);
}

/// Opcode group for wire metrics.
#[derive(Clone, Copy, Debug)]
pub(crate) enum MaterialBatchWireKind {
    SetFloat4x4,
    SetFloatArray,
    SetFloat4Array,
}
