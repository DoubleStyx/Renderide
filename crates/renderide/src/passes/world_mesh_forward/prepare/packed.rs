//! Output retained after packing one view's forward draws.

use std::sync::Arc;

use crate::world_mesh::{InstancePlan, WorldMeshDrawList};

use super::super::MaterialBatchPacket;

pub(super) struct PackedForwardDraws {
    pub(super) draws: WorldMeshDrawList,
    pub(super) plan: Arc<InstancePlan>,
    pub(super) overlay_view_proj: glam::Mat4,
    pub(super) precomputed_batches: Vec<MaterialBatchPacket>,
}
