//! Runtime reflection-probe cubemaps captured for OnChanges probes.

use std::sync::Arc;

use hashbrown::{HashMap, HashSet};

use crate::scene::RenderSpaceId;

/// Stable identity for one host reflection probe capture slot.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(crate) struct RuntimeReflectionProbeCaptureKey {
    /// Render space that owns the probe.
    pub(crate) space_id: RenderSpaceId,
    /// Dense reflection-probe renderable index inside the render space.
    pub(crate) renderable_index: i32,
}

/// Newly captured runtime cubemap for an OnChanges reflection probe.
pub(crate) struct RuntimeReflectionProbeCapture {
    /// Capture slot identity.
    pub(crate) key: RuntimeReflectionProbeCaptureKey,
    /// Monotonic renderer-side capture generation.
    pub(crate) generation: u64,
    /// Cubemap face edge in texels.
    pub(crate) face_size: u32,
    /// Number of mips allocated on the captured texture.
    pub(crate) mip_levels: u32,
    /// Captured texture kept alive with the source view.
    pub(crate) texture: Arc<wgpu::Texture>,
    /// Cube-dimension view over the captured texture.
    pub(crate) view: Arc<wgpu::TextureView>,
    /// 2D-array view over the captured texture.
    pub(crate) array_view: Arc<wgpu::TextureView>,
}

/// Latest captured source for one runtime OnChanges reflection probe.
#[derive(Clone)]
pub(crate) struct RuntimeReflectionProbeCaptureSource {
    /// Capture slot identity.
    pub(crate) key: RuntimeReflectionProbeCaptureKey,
    /// Monotonic renderer-side capture generation.
    pub(crate) generation: u64,
    /// Cubemap face edge in texels.
    pub(crate) face_size: u32,
    /// Number of mips allocated on the captured texture.
    pub(crate) mip_levels: u32,
    /// Captured texture kept alive with the source view.
    pub(crate) texture: Arc<wgpu::Texture>,
    /// Cube-dimension view over the captured texture.
    pub(crate) view: Arc<wgpu::TextureView>,
    /// 2D-array view over the captured texture.
    pub(crate) array_view: Arc<wgpu::TextureView>,
}

/// Latest runtime cubemap captures keyed by host probe identity.
#[derive(Default)]
pub(crate) struct RuntimeReflectionProbeCaptureStore {
    captures: HashMap<RuntimeReflectionProbeCaptureKey, RuntimeReflectionProbeCaptureSource>,
}

impl RuntimeReflectionProbeCaptureStore {
    /// Stores the latest capture for a probe, replacing any older generation.
    pub(crate) fn insert(&mut self, capture: RuntimeReflectionProbeCapture) {
        self.captures.insert(
            capture.key,
            RuntimeReflectionProbeCaptureSource {
                key: capture.key,
                generation: capture.generation,
                face_size: capture.face_size,
                mip_levels: capture.mip_levels,
                texture: capture.texture,
                view: capture.view,
                array_view: capture.array_view,
            },
        );
    }

    /// Returns the latest captured source for a probe.
    pub(crate) fn get(
        &self,
        key: RuntimeReflectionProbeCaptureKey,
    ) -> Option<&RuntimeReflectionProbeCaptureSource> {
        self.captures.get(&key)
    }

    /// Removes captures for probes that are no longer present as active OnChanges probes.
    pub(crate) fn retain_active(&mut self, active: &HashSet<RuntimeReflectionProbeCaptureKey>) {
        self.captures.retain(|key, _capture| active.contains(key));
    }

    /// Removes captures owned by closed render spaces.
    pub(crate) fn purge_spaces(&mut self, spaces: &HashSet<RenderSpaceId>) -> usize {
        let before = self.captures.len();
        self.captures
            .retain(|key, _capture| !spaces.contains(&key.space_id));
        before.saturating_sub(self.captures.len())
    }
}
