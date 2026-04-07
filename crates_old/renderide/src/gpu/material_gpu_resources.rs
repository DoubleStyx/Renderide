//! Per-material GPU resource table: cached bind groups for native UI and world-unlit draws.
//!
//! Invalidation follows host texture lifecycle ([`crate::gpu::GpuState::drop_texture2d`]); cache keys
//! include resolved texture asset ids and whether GPU views were available at bind creation (see
//! [`NativeUiMaterialBindCache`]).
//!
//! This type is a **newtype** over [`NativeUiMaterialBindCache`] so call sites can migrate to a
//! stable name without changing bind order or shader layouts.

use std::ops::{Deref, DerefMut};

use super::native_ui_bind_cache::NativeUiMaterialBindCache;

/// Cached bind groups and writers for material-driven fragment bindings (UI / world unlit).
#[derive(Default)]
pub struct MaterialGpuResources(pub NativeUiMaterialBindCache);

impl Deref for MaterialGpuResources {
    type Target = NativeUiMaterialBindCache;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for MaterialGpuResources {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
