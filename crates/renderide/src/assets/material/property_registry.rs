//! Process-stable shader property ids for [`crate::shared::MaterialPropertyIdRequest`].
//!
//! Unity’s Renderite path uses `Shader.PropertyToID`. This renderer assigns opaque integers; the
//! host must use the returned [`crate::shared::MaterialPropertyIdResult`] values in subsequent
//! [`crate::shared::MaterialsUpdateBatch`] records.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Callback invoked when the host resolves a material property name (see [`PropertyIdRegistry`]).
pub type MaterialPropertySemanticHook = Arc<dyn Fn(&str, i32) + Send + Sync>;

/// Intern table and optional name→semantics hooks (e.g. mapping `_MainTex` to a material family’s slot).
///
/// Hooks are invoked on every host property-id **request** for a non-empty name (including when the
/// name was already interned), matching the “apply each request row” behavior of the legacy renderer.
pub struct PropertyIdRegistry {
    inner: Mutex<PropertyIdRegistryInner>,
}

struct PropertyIdRegistryInner {
    next_id: i32,
    names: HashMap<String, i32>,
    semantic_hooks: Vec<MaterialPropertySemanticHook>,
}

impl PropertyIdRegistry {
    /// Builds a registry starting at property id `1` (id `0` means “no property” / empty name).
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(PropertyIdRegistryInner {
                next_id: 1,
                names: HashMap::new(),
                semantic_hooks: Vec::new(),
            }),
        }
    }

    /// Registers a callback invoked for every name in each [`crate::shared::MaterialPropertyIdRequest`].
    pub fn add_semantic_hook(&self, hook: MaterialPropertySemanticHook) {
        let mut g = match self.inner.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        g.semantic_hooks.push(hook);
    }

    /// Returns the stable id for `name`, allocating on first sight.
    pub fn intern(&self, name: &str) -> i32 {
        if name.is_empty() {
            return 0;
        }
        let mut g = match self.inner.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        if let Some(&id) = g.names.get(name) {
            return id;
        }
        let id = g.next_id;
        g.next_id = g.next_id.saturating_add(1).max(1);
        g.names.insert(name.to_string(), id);
        id
    }

    /// Interns then runs semantic hooks (use from `MaterialPropertyIdRequest` handling).
    pub fn intern_for_host_request(&self, name: &str) -> i32 {
        let id = self.intern(name);
        if name.is_empty() {
            return id;
        }
        let hooks: Vec<MaterialPropertySemanticHook> = {
            let g = match self.inner.lock() {
                Ok(g) => g,
                Err(poisoned) => poisoned.into_inner(),
            };
            g.semantic_hooks.clone()
        };
        for h in hooks {
            h(name, id);
        }
        id
    }
}

impl Default for PropertyIdRegistry {
    fn default() -> Self {
        Self::new()
    }
}
