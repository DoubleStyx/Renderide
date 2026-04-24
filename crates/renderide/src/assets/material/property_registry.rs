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
/// name was already interned); callers that stream per-row request batches see every row observed.
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicI32, Ordering};

    #[test]
    fn empty_name_returns_zero() {
        let reg = PropertyIdRegistry::new();
        assert_eq!(reg.intern(""), 0);
    }

    #[test]
    fn first_nonempty_name_starts_at_one() {
        let reg = PropertyIdRegistry::new();
        assert_eq!(reg.intern("_MainTex"), 1);
    }

    #[test]
    fn repeated_name_returns_same_id() {
        let reg = PropertyIdRegistry::new();
        let first = reg.intern("_Color");
        let second = reg.intern("_Color");
        assert_eq!(first, second);
    }

    #[test]
    fn distinct_names_get_distinct_increasing_ids() {
        let reg = PropertyIdRegistry::new();
        let a = reg.intern("_A");
        let b = reg.intern("_B");
        let c = reg.intern("_C");
        assert_eq!((a, b, c), (1, 2, 3));
    }

    #[test]
    fn intern_for_host_request_fires_hook_each_time() {
        let reg = PropertyIdRegistry::new();
        let counter = Arc::new(AtomicI32::new(0));
        let last_id = Arc::new(AtomicI32::new(0));
        let counter_h = counter.clone();
        let last_id_h = last_id.clone();
        reg.add_semantic_hook(Arc::new(move |_name, id| {
            counter_h.fetch_add(1, Ordering::SeqCst);
            last_id_h.store(id, Ordering::SeqCst);
        }));

        let id1 = reg.intern_for_host_request("_MainTex");
        let id2 = reg.intern_for_host_request("_MainTex");
        assert_eq!(id1, id2);
        assert_eq!(counter.load(Ordering::SeqCst), 2);
        assert_eq!(last_id.load(Ordering::SeqCst), id1);
    }

    #[test]
    fn intern_for_host_request_empty_name_skips_hook() {
        let reg = PropertyIdRegistry::new();
        let counter = Arc::new(AtomicI32::new(0));
        let counter_h = counter.clone();
        reg.add_semantic_hook(Arc::new(move |_name, _id| {
            counter_h.fetch_add(1, Ordering::SeqCst);
        }));

        assert_eq!(reg.intern_for_host_request(""), 0);
        assert_eq!(counter.load(Ordering::SeqCst), 0);
    }
}
