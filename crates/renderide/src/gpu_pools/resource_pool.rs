//! Generic resident GPU resource pool mechanics shared by every concrete asset pool.
//!
//! The algebra factored out:
//!
//! * [`GpuResourcePool<T, A>`] holds a `HashMap<i32, T>` plus a [`VramAccounting`] tally and an
//!   access policy `A: PoolResourceAccess`. Pool mechanics (insert / remove / get / iter /
//!   accounting deltas) live here once.
//! * [`PoolResourceAccess`] tags a pool with its [`VramResourceKind`] (which accounting bucket
//!   to charge) and an access-notification hook (for future streaming / eviction).
//! * [`StreamingAccess`] implements the trait for pools that carry a [`StreamingPolicy`]
//!   (mesh + the three host texture pools). It dispatches `note_access` to the per-kind hook on
//!   the policy.
//! * [`UntrackedAccess`] implements the trait for pools whose host drives evictions explicitly
//!   (render-target + video). The notify hook is a no-op.
//!
//! Two facade macros -- `impl_streaming_pool_facade!` and `impl_resident_pool_facade!` -- emit
//! the same canonical method vocabulary (`insert` / `remove` / `get` / `get_mut` / `iter` /
//! `len` / `is_empty` / `accounting` / `accounting_mut`) on each concrete pool newtype. The
//! streaming variant additionally emits `streaming_mut()`.

use hashbrown::HashMap;
use hashbrown::hash_map::Entry;

use super::{GpuResource, NoopStreamingPolicy, StreamingPolicy, VramAccounting, VramResourceKind};

/// Access policy invoked by [`GpuResourcePool`] when a resource is inserted, replaced, or
/// touched. The trait reports which accounting bucket the pool charges and dispatches access
/// notifications to whatever streaming or eviction policy the implementation carries.
pub(crate) trait PoolResourceAccess {
    /// VRAM accounting bucket charged by this access policy.
    fn kind(&self) -> VramResourceKind;

    /// Records an access for future streaming or eviction policies. Implementations without a
    /// streaming policy should treat this as a no-op.
    fn note_access(&mut self, asset_id: i32);
}

/// Streaming-aware access for a single resource kind. Used by mesh and the three host texture
/// pools -- every pool whose host drives priority hints rather than explicit deletes.
pub(crate) struct StreamingAccess {
    /// Accounting bucket: [`VramResourceKind::Mesh`] or [`VramResourceKind::Texture`].
    kind: VramResourceKind,
    /// Streaming / eviction policy receiving access notifications.
    streaming: Box<dyn StreamingPolicy>,
}

impl StreamingAccess {
    /// Mesh-pool access wired to `streaming`.
    pub(crate) fn mesh(streaming: Box<dyn StreamingPolicy>) -> Self {
        Self {
            kind: VramResourceKind::Mesh,
            streaming,
        }
    }

    /// Mesh-pool access using [`NoopStreamingPolicy`].
    pub(crate) fn mesh_noop() -> Self {
        Self::mesh(Box::new(NoopStreamingPolicy))
    }

    /// Texture-pool access wired to `streaming`.
    pub(crate) fn texture(streaming: Box<dyn StreamingPolicy>) -> Self {
        Self {
            kind: VramResourceKind::Texture,
            streaming,
        }
    }

    /// Texture-pool access using [`NoopStreamingPolicy`].
    pub(crate) fn texture_noop() -> Self {
        Self::texture(Box::new(NoopStreamingPolicy))
    }

    /// Mutable streaming policy for callers that record evictions or query suggestions.
    pub(crate) fn streaming_mut(&mut self) -> &mut dyn StreamingPolicy {
        self.streaming.as_mut()
    }
}

impl PoolResourceAccess for StreamingAccess {
    fn kind(&self) -> VramResourceKind {
        self.kind
    }

    fn note_access(&mut self, asset_id: i32) {
        match self.kind {
            VramResourceKind::Mesh => self.streaming.note_mesh_access(asset_id),
            VramResourceKind::Texture => self.streaming.note_texture_access(asset_id),
        }
    }
}

/// Access policy without streaming hooks -- render-target and video pools, whose host drives
/// evictions through explicit delete commands rather than through priority hints.
#[derive(Debug, Clone, Copy)]
pub(crate) struct UntrackedAccess {
    /// Accounting bucket the pool charges (textures or meshes).
    kind: VramResourceKind,
}

impl UntrackedAccess {
    /// Creates an untracked access policy charging into `kind`.
    pub(crate) fn new(kind: VramResourceKind) -> Self {
        Self { kind }
    }
}

impl PoolResourceAccess for UntrackedAccess {
    fn kind(&self) -> VramResourceKind {
        self.kind
    }

    fn note_access(&mut self, _asset_id: i32) {}
}

/// Common resident-resource table with VRAM accounting and a typed access policy.
#[derive(Debug)]
pub(crate) struct GpuResourcePool<T, A>
where
    T: GpuResource,
    A: PoolResourceAccess,
{
    /// Resident GPU resources keyed by host asset id.
    resources: HashMap<i32, T>,
    /// Running VRAM totals for entries in [`Self::resources`].
    accounting: VramAccounting,
    /// Type-specific access behavior for streaming and accounting bucket selection.
    access: A,
}

impl<T, A> GpuResourcePool<T, A>
where
    T: GpuResource,
    A: PoolResourceAccess,
{
    /// Creates an empty resident table using `access` for accounting and streaming hooks.
    pub(crate) fn new(access: A) -> Self {
        Self {
            resources: HashMap::new(),
            accounting: VramAccounting::default(),
            access,
        }
    }

    /// VRAM accounting totals for resident resources.
    pub(crate) fn accounting(&self) -> &VramAccounting {
        &self.accounting
    }

    /// Mutable VRAM accounting for explicit accounting adjustments.
    pub(crate) fn accounting_mut(&mut self) -> &mut VramAccounting {
        &mut self.accounting
    }

    /// Mutable access policy (used by streaming-enabled facades to expose `streaming_mut()`).
    pub(crate) fn access_mut(&mut self) -> &mut A {
        &mut self.access
    }

    /// Inserts or replaces a resident resource and returns whether an entry already existed.
    pub(crate) fn insert(&mut self, resource: T) -> bool {
        let id = resource.asset_id();
        let bytes = resource.resident_bytes();
        let kind = self.access.kind();
        let existed_before = match self.resources.entry(id) {
            Entry::Occupied(mut entry) => {
                let old = entry.insert(resource);
                self.accounting
                    .on_resident_removed(kind, old.resident_bytes());
                true
            }
            Entry::Vacant(entry) => {
                entry.insert(resource);
                false
            }
        };

        self.accounting.on_resident_added(kind, bytes);
        self.access.note_access(id);
        existed_before
    }

    /// Removes a resident resource by host asset id and returns whether it existed.
    pub(crate) fn remove(&mut self, asset_id: i32) -> bool {
        let Some(old) = self.resources.remove(&asset_id) else {
            return false;
        };
        self.accounting
            .on_resident_removed(self.access.kind(), old.resident_bytes());
        true
    }

    /// Borrows a resident resource by host asset id.
    #[inline]
    pub(crate) fn get(&self, asset_id: i32) -> Option<&T> {
        self.resources.get(&asset_id)
    }

    /// Mutably borrows a resident resource by host asset id.
    #[inline]
    pub(crate) fn get_mut(&mut self, asset_id: i32) -> Option<&mut T> {
        self.resources.get_mut(&asset_id)
    }

    /// Borrows all resident resources for iteration and diagnostics.
    #[inline]
    pub(crate) fn resources(&self) -> &HashMap<i32, T> {
        &self.resources
    }

    /// Number of resident resources.
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.resources.len()
    }

    /// Whether the pool has no resident resources.
    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.resources.is_empty()
    }

    /// Applies a byte-size delta after an in-place resource mutation.
    pub(crate) fn account_resident_delta(&mut self, before: u64, after: u64) {
        let kind = self.access.kind();
        if after > before {
            self.accounting.on_resident_added(kind, after - before);
        } else if before > after {
            self.accounting.on_resident_removed(kind, before - after);
        }
    }

    /// Records a resource access without changing residency.
    pub(crate) fn note_access(&mut self, asset_id: i32) {
        self.access.note_access(asset_id);
    }
}

/// Implements the unified resident-pool facade for a concrete pool whose `inner` is
/// `GpuResourcePool<R, StreamingAccess>`. Emits the canonical method vocabulary plus
/// `streaming_mut()` and the `new(streaming) / default_pool()` pair.
///
/// The `$access_with` and `$access_noop` arms select the per-kind constructors on
/// [`StreamingAccess`] (mesh vs texture).
macro_rules! impl_streaming_pool_facade {
    ($pool:ty, $resource:ty, $access_with:expr, $access_noop:expr $(,)?) => {
        impl $pool {
            /// Creates an empty pool with the given streaming policy.
            pub fn new(streaming: Box<dyn $crate::gpu_pools::StreamingPolicy>) -> Self {
                let access_with: fn(
                    Box<dyn $crate::gpu_pools::StreamingPolicy>,
                ) -> $crate::gpu_pools::resource_pool::StreamingAccess = $access_with;
                Self {
                    inner: $crate::gpu_pools::resource_pool::GpuResourcePool::new(access_with(
                        streaming,
                    )),
                }
            }

            /// Default pool with [`crate::gpu_pools::NoopStreamingPolicy`].
            pub fn default_pool() -> Self {
                let access_noop: fn() -> $crate::gpu_pools::resource_pool::StreamingAccess =
                    $access_noop;
                Self {
                    inner: $crate::gpu_pools::resource_pool::GpuResourcePool::new(access_noop()),
                }
            }

            /// VRAM accounting for resident resources.
            #[inline]
            pub fn accounting(&self) -> &$crate::gpu_pools::VramAccounting {
                self.inner.accounting()
            }

            /// Mutable VRAM totals (insert/remove update accounting).
            #[inline]
            pub fn accounting_mut(&mut self) -> &mut $crate::gpu_pools::VramAccounting {
                self.inner.accounting_mut()
            }

            /// Streaming policy hook for eviction suggestions.
            #[inline]
            pub fn streaming_mut(&mut self) -> &mut dyn $crate::gpu_pools::StreamingPolicy {
                self.inner.access_mut().streaming_mut()
            }

            /// Inserts or replaces a resource. Returns `true` if an entry was replaced.
            #[inline]
            pub fn insert(&mut self, resource: $resource) -> bool {
                self.inner.insert(resource)
            }

            /// Removes a resource by host asset id; returns `true` if it was present.
            #[inline]
            pub fn remove(&mut self, asset_id: i32) -> bool {
                self.inner.remove(asset_id)
            }

            /// Borrows a resident resource by host asset id.
            #[inline]
            pub fn get(&self, asset_id: i32) -> Option<&$resource> {
                self.inner.get(asset_id)
            }

            /// Mutably borrows a resident resource (uploads, property updates).
            #[inline]
            pub fn get_mut(&mut self, asset_id: i32) -> Option<&mut $resource> {
                self.inner.get_mut(asset_id)
            }

            /// Iterates resident resources for iteration and HUD stats.
            #[inline]
            pub fn iter(&self) -> impl Iterator<Item = &$resource> {
                self.inner.resources().values()
            }

            /// Borrows the resident map for callers that need keyed access (HUD lookups).
            #[inline]
            pub fn as_map(&self) -> &hashbrown::HashMap<i32, $resource> {
                self.inner.resources()
            }

            /// Number of resident entries.
            #[inline]
            pub fn len(&self) -> usize {
                self.inner.len()
            }

            /// Whether the pool has no resident entries.
            #[inline]
            pub fn is_empty(&self) -> bool {
                self.inner.is_empty()
            }
        }
    };
}

/// Implements the unified resident-pool facade for a concrete pool whose `inner` is
/// `GpuResourcePool<R, UntrackedAccess>`. Same vocabulary as the streaming variant minus
/// `streaming_mut()`, and `new()` takes no arguments.
macro_rules! impl_resident_pool_facade {
    ($pool:ty, $resource:ty, $kind:expr $(,)?) => {
        impl $pool {
            /// Creates an empty pool.
            pub fn new() -> Self {
                Self {
                    inner: $crate::gpu_pools::resource_pool::GpuResourcePool::new(
                        $crate::gpu_pools::resource_pool::UntrackedAccess::new($kind),
                    ),
                }
            }

            /// VRAM accounting for resident resources.
            #[inline]
            pub fn accounting(&self) -> &$crate::gpu_pools::VramAccounting {
                self.inner.accounting()
            }

            /// Mutable VRAM totals (insert/remove update accounting).
            #[inline]
            pub fn accounting_mut(&mut self) -> &mut $crate::gpu_pools::VramAccounting {
                self.inner.accounting_mut()
            }

            /// Inserts or replaces a resource. Returns `true` if an entry was replaced.
            #[inline]
            pub fn insert(&mut self, resource: $resource) -> bool {
                self.inner.insert(resource)
            }

            /// Removes a resource by host asset id; returns `true` if it was present.
            #[inline]
            pub fn remove(&mut self, asset_id: i32) -> bool {
                self.inner.remove(asset_id)
            }

            /// Borrows a resident resource by host asset id.
            #[inline]
            pub fn get(&self, asset_id: i32) -> Option<&$resource> {
                self.inner.get(asset_id)
            }

            /// Mutably borrows a resident resource (uploads, property updates).
            #[inline]
            pub fn get_mut(&mut self, asset_id: i32) -> Option<&mut $resource> {
                self.inner.get_mut(asset_id)
            }

            /// Iterates resident resources for iteration and HUD stats.
            #[inline]
            pub fn iter(&self) -> impl Iterator<Item = &$resource> {
                self.inner.resources().values()
            }

            /// Borrows the resident map for callers that need keyed access (HUD lookups).
            #[inline]
            pub fn as_map(&self) -> &hashbrown::HashMap<i32, $resource> {
                self.inner.resources()
            }

            /// Number of resident entries.
            #[inline]
            pub fn len(&self) -> usize {
                self.inner.len()
            }

            /// Whether the pool has no resident entries.
            #[inline]
            pub fn is_empty(&self) -> bool {
                self.inner.is_empty()
            }
        }

        impl Default for $pool {
            fn default() -> Self {
                Self::new()
            }
        }
    };
}

pub(crate) use impl_resident_pool_facade;
pub(crate) use impl_streaming_pool_facade;

#[cfg(test)]
mod tests {
    //! Unit tests for generic resident GPU resource pool mechanics.

    use super::{GpuResourcePool, PoolResourceAccess, UntrackedAccess};
    use crate::gpu_pools::{GpuResource, VramResourceKind};

    /// Fake resident resource used to test generic pool behavior without GPU handles.
    #[derive(Debug)]
    struct TestResource {
        /// Host asset id.
        asset_id: i32,
        /// Resident byte count.
        resident_bytes: u64,
    }

    impl TestResource {
        /// Creates a fake resident resource.
        fn new(asset_id: i32, resident_bytes: u64) -> Self {
            Self {
                asset_id,
                resident_bytes,
            }
        }
    }

    impl GpuResource for TestResource {
        fn resident_bytes(&self) -> u64 {
            self.resident_bytes
        }

        fn asset_id(&self) -> i32 {
            self.asset_id
        }
    }

    /// Tracking access policy that records observed asset ids.
    #[derive(Debug, Default)]
    struct RecordingAccess {
        /// Asset ids observed through insert/touch hooks.
        touched: Vec<i32>,
    }

    impl PoolResourceAccess for RecordingAccess {
        fn kind(&self) -> VramResourceKind {
            VramResourceKind::Texture
        }

        fn note_access(&mut self, asset_id: i32) {
            self.touched.push(asset_id);
        }
    }

    /// Creates an empty pool with a recording access policy.
    fn recording_pool() -> GpuResourcePool<TestResource, RecordingAccess> {
        GpuResourcePool::new(RecordingAccess::default())
    }

    /// Creates an empty pool with the texture-tagged untracked access policy.
    fn texture_pool() -> GpuResourcePool<TestResource, UntrackedAccess> {
        GpuResourcePool::new(UntrackedAccess::new(VramResourceKind::Texture))
    }

    /// Insert adds bytes and records an access through the policy.
    #[test]
    fn insert_tracks_accounting_and_access() {
        let mut pool = recording_pool();

        assert!(!pool.insert(TestResource::new(7, 128)));

        assert_eq!(pool.accounting().texture_resident_bytes(), 128);
        assert_eq!(pool.accounting().total_resident_bytes(), 128);
        assert_eq!(pool.access_mut().touched.as_slice(), &[7]);
    }

    /// Replacement subtracts the old resource before adding the new one.
    #[test]
    fn replacement_rebalances_accounting() {
        let mut pool = texture_pool();
        assert!(!pool.insert(TestResource::new(7, 128)));
        assert!(pool.insert(TestResource::new(7, 64)));

        assert_eq!(pool.accounting().texture_resident_bytes(), 64);
        assert_eq!(pool.accounting().total_resident_bytes(), 64);
    }

    /// Removal subtracts bytes and reports whether an entry existed.
    #[test]
    fn remove_updates_accounting_and_reports_presence() {
        let mut pool = texture_pool();
        assert!(!pool.insert(TestResource::new(7, 128)));

        assert!(pool.remove(7));
        assert!(!pool.remove(7));

        assert_eq!(pool.accounting().texture_resident_bytes(), 0);
        assert_eq!(pool.accounting().total_resident_bytes(), 0);
    }

    /// Length, emptiness, and map access reflect resident entries.
    #[test]
    fn resident_map_access_reflects_entries() {
        let mut pool = texture_pool();
        assert!(pool.is_empty());

        assert!(!pool.insert(TestResource::new(7, 128)));

        assert_eq!(pool.len(), 1);
        assert!(!pool.is_empty());
        assert!(pool.get(7).is_some());
        assert!(pool.resources().contains_key(&7));
    }

    /// Explicit byte deltas update accounting after in-place resource mutation.
    #[test]
    fn resident_delta_adjusts_accounting() {
        let mut pool = texture_pool();
        assert!(!pool.insert(TestResource::new(7, 128)));

        pool.account_resident_delta(128, 192);
        assert_eq!(pool.accounting().texture_resident_bytes(), 192);

        pool.account_resident_delta(192, 64);
        assert_eq!(pool.accounting().texture_resident_bytes(), 64);
    }

    /// Mesh-tagged untracked access routes bytes into the mesh subtotal.
    #[test]
    fn mesh_tagged_untracked_pool_routes_into_mesh_subtotal() {
        let mut pool: GpuResourcePool<TestResource, UntrackedAccess> =
            GpuResourcePool::new(UntrackedAccess::new(VramResourceKind::Mesh));
        assert!(!pool.insert(TestResource::new(1, 256)));
        assert_eq!(pool.accounting().mesh_resident_bytes(), 256);
        assert_eq!(pool.accounting().texture_resident_bytes(), 0);
        assert_eq!(pool.accounting().total_resident_bytes(), 256);
    }
}
