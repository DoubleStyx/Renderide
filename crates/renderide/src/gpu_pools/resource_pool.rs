//! Generic resident GPU resource pool mechanics shared by concrete asset pools.

use hashbrown::HashMap;
use hashbrown::hash_map::Entry;

use super::{GpuResource, NoopStreamingPolicy, StreamingPolicy, VramAccounting, VramResourceKind};

/// Type-specific behavior invoked by [`GpuResourcePool`] when a resource is inserted or touched.
pub(crate) trait PoolResourceAccess {
    /// VRAM accounting bucket used for this resource family.
    const RESOURCE_KIND: VramResourceKind;

    /// Records an access for future streaming or eviction policies.
    fn note_access(&mut self, asset_id: i32);
}

/// Common resident-resource table with VRAM accounting and type-specific access hooks.
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
    /// Type-specific access behavior for streaming and accounting.
    access: A,
}

impl<T, A> GpuResourcePool<T, A>
where
    T: GpuResource,
    A: PoolResourceAccess,
{
    /// Creates an empty resident table using `access` for type-specific behavior.
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

    /// Type-specific access policy.
    pub(crate) fn access_mut(&mut self) -> &mut A {
        &mut self.access
    }

    /// Inserts or replaces a resident resource and returns whether an entry already existed.
    pub(crate) fn insert(&mut self, resource: T) -> bool {
        let id = resource.asset_id();
        let bytes = resource.resident_bytes();
        let existed_before = match self.resources.entry(id) {
            Entry::Occupied(mut entry) => {
                let old = entry.insert(resource);
                self.accounting
                    .on_resident_removed(A::RESOURCE_KIND, old.resident_bytes());
                true
            }
            Entry::Vacant(entry) => {
                entry.insert(resource);
                false
            }
        };

        self.accounting.on_resident_added(A::RESOURCE_KIND, bytes);
        self.access.note_access(id);
        existed_before
    }

    /// Removes a resident resource by host asset id and returns whether it existed.
    pub(crate) fn remove(&mut self, asset_id: i32) -> bool {
        let Some(old) = self.resources.remove(&asset_id) else {
            return false;
        };
        self.accounting
            .on_resident_removed(A::RESOURCE_KIND, old.resident_bytes());
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
        if after > before {
            self.accounting
                .on_resident_added(A::RESOURCE_KIND, after - before);
        } else if before > after {
            self.accounting
                .on_resident_removed(A::RESOURCE_KIND, before - after);
        }
    }

    /// Records a resource access without changing residency.
    pub(crate) fn note_access(&mut self, asset_id: i32) {
        self.access.note_access(asset_id);
    }
}

/// Mesh-specific access behavior backed by a streaming policy.
pub(crate) struct MeshPoolAccess {
    /// Streaming and eviction hooks for mesh residency.
    streaming: Box<dyn StreamingPolicy>,
}

impl MeshPoolAccess {
    /// Creates mesh access behavior with `streaming`.
    pub(crate) fn new(streaming: Box<dyn StreamingPolicy>) -> Self {
        Self { streaming }
    }

    /// Creates mesh access behavior with [`NoopStreamingPolicy`].
    pub(crate) fn noop() -> Self {
        Self::new(Box::new(NoopStreamingPolicy))
    }

    /// Mutable streaming policy hook for mesh residency.
    pub(crate) fn streaming_mut(&mut self) -> &mut dyn StreamingPolicy {
        self.streaming.as_mut()
    }
}

impl PoolResourceAccess for MeshPoolAccess {
    const RESOURCE_KIND: VramResourceKind = VramResourceKind::Mesh;

    fn note_access(&mut self, asset_id: i32) {
        self.streaming.note_mesh_access(asset_id);
    }
}

/// Texture-specific access behavior backed by a streaming policy.
pub(crate) struct TexturePoolAccess {
    /// Streaming and eviction hooks for texture residency.
    streaming: Box<dyn StreamingPolicy>,
}

impl TexturePoolAccess {
    /// Creates texture access behavior with `streaming`.
    pub(crate) fn new(streaming: Box<dyn StreamingPolicy>) -> Self {
        Self { streaming }
    }

    /// Creates texture access behavior with [`NoopStreamingPolicy`].
    pub(crate) fn noop() -> Self {
        Self::new(Box::new(NoopStreamingPolicy))
    }

    /// Mutable streaming policy hook for texture residency.
    pub(crate) fn streaming_mut(&mut self) -> &mut dyn StreamingPolicy {
        self.streaming.as_mut()
    }
}

impl PoolResourceAccess for TexturePoolAccess {
    const RESOURCE_KIND: VramResourceKind = VramResourceKind::Texture;

    fn note_access(&mut self, asset_id: i32) {
        self.streaming.note_texture_access(asset_id);
    }
}

/// Implements the common resident texture-pool facade over [`GpuResourcePool`].
macro_rules! impl_texture_pool_facade {
    ($pool:ty, $resource:ty) => {
        impl $pool {
            /// Creates an empty pool with the given streaming policy.
            pub fn new(streaming: Box<dyn $crate::gpu_pools::StreamingPolicy>) -> Self {
                Self {
                    inner: $crate::gpu_pools::resource_pool::GpuResourcePool::new(
                        $crate::gpu_pools::resource_pool::TexturePoolAccess::new(streaming),
                    ),
                }
            }

            /// Default pool with [`crate::gpu_pools::NoopStreamingPolicy`].
            pub fn default_pool() -> Self {
                Self {
                    inner: $crate::gpu_pools::resource_pool::GpuResourcePool::new(
                        $crate::gpu_pools::resource_pool::TexturePoolAccess::noop(),
                    ),
                }
            }

            /// VRAM accounting for resident textures.
            pub fn accounting(&self) -> &$crate::gpu_pools::VramAccounting {
                self.inner.accounting()
            }

            /// Mutable VRAM totals (insert/remove update accounting).
            pub fn accounting_mut(&mut self) -> &mut $crate::gpu_pools::VramAccounting {
                self.inner.accounting_mut()
            }

            /// Streaming policy for mip eviction suggestions.
            pub fn streaming_mut(&mut self) -> &mut dyn $crate::gpu_pools::StreamingPolicy {
                self.inner.access_mut().streaming_mut()
            }

            /// Inserts or replaces a texture. Returns `true` if a previous entry was replaced.
            pub fn insert_texture(&mut self, tex: $resource) -> bool {
                self.inner.insert(tex)
            }

            /// Removes a texture by host id; returns `true` if it was present.
            pub fn remove_texture(&mut self, asset_id: i32) -> bool {
                self.inner.remove(asset_id)
            }

            /// Borrows a resident texture by host asset id.
            #[inline]
            pub fn get_texture(&self, asset_id: i32) -> Option<&$resource> {
                self.inner.get(asset_id)
            }

            /// Mutably borrows a resident texture (mip uploads, property changes).
            #[inline]
            pub fn get_texture_mut(&mut self, asset_id: i32) -> Option<&mut $resource> {
                self.inner.get_mut(asset_id)
            }

            /// Full map for iteration and HUD stats.
            #[inline]
            pub fn textures(&self) -> &hashbrown::HashMap<i32, $resource> {
                self.inner.resources()
            }

            /// Number of resident texture entries in the pool.
            #[inline]
            pub fn resident_texture_count(&self) -> usize {
                self.inner.len()
            }
        }
    };
}

pub(crate) use impl_texture_pool_facade;

/// Render-texture access behavior without streaming hooks.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct RenderTexturePoolAccess;

impl PoolResourceAccess for RenderTexturePoolAccess {
    const RESOURCE_KIND: VramResourceKind = VramResourceKind::Texture;

    fn note_access(&mut self, _asset_id: i32) {}
}

#[cfg(test)]
mod tests {
    //! Unit tests for generic resident GPU resource pool mechanics.

    use super::{GpuResourcePool, PoolResourceAccess};
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

    /// Fake access behavior that records touched asset ids.
    #[derive(Debug, Default)]
    struct TestAccess {
        /// Asset ids observed through insert/touch hooks.
        touched: Vec<i32>,
    }

    impl PoolResourceAccess for TestAccess {
        const RESOURCE_KIND: VramResourceKind = VramResourceKind::Texture;

        fn note_access(&mut self, asset_id: i32) {
            self.touched.push(asset_id);
        }
    }

    /// Creates an empty fake resource pool.
    fn test_pool() -> GpuResourcePool<TestResource, TestAccess> {
        GpuResourcePool::new(TestAccess::default())
    }

    /// Insert adds bytes and records an access.
    #[test]
    fn insert_tracks_accounting_and_access() {
        let mut pool = test_pool();

        assert!(!pool.insert(TestResource::new(7, 128)));

        assert_eq!(pool.accounting().texture_resident_bytes(), 128);
        assert_eq!(pool.accounting().total_resident_bytes(), 128);
        assert_eq!(pool.access_mut().touched.as_slice(), &[7]);
    }

    /// Replacement subtracts the old resource before adding the new one.
    #[test]
    fn replacement_rebalances_accounting() {
        let mut pool = test_pool();
        assert!(!pool.insert(TestResource::new(7, 128)));
        assert!(pool.insert(TestResource::new(7, 64)));

        assert_eq!(pool.accounting().texture_resident_bytes(), 64);
        assert_eq!(pool.accounting().total_resident_bytes(), 64);
    }

    /// Removal subtracts bytes and reports whether an entry existed.
    #[test]
    fn remove_updates_accounting_and_reports_presence() {
        let mut pool = test_pool();
        assert!(!pool.insert(TestResource::new(7, 128)));

        assert!(pool.remove(7));
        assert!(!pool.remove(7));

        assert_eq!(pool.accounting().texture_resident_bytes(), 0);
        assert_eq!(pool.accounting().total_resident_bytes(), 0);
    }

    /// Length, emptiness, and map access reflect resident entries.
    #[test]
    fn resident_map_access_reflects_entries() {
        let mut pool = test_pool();
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
        let mut pool = test_pool();
        assert!(!pool.insert(TestResource::new(7, 128)));

        pool.account_resident_delta(128, 192);
        assert_eq!(pool.accounting().texture_resident_bytes(), 192);

        pool.account_resident_delta(192, 64);
        assert_eq!(pool.accounting().texture_resident_bytes(), 64);
    }
}
