//! Shared GPU-resource lifecycle primitives.
//!
//! This module owns small reusable mechanics for renderer resource reuse: keyed caches,
//! one-shot GPU-resource slots, sharded LRUs, and common cache counters. It intentionally does
//! not own subsystem-specific invalidation, device-limit validation, upload fencing, or residency
//! policy.

mod cache;
mod once;
mod sharded_lru;
mod stats;
mod texture_view;

pub(crate) use cache::{BindGroupMap, RenderPipelineMap};
pub(crate) use once::OnceGpu;
pub(crate) use sharded_lru::ShardedLru;
pub(crate) use stats::{AtomicCacheCounters, CacheCounters, CacheStats};
pub(crate) use texture_view::{TextureViewCache, TextureViewDescriptorKey};
