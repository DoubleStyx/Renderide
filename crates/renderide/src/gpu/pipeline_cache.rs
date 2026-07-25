//! Process-global persistent GPU pipeline cache.
//!
//! Seeded from disk at device init and written back at shutdown so compiled shader pipelines are
//! reused across launches instead of recompiled every session. Created only when the adapter
//! reports [`wgpu::Features::PIPELINE_CACHE`]; pipeline-creation sites pass [`pipeline_cache`] and
//! degrade to no cache when the feature is absent or the file is missing.

use std::path::PathBuf;
use std::sync::OnceLock;

/// Global slot set once by [`init_pipeline_cache`].
static PIPELINE_CACHE: OnceLock<CacheSlot> = OnceLock::new();

/// The live cache and the disk path it persists to.
struct CacheSlot {
    cache: wgpu::PipelineCache,
    path: PathBuf,
}

/// Cache handle for a pipeline descriptor's `cache` field, or [`None`] when unsupported.
#[inline]
pub(crate) fn pipeline_cache() -> Option<&'static wgpu::PipelineCache> {
    PIPELINE_CACHE.get().map(|slot| &slot.cache)
}

/// Creates the persistent pipeline cache for `device`, seeded from disk when a prior blob exists.
///
/// No-op when the device lacks [`wgpu::Features::PIPELINE_CACHE`], the adapter has no cache key, or
/// the cache was already initialized.
pub(crate) fn init_pipeline_cache(device: &wgpu::Device, adapter: &wgpu::Adapter) {
    if PIPELINE_CACHE.get().is_some()
        || !device.features().contains(wgpu::Features::PIPELINE_CACHE)
    {
        return;
    }
    let Some(key) = wgpu::util::pipeline_cache_key(&adapter.get_info()) else {
        return;
    };
    let path = cache_path(&key);
    let data = std::fs::read(&path).ok();
    // SAFETY: `data`, when present, was written by a prior `get_data` call on this crate's cache.
    // `fallback: true` makes wgpu ignore data that does not match the current adapter or driver.
    let cache = unsafe {
        device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
            label: Some("renderide_pipeline_cache"),
            data: data.as_deref(),
            fallback: true,
        })
    };
    let _ = PIPELINE_CACHE.set(CacheSlot { cache, path });
    logger::info!("pipeline cache initialized (seeded={})", data.is_some());
}

/// Writes the current pipeline cache blob to disk. Call once at shutdown.
pub(crate) fn persist_pipeline_cache() {
    let Some(slot) = PIPELINE_CACHE.get() else {
        return;
    };
    let Some(data) = slot.cache.get_data() else {
        return;
    };
    if let Some(parent) = slot.path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let temp = slot.path.with_extension("tmp");
    if std::fs::write(&temp, &data).is_ok() && std::fs::rename(&temp, &slot.path).is_err() {
        let _ = std::fs::remove_file(&temp);
    }
}

/// Per-adapter cache file under the platform local-data directory.
fn cache_path(key: &str) -> PathBuf {
    let base = std::env::var_os("LOCALAPPDATA")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    base.join("renderide").join("pipeline_cache").join(key)
}
