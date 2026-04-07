//! Generic asset manager for storing assets by handle.
//!
//! Extension point for asset loading, texture/mesh.

use std::collections::HashMap;

use super::Asset;
use super::AssetId;

/// Manages a collection of assets of type `T` keyed by `AssetId`.
/// Mirrors Unity's per-type asset managers (e.g. MaterialAssetManager).
pub struct AssetManager<T: Asset> {
    assets: HashMap<AssetId, T>,
}

impl<T: Asset> AssetManager<T> {
    /// Creates a new empty manager.
    pub fn new() -> Self {
        Self {
            assets: HashMap::new(),
        }
    }

    /// Returns a reference to the asset with the given id, if present.
    pub fn get(&self, id: AssetId) -> Option<&T> {
        self.assets.get(&id)
    }

    /// Returns a mutable reference to the asset with the given id, if present.
    pub fn get_mut(&mut self, id: AssetId) -> Option<&mut T> {
        self.assets.get_mut(&id)
    }

    /// Inserts an asset, using its `id()` as the key.
    /// Returns the previous asset if one existed.
    pub fn insert(&mut self, asset: T) -> Option<T> {
        self.assets.insert(asset.id(), asset)
    }

    /// Removes the asset with the given id.
    /// Returns the removed asset if present.
    pub fn remove(&mut self, id: AssetId) -> Option<T> {
        self.assets.remove(&id)
    }

    /// Returns the number of assets in this manager.
    pub fn len(&self) -> usize {
        self.assets.len()
    }

    /// Iterates over all stored assets (hash map iteration order is undefined).
    pub fn values(&self) -> impl Iterator<Item = &T> + '_ {
        self.assets.values()
    }

    /// Returns true if no assets are stored.
    pub fn is_empty(&self) -> bool {
        self.assets.is_empty()
    }

    /// Returns true if an asset with the given id exists.
    pub fn contains_key(&self, id: AssetId) -> bool {
        self.assets.contains_key(&id)
    }
}

impl<T: Asset> Default for AssetManager<T> {
    fn default() -> Self {
        Self::new()
    }
}
