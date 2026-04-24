//! Shader permutations for variant-specific WGSL and raster pipeline keys.
//!
//! Production renderers compile **variant-specific** WGSL by baking `#ifdef`-style choices into the
//! source string (or templating) before [`wgpu::Device::create_shader_module`]. [`ShaderPermutation`]
//! selects those static features (e.g. multiview). Cached [`wgpu::RenderPipeline`] instances for
//! materials are owned by [`crate::materials::MaterialPipelineCache`] ([`crate::materials::cache`]),
//! keyed by [`crate::materials::MaterialPipelineCacheKey`] (permutation + surface format + layout).
//!
//! Longer-term permutation management (feature growth, lazy compile, pruning) is outlined in the
//! repository doc `docs/shader_permutation_strategy.md` (planning reference only).

pub mod raster;

pub use raster::SHADER_PERM_MULTIVIEW_STEREO;

/// Bit flags selecting static shader features (depth-only, alpha clip, multiview stereo, etc.).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ShaderPermutation(pub u32);

#[cfg(test)]
mod tests {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    use super::{ShaderPermutation, SHADER_PERM_MULTIVIEW_STEREO};

    /// Hashes `value` with the standard library's default hasher.
    fn hash_of<T: Hash>(value: &T) -> u64 {
        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    }

    /// `ShaderPermutation::default()` is the zero-bit permutation (the non-multiview baseline).
    #[test]
    fn default_is_zero_bits() {
        assert_eq!(ShaderPermutation::default(), ShaderPermutation(0));
    }

    /// Distinct permutation values must compare non-equal and hash distinctly so they do not
    /// collide in pipeline caches.
    #[test]
    fn distinct_permutations_are_non_equal_and_hash_distinctly() {
        let a = ShaderPermutation(0);
        let b = ShaderPermutation(1);
        assert_ne!(a, b);
        assert_ne!(hash_of(&a), hash_of(&b));
    }

    /// Guards the multiview feature bit layout other modules key off of.
    #[test]
    fn multiview_stereo_bit_is_one() {
        assert_eq!(SHADER_PERM_MULTIVIEW_STEREO, ShaderPermutation(1));
    }
}
