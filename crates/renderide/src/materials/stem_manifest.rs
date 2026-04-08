//! Build-time [`crate::embedded_shaders::SHADER_MANIFEST_JSON`] → Unity logical name → target WGSL stem.

use std::collections::HashMap;
use std::sync::OnceLock;

use serde::Deserialize;

use crate::assets::util::normalize_unity_shader_lookup_key;

/// One material entry from `shaders/target/manifest.json`.
#[derive(Debug, Deserialize)]
pub struct ShaderManifestMaterialEntry {
    /// File stem under `shaders/target/<stem>.wgsl`.
    pub stem: String,
    /// Unity `Shader "…"` keys (normalized at lookup time).
    #[serde(default)]
    pub unity_names: Vec<String>,
}

/// Parsed shader manifest embedded at build time.
#[derive(Debug, Deserialize)]
pub struct ShaderManifest {
    pub materials: Vec<ShaderManifestMaterialEntry>,
    #[serde(default)]
    pub globals_module: Option<String>,
}

impl ShaderManifest {
    /// Parses [`crate::embedded_shaders::SHADER_MANIFEST_JSON`].
    pub fn from_embedded_json() -> Self {
        serde_json::from_str(crate::embedded_shaders::SHADER_MANIFEST_JSON)
            .expect("SHADER_MANIFEST_JSON must match ShaderManifest schema")
    }

    /// Maps normalized Unity shader keys to target stems (first manifest match wins).
    pub fn unity_name_to_stem_map(manifest: &Self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        for entry in &manifest.materials {
            for name in &entry.unity_names {
                let key = normalize_unity_shader_lookup_key(name);
                m.entry(key).or_insert_with(|| entry.stem.clone());
            }
        }
        m
    }
}

/// Resolves a Unity-style logical shader name to a composed WGSL stem (`shaders/target/<stem>.wgsl`).
#[derive(Debug)]
pub struct StemResolver {
    unity_to_stem: HashMap<String, String>,
}

impl StemResolver {
    /// Builds the lookup table from the embedded manifest.
    pub fn from_embedded_manifest() -> Self {
        let manifest = ShaderManifest::from_embedded_json();
        Self {
            unity_to_stem: ShaderManifest::unity_name_to_stem_map(&manifest),
        }
    }

    /// Resolves a host-provided Unity shader name to a target stem, if listed in the manifest.
    pub fn stem_for_unity_name(&self, unity_shader_name: &str) -> Option<&str> {
        let key = normalize_unity_shader_lookup_key(unity_shader_name);
        self.unity_to_stem.get(&key).map(String::as_str)
    }
}

/// Returns the composed WGSL stem for `name` when it appears in the embedded manifest (used for routing).
///
/// Uses a process-wide [`StemResolver`] built from [`ShaderManifest::from_embedded_json`].
pub fn manifest_stem_for_unity_name(name: &str) -> Option<String> {
    static RESOLVER: OnceLock<StemResolver> = OnceLock::new();
    RESOLVER
        .get_or_init(StemResolver::from_embedded_manifest)
        .stem_for_unity_name(name)
        .map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_manifest_parses() {
        let m = ShaderManifest::from_embedded_json();
        assert!(!m.materials.is_empty());
    }

    #[test]
    fn manifest_stem_resolves_unlit() {
        let s = super::manifest_stem_for_unity_name("Unlit").expect("manifest lists Unlit");
        assert!(
            s.starts_with("world_unlit"),
            "expected world_unlit stem, got {s}"
        );
    }
}
