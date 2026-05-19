//! Central shader/material asset graph state for routing, source generations, and future compiler hooks.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use hashbrown::HashMap;
use parking_lot::Mutex;

use crate::materials::embedded::stem_metadata::embedded_composed_stem_for_permutation;
use crate::materials::{RasterPipelineKind, ShaderPermutation};

use super::router::MaterialRouter;

/// Generation assigned to a shader source before any development reload has occurred.
const INITIAL_SHADER_SOURCE_GENERATION: u64 = 1;

/// Key for one composed embedded shader source variant in the material asset graph.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct EmbeddedShaderSourceKey {
    /// Base embedded stem requested by routing, such as `unlit_default`.
    pub(crate) base_stem: String,
    /// Shader permutation used to select the composed WGSL target.
    pub(crate) permutation: ShaderPermutation,
}

impl EmbeddedShaderSourceKey {
    /// Builds a source key from a base embedded material stem and shader permutation.
    pub(crate) fn new(base_stem: impl Into<String>, permutation: ShaderPermutation) -> Self {
        Self {
            base_stem: base_stem.into(),
            permutation,
        }
    }

    /// Returns the composed target stem used to find WGSL source and metadata.
    pub(crate) fn composed_stem(&self) -> String {
        embedded_composed_stem_for_permutation(&self.base_stem, self.permutation)
    }
}

/// Source generation and optional development override for one composed embedded shader target.
#[derive(Clone, Debug)]
struct EmbeddedShaderSourceNode {
    /// Composed target stem, such as `unlit_multiview`.
    composed_stem: String,
    /// Monotonic generation included in material pipeline cache keys.
    generation: u64,
    /// Last modification time observed for development hot reload.
    last_modified: Option<SystemTime>,
    /// Runtime WGSL override loaded from `shaders/target` when development reload is enabled.
    source_override: Option<Arc<str>>,
}

impl EmbeddedShaderSourceNode {
    /// Creates an embedded source node for a composed target.
    fn new(composed_stem: String) -> Self {
        Self {
            composed_stem,
            generation: INITIAL_SHADER_SOURCE_GENERATION,
            last_modified: None,
            source_override: None,
        }
    }

    /// Bumps the source generation and replaces the development source override.
    fn reload(&mut self, modified: SystemTime, source: Arc<str>) {
        self.generation = self
            .generation
            .wrapping_add(1)
            .max(INITIAL_SHADER_SOURCE_GENERATION);
        self.last_modified = Some(modified);
        self.source_override = Some(source);
    }
}

/// Host shader asset route tracked by the material asset graph.
#[derive(Clone, Debug)]
struct ShaderAssetNode {
    /// Resolved raster pipeline kind for this host shader asset.
    pipeline: RasterPipelineKind,
    /// Shader asset filename extracted from the uploaded shader asset, when available.
    shader_asset_name: Option<String>,
    /// Shader-specific variant bits parsed from the uploaded shader asset, when available.
    shader_variant_bits: Option<u32>,
    /// Monotonic route generation for this shader asset node.
    generation: u64,
}

impl ShaderAssetNode {
    /// Builds a shader asset graph node from resolved route data.
    fn new(
        pipeline: RasterPipelineKind,
        shader_asset_name: Option<String>,
        shader_variant_bits: Option<u32>,
        generation: u64,
    ) -> Self {
        Self {
            pipeline,
            shader_asset_name,
            shader_variant_bits,
            generation,
        }
    }
}

/// Scalar shape for a future global shader uniform entry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GlobalUniformValueType {
    /// Single `f32` value.
    Float,
    /// Four-lane `f32` vector.
    Vec4,
    /// Single `u32` value.
    Uint,
    /// Four-by-four `f32` matrix.
    Mat4,
}

/// Registered global shader uniform dependency hook.
#[derive(Clone, Debug)]
struct GlobalUniformNode {
    /// Stable uniform name used by shader/compiler metadata.
    name: String,
    /// Uniform value type.
    value_type: GlobalUniformValueType,
    /// Monotonic generation for data or declaration changes.
    generation: u64,
}

impl GlobalUniformNode {
    /// Builds a global uniform node.
    fn new(name: impl Into<String>, value_type: GlobalUniformValueType, generation: u64) -> Self {
        Self {
            name: name.into(),
            value_type,
            generation,
        }
    }
}

/// Source lookup result used by the pipeline cache.
#[derive(Clone, Debug)]
pub(crate) struct MaterialShaderSourceSnapshot {
    /// Source generation included in pipeline cache keys.
    pub(crate) generation: u64,
    /// Optional WGSL override loaded from local shader targets for development reload.
    pub(crate) source_override: Option<Arc<str>>,
}

/// Development hot reload result for one polling pass.
#[derive(Clone, Debug, Default)]
pub(crate) struct MaterialShaderHotReloadReport {
    /// Composed stems whose source generation changed.
    pub(crate) reloaded_stems: Vec<String>,
    /// Recoverable filesystem or source read errors observed while polling.
    pub(crate) errors: Vec<String>,
}

impl MaterialShaderHotReloadReport {
    /// Returns `true` when no source changed and no errors were reported.
    pub(crate) fn is_empty(&self) -> bool {
        self.reloaded_stems.is_empty() && self.errors.is_empty()
    }
}

/// Plain-data diagnostic snapshot of the material shader graph.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct MaterialShaderGraphDiagnosticSnapshot {
    /// Host shader asset nodes currently registered.
    pub(crate) shader_nodes: usize,
    /// Embedded WGSL source nodes touched by routing, warmup, or pipeline lookup.
    pub(crate) embedded_source_nodes: usize,
    /// Registered global shader uniform hooks.
    pub(crate) global_uniforms: usize,
    /// Registered shader routes that resolve to embedded material stems.
    pub(crate) embedded_shader_routes: usize,
    /// Registered shader routes carrying shader variant bits.
    pub(crate) shader_variant_routes: usize,
    /// Total bytes across shader asset names retained by graph nodes.
    pub(crate) shader_asset_name_bytes: usize,
    /// Wrapping sum of shader route generations.
    pub(crate) shader_route_generation_sum: u64,
    /// Total bytes across registered global uniform names.
    pub(crate) global_uniform_name_bytes: usize,
    /// Bit mask of registered global uniform value types.
    pub(crate) global_uniform_type_mask: u32,
    /// Wrapping sum of global uniform generations.
    pub(crate) global_uniform_generation_sum: u64,
    /// Total route/source/global invalidations observed by the graph.
    pub(crate) invalidations: u64,
    /// Development WGSL source invalidations observed by the graph.
    pub(crate) source_invalidations: u64,
    /// Whether development WGSL hot reload is enabled.
    pub(crate) dev_hot_reload_enabled: bool,
    /// Last successful development reload composed stem, when any.
    pub(crate) last_dev_reload_stem: Option<String>,
    /// Last development reload error, when any.
    pub(crate) last_dev_reload_error: Option<String>,
}

/// Mutable source-side graph state protected for read-only pipeline lookups.
#[derive(Debug)]
struct ShaderSourceGraphState {
    /// Embedded source nodes keyed by base stem plus permutation.
    embedded_sources: HashMap<EmbeddedShaderSourceKey, EmbeddedShaderSourceNode>,
    /// Development reload target directory, usually `crates/renderide/shaders/target`.
    dev_hot_reload_target_dir: PathBuf,
    /// Whether development reload polling is active.
    dev_hot_reload_enabled: bool,
    /// Number of source invalidations observed.
    source_invalidations: u64,
    /// Last successfully reloaded composed stem.
    last_dev_reload_stem: Option<String>,
    /// Last development reload error.
    last_dev_reload_error: Option<String>,
}

impl ShaderSourceGraphState {
    /// Builds empty source state for the default shader target directory.
    fn new() -> Self {
        Self {
            embedded_sources: HashMap::new(),
            dev_hot_reload_target_dir: default_shader_target_dir(),
            dev_hot_reload_enabled: false,
            source_invalidations: 0,
            last_dev_reload_stem: None,
            last_dev_reload_error: None,
        }
    }

    /// Returns the node for `key`, inserting the initial generation when first touched.
    fn node_for_key(&mut self, key: EmbeddedShaderSourceKey) -> &mut EmbeddedShaderSourceNode {
        self.embedded_sources
            .entry(key)
            .or_insert_with_key(|key| EmbeddedShaderSourceNode::new(key.composed_stem()))
    }

    /// Returns a shader source snapshot for a key, inserting the source node when needed.
    fn source_snapshot_for_key(
        &mut self,
        key: EmbeddedShaderSourceKey,
    ) -> MaterialShaderSourceSnapshot {
        let node = self.node_for_key(key);
        MaterialShaderSourceSnapshot {
            generation: node.generation,
            source_override: node.source_override.clone(),
        }
    }

    /// Polls active source nodes for development WGSL file changes.
    fn poll_dev_hot_reload(&mut self) -> MaterialShaderHotReloadReport {
        let mut report = MaterialShaderHotReloadReport::default();
        if !self.dev_hot_reload_enabled {
            return report;
        }

        let target_dir = self.dev_hot_reload_target_dir.clone();
        for node in self.embedded_sources.values_mut() {
            let path = target_dir.join(format!("{}.wgsl", node.composed_stem));
            let Ok(metadata) = std::fs::metadata(&path) else {
                continue;
            };
            let Ok(modified) = metadata.modified() else {
                continue;
            };
            if node.last_modified.is_some_and(|seen| seen >= modified) {
                continue;
            }
            match std::fs::read_to_string(&path) {
                Ok(source) => {
                    node.reload(modified, Arc::from(source.into_boxed_str()));
                    self.source_invalidations = self.source_invalidations.saturating_add(1);
                    self.last_dev_reload_stem = Some(node.composed_stem.clone());
                    self.last_dev_reload_error = None;
                    report.reloaded_stems.push(node.composed_stem.clone());
                }
                Err(error) => {
                    let message = format!("{}: {error}", path.display());
                    self.last_dev_reload_error = Some(message.clone());
                    report.errors.push(message);
                }
            }
        }
        report
    }
}

/// Central graph for material shader assets, source generations, and future compiler dependencies.
#[derive(Debug)]
pub(crate) struct MaterialAssetGraph {
    /// Host shader router used by draw collection and material binding.
    router: MaterialRouter,
    /// Host shader asset nodes keyed by host asset id.
    shader_nodes: HashMap<i32, ShaderAssetNode>,
    /// Development source and source-generation state.
    source_state: Mutex<ShaderSourceGraphState>,
    /// Registered global uniform hooks keyed by uniform name.
    global_uniforms: HashMap<String, GlobalUniformNode>,
    /// Route/global invalidation counter.
    invalidations: u64,
    /// Next node generation for shader and global nodes.
    next_generation: u64,
}

impl MaterialAssetGraph {
    /// Creates an empty material asset graph with a fallback raster route.
    pub(crate) fn new(fallback: RasterPipelineKind) -> Self {
        Self {
            router: MaterialRouter::new(fallback),
            shader_nodes: HashMap::new(),
            source_state: Mutex::new(ShaderSourceGraphState::new()),
            global_uniforms: HashMap::new(),
            invalidations: 0,
            next_generation: INITIAL_SHADER_SOURCE_GENERATION,
        }
    }

    /// Returns the shader router used by world-mesh material resolution.
    pub(crate) fn router(&self) -> &MaterialRouter {
        &self.router
    }

    /// Registers or replaces a host shader route.
    pub(crate) fn register_shader_route(
        &mut self,
        shader_asset_id: i32,
        pipeline: RasterPipelineKind,
        shader_asset_name: Option<String>,
        shader_variant_bits: Option<u32>,
    ) {
        self.router.set_shader_route(
            shader_asset_id,
            pipeline.clone(),
            shader_asset_name.clone(),
            shader_variant_bits,
        );
        match &pipeline {
            RasterPipelineKind::EmbeddedStem(stem) => {
                self.router
                    .set_shader_stem(shader_asset_id, stem.to_string());
                self.ensure_source_node(stem.as_ref(), ShaderPermutation::default());
            }
            RasterPipelineKind::Null => {
                self.router.remove_shader_stem(shader_asset_id);
            }
        }
        let generation = self.bump_generation();
        self.shader_nodes.insert(
            shader_asset_id,
            ShaderAssetNode::new(pipeline, shader_asset_name, shader_variant_bits, generation),
        );
        self.note_invalidation();
    }

    /// Removes a host shader route.
    pub(crate) fn unregister_shader_route(&mut self, shader_asset_id: i32) {
        self.router.remove_shader_route(shader_asset_id);
        if self.shader_nodes.remove(&shader_asset_id).is_some() {
            self.note_invalidation();
        }
    }

    /// Returns sorted shader routes for diagnostics.
    pub(crate) fn shader_routes_for_hud(
        &self,
    ) -> Vec<(i32, RasterPipelineKind, Option<String>, Option<u32>)> {
        self.router.routes_sorted_for_hud()
    }

    /// Returns the composed stem recorded for a host shader asset.
    pub(crate) fn stem_for_shader_asset(&self, shader_asset_id: i32) -> Option<&str> {
        self.router.stem_for_shader_asset(shader_asset_id)
    }

    /// Returns the shader variant bits recorded for a host shader asset.
    pub(crate) fn variant_bits_for_shader_asset(&self, shader_asset_id: i32) -> Option<u32> {
        self.router.variant_bits_for_shader_asset(shader_asset_id)
    }

    /// Returns a shader source snapshot for a raster kind and shader permutation.
    pub(crate) fn shader_source_snapshot(
        &self,
        kind: &RasterPipelineKind,
        permutation: ShaderPermutation,
    ) -> MaterialShaderSourceSnapshot {
        let base_stem = match kind {
            RasterPipelineKind::EmbeddedStem(stem) => stem.as_ref(),
            RasterPipelineKind::Null => "null_default",
        };
        self.ensure_source_node(base_stem, permutation)
    }

    /// Enables or disables development WGSL hot reload polling.
    pub(crate) fn set_dev_hot_reload_enabled(&mut self, enabled: bool) {
        let mut state = self.source_state.lock();
        if state.dev_hot_reload_enabled != enabled {
            state.dev_hot_reload_enabled = enabled;
            drop(state);
            self.note_invalidation();
        }
    }

    /// Polls local WGSL targets for development reload changes.
    pub(crate) fn poll_dev_hot_reload(&mut self) -> MaterialShaderHotReloadReport {
        let mut state = self.source_state.lock();
        let report = state.poll_dev_hot_reload();
        let changed = !report.reloaded_stems.is_empty();
        drop(state);
        if changed {
            self.router.bump_generation_for_shader_dependency();
            self.note_invalidation();
        }
        report
    }

    /// Registers a typed global shader uniform dependency hook.
    pub(crate) fn register_global_uniform(
        &mut self,
        name: impl Into<String>,
        value_type: GlobalUniformValueType,
    ) {
        let name = name.into();
        let generation = self.bump_generation();
        self.global_uniforms.insert(
            name.clone(),
            GlobalUniformNode::new(name, value_type, generation),
        );
        self.note_invalidation();
    }

    /// Captures graph diagnostics.
    pub(crate) fn diagnostic_snapshot(&self) -> MaterialShaderGraphDiagnosticSnapshot {
        let state = self.source_state.lock();
        let mut embedded_shader_routes = 0usize;
        let mut shader_variant_routes = 0usize;
        let mut shader_asset_name_bytes = 0usize;
        let mut shader_route_generation_sum = 0u64;
        for node in self.shader_nodes.values() {
            if matches!(node.pipeline, RasterPipelineKind::EmbeddedStem(_)) {
                embedded_shader_routes = embedded_shader_routes.saturating_add(1);
            }
            if node.shader_variant_bits.is_some() {
                shader_variant_routes = shader_variant_routes.saturating_add(1);
            }
            shader_asset_name_bytes = shader_asset_name_bytes
                .saturating_add(node.shader_asset_name.as_ref().map_or(0, String::len));
            shader_route_generation_sum = shader_route_generation_sum.wrapping_add(node.generation);
        }
        let mut global_uniform_name_bytes = 0usize;
        let mut global_uniform_type_mask = 0u32;
        let mut global_uniform_generation_sum = 0u64;
        for node in self.global_uniforms.values() {
            global_uniform_name_bytes = global_uniform_name_bytes.saturating_add(node.name.len());
            global_uniform_type_mask |= global_uniform_type_bit(node.value_type);
            global_uniform_generation_sum =
                global_uniform_generation_sum.wrapping_add(node.generation);
        }
        MaterialShaderGraphDiagnosticSnapshot {
            shader_nodes: self.shader_nodes.len(),
            embedded_source_nodes: state.embedded_sources.len(),
            global_uniforms: self.global_uniforms.len(),
            embedded_shader_routes,
            shader_variant_routes,
            shader_asset_name_bytes,
            shader_route_generation_sum,
            global_uniform_name_bytes,
            global_uniform_type_mask,
            global_uniform_generation_sum,
            invalidations: self.invalidations,
            source_invalidations: state.source_invalidations,
            dev_hot_reload_enabled: state.dev_hot_reload_enabled,
            last_dev_reload_stem: state.last_dev_reload_stem.clone(),
            last_dev_reload_error: state.last_dev_reload_error.clone(),
        }
    }

    /// Ensures a source node exists and returns its current source snapshot.
    fn ensure_source_node(
        &self,
        base_stem: &str,
        permutation: ShaderPermutation,
    ) -> MaterialShaderSourceSnapshot {
        self.source_state
            .lock()
            .source_snapshot_for_key(EmbeddedShaderSourceKey::new(base_stem, permutation))
    }

    /// Bumps and returns the next graph node generation.
    fn bump_generation(&mut self) -> u64 {
        self.next_generation = self.next_generation.wrapping_add(1);
        self.next_generation
    }

    /// Records a material graph invalidation.
    fn note_invalidation(&mut self) {
        self.invalidations = self.invalidations.saturating_add(1);
    }

    /// Overrides the development hot reload target directory in tests.
    #[cfg(test)]
    pub(crate) fn set_dev_hot_reload_target_dir_for_tests(&self, dir: impl Into<PathBuf>) {
        self.source_state.lock().dev_hot_reload_target_dir = dir.into();
    }
}

/// Returns a compact bit for one global uniform value type.
fn global_uniform_type_bit(value_type: GlobalUniformValueType) -> u32 {
    match value_type {
        GlobalUniformValueType::Float => 1 << 0,
        GlobalUniformValueType::Vec4 => 1 << 1,
        GlobalUniformValueType::Uint => 1 << 2,
        GlobalUniformValueType::Mat4 => 1 << 3,
    }
}

/// Returns the default shader target directory for development hot reload.
fn default_shader_target_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("shaders/target")
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    use super::{GlobalUniformValueType, MaterialAssetGraph};
    use crate::materials::{RasterPipelineKind, ShaderPermutation};

    #[test]
    fn register_shader_route_tracks_router_and_graph_node() {
        let mut graph = MaterialAssetGraph::new(RasterPipelineKind::Null);
        let route = RasterPipelineKind::EmbeddedStem(Arc::from("unlit_default"));

        graph.register_shader_route(7, route.clone(), Some("unlit".to_string()), Some(0x20));

        assert_eq!(graph.router().pipeline_for_shader_asset(7), route);
        assert_eq!(graph.variant_bits_for_shader_asset(7), Some(0x20));
        assert_eq!(graph.diagnostic_snapshot().shader_nodes, 1);
        assert_eq!(graph.diagnostic_snapshot().embedded_source_nodes, 1);
    }

    #[test]
    fn global_uniform_registration_tracks_dependency_hook() {
        let mut graph = MaterialAssetGraph::new(RasterPipelineKind::Null);

        graph.register_global_uniform("Renderide_TestFloat", GlobalUniformValueType::Float);

        assert_eq!(graph.diagnostic_snapshot().global_uniforms, 1);
        assert_eq!(graph.diagnostic_snapshot().invalidations, 1);
    }

    #[test]
    fn dev_hot_reload_loads_changed_target_source() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("unlit_default.wgsl");
        fs::write(&path, "fn first() {}\n").expect("write target");

        let mut graph = MaterialAssetGraph::new(RasterPipelineKind::Null);
        graph.set_dev_hot_reload_target_dir_for_tests(dir.path());
        graph.set_dev_hot_reload_enabled(true);
        let kind = RasterPipelineKind::EmbeddedStem(Arc::from("unlit_default"));
        let before = graph.shader_source_snapshot(&kind, ShaderPermutation::default());

        let report = graph.poll_dev_hot_reload();
        let after_first = graph.shader_source_snapshot(&kind, ShaderPermutation::default());
        thread::sleep(Duration::from_millis(2));
        fs::write(&path, "fn second() {}\n").expect("rewrite target");
        let report_second = graph.poll_dev_hot_reload();
        let after_second = graph.shader_source_snapshot(&kind, ShaderPermutation::default());

        assert!(report.errors.is_empty());
        assert_eq!(report.reloaded_stems, vec!["unlit_default".to_string()]);
        assert!(report_second.errors.is_empty());
        assert_eq!(
            report_second.reloaded_stems,
            vec!["unlit_default".to_string()]
        );
        assert_ne!(before.generation, after_first.generation);
        assert_ne!(after_first.generation, after_second.generation);
        assert_eq!(
            after_second.source_override.as_deref(),
            Some("fn second() {}\n")
        );
    }
}
