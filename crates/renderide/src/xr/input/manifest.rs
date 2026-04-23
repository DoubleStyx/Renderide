//! Data-driven OpenXR action and interaction profile binding manifest.
//!
//! Parses `actions.toml` (action set + action list) and `bindings/*.toml` (per-profile binding
//! tables) at session init, so adding an interaction profile is a data change rather than a Rust
//! edit. The runtime pipeline feeds the parsed [`ActionManifest`] into
//! [`super::openxr_actions::create_openxr_input_parts`] to create typed [`openxr::Action`] handles,
//! and into [`super::bindings::apply_suggested_interaction_bindings`] to submit suggested bindings
//! per profile.
//!
//! Assets live in `crates/renderide/assets/xr/` at source, and are copied next to the binary by
//! [`crate::build`](../../../build.rs) so the same resolution pattern used for `config.toml`
//! locates them at runtime.
//!
//! Validation at parse time rejects manifests that would otherwise produce confusing runtime
//! failures — unknown action references, duplicated action ids, misrouted haptic bindings, or
//! unknown extension gate names.
//!
//! Parse is pure (`parse_action_manifest` / `parse_binding_profile` take string slices) so unit
//! tests do not touch the filesystem. Disk loading is confined to [`load_manifest`].
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;

/// Errors produced while parsing or validating action/binding manifests.
#[derive(Debug, thiserror::Error)]
pub enum ManifestError {
    /// TOML syntax error in `actions.toml`.
    #[error("failed to parse actions manifest: {0}")]
    ActionsToml(#[source] toml::de::Error),
    /// TOML syntax error in a per-profile `bindings/*.toml`.
    #[error("failed to parse binding file {file}: {source}")]
    BindingToml {
        /// File whose contents failed to parse.
        file: String,
        /// Underlying TOML deserializer error.
        #[source]
        source: toml::de::Error,
    },
    /// Two actions share the same id within the action set.
    #[error("duplicate action id '{0}' in actions manifest")]
    DuplicateAction(String),
    /// A binding references an action not declared in `actions.toml`.
    #[error("binding in profile '{profile}' references unknown action '{action}'")]
    UnknownAction {
        /// Interaction profile path the offending binding lives under.
        profile: String,
        /// Action id that could not be resolved against the action manifest.
        action: String,
    },
    /// A haptic action is bound to an input path instead of `/output/haptic`.
    #[error("haptic action '{action}' in profile '{profile}' bound to non-haptic path '{path}'")]
    HapticOnWrongPath {
        /// Interaction profile path the offending binding lives under.
        profile: String,
        /// Haptic action id whose binding path is invalid.
        action: String,
        /// Offending OpenXR path.
        path: String,
    },
    /// A non-haptic action is bound to an `/output/haptic` path.
    #[error(
        "non-haptic action '{action}' in profile '{profile}' bound to haptic output path '{path}'"
    )]
    NonHapticOnHapticPath {
        /// Interaction profile path the offending binding lives under.
        profile: String,
        /// Action id bound to a haptic output path.
        action: String,
        /// Offending OpenXR path.
        path: String,
    },
    /// A binding file declared an extension gate that no [`ExtensionGate`] variant knows about.
    #[error("unknown extension_gate '{gate}' in profile '{profile}'")]
    UnknownExtensionGate {
        /// Interaction profile whose file declared the gate.
        profile: String,
        /// Offending gate name.
        gate: String,
    },
    /// Two binding files declared the same profile path.
    #[error("duplicate profile path '{0}' across binding files")]
    DuplicateProfile(String),
    /// Filesystem IO error while reading a manifest file.
    #[error("failed to read {path}: {source}")]
    Io {
        /// Path being read.
        path: String,
        /// Underlying io error.
        #[source]
        source: std::io::Error,
    },
    /// No `actions.toml` was found in any search location.
    #[error(
        "OpenXR action manifest not found; searched: {}",
        .searched.join(", ")
    )]
    ActionsManifestMissing {
        /// Paths checked before giving up.
        searched: Vec<String>,
    },
    /// A required `bindings/` directory was not found or empty.
    #[error("OpenXR bindings directory is missing or empty at {path}")]
    BindingsDirMissing {
        /// Directory that was expected to contain `*.toml` profiles.
        path: String,
    },
}

/// Declared [`openxr::Action`] payload type for an entry in `actions.toml`.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ActionType {
    /// `xr::Action<xr::Posef>` — a tracked pose used to build an [`openxr::Space`].
    Pose,
    /// `xr::Action<bool>` — digital button / touch state.
    Bool,
    /// `xr::Action<f32>` — analog axis such as trigger pull.
    Float,
    /// `xr::Action<xr::Vector2f>` — 2D axis such as thumbstick or trackpad.
    Vector2f,
    /// Haptic output driven via [`openxr::Action::apply_feedback`].
    Haptic,
}

/// Top-level action set metadata from `[action_set]`.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct ActionSetDef {
    /// Stable identifier passed to [`openxr::Instance::create_action_set`].
    pub id: String,
    /// Human-readable label shown in runtime binding UIs.
    pub localized_name: String,
    /// Action set priority; higher values win during binding resolution. Default 0.
    #[serde(default)]
    pub priority: u32,
}

/// One `[[action]]` entry from `actions.toml`.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct ActionDef {
    /// Stable identifier used to reference the action from a binding entry.
    pub id: String,
    /// OpenXR action payload type.
    #[serde(rename = "type")]
    pub ty: ActionType,
    /// Human-readable label shown in runtime binding UIs.
    pub localized_name: String,
}

/// Parsed, unvalidated contents of `actions.toml` — pass to [`ActionManifest::from_parsed`] to validate.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
struct ActionManifestRaw {
    action_set: ActionSetDef,
    action: Vec<ActionDef>,
}

/// Validated action manifest — every [`ActionDef`] has a unique id and a known type.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActionManifest {
    /// Action set metadata.
    pub action_set: ActionSetDef,
    /// All actions in declaration order.
    pub actions: Vec<ActionDef>,
    /// Fast lookup from action id to its index in `actions`.
    id_index: HashMap<String, usize>,
}

impl ActionManifest {
    /// Validates a parsed raw manifest and builds the id lookup index.
    fn from_parsed(raw: ActionManifestRaw) -> Result<Self, ManifestError> {
        let mut id_index = HashMap::with_capacity(raw.action.len());
        for (idx, action) in raw.action.iter().enumerate() {
            if id_index.insert(action.id.clone(), idx).is_some() {
                return Err(ManifestError::DuplicateAction(action.id.clone()));
            }
        }
        Ok(Self {
            action_set: raw.action_set,
            actions: raw.action,
            id_index,
        })
    }

    /// Returns the type of the action with `id`, or `None` if not declared.
    pub fn action_type(&self, id: &str) -> Option<ActionType> {
        self.id_index.get(id).map(|&i| self.actions[i].ty)
    }

    /// Returns the declared action by id.
    pub fn get(&self, id: &str) -> Option<&ActionDef> {
        self.id_index.get(id).map(|&i| &self.actions[i])
    }

    /// True when the manifest declares at least one haptic action.
    pub fn has_haptic(&self) -> bool {
        self.actions.iter().any(|a| a.ty == ActionType::Haptic)
    }
}

/// Parses `actions.toml` source text and validates the action inventory.
pub fn parse_action_manifest(toml_src: &str) -> Result<ActionManifest, ManifestError> {
    let raw: ActionManifestRaw = toml::from_str(toml_src).map_err(ManifestError::ActionsToml)?;
    ActionManifest::from_parsed(raw)
}

/// Identifier for an OpenXR extension that gates a profile's binding submission.
///
/// Each variant maps one-to-one to a field of [`super::bindings::ProfileExtensionGates`] — profiles
/// declaring an unknown variant are rejected at parse time so a typo in a binding TOML cannot
/// silently skip binding suggestion at runtime.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ExtensionGate {
    /// `XR_KHR_generic_controller`.
    KhrGenericController,
    /// `XR_BD_controller_interaction` — gates both Pico 4 and Pico Neo3.
    BdController,
    /// `XR_EXT_hp_mixed_reality_controller`.
    ExtHpMixedRealityController,
    /// `XR_EXT_samsung_odyssey_controller`.
    ExtSamsungOdysseyController,
    /// `XR_HTC_vive_cosmos_controller_interaction`.
    HtcViveCosmosControllerInteraction,
    /// `XR_HTC_vive_focus3_controller_interaction`.
    HtcViveFocus3ControllerInteraction,
    /// `XR_FB_touch_controller_pro`.
    FbTouchControllerPro,
    /// `XR_META_touch_controller_plus`.
    MetaTouchControllerPlus,
}

impl ExtensionGate {
    /// Resolves a TOML `extension_gate` string into the typed enum.
    fn from_str(raw: &str) -> Option<Self> {
        let gate = match raw {
            "khr_generic_controller" => Self::KhrGenericController,
            "bd_controller" => Self::BdController,
            "ext_hp_mixed_reality_controller" => Self::ExtHpMixedRealityController,
            "ext_samsung_odyssey_controller" => Self::ExtSamsungOdysseyController,
            "htc_vive_cosmos_controller_interaction" => Self::HtcViveCosmosControllerInteraction,
            "htc_vive_focus3_controller_interaction" => Self::HtcViveFocus3ControllerInteraction,
            "fb_touch_controller_pro" => Self::FbTouchControllerPro,
            "meta_touch_controller_plus" => Self::MetaTouchControllerPlus,
            _ => return None,
        };
        Some(gate)
    }
}

/// One `[[binding]]` entry from a profile file.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct BindingEntry {
    /// Action id the binding targets; must exist in [`ActionManifest`].
    pub action: String,
    /// Full OpenXR path (e.g. `/user/hand/left/input/trigger/value`).
    pub path: String,
}

/// Raw per-profile binding file (see `bindings/*.toml`).
#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
struct BindingProfileRaw {
    profile: String,
    #[serde(default)]
    extension_gate: Option<String>,
    #[serde(default)]
    binding: Vec<BindingEntry>,
}

/// Validated per-profile binding table.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BindingProfile {
    /// Full interaction profile path (`/interaction_profiles/...`).
    pub profile: String,
    /// Optional extension required for the runtime to accept this profile.
    pub extension_gate: Option<ExtensionGate>,
    /// `(action, input/output path)` pairs submitted via `xrSuggestInteractionProfileBindings`.
    pub bindings: Vec<BindingEntry>,
}

/// Parses a single profile file, without validating against an action manifest.
fn parse_binding_profile(
    file_label: &str,
    toml_src: &str,
) -> Result<BindingProfile, ManifestError> {
    let raw: BindingProfileRaw =
        toml::from_str(toml_src).map_err(|e| ManifestError::BindingToml {
            file: file_label.to_string(),
            source: e,
        })?;
    let extension_gate = match raw.extension_gate {
        None => None,
        Some(ref name) => match ExtensionGate::from_str(name) {
            Some(g) => Some(g),
            None => {
                return Err(ManifestError::UnknownExtensionGate {
                    profile: raw.profile.clone(),
                    gate: name.clone(),
                });
            }
        },
    };
    Ok(BindingProfile {
        profile: raw.profile,
        extension_gate,
        bindings: raw.binding,
    })
}

/// Returns `true` when `path` ends with a component denoting a haptic output.
fn is_haptic_path(path: &str) -> bool {
    path.ends_with("/output/haptic")
}

/// Checks that every binding in `profile` resolves to a known action and that haptic actions
/// bind only to haptic output paths (and vice versa).
fn validate_profile(
    actions: &ActionManifest,
    profile: &BindingProfile,
) -> Result<(), ManifestError> {
    for binding in &profile.bindings {
        let ty =
            actions
                .action_type(&binding.action)
                .ok_or_else(|| ManifestError::UnknownAction {
                    profile: profile.profile.clone(),
                    action: binding.action.clone(),
                })?;
        let path_is_haptic = is_haptic_path(&binding.path);
        if ty == ActionType::Haptic && !path_is_haptic {
            return Err(ManifestError::HapticOnWrongPath {
                profile: profile.profile.clone(),
                action: binding.action.clone(),
                path: binding.path.clone(),
            });
        }
        if ty != ActionType::Haptic && path_is_haptic {
            return Err(ManifestError::NonHapticOnHapticPath {
                profile: profile.profile.clone(),
                action: binding.action.clone(),
                path: binding.path.clone(),
            });
        }
    }
    Ok(())
}

/// Parsed and validated complete manifest: actions plus every profile's bindings.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Manifest {
    /// Action inventory.
    pub actions: ActionManifest,
    /// Per-interaction-profile binding tables, in load order.
    pub profiles: Vec<BindingProfile>,
}

/// Builds a validated [`Manifest`] from parsed actions and a list of parsed profile files.
///
/// Each element of `profile_sources` is `(file_label, file_contents)`. The label is surfaced in
/// diagnostics to point at whichever TOML file contains a validation failure.
pub fn build_manifest(
    actions_src: &str,
    profile_sources: &[(&str, &str)],
) -> Result<Manifest, ManifestError> {
    let actions = parse_action_manifest(actions_src)?;
    let mut profiles = Vec::with_capacity(profile_sources.len());
    let mut seen_profile_paths: HashMap<String, usize> = HashMap::new();

    for (label, src) in profile_sources {
        let profile = parse_binding_profile(label, src)?;
        if seen_profile_paths
            .insert(profile.profile.clone(), profiles.len())
            .is_some()
        {
            return Err(ManifestError::DuplicateProfile(profile.profile));
        }
        validate_profile(&actions, &profile)?;
        profiles.push(profile);
    }

    Ok(Manifest { actions, profiles })
}

// ---------------------------------------------------------------------------
// Disk loader
// ---------------------------------------------------------------------------

/// Environment override: set to a directory containing `actions.toml` and `bindings/` to bypass search.
const ENV_XR_ASSETS: &str = "RENDERIDE_XR_ASSETS";
/// Canonical action manifest filename inside the XR assets directory.
const ACTIONS_FILE: &str = "actions.toml";
/// Canonical bindings subdirectory inside the XR assets directory.
const BINDINGS_DIR: &str = "bindings";

/// Enumerates directories that might contain `actions.toml` plus `bindings/`.
fn xr_assets_search_candidates() -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    let push_unique = |v: &mut Vec<PathBuf>, p: PathBuf| {
        if !v.iter().any(|x| x == &p) {
            v.push(p);
        }
    };

    if let Ok(raw) = std::env::var(ENV_XR_ASSETS) {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            push_unique(&mut out, PathBuf::from(trimmed));
        }
    }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            push_unique(&mut out, dir.join("xr"));
            if let Some(parent) = dir.parent() {
                push_unique(&mut out, parent.join("xr"));
            }
        }
    }

    if let Ok(cwd) = std::env::current_dir() {
        if let Some(root) = crate::config::find_renderide_workspace_root(&cwd) {
            push_unique(&mut out, root.join("crates/renderide/assets/xr"));
        }
        push_unique(&mut out, cwd.join("xr"));
    }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            if let Some(root) = crate::config::find_renderide_workspace_root(dir) {
                push_unique(&mut out, root.join("crates/renderide/assets/xr"));
            }
        }
    }

    out
}

/// Chosen XR assets directory.
#[derive(Clone, Debug)]
pub struct XrAssetsLocation {
    /// Directory containing `actions.toml` and `bindings/`.
    pub root: PathBuf,
}

/// Locates the XR assets directory, returning the first candidate that has an `actions.toml`.
pub fn resolve_xr_assets_dir() -> Result<XrAssetsLocation, ManifestError> {
    let attempted = xr_assets_search_candidates();
    for candidate in &attempted {
        if candidate.join(ACTIONS_FILE).is_file() {
            return Ok(XrAssetsLocation {
                root: candidate.clone(),
            });
        }
    }
    Err(ManifestError::ActionsManifestMissing {
        searched: attempted.iter().map(|p| p.display().to_string()).collect(),
    })
}

/// Reads one file from disk, wrapping io errors with the path for diagnostics.
fn read_file(path: &Path) -> Result<String, ManifestError> {
    std::fs::read_to_string(path).map_err(|e| ManifestError::Io {
        path: path.display().to_string(),
        source: e,
    })
}

/// Lists every `.toml` file inside a `bindings/` directory, sorted for deterministic diagnostics.
fn list_binding_files(dir: &Path) -> Result<Vec<PathBuf>, ManifestError> {
    if !dir.is_dir() {
        return Err(ManifestError::BindingsDirMissing {
            path: dir.display().to_string(),
        });
    }
    let mut files = Vec::new();
    let entries = std::fs::read_dir(dir).map_err(|e| ManifestError::Io {
        path: dir.display().to_string(),
        source: e,
    })?;
    for entry in entries {
        let entry = entry.map_err(|e| ManifestError::Io {
            path: dir.display().to_string(),
            source: e,
        })?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("toml") {
            files.push(path);
        }
    }
    if files.is_empty() {
        return Err(ManifestError::BindingsDirMissing {
            path: dir.display().to_string(),
        });
    }
    files.sort();
    Ok(files)
}

/// Reads and validates the action + binding manifests from the resolved XR assets directory.
pub fn load_manifest() -> Result<(Manifest, XrAssetsLocation), ManifestError> {
    let location = resolve_xr_assets_dir()?;
    let actions_path = location.root.join(ACTIONS_FILE);
    let actions_src = read_file(&actions_path)?;

    let bindings_dir = location.root.join(BINDINGS_DIR);
    let binding_paths = list_binding_files(&bindings_dir)?;

    let mut sources: Vec<(String, String)> = Vec::with_capacity(binding_paths.len());
    for path in &binding_paths {
        let src = read_file(path)?;
        sources.push((path.display().to_string(), src));
    }

    let source_refs: Vec<(&str, &str)> = sources
        .iter()
        .map(|(label, src)| (label.as_str(), src.as_str()))
        .collect();

    let manifest = build_manifest(&actions_src, &source_refs)?;
    Ok((manifest, location))
}

#[cfg(test)]
mod tests {
    use super::*;

    const ACTIONS_OK: &str = r#"
[action_set]
id = "renderide_input"
localized_name = "Renderide VR input"
priority = 0

[[action]]
id = "left_grip_pose"
type = "pose"
localized_name = "Left grip pose"

[[action]]
id = "left_trigger"
type = "float"
localized_name = "Left trigger"

[[action]]
id = "left_haptic"
type = "haptic"
localized_name = "Left haptic"
"#;

    const PROFILE_OK: &str = r#"
profile = "/interaction_profiles/oculus/touch_controller"

[[binding]]
action = "left_grip_pose"
path = "/user/hand/left/input/grip/pose"

[[binding]]
action = "left_trigger"
path = "/user/hand/left/input/trigger/value"

[[binding]]
action = "left_haptic"
path = "/user/hand/left/output/haptic"
"#;

    #[test]
    fn parses_valid_action_manifest() {
        let m = parse_action_manifest(ACTIONS_OK).expect("parse");
        assert_eq!(m.action_set.id, "renderide_input");
        assert_eq!(m.actions.len(), 3);
        assert_eq!(m.action_type("left_trigger"), Some(ActionType::Float));
        assert!(m.has_haptic());
    }

    #[test]
    fn rejects_duplicate_action_id() {
        let src = r#"
[action_set]
id = "renderide_input"
localized_name = "Renderide VR input"

[[action]]
id = "dup"
type = "bool"
localized_name = "Dup"

[[action]]
id = "dup"
type = "float"
localized_name = "Dup again"
"#;
        let err = parse_action_manifest(src).expect_err("should fail");
        match err {
            ManifestError::DuplicateAction(id) => assert_eq!(id, "dup"),
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn parses_valid_profile_file() {
        let m = build_manifest(ACTIONS_OK, &[("oculus.toml", PROFILE_OK)]).expect("manifest");
        assert_eq!(m.profiles.len(), 1);
        assert_eq!(
            m.profiles[0].profile,
            "/interaction_profiles/oculus/touch_controller"
        );
        assert_eq!(m.profiles[0].extension_gate, None);
        assert_eq!(m.profiles[0].bindings.len(), 3);
    }

    #[test]
    fn rejects_unknown_action_ref() {
        let bad = r#"
profile = "/interaction_profiles/oculus/touch_controller"

[[binding]]
action = "not_declared"
path = "/user/hand/left/input/trigger/value"
"#;
        let err = build_manifest(ACTIONS_OK, &[("p.toml", bad)]).expect_err("should fail");
        match err {
            ManifestError::UnknownAction { action, .. } => {
                assert_eq!(action, "not_declared");
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn rejects_haptic_on_non_haptic_path() {
        let bad = r#"
profile = "/interaction_profiles/oculus/touch_controller"

[[binding]]
action = "left_haptic"
path = "/user/hand/left/input/trigger/value"
"#;
        let err = build_manifest(ACTIONS_OK, &[("p.toml", bad)]).expect_err("should fail");
        match err {
            ManifestError::HapticOnWrongPath { action, .. } => {
                assert_eq!(action, "left_haptic");
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn rejects_non_haptic_on_haptic_path() {
        let bad = r#"
profile = "/interaction_profiles/oculus/touch_controller"

[[binding]]
action = "left_trigger"
path = "/user/hand/left/output/haptic"
"#;
        let err = build_manifest(ACTIONS_OK, &[("p.toml", bad)]).expect_err("should fail");
        match err {
            ManifestError::NonHapticOnHapticPath { action, .. } => {
                assert_eq!(action, "left_trigger");
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn rejects_unknown_extension_gate() {
        let bad = r#"
profile = "/interaction_profiles/vendor/bogus_controller"
extension_gate = "totally_made_up_extension"

[[binding]]
action = "left_trigger"
path = "/user/hand/left/input/trigger/value"
"#;
        let err = build_manifest(ACTIONS_OK, &[("p.toml", bad)]).expect_err("should fail");
        match err {
            ManifestError::UnknownExtensionGate { gate, .. } => {
                assert_eq!(gate, "totally_made_up_extension");
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn rejects_duplicate_profile_path() {
        let err = build_manifest(
            ACTIONS_OK,
            &[("first.toml", PROFILE_OK), ("second.toml", PROFILE_OK)],
        )
        .expect_err("should fail");
        match err {
            ManifestError::DuplicateProfile(path) => {
                assert_eq!(path, "/interaction_profiles/oculus/touch_controller");
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    /// Asserts the shipped `assets/xr/*` files parse cleanly and cover every expected profile.
    ///
    /// Anchors the transliteration so future edits that typo an action id, mis-route a haptic
    /// binding, or drop a profile fail fast at `cargo test` rather than only at session init.
    #[test]
    fn shipped_manifest_loads() {
        const ACTIONS: &str = include_str!("../../../assets/xr/actions.toml");
        const TOUCH: &str =
            include_str!("../../../assets/xr/bindings/oculus_touch_controller.toml");
        const INDEX: &str = include_str!("../../../assets/xr/bindings/valve_index_controller.toml");
        const VIVE: &str = include_str!("../../../assets/xr/bindings/htc_vive_controller.toml");
        const VIVE_COSMOS: &str =
            include_str!("../../../assets/xr/bindings/htc_vive_cosmos_controller.toml");
        const VIVE_FOCUS3: &str =
            include_str!("../../../assets/xr/bindings/htc_vive_focus3_controller.toml");
        const WMR: &str =
            include_str!("../../../assets/xr/bindings/microsoft_motion_controller.toml");
        const HP: &str =
            include_str!("../../../assets/xr/bindings/hp_mixed_reality_controller.toml");
        const SAMSUNG: &str =
            include_str!("../../../assets/xr/bindings/samsung_odyssey_controller.toml");
        const PICO4: &str =
            include_str!("../../../assets/xr/bindings/bytedance_pico4_controller.toml");
        const PICO_NEO3: &str =
            include_str!("../../../assets/xr/bindings/bytedance_pico_neo3_controller.toml");
        const TOUCH_PRO: &str =
            include_str!("../../../assets/xr/bindings/facebook_touch_controller_pro.toml");
        const TOUCH_PLUS: &str =
            include_str!("../../../assets/xr/bindings/meta_touch_controller_plus.toml");
        const GENERIC: &str =
            include_str!("../../../assets/xr/bindings/khr_generic_controller.toml");
        const SIMPLE: &str = include_str!("../../../assets/xr/bindings/khr_simple_controller.toml");

        let sources = [
            ("oculus_touch_controller.toml", TOUCH),
            ("valve_index_controller.toml", INDEX),
            ("htc_vive_controller.toml", VIVE),
            ("htc_vive_cosmos_controller.toml", VIVE_COSMOS),
            ("htc_vive_focus3_controller.toml", VIVE_FOCUS3),
            ("microsoft_motion_controller.toml", WMR),
            ("hp_mixed_reality_controller.toml", HP),
            ("samsung_odyssey_controller.toml", SAMSUNG),
            ("bytedance_pico4_controller.toml", PICO4),
            ("bytedance_pico_neo3_controller.toml", PICO_NEO3),
            ("facebook_touch_controller_pro.toml", TOUCH_PRO),
            ("meta_touch_controller_plus.toml", TOUCH_PLUS),
            ("khr_generic_controller.toml", GENERIC),
            ("khr_simple_controller.toml", SIMPLE),
        ];
        let manifest = build_manifest(ACTIONS, &sources).expect("shipped manifest validates");

        assert_eq!(
            manifest.profiles.len(),
            14,
            "expected 14 shipped profiles, got {}",
            manifest.profiles.len()
        );
        assert!(
            manifest.actions.has_haptic(),
            "shipped actions.toml should declare a haptic action"
        );

        let profile_paths: std::collections::HashSet<&str> = manifest
            .profiles
            .iter()
            .map(|p| p.profile.as_str())
            .collect();
        for expected in [
            "/interaction_profiles/oculus/touch_controller",
            "/interaction_profiles/valve/index_controller",
            "/interaction_profiles/htc/vive_controller",
            "/interaction_profiles/htc/vive_cosmos_controller",
            "/interaction_profiles/htc/vive_focus3_controller",
            "/interaction_profiles/microsoft/motion_controller",
            "/interaction_profiles/hp/mixed_reality_controller",
            "/interaction_profiles/samsung/odyssey_controller",
            "/interaction_profiles/bytedance/pico4_controller",
            "/interaction_profiles/bytedance/pico_neo3_controller",
            "/interaction_profiles/facebook/touch_controller_pro",
            "/interaction_profiles/meta/touch_controller_plus",
            "/interaction_profiles/khr/generic_controller",
            "/interaction_profiles/khr/simple_controller",
        ] {
            assert!(
                profile_paths.contains(expected),
                "shipped manifest missing expected profile {expected}"
            );
        }
    }

    #[test]
    fn accepts_known_extension_gate() {
        let src = r#"
profile = "/interaction_profiles/hp/mixed_reality_controller"
extension_gate = "ext_hp_mixed_reality_controller"

[[binding]]
action = "left_trigger"
path = "/user/hand/left/input/trigger/value"
"#;
        let m = build_manifest(ACTIONS_OK, &[("p.toml", src)]).expect("manifest");
        assert_eq!(
            m.profiles[0].extension_gate,
            Some(ExtensionGate::ExtHpMixedRealityController)
        );
    }
}
