use hashbrown::HashMap;
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
    /// `xr::Action<xr::Posef>` -- a tracked pose used to build an [`openxr::Space`].
    Pose,
    /// `xr::Action<bool>` -- digital button / touch state.
    Bool,
    /// `xr::Action<f32>` -- analog axis such as trigger pull.
    Float,
    /// `xr::Action<xr::Vector2f>` -- 2D axis such as thumbstick or trackpad.
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

/// Parsed, unvalidated contents of `actions.toml` -- pass to [`ActionManifest::from_parsed`] to validate.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub(super) struct ActionManifestRaw {
    action_set: ActionSetDef,
    action: Vec<ActionDef>,
}

/// Validated action manifest -- every [`ActionDef`] has a unique id and a known type.
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
    pub(super) fn from_parsed(raw: ActionManifestRaw) -> Result<Self, ManifestError> {
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

/// Identifier for an OpenXR extension that gates a profile's binding submission.
///
/// Each variant maps one-to-one to a field of [`crate::xr::input::bindings::ProfileExtensionGates`] -- profiles
/// declaring an unknown variant are rejected at parse time so a typo in a binding TOML cannot
/// silently skip binding suggestion at runtime.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ExtensionGate {
    /// `XR_KHR_generic_controller`.
    KhrGenericController,
    /// `XR_BD_controller_interaction` -- gates both Pico 4 and Pico Neo3.
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
    pub(super) fn from_str(raw: &str) -> Option<Self> {
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

/// Parsed and validated complete manifest: actions plus every profile's bindings.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Manifest {
    /// Action inventory.
    pub actions: ActionManifest,
    /// Per-interaction-profile binding tables, in load order.
    pub profiles: Vec<BindingProfile>,
}
