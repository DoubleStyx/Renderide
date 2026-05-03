//! OpenXR interaction profile binding suggestions, driven by the data-only TOML manifest.
//!
//! The per-profile binding tables previously hardcoded in this file now live under
//! `crates/renderide/assets/xr/bindings/` and are loaded via [`super::manifest`]. This module owns
//! the extension-gating logic, the [`ActionHandleRef`] dispatch helper that turns a parsed binding
//! entry into a typed [`xr::Binding`], and [`apply_suggested_interaction_bindings`] which submits
//! each profile table to the runtime.

use std::collections::HashMap;

use openxr as xr;

use super::manifest::{ExtensionGate, Manifest};
use super::openxr_actions::OpenxrInputActions;

/// Per-extension flags gating which profile binding suggestions are attempted.
///
/// Each flag tracks whether the runtime exposed (and the application enabled) the matching
/// OpenXR extension that registers the profile path. Profiles whose extension was not enabled
/// are skipped in [`apply_suggested_interaction_bindings`] so the runtime does not log an error
/// for an unknown profile. Populated by `bootstrap` from the enabled `xr::ExtensionSet`.
pub struct ProfileExtensionGates {
    /// `XR_KHR_generic_controller`.
    pub khr_generic_controller: bool,
    /// `XR_BD_controller_interaction` -- gates both Pico 4 and Pico Neo3.
    pub bd_controller: bool,
    /// `XR_EXT_hp_mixed_reality_controller`.
    pub ext_hp_mixed_reality_controller: bool,
    /// `XR_EXT_samsung_odyssey_controller`.
    pub ext_samsung_odyssey_controller: bool,
    /// `XR_HTC_vive_cosmos_controller_interaction`.
    pub htc_vive_cosmos_controller_interaction: bool,
    /// `XR_HTC_vive_focus3_controller_interaction`.
    pub htc_vive_focus3_controller_interaction: bool,
    /// `XR_FB_touch_controller_pro`.
    pub fb_touch_controller_pro: bool,
    /// `XR_META_touch_controller_plus`.
    pub meta_touch_controller_plus: bool,
}

impl ProfileExtensionGates {
    /// Returns `true` when the extension advertised by `gate` was enabled on the instance.
    pub fn is_enabled(&self, gate: ExtensionGate) -> bool {
        match gate {
            ExtensionGate::KhrGenericController => self.khr_generic_controller,
            ExtensionGate::BdController => self.bd_controller,
            ExtensionGate::ExtHpMixedRealityController => self.ext_hp_mixed_reality_controller,
            ExtensionGate::ExtSamsungOdysseyController => self.ext_samsung_odyssey_controller,
            ExtensionGate::HtcViveCosmosControllerInteraction => {
                self.htc_vive_cosmos_controller_interaction
            }
            ExtensionGate::HtcViveFocus3ControllerInteraction => {
                self.htc_vive_focus3_controller_interaction
            }
            ExtensionGate::FbTouchControllerPro => self.fb_touch_controller_pro,
            ExtensionGate::MetaTouchControllerPlus => self.meta_touch_controller_plus,
        }
    }
}

/// Typed reference to a created [`xr::Action`], used to construct [`xr::Binding`] values by id.
///
/// The manifest is string-keyed; this enum carries the type information needed to call the
/// matching generic [`xr::Binding::new`] overload at submission time.
#[derive(Clone, Copy)]
pub(super) enum ActionHandleRef<'a> {
    /// Tracked pose handle (grip or aim).
    Pose(&'a xr::Action<xr::Posef>),
    /// Digital state handle.
    Bool(&'a xr::Action<bool>),
    /// Analog axis handle.
    Float(&'a xr::Action<f32>),
    /// Two-axis handle (thumbstick, trackpad).
    Vector2f(&'a xr::Action<xr::Vector2f>),
    /// Haptic output handle.
    Haptic(&'a xr::Action<xr::Haptic>),
}

impl<'a> ActionHandleRef<'a> {
    /// Dispatches to the typed [`xr::Binding::new`] constructor.
    fn to_binding(self, path: xr::Path) -> xr::Binding<'a> {
        match self {
            Self::Pose(a) => xr::Binding::new(a, path),
            Self::Bool(a) => xr::Binding::new(a, path),
            Self::Float(a) => xr::Binding::new(a, path),
            Self::Vector2f(a) => xr::Binding::new(a, path),
            Self::Haptic(a) => xr::Binding::new(a, path),
        }
    }
}

/// Submits every manifest-declared binding table to the runtime, honouring extension gates.
///
/// Succeeds if **any** profile accepted bindings; returns the last error otherwise. Each profile
/// result is logged separately (info on accept, warn on reject) so runtime mismatches -- e.g. a
/// profile rejected because the runtime does not recognise a path -- are diagnosable rather than
/// silently swallowed.
///
/// `actions_by_id` must cover every action id referenced in `manifest.profiles[*].bindings`; the
/// manifest parser has already validated this invariant at load time.
pub(super) fn apply_suggested_interaction_bindings(
    instance: &xr::Instance,
    manifest: &Manifest,
    actions_by_id: &HashMap<String, ActionHandleRef<'_>>,
    gates: &ProfileExtensionGates,
) -> Result<(), xr::sys::Result> {
    let mut any_accepted = false;
    let mut last_err: Option<xr::sys::Result> = None;

    for profile in &manifest.profiles {
        if let Some(gate) = profile.extension_gate
            && !gates.is_enabled(gate)
        {
            continue;
        }

        let profile_path = instance.string_to_path(&profile.profile)?;

        let mut bindings: Vec<xr::Binding<'_>> = Vec::with_capacity(profile.bindings.len());
        for entry in &profile.bindings {
            let Some(handle) = actions_by_id.get(entry.action.as_str()) else {
                logger::error!(
                    "OpenXR manifest invariant: binding for '{}' in '{}' has no matching action handle",
                    entry.action,
                    profile.profile
                );
                return Err(xr::sys::Result::ERROR_VALIDATION_FAILURE);
            };
            let binding_path = instance.string_to_path(&entry.path)?;
            bindings.push(handle.to_binding(binding_path));
        }

        match instance.suggest_interaction_profile_bindings(profile_path, &bindings) {
            Ok(()) => {
                any_accepted = true;
                logger::info!("OpenXR binding suggestion accepted: {}", profile.profile);
            }
            Err(e) => {
                last_err = Some(e);
                logger::warn!(
                    "OpenXR binding suggestion rejected: {}: {e:?}",
                    profile.profile
                );
            }
        }
    }

    if !any_accepted {
        return Err(last_err.unwrap_or(xr::sys::Result::ERROR_PATH_UNSUPPORTED));
    }
    Ok(())
}

/// Builds a `HashMap<id, ActionHandleRef>` over every action owned by [`OpenxrInputActions`] so the
/// binding loop can look up handles by the string ids declared in the manifest.
pub(super) fn build_action_handle_map(
    actions: &OpenxrInputActions,
) -> HashMap<String, ActionHandleRef<'_>> {
    let mut map = HashMap::with_capacity(44);
    macro_rules! put {
        ($variant:ident, $field:ident) => {
            map.insert(
                stringify!($field).into(),
                ActionHandleRef::$variant(&actions.$field),
            );
        };
    }

    put!(Pose, left_grip_pose);
    put!(Pose, right_grip_pose);
    put!(Pose, left_aim_pose);
    put!(Pose, right_aim_pose);

    put!(Float, left_trigger);
    put!(Float, right_trigger);
    put!(Bool, left_trigger_touch);
    put!(Bool, right_trigger_touch);
    put!(Bool, left_trigger_click);
    put!(Bool, right_trigger_click);

    put!(Float, left_squeeze);
    put!(Float, right_squeeze);
    put!(Bool, left_squeeze_click);
    put!(Bool, right_squeeze_click);

    put!(Vector2f, left_thumbstick);
    put!(Vector2f, right_thumbstick);
    put!(Bool, left_thumbstick_touch);
    put!(Bool, right_thumbstick_touch);
    put!(Bool, left_thumbstick_click);
    put!(Bool, right_thumbstick_click);

    put!(Vector2f, left_trackpad);
    put!(Vector2f, right_trackpad);
    put!(Bool, left_trackpad_touch);
    put!(Bool, right_trackpad_touch);
    put!(Bool, left_trackpad_click);
    put!(Bool, right_trackpad_click);
    put!(Float, left_trackpad_force);
    put!(Float, right_trackpad_force);

    put!(Bool, left_primary);
    put!(Bool, right_primary);
    put!(Bool, left_secondary);
    put!(Bool, right_secondary);
    put!(Bool, left_primary_touch);
    put!(Bool, right_primary_touch);
    put!(Bool, left_secondary_touch);
    put!(Bool, right_secondary_touch);

    put!(Bool, left_menu);
    put!(Bool, right_menu);

    put!(Bool, left_thumbrest_touch);
    put!(Bool, right_thumbrest_touch);

    put!(Bool, left_select);
    put!(Bool, right_select);

    put!(Haptic, left_haptic);
    put!(Haptic, right_haptic);

    map
}

#[cfg(test)]
mod tests {
    use super::super::manifest::ExtensionGate;
    use super::*;

    fn gates(enabled: bool) -> ProfileExtensionGates {
        ProfileExtensionGates {
            khr_generic_controller: enabled,
            bd_controller: enabled,
            ext_hp_mixed_reality_controller: enabled,
            ext_samsung_odyssey_controller: enabled,
            htc_vive_cosmos_controller_interaction: enabled,
            htc_vive_focus3_controller_interaction: enabled,
            fb_touch_controller_pro: enabled,
            meta_touch_controller_plus: enabled,
        }
    }

    #[test]
    fn extension_gates_report_disabled_profiles() {
        let gates = gates(false);

        assert!(!gates.is_enabled(ExtensionGate::KhrGenericController));
        assert!(!gates.is_enabled(ExtensionGate::BdController));
        assert!(!gates.is_enabled(ExtensionGate::ExtHpMixedRealityController));
        assert!(!gates.is_enabled(ExtensionGate::ExtSamsungOdysseyController));
        assert!(!gates.is_enabled(ExtensionGate::HtcViveCosmosControllerInteraction));
        assert!(!gates.is_enabled(ExtensionGate::HtcViveFocus3ControllerInteraction));
        assert!(!gates.is_enabled(ExtensionGate::FbTouchControllerPro));
        assert!(!gates.is_enabled(ExtensionGate::MetaTouchControllerPlus));
    }

    #[test]
    fn extension_gates_report_enabled_profiles() {
        let gates = gates(true);

        assert!(gates.is_enabled(ExtensionGate::KhrGenericController));
        assert!(gates.is_enabled(ExtensionGate::BdController));
        assert!(gates.is_enabled(ExtensionGate::ExtHpMixedRealityController));
        assert!(gates.is_enabled(ExtensionGate::ExtSamsungOdysseyController));
        assert!(gates.is_enabled(ExtensionGate::HtcViveCosmosControllerInteraction));
        assert!(gates.is_enabled(ExtensionGate::HtcViveFocus3ControllerInteraction));
        assert!(gates.is_enabled(ExtensionGate::FbTouchControllerPro));
        assert!(gates.is_enabled(ExtensionGate::MetaTouchControllerPlus));
    }
}
