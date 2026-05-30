//! Data-driven OpenXR extension negotiation for the bootstrap path.
//!
//! `XR_KHR_vulkan_enable2` is checked up front by [`super::instance::create_openxr_instance`]
//! because the bootstrap cannot proceed without it. Every other extension is optional and
//! described by an [`OpenxrExtensionEntry`] in [`OPTIONAL_EXTENSIONS`]: when the runtime
//! advertises it, the matching `xr::ExtensionSet` field is enabled and (for controller profiles)
//! the corresponding [`ProfileExtensionGates`] field is set, telling
//! [`super::super::input::OpenxrInput`] which profile binding tables to attempt.

use openxr as xr;

use super::super::input::ProfileExtensionGates;

/// Static description of one optional OpenXR extension.
///
/// `enable` mirrors the runtime-advertised availability into the `enabled` set; the renderer
/// never gates an extension off when the runtime supports it. `feeds_profile_gate` is `Some`
/// for controller-profile extensions whose state must propagate to
/// [`ProfileExtensionGates`], and `None` for purely diagnostic extensions like `XR_EXT_debug_utils`.
struct OpenxrExtensionEntry {
    /// Stable display name used by [`enabled_extension_summary`].
    log_name: &'static str,
    /// Returns whether `set` advertises this extension.
    is_available: fn(&xr::ExtensionSet) -> bool,
    /// Copies `available` into the matching `enabled` flag.
    enable: fn(&mut xr::ExtensionSet, bool),
    /// Optional propagation into [`ProfileExtensionGates`]. `None` for extensions that do not
    /// gate any binding profile.
    feeds_profile_gate: Option<fn(&mut ProfileExtensionGates, bool)>,
}

/// Optional extensions enabled when the runtime advertises them.
///
/// Order matches the historical [`enabled_extension_summary`] output so logs stay stable.
const OPTIONAL_EXTENSIONS: &[OpenxrExtensionEntry] = &[
    OpenxrExtensionEntry {
        log_name: "EXT_debug_utils",
        is_available: |set| set.ext_debug_utils,
        enable: |set, v| set.ext_debug_utils = v,
        feeds_profile_gate: None,
    },
    OpenxrExtensionEntry {
        log_name: "KHR_composition_layer_depth",
        is_available: |set| set.khr_composition_layer_depth,
        enable: |set, v| set.khr_composition_layer_depth = v,
        feeds_profile_gate: None,
    },
    OpenxrExtensionEntry {
        log_name: "KHR_generic_controller",
        is_available: |set| set.khr_generic_controller,
        enable: |set, v| set.khr_generic_controller = v,
        feeds_profile_gate: Some(|gates, v| gates.khr_generic_controller = v),
    },
    OpenxrExtensionEntry {
        log_name: "BD_controller_interaction",
        is_available: |set| set.bd_controller_interaction,
        enable: |set, v| set.bd_controller_interaction = v,
        feeds_profile_gate: Some(|gates, v| gates.bd_controller = v),
    },
    OpenxrExtensionEntry {
        log_name: "EXT_hp_mixed_reality_controller",
        is_available: |set| set.ext_hp_mixed_reality_controller,
        enable: |set, v| set.ext_hp_mixed_reality_controller = v,
        feeds_profile_gate: Some(|gates, v| gates.ext_hp_mixed_reality_controller = v),
    },
    OpenxrExtensionEntry {
        log_name: "EXT_samsung_odyssey_controller",
        is_available: |set| set.ext_samsung_odyssey_controller,
        enable: |set, v| set.ext_samsung_odyssey_controller = v,
        feeds_profile_gate: Some(|gates, v| gates.ext_samsung_odyssey_controller = v),
    },
    OpenxrExtensionEntry {
        log_name: "HTC_vive_cosmos_controller_interaction",
        is_available: |set| set.htc_vive_cosmos_controller_interaction,
        enable: |set, v| set.htc_vive_cosmos_controller_interaction = v,
        feeds_profile_gate: Some(|gates, v| gates.htc_vive_cosmos_controller_interaction = v),
    },
    OpenxrExtensionEntry {
        log_name: "HTC_vive_focus3_controller_interaction",
        is_available: |set| set.htc_vive_focus3_controller_interaction,
        enable: |set, v| set.htc_vive_focus3_controller_interaction = v,
        feeds_profile_gate: Some(|gates, v| gates.htc_vive_focus3_controller_interaction = v),
    },
    OpenxrExtensionEntry {
        log_name: "FB_touch_controller_pro",
        is_available: |set| set.fb_touch_controller_pro,
        enable: |set, v| set.fb_touch_controller_pro = v,
        feeds_profile_gate: Some(|gates, v| gates.fb_touch_controller_pro = v),
    },
    OpenxrExtensionEntry {
        log_name: "META_touch_controller_plus",
        is_available: |set| set.meta_touch_controller_plus,
        enable: |set, v| set.meta_touch_controller_plus = v,
        feeds_profile_gate: Some(|gates, v| gates.meta_touch_controller_plus = v),
    },
    OpenxrExtensionEntry {
        log_name: "EXT_palm_pose",
        is_available: |set| set.ext_palm_pose,
        enable: |set, v| set.ext_palm_pose = v,
        feeds_profile_gate: Some(|gates, v| gates.palm_pose = v),
    },
    OpenxrExtensionEntry {
        log_name: "EXT_hand_tracking",
        is_available: |set| set.ext_hand_tracking,
        enable: |set, v| set.ext_hand_tracking = v,
        feeds_profile_gate: Some(|gates, v| gates.hand_tracking_ext = v),
    },
];

/// Mirrors each [`OPTIONAL_EXTENSIONS`] entry from `available` into `enabled` and, for
/// profile-gating entries, into `gates`. Caller is responsible for handling the mandatory
/// `XR_KHR_vulkan_enable2` precondition separately.
pub(super) fn enable_optional_extensions(
    available: &xr::ExtensionSet,
    enabled: &mut xr::ExtensionSet,
    gates: &mut ProfileExtensionGates,
) {
    for entry in OPTIONAL_EXTENSIONS {
        let advertised = (entry.is_available)(available);
        (entry.enable)(enabled, advertised);
        if let Some(gate_fn) = entry.feeds_profile_gate {
            gate_fn(gates, advertised);
        }
    }
}

/// Empty profile gates: every controller extension defaulted to `false`. Callers feed this
/// into [`enable_optional_extensions`] so gating flips on along with the matching extension.
pub(super) fn empty_profile_gates() -> ProfileExtensionGates {
    ProfileExtensionGates {
        khr_generic_controller: false,
        bd_controller: false,
        ext_hp_mixed_reality_controller: false,
        ext_samsung_odyssey_controller: false,
        htc_vive_cosmos_controller_interaction: false,
        htc_vive_focus3_controller_interaction: false,
        fb_touch_controller_pro: false,
        meta_touch_controller_plus: false,
        palm_pose: false,
        hand_tracking_ext: false,
    }
}

/// Comma-separated list of enabled extension names, in the historical log order.
///
/// `KHR_vulkan_enable2` leads the list when it is enabled, followed by every advertised entry in
/// [`OPTIONAL_EXTENSIONS`] order.
pub(super) fn enabled_extension_summary(enabled: &xr::ExtensionSet) -> String {
    let mut names: Vec<&'static str> = Vec::new();
    if enabled.khr_vulkan_enable2 {
        names.push("KHR_vulkan_enable2");
    }
    for entry in OPTIONAL_EXTENSIONS {
        if (entry.is_available)(enabled) {
            names.push(entry.log_name);
        }
    }
    names.join(",")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_available() -> xr::ExtensionSet {
        let mut set = xr::ExtensionSet::default();
        set.khr_vulkan_enable2 = true;
        set.ext_debug_utils = true;
        set.khr_composition_layer_depth = true;
        set.khr_generic_controller = true;
        set.bd_controller_interaction = true;
        set.ext_hp_mixed_reality_controller = true;
        set.ext_samsung_odyssey_controller = true;
        set.htc_vive_cosmos_controller_interaction = true;
        set.htc_vive_focus3_controller_interaction = true;
        set.fb_touch_controller_pro = true;
        set.meta_touch_controller_plus = true;
        set.ext_palm_pose = true;
        set.ext_hand_tracking = true;
        set
    }

    #[test]
    fn enable_optional_propagates_every_advertised_extension() {
        let available = all_available();
        let mut enabled = xr::ExtensionSet::default();
        let mut gates = empty_profile_gates();
        enable_optional_extensions(&available, &mut enabled, &mut gates);
        assert!(enabled.ext_debug_utils);
        assert!(enabled.khr_composition_layer_depth);
        assert!(enabled.khr_generic_controller);
        assert!(enabled.bd_controller_interaction);
        assert!(enabled.ext_hp_mixed_reality_controller);
        assert!(enabled.ext_samsung_odyssey_controller);
        assert!(enabled.htc_vive_cosmos_controller_interaction);
        assert!(enabled.htc_vive_focus3_controller_interaction);
        assert!(enabled.fb_touch_controller_pro);
        assert!(enabled.meta_touch_controller_plus);
        assert!(enabled.ext_palm_pose);
        assert!(enabled.ext_hand_tracking);
    }

    #[test]
    fn enable_optional_propagates_every_profile_gate() {
        let available = all_available();
        let mut enabled = xr::ExtensionSet::default();
        let mut gates = empty_profile_gates();
        enable_optional_extensions(&available, &mut enabled, &mut gates);
        assert!(gates.khr_generic_controller);
        assert!(gates.bd_controller);
        assert!(gates.ext_hp_mixed_reality_controller);
        assert!(gates.ext_samsung_odyssey_controller);
        assert!(gates.htc_vive_cosmos_controller_interaction);
        assert!(gates.htc_vive_focus3_controller_interaction);
        assert!(gates.fb_touch_controller_pro);
        assert!(gates.meta_touch_controller_plus);
        assert!(gates.palm_pose);
        assert!(gates.hand_tracking_ext);
    }

    #[test]
    fn enable_optional_leaves_unadvertised_extensions_off() {
        let available = xr::ExtensionSet::default();
        let mut enabled = xr::ExtensionSet::default();
        let mut gates = empty_profile_gates();
        enable_optional_extensions(&available, &mut enabled, &mut gates);
        assert!(!enabled.ext_debug_utils);
        assert!(!enabled.khr_composition_layer_depth);
        assert!(!enabled.khr_generic_controller);
        assert!(!enabled.ext_hand_tracking);
        assert!(!gates.khr_generic_controller);
        assert!(!gates.bd_controller);
        assert!(!gates.palm_pose);
        assert!(!gates.hand_tracking_ext);
    }

    #[test]
    fn enable_optional_does_not_touch_vulkan_enable2() {
        // The mandatory extension is enabled separately in `create_openxr_instance` after the
        // precondition check; the optional loop must not flip it.
        let available = all_available();
        let mut enabled = xr::ExtensionSet::default();
        let mut gates = empty_profile_gates();
        enable_optional_extensions(&available, &mut enabled, &mut gates);
        assert!(!enabled.khr_vulkan_enable2);
    }

    #[test]
    fn summary_matches_legacy_order_when_all_enabled() {
        let mut enabled = all_available();
        // The function expects the caller to have already enabled the mandatory extension.
        enabled.khr_vulkan_enable2 = true;
        let summary = enabled_extension_summary(&enabled);
        assert_eq!(
            summary,
            "KHR_vulkan_enable2,EXT_debug_utils,KHR_composition_layer_depth,KHR_generic_controller,\
             BD_controller_interaction,EXT_hp_mixed_reality_controller,\
             EXT_samsung_odyssey_controller,HTC_vive_cosmos_controller_interaction,\
             HTC_vive_focus3_controller_interaction,FB_touch_controller_pro,\
             META_touch_controller_plus,EXT_palm_pose,EXT_hand_tracking"
        );
    }

    #[test]
    fn summary_omits_disabled_entries() {
        let mut enabled = xr::ExtensionSet::default();
        enabled.khr_vulkan_enable2 = true;
        enabled.ext_debug_utils = true;
        enabled.bd_controller_interaction = true;
        let summary = enabled_extension_summary(&enabled);
        assert_eq!(
            summary,
            "KHR_vulkan_enable2,EXT_debug_utils,BD_controller_interaction"
        );
    }

    #[test]
    fn summary_is_empty_when_nothing_enabled() {
        let summary = enabled_extension_summary(&xr::ExtensionSet::default());
        assert_eq!(summary, "");
    }
}
