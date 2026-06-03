use super::parser::{build_manifest, parse_action_manifest};
use super::types::{ActionType, ExtensionGate, Manifest, ManifestError};

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

fn shipped_manifest() -> Manifest {
    const ACTIONS: &str = include_str!("../../../../assets/xr/actions.toml");
    const TOUCH: &str = include_str!("../../../../assets/xr/bindings/oculus_touch_controller.toml");
    const INDEX: &str = include_str!("../../../../assets/xr/bindings/valve_index_controller.toml");
    const VIVE: &str = include_str!("../../../../assets/xr/bindings/htc_vive_controller.toml");
    const VIVE_TRACKER: &str = include_str!("../../../../assets/xr/bindings/htc_vive_tracker.toml");
    const VIVE_COSMOS: &str =
        include_str!("../../../../assets/xr/bindings/htc_vive_cosmos_controller.toml");
    const VIVE_FOCUS3: &str =
        include_str!("../../../../assets/xr/bindings/htc_vive_focus3_controller.toml");
    const WMR: &str =
        include_str!("../../../../assets/xr/bindings/microsoft_motion_controller.toml");
    const HP: &str =
        include_str!("../../../../assets/xr/bindings/hp_mixed_reality_controller.toml");
    const SAMSUNG: &str =
        include_str!("../../../../assets/xr/bindings/samsung_odyssey_controller.toml");
    const PICO4: &str =
        include_str!("../../../../assets/xr/bindings/bytedance_pico4_controller.toml");
    const PICO_NEO3: &str =
        include_str!("../../../../assets/xr/bindings/bytedance_pico_neo3_controller.toml");
    const TOUCH_PRO: &str =
        include_str!("../../../../assets/xr/bindings/facebook_touch_controller_pro.toml");
    const TOUCH_PLUS: &str =
        include_str!("../../../../assets/xr/bindings/meta_touch_controller_plus.toml");
    const GENERIC: &str =
        include_str!("../../../../assets/xr/bindings/khr_generic_controller.toml");
    const SIMPLE: &str = include_str!("../../../../assets/xr/bindings/khr_simple_controller.toml");

    let sources = [
        ("oculus_touch_controller.toml", TOUCH),
        ("valve_index_controller.toml", INDEX),
        ("htc_vive_controller.toml", VIVE),
        ("htc_vive_tracker.toml", VIVE_TRACKER),
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
    build_manifest(ACTIONS, &sources).expect("shipped manifest validates")
}

/// Asserts the shipped `assets/xr/*` files parse cleanly and cover every expected profile.
///
/// Anchors the transliteration so future edits that typo an action id, mis-route a haptic
/// binding, or drop a profile fail fast at `cargo test` rather than only at session init.
#[test]
fn shipped_manifest_loads() {
    let manifest = shipped_manifest();

    assert_eq!(
        manifest.profiles.len(),
        15,
        "expected 15 shipped profiles, got {}",
        manifest.profiles.len()
    );
    assert!(
        manifest.actions.has_haptic(),
        "shipped actions.toml should declare a haptic action"
    );
    let palm_bindings = manifest
        .profiles
        .iter()
        .flat_map(|p| p.bindings.iter())
        .filter(|b| b.path.contains("/input/palm_ext/pose"))
        .count();
    assert_eq!(
        palm_bindings, 24,
        "expected left/right palm_ext bindings on concrete shipped profiles"
    );
    for binding in manifest
        .profiles
        .iter()
        .flat_map(|p| p.bindings.iter())
        .filter(|b| b.path.contains("/input/palm_ext/pose"))
    {
        assert_eq!(
            binding.extension_gate,
            Some(ExtensionGate::PalmPose),
            "palm_ext binding must be gated by XR_EXT_palm_pose"
        );
    }

    let profile_paths: hashbrown::HashSet<&str> = manifest
        .profiles
        .iter()
        .map(|p| p.profile.as_str())
        .collect();
    for expected in [
        "/interaction_profiles/oculus/touch_controller",
        "/interaction_profiles/valve/index_controller",
        "/interaction_profiles/htc/vive_controller",
        "/interaction_profiles/htc/vive_tracker_htcx",
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
fn shipped_vive_tracker_profile_is_body_pose_only() {
    let manifest = shipped_manifest();
    let tracker_profile = manifest
        .profiles
        .iter()
        .find(|profile| profile.profile == "/interaction_profiles/htc/vive_tracker_htcx")
        .expect("tracker profile");
    assert_eq!(
        tracker_profile.extension_gate,
        Some(ExtensionGate::HtcxViveTrackerInteraction)
    );
    assert_eq!(tracker_profile.bindings.len(), 14);
    for binding in &tracker_profile.bindings {
        assert!(binding.action.starts_with("tracker_"));
        assert!(binding.action.ends_with("_grip_pose"));
        assert!(binding.path.starts_with("/user/vive_tracker_htcx/role/"));
        assert!(binding.path.ends_with("/input/grip/pose"));
        assert_eq!(binding.extension_gate, None);
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

#[test]
fn accepts_tracker_extension_gate() {
    let src = r#"
profile = "/interaction_profiles/htc/vive_tracker_htcx"
extension_gate = "htcx_vive_tracker_interaction"

[[binding]]
action = "left_grip_pose"
path = "/user/vive_tracker_htcx/role/waist/input/grip/pose"
"#;
    let m = build_manifest(ACTIONS_OK, &[("p.toml", src)]).expect("manifest");
    assert_eq!(
        m.profiles[0].extension_gate,
        Some(ExtensionGate::HtcxViveTrackerInteraction)
    );
}

#[test]
fn accepts_binding_level_extension_gate() {
    let src = r#"
profile = "/interaction_profiles/oculus/touch_controller"

[[binding]]
action = "left_grip_pose"
path = "/user/hand/left/input/palm_ext/pose"
extension_gate = "palm_pose"
"#;
    let m = build_manifest(ACTIONS_OK, &[("p.toml", src)]).expect("manifest");
    assert_eq!(
        m.profiles[0].bindings[0].extension_gate,
        Some(ExtensionGate::PalmPose)
    );
}
