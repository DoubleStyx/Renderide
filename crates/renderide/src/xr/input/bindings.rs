//! OpenXR interaction profile binding suggestions for the Renderide action set.

use openxr as xr;

/// All [`xr::Path`] values used when calling [`openxr::Instance::suggest_interaction_profile_bindings`].
pub(super) struct BindingPaths {
    pub(super) left_grip_pose: xr::Path,
    pub(super) right_grip_pose: xr::Path,
    pub(super) left_aim_pose: xr::Path,
    pub(super) right_aim_pose: xr::Path,
    pub(super) left_trigger_value: xr::Path,
    pub(super) right_trigger_value: xr::Path,
    pub(super) left_trigger_touch: xr::Path,
    pub(super) right_trigger_touch: xr::Path,
    pub(super) left_trigger_click: xr::Path,
    pub(super) right_trigger_click: xr::Path,
    pub(super) left_squeeze_value: xr::Path,
    pub(super) right_squeeze_value: xr::Path,
    pub(super) left_squeeze_click: xr::Path,
    pub(super) right_squeeze_click: xr::Path,
    pub(super) left_thumbstick: xr::Path,
    pub(super) right_thumbstick: xr::Path,
    pub(super) left_thumbstick_touch: xr::Path,
    pub(super) right_thumbstick_touch: xr::Path,
    pub(super) left_thumbstick_click: xr::Path,
    pub(super) right_thumbstick_click: xr::Path,
    pub(super) left_trackpad: xr::Path,
    pub(super) right_trackpad: xr::Path,
    pub(super) left_trackpad_touch: xr::Path,
    pub(super) right_trackpad_touch: xr::Path,
    pub(super) left_trackpad_click: xr::Path,
    pub(super) right_trackpad_click: xr::Path,
    pub(super) left_trackpad_force: xr::Path,
    pub(super) right_trackpad_force: xr::Path,
    pub(super) left_x_click: xr::Path,
    pub(super) left_y_click: xr::Path,
    pub(super) left_x_touch: xr::Path,
    pub(super) left_y_touch: xr::Path,
    pub(super) left_a_click: xr::Path,
    pub(super) left_b_click: xr::Path,
    pub(super) left_a_touch: xr::Path,
    pub(super) left_b_touch: xr::Path,
    pub(super) right_a_click: xr::Path,
    pub(super) right_b_click: xr::Path,
    pub(super) right_a_touch: xr::Path,
    pub(super) right_b_touch: xr::Path,
    pub(super) left_menu_click: xr::Path,
    pub(super) right_menu_click: xr::Path,
    pub(super) left_thumbrest_touch: xr::Path,
    pub(super) right_thumbrest_touch: xr::Path,
    pub(super) left_select_click: xr::Path,
    pub(super) right_select_click: xr::Path,
}

/// Registered OpenXR interaction profile paths (e.g. Oculus Touch, Index).
pub(super) struct InteractionProfilePaths {
    pub(super) oculus_touch: xr::Path,
    pub(super) valve_index: xr::Path,
    pub(super) htc_vive: xr::Path,
    pub(super) microsoft_motion: xr::Path,
    pub(super) generic_controller: xr::Path,
    pub(super) simple_controller: xr::Path,
    /// [`XR_BD_controller_interaction`](https://registry.khronos.org/OpenXR/specs/1.0/html/xrspec.html#XR_BD_controller_interaction) — Pico 4.
    pub(super) pico4_controller: xr::Path,
    /// [`XR_BD_controller_interaction`](https://registry.khronos.org/OpenXR/specs/1.0/html/xrspec.html#XR_BD_controller_interaction) — Pico Neo3.
    pub(super) pico_neo3_controller: xr::Path,
    /// [`XR_EXT_hp_mixed_reality_controller`](https://registry.khronos.org/OpenXR/specs/1.0/html/xrspec.html#XR_EXT_hp_mixed_reality_controller).
    pub(super) hp_reverb_g2: xr::Path,
    /// [`XR_EXT_samsung_odyssey_controller`](https://registry.khronos.org/OpenXR/specs/1.0/html/xrspec.html#XR_EXT_samsung_odyssey_controller).
    pub(super) samsung_odyssey: xr::Path,
    /// [`XR_HTC_vive_cosmos_controller_interaction`](https://registry.khronos.org/OpenXR/specs/1.0/html/xrspec.html#XR_HTC_vive_cosmos_controller_interaction).
    pub(super) htc_vive_cosmos: xr::Path,
    /// [`XR_HTC_vive_focus3_controller_interaction`](https://registry.khronos.org/OpenXR/specs/1.0/html/xrspec.html#XR_HTC_vive_focus3_controller_interaction).
    pub(super) htc_vive_focus3: xr::Path,
    /// [`XR_FB_touch_controller_pro`](https://registry.khronos.org/OpenXR/specs/1.0/html/xrspec.html#XR_FB_touch_controller_pro) — Quest Pro / Touch Pro.
    pub(super) meta_touch_pro: xr::Path,
    /// [`XR_META_touch_controller_plus`](https://registry.khronos.org/OpenXR/specs/1.0/html/xrspec.html#XR_META_touch_controller_plus) — Quest 3 / Touch Plus.
    pub(super) meta_touch_plus: xr::Path,
}

/// Per-extension flags gating which profile binding suggestions are attempted.
///
/// Each flag tracks whether the runtime exposed (and the application enabled) the matching
/// OpenXR extension that registers the profile path. Profiles whose extension was not enabled
/// are skipped in [`apply_suggested_interaction_bindings`] so the runtime does not log an error
/// for an unknown profile. Populated by `bootstrap` from the enabled `xr::ExtensionSet`.
pub struct ProfileExtensionGates {
    /// `XR_KHR_generic_controller`.
    pub khr_generic_controller: bool,
    /// `XR_BD_controller_interaction` — gates both Pico 4 and Pico Neo3.
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

/// References to every action participating in binding suggestions.
pub(super) struct ActionRefs<'a> {
    pub(super) left_grip_pose: &'a xr::Action<xr::Posef>,
    pub(super) right_grip_pose: &'a xr::Action<xr::Posef>,
    pub(super) left_aim_pose: &'a xr::Action<xr::Posef>,
    pub(super) right_aim_pose: &'a xr::Action<xr::Posef>,
    pub(super) left_trigger: &'a xr::Action<f32>,
    pub(super) right_trigger: &'a xr::Action<f32>,
    pub(super) left_trigger_touch: &'a xr::Action<bool>,
    pub(super) right_trigger_touch: &'a xr::Action<bool>,
    pub(super) left_trigger_click: &'a xr::Action<bool>,
    pub(super) right_trigger_click: &'a xr::Action<bool>,
    pub(super) left_squeeze: &'a xr::Action<f32>,
    pub(super) right_squeeze: &'a xr::Action<f32>,
    pub(super) left_squeeze_click: &'a xr::Action<bool>,
    pub(super) right_squeeze_click: &'a xr::Action<bool>,
    pub(super) left_thumbstick: &'a xr::Action<xr::Vector2f>,
    pub(super) right_thumbstick: &'a xr::Action<xr::Vector2f>,
    pub(super) left_thumbstick_touch: &'a xr::Action<bool>,
    pub(super) right_thumbstick_touch: &'a xr::Action<bool>,
    pub(super) left_thumbstick_click: &'a xr::Action<bool>,
    pub(super) right_thumbstick_click: &'a xr::Action<bool>,
    pub(super) left_trackpad: &'a xr::Action<xr::Vector2f>,
    pub(super) right_trackpad: &'a xr::Action<xr::Vector2f>,
    pub(super) left_trackpad_touch: &'a xr::Action<bool>,
    pub(super) right_trackpad_touch: &'a xr::Action<bool>,
    pub(super) left_trackpad_click: &'a xr::Action<bool>,
    pub(super) right_trackpad_click: &'a xr::Action<bool>,
    pub(super) left_trackpad_force: &'a xr::Action<f32>,
    pub(super) right_trackpad_force: &'a xr::Action<f32>,
    pub(super) left_primary: &'a xr::Action<bool>,
    pub(super) right_primary: &'a xr::Action<bool>,
    pub(super) left_secondary: &'a xr::Action<bool>,
    pub(super) right_secondary: &'a xr::Action<bool>,
    pub(super) left_primary_touch: &'a xr::Action<bool>,
    pub(super) right_primary_touch: &'a xr::Action<bool>,
    pub(super) left_secondary_touch: &'a xr::Action<bool>,
    pub(super) right_secondary_touch: &'a xr::Action<bool>,
    pub(super) left_menu: &'a xr::Action<bool>,
    pub(super) right_menu: &'a xr::Action<bool>,
    pub(super) left_thumbrest_touch: &'a xr::Action<bool>,
    pub(super) right_thumbrest_touch: &'a xr::Action<bool>,
    pub(super) left_select: &'a xr::Action<bool>,
    pub(super) right_select: &'a xr::Action<bool>,
}

/// Oculus Touch interaction profile suggestions for the shared action set.
fn oculus_touch_bindings<'a>(a: &'a ActionRefs<'a>, p: &'a BindingPaths) -> Vec<xr::Binding<'a>> {
    vec![
        xr::Binding::new(a.left_grip_pose, p.left_grip_pose),
        xr::Binding::new(a.right_grip_pose, p.right_grip_pose),
        xr::Binding::new(a.left_aim_pose, p.left_aim_pose),
        xr::Binding::new(a.right_aim_pose, p.right_aim_pose),
        xr::Binding::new(a.left_trigger, p.left_trigger_value),
        xr::Binding::new(a.right_trigger, p.right_trigger_value),
        xr::Binding::new(a.left_trigger_touch, p.left_trigger_touch),
        xr::Binding::new(a.right_trigger_touch, p.right_trigger_touch),
        xr::Binding::new(a.left_squeeze, p.left_squeeze_value),
        xr::Binding::new(a.right_squeeze, p.right_squeeze_value),
        xr::Binding::new(a.left_thumbstick, p.left_thumbstick),
        xr::Binding::new(a.right_thumbstick, p.right_thumbstick),
        xr::Binding::new(a.left_thumbstick_touch, p.left_thumbstick_touch),
        xr::Binding::new(a.right_thumbstick_touch, p.right_thumbstick_touch),
        xr::Binding::new(a.left_thumbstick_click, p.left_thumbstick_click),
        xr::Binding::new(a.right_thumbstick_click, p.right_thumbstick_click),
        xr::Binding::new(a.left_primary, p.left_x_click),
        xr::Binding::new(a.left_secondary, p.left_y_click),
        xr::Binding::new(a.right_primary, p.right_a_click),
        xr::Binding::new(a.right_secondary, p.right_b_click),
        xr::Binding::new(a.left_primary_touch, p.left_x_touch),
        xr::Binding::new(a.left_secondary_touch, p.left_y_touch),
        xr::Binding::new(a.right_primary_touch, p.right_a_touch),
        xr::Binding::new(a.right_secondary_touch, p.right_b_touch),
        xr::Binding::new(a.left_menu, p.left_menu_click),
        xr::Binding::new(a.left_thumbrest_touch, p.left_thumbrest_touch),
        xr::Binding::new(a.right_thumbrest_touch, p.right_thumbrest_touch),
    ]
}

/// ByteDance Pico Neo3 (`XR_BD_controller_interaction`); bilateral `menu/click` and no thumbrest paths.
fn pico_neo3_controller_bindings<'a>(
    a: &'a ActionRefs<'a>,
    p: &'a BindingPaths,
) -> Vec<xr::Binding<'a>> {
    vec![
        xr::Binding::new(a.left_grip_pose, p.left_grip_pose),
        xr::Binding::new(a.right_grip_pose, p.right_grip_pose),
        xr::Binding::new(a.left_aim_pose, p.left_aim_pose),
        xr::Binding::new(a.right_aim_pose, p.right_aim_pose),
        xr::Binding::new(a.left_trigger, p.left_trigger_value),
        xr::Binding::new(a.right_trigger, p.right_trigger_value),
        xr::Binding::new(a.left_trigger_touch, p.left_trigger_touch),
        xr::Binding::new(a.right_trigger_touch, p.right_trigger_touch),
        xr::Binding::new(a.left_trigger_click, p.left_trigger_click),
        xr::Binding::new(a.right_trigger_click, p.right_trigger_click),
        xr::Binding::new(a.left_squeeze, p.left_squeeze_value),
        xr::Binding::new(a.right_squeeze, p.right_squeeze_value),
        xr::Binding::new(a.left_squeeze_click, p.left_squeeze_click),
        xr::Binding::new(a.right_squeeze_click, p.right_squeeze_click),
        xr::Binding::new(a.left_thumbstick, p.left_thumbstick),
        xr::Binding::new(a.right_thumbstick, p.right_thumbstick),
        xr::Binding::new(a.left_thumbstick_touch, p.left_thumbstick_touch),
        xr::Binding::new(a.right_thumbstick_touch, p.right_thumbstick_touch),
        xr::Binding::new(a.left_thumbstick_click, p.left_thumbstick_click),
        xr::Binding::new(a.right_thumbstick_click, p.right_thumbstick_click),
        xr::Binding::new(a.left_primary, p.left_x_click),
        xr::Binding::new(a.left_secondary, p.left_y_click),
        xr::Binding::new(a.right_primary, p.right_a_click),
        xr::Binding::new(a.right_secondary, p.right_b_click),
        xr::Binding::new(a.left_primary_touch, p.left_x_touch),
        xr::Binding::new(a.left_secondary_touch, p.left_y_touch),
        xr::Binding::new(a.right_primary_touch, p.right_a_touch),
        xr::Binding::new(a.right_secondary_touch, p.right_b_touch),
        xr::Binding::new(a.left_menu, p.left_menu_click),
        xr::Binding::new(a.right_menu, p.right_menu_click),
    ]
}

/// ByteDance Pico 4 (`XR_BD_controller_interaction`); `menu/click` is left-only per spec.
fn pico4_controller_bindings<'a>(
    a: &'a ActionRefs<'a>,
    p: &'a BindingPaths,
) -> Vec<xr::Binding<'a>> {
    vec![
        xr::Binding::new(a.left_grip_pose, p.left_grip_pose),
        xr::Binding::new(a.right_grip_pose, p.right_grip_pose),
        xr::Binding::new(a.left_aim_pose, p.left_aim_pose),
        xr::Binding::new(a.right_aim_pose, p.right_aim_pose),
        xr::Binding::new(a.left_trigger, p.left_trigger_value),
        xr::Binding::new(a.right_trigger, p.right_trigger_value),
        xr::Binding::new(a.left_trigger_touch, p.left_trigger_touch),
        xr::Binding::new(a.right_trigger_touch, p.right_trigger_touch),
        xr::Binding::new(a.left_trigger_click, p.left_trigger_click),
        xr::Binding::new(a.right_trigger_click, p.right_trigger_click),
        xr::Binding::new(a.left_squeeze, p.left_squeeze_value),
        xr::Binding::new(a.right_squeeze, p.right_squeeze_value),
        xr::Binding::new(a.left_squeeze_click, p.left_squeeze_click),
        xr::Binding::new(a.right_squeeze_click, p.right_squeeze_click),
        xr::Binding::new(a.left_thumbstick, p.left_thumbstick),
        xr::Binding::new(a.right_thumbstick, p.right_thumbstick),
        xr::Binding::new(a.left_thumbstick_touch, p.left_thumbstick_touch),
        xr::Binding::new(a.right_thumbstick_touch, p.right_thumbstick_touch),
        xr::Binding::new(a.left_thumbstick_click, p.left_thumbstick_click),
        xr::Binding::new(a.right_thumbstick_click, p.right_thumbstick_click),
        xr::Binding::new(a.left_primary, p.left_x_click),
        xr::Binding::new(a.left_secondary, p.left_y_click),
        xr::Binding::new(a.right_primary, p.right_a_click),
        xr::Binding::new(a.right_secondary, p.right_b_click),
        xr::Binding::new(a.left_primary_touch, p.left_x_touch),
        xr::Binding::new(a.left_secondary_touch, p.left_y_touch),
        xr::Binding::new(a.right_primary_touch, p.right_a_touch),
        xr::Binding::new(a.right_secondary_touch, p.right_b_touch),
        xr::Binding::new(a.left_menu, p.left_menu_click),
    ]
}

/// Valve Index controller profile.
fn valve_index_bindings<'a>(a: &'a ActionRefs<'a>, p: &'a BindingPaths) -> Vec<xr::Binding<'a>> {
    vec![
        xr::Binding::new(a.left_grip_pose, p.left_grip_pose),
        xr::Binding::new(a.right_grip_pose, p.right_grip_pose),
        xr::Binding::new(a.left_aim_pose, p.left_aim_pose),
        xr::Binding::new(a.right_aim_pose, p.right_aim_pose),
        xr::Binding::new(a.left_trigger, p.left_trigger_value),
        xr::Binding::new(a.right_trigger, p.right_trigger_value),
        xr::Binding::new(a.left_trigger_touch, p.left_trigger_touch),
        xr::Binding::new(a.right_trigger_touch, p.right_trigger_touch),
        xr::Binding::new(a.left_trigger_click, p.left_trigger_click),
        xr::Binding::new(a.right_trigger_click, p.right_trigger_click),
        xr::Binding::new(a.left_squeeze, p.left_squeeze_value),
        xr::Binding::new(a.right_squeeze, p.right_squeeze_value),
        xr::Binding::new(a.left_thumbstick, p.left_thumbstick),
        xr::Binding::new(a.right_thumbstick, p.right_thumbstick),
        xr::Binding::new(a.left_thumbstick_touch, p.left_thumbstick_touch),
        xr::Binding::new(a.right_thumbstick_touch, p.right_thumbstick_touch),
        xr::Binding::new(a.left_thumbstick_click, p.left_thumbstick_click),
        xr::Binding::new(a.right_thumbstick_click, p.right_thumbstick_click),
        xr::Binding::new(a.left_trackpad, p.left_trackpad),
        xr::Binding::new(a.right_trackpad, p.right_trackpad),
        xr::Binding::new(a.left_trackpad_touch, p.left_trackpad_touch),
        xr::Binding::new(a.right_trackpad_touch, p.right_trackpad_touch),
        xr::Binding::new(a.left_trackpad_force, p.left_trackpad_force),
        xr::Binding::new(a.right_trackpad_force, p.right_trackpad_force),
        xr::Binding::new(a.left_primary, p.left_a_click),
        xr::Binding::new(a.left_secondary, p.left_b_click),
        xr::Binding::new(a.right_primary, p.right_a_click),
        xr::Binding::new(a.right_secondary, p.right_b_click),
        xr::Binding::new(a.left_primary_touch, p.left_a_touch),
        xr::Binding::new(a.left_secondary_touch, p.left_b_touch),
        xr::Binding::new(a.right_primary_touch, p.right_a_touch),
        xr::Binding::new(a.right_secondary_touch, p.right_b_touch),
    ]
}

/// HTC Vive wand profile.
fn htc_vive_bindings<'a>(a: &'a ActionRefs<'a>, p: &'a BindingPaths) -> Vec<xr::Binding<'a>> {
    vec![
        xr::Binding::new(a.left_grip_pose, p.left_grip_pose),
        xr::Binding::new(a.right_grip_pose, p.right_grip_pose),
        xr::Binding::new(a.left_aim_pose, p.left_aim_pose),
        xr::Binding::new(a.right_aim_pose, p.right_aim_pose),
        xr::Binding::new(a.left_trigger, p.left_trigger_value),
        xr::Binding::new(a.right_trigger, p.right_trigger_value),
        xr::Binding::new(a.left_trigger_click, p.left_trigger_click),
        xr::Binding::new(a.right_trigger_click, p.right_trigger_click),
        xr::Binding::new(a.left_squeeze_click, p.left_squeeze_click),
        xr::Binding::new(a.right_squeeze_click, p.right_squeeze_click),
        xr::Binding::new(a.left_trackpad, p.left_trackpad),
        xr::Binding::new(a.right_trackpad, p.right_trackpad),
        xr::Binding::new(a.left_trackpad_touch, p.left_trackpad_touch),
        xr::Binding::new(a.right_trackpad_touch, p.right_trackpad_touch),
        xr::Binding::new(a.left_trackpad_click, p.left_trackpad_click),
        xr::Binding::new(a.right_trackpad_click, p.right_trackpad_click),
        xr::Binding::new(a.left_menu, p.left_menu_click),
        xr::Binding::new(a.right_menu, p.right_menu_click),
    ]
}

/// Windows Mixed Reality motion controllers.
fn microsoft_motion_bindings<'a>(
    a: &'a ActionRefs<'a>,
    p: &'a BindingPaths,
) -> Vec<xr::Binding<'a>> {
    vec![
        xr::Binding::new(a.left_grip_pose, p.left_grip_pose),
        xr::Binding::new(a.right_grip_pose, p.right_grip_pose),
        xr::Binding::new(a.left_aim_pose, p.left_aim_pose),
        xr::Binding::new(a.right_aim_pose, p.right_aim_pose),
        xr::Binding::new(a.left_trigger, p.left_trigger_value),
        xr::Binding::new(a.right_trigger, p.right_trigger_value),
        xr::Binding::new(a.left_squeeze_click, p.left_squeeze_click),
        xr::Binding::new(a.right_squeeze_click, p.right_squeeze_click),
        xr::Binding::new(a.left_thumbstick, p.left_thumbstick),
        xr::Binding::new(a.right_thumbstick, p.right_thumbstick),
        xr::Binding::new(a.left_thumbstick_click, p.left_thumbstick_click),
        xr::Binding::new(a.right_thumbstick_click, p.right_thumbstick_click),
        xr::Binding::new(a.left_trackpad, p.left_trackpad),
        xr::Binding::new(a.right_trackpad, p.right_trackpad),
        xr::Binding::new(a.left_trackpad_touch, p.left_trackpad_touch),
        xr::Binding::new(a.right_trackpad_touch, p.right_trackpad_touch),
        xr::Binding::new(a.left_trackpad_click, p.left_trackpad_click),
        xr::Binding::new(a.right_trackpad_click, p.right_trackpad_click),
        xr::Binding::new(a.left_menu, p.left_menu_click),
        xr::Binding::new(a.right_menu, p.right_menu_click),
    ]
}

/// `XR_KHR_generic_controller` minimal suggestions (tracked pose + triggers + sticks + face buttons).
fn generic_controller_bindings<'a>(
    a: &'a ActionRefs<'a>,
    p: &'a BindingPaths,
) -> Vec<xr::Binding<'a>> {
    vec![
        xr::Binding::new(a.left_grip_pose, p.left_grip_pose),
        xr::Binding::new(a.right_grip_pose, p.right_grip_pose),
        xr::Binding::new(a.left_aim_pose, p.left_aim_pose),
        xr::Binding::new(a.right_aim_pose, p.right_aim_pose),
        xr::Binding::new(a.left_trigger, p.left_trigger_value),
        xr::Binding::new(a.right_trigger, p.right_trigger_value),
        xr::Binding::new(a.left_squeeze, p.left_squeeze_value),
        xr::Binding::new(a.right_squeeze, p.right_squeeze_value),
        xr::Binding::new(a.left_thumbstick, p.left_thumbstick),
        xr::Binding::new(a.right_thumbstick, p.right_thumbstick),
        xr::Binding::new(a.left_thumbstick_click, p.left_thumbstick_click),
        xr::Binding::new(a.right_thumbstick_click, p.right_thumbstick_click),
        xr::Binding::new(a.left_primary, p.left_select_click),
        xr::Binding::new(a.right_primary, p.right_select_click),
        xr::Binding::new(a.left_secondary, p.left_menu_click),
        xr::Binding::new(a.right_secondary, p.right_menu_click),
    ]
}

/// OpenXR `simple_controller` profile.
fn simple_controller_bindings<'a>(
    a: &'a ActionRefs<'a>,
    p: &'a BindingPaths,
) -> Vec<xr::Binding<'a>> {
    vec![
        xr::Binding::new(a.left_grip_pose, p.left_grip_pose),
        xr::Binding::new(a.right_grip_pose, p.right_grip_pose),
        xr::Binding::new(a.left_aim_pose, p.left_aim_pose),
        xr::Binding::new(a.right_aim_pose, p.right_aim_pose),
        xr::Binding::new(a.left_select, p.left_select_click),
        xr::Binding::new(a.right_select, p.right_select_click),
        xr::Binding::new(a.left_menu, p.left_menu_click),
        xr::Binding::new(a.right_menu, p.right_menu_click),
    ]
}

/// HP Reverb G2 (`XR_EXT_hp_mixed_reality_controller`); WMR ergonomics with Touch-style face buttons,
/// no trackpad.
fn hp_reverb_g2_bindings<'a>(a: &'a ActionRefs<'a>, p: &'a BindingPaths) -> Vec<xr::Binding<'a>> {
    vec![
        xr::Binding::new(a.left_grip_pose, p.left_grip_pose),
        xr::Binding::new(a.right_grip_pose, p.right_grip_pose),
        xr::Binding::new(a.left_aim_pose, p.left_aim_pose),
        xr::Binding::new(a.right_aim_pose, p.right_aim_pose),
        xr::Binding::new(a.left_trigger, p.left_trigger_value),
        xr::Binding::new(a.right_trigger, p.right_trigger_value),
        xr::Binding::new(a.left_squeeze, p.left_squeeze_value),
        xr::Binding::new(a.right_squeeze, p.right_squeeze_value),
        xr::Binding::new(a.left_thumbstick, p.left_thumbstick),
        xr::Binding::new(a.right_thumbstick, p.right_thumbstick),
        xr::Binding::new(a.left_thumbstick_click, p.left_thumbstick_click),
        xr::Binding::new(a.right_thumbstick_click, p.right_thumbstick_click),
        xr::Binding::new(a.left_primary, p.left_x_click),
        xr::Binding::new(a.left_secondary, p.left_y_click),
        xr::Binding::new(a.right_primary, p.right_a_click),
        xr::Binding::new(a.right_secondary, p.right_b_click),
        xr::Binding::new(a.left_menu, p.left_menu_click),
        xr::Binding::new(a.right_menu, p.right_menu_click),
    ]
}

/// Samsung Odyssey (`XR_EXT_samsung_odyssey_controller`); identical layout to Microsoft Motion.
fn samsung_odyssey_bindings<'a>(
    a: &'a ActionRefs<'a>,
    p: &'a BindingPaths,
) -> Vec<xr::Binding<'a>> {
    microsoft_motion_bindings(a, p)
}

/// HTC Vive Cosmos (`XR_HTC_vive_cosmos_controller_interaction`); thumbstick + face buttons,
/// no trackpad. `menu/click` is left-only on Cosmos; the right hand exposes `system/click`
/// instead, which we do not currently bind.
fn htc_vive_cosmos_bindings<'a>(
    a: &'a ActionRefs<'a>,
    p: &'a BindingPaths,
) -> Vec<xr::Binding<'a>> {
    vec![
        xr::Binding::new(a.left_grip_pose, p.left_grip_pose),
        xr::Binding::new(a.right_grip_pose, p.right_grip_pose),
        xr::Binding::new(a.left_aim_pose, p.left_aim_pose),
        xr::Binding::new(a.right_aim_pose, p.right_aim_pose),
        xr::Binding::new(a.left_trigger, p.left_trigger_value),
        xr::Binding::new(a.right_trigger, p.right_trigger_value),
        xr::Binding::new(a.left_trigger_click, p.left_trigger_click),
        xr::Binding::new(a.right_trigger_click, p.right_trigger_click),
        xr::Binding::new(a.left_squeeze_click, p.left_squeeze_click),
        xr::Binding::new(a.right_squeeze_click, p.right_squeeze_click),
        xr::Binding::new(a.left_thumbstick, p.left_thumbstick),
        xr::Binding::new(a.right_thumbstick, p.right_thumbstick),
        xr::Binding::new(a.left_thumbstick_click, p.left_thumbstick_click),
        xr::Binding::new(a.right_thumbstick_click, p.right_thumbstick_click),
        xr::Binding::new(a.left_primary, p.left_x_click),
        xr::Binding::new(a.left_secondary, p.left_y_click),
        xr::Binding::new(a.right_primary, p.right_a_click),
        xr::Binding::new(a.right_secondary, p.right_b_click),
        xr::Binding::new(a.left_menu, p.left_menu_click),
    ]
}

/// HTC Vive Focus 3 (`XR_HTC_vive_focus3_controller_interaction`); standalone Vive controller with
/// Touch-style face buttons, thumbrest touch, and analog squeeze. `menu/click` is left-only;
/// the right hand exposes `system/click` instead, which we do not currently bind.
fn htc_vive_focus3_bindings<'a>(
    a: &'a ActionRefs<'a>,
    p: &'a BindingPaths,
) -> Vec<xr::Binding<'a>> {
    vec![
        xr::Binding::new(a.left_grip_pose, p.left_grip_pose),
        xr::Binding::new(a.right_grip_pose, p.right_grip_pose),
        xr::Binding::new(a.left_aim_pose, p.left_aim_pose),
        xr::Binding::new(a.right_aim_pose, p.right_aim_pose),
        xr::Binding::new(a.left_trigger, p.left_trigger_value),
        xr::Binding::new(a.right_trigger, p.right_trigger_value),
        xr::Binding::new(a.left_trigger_touch, p.left_trigger_touch),
        xr::Binding::new(a.right_trigger_touch, p.right_trigger_touch),
        xr::Binding::new(a.left_squeeze, p.left_squeeze_value),
        xr::Binding::new(a.right_squeeze, p.right_squeeze_value),
        xr::Binding::new(a.left_squeeze_click, p.left_squeeze_click),
        xr::Binding::new(a.right_squeeze_click, p.right_squeeze_click),
        xr::Binding::new(a.left_thumbstick, p.left_thumbstick),
        xr::Binding::new(a.right_thumbstick, p.right_thumbstick),
        xr::Binding::new(a.left_thumbstick_touch, p.left_thumbstick_touch),
        xr::Binding::new(a.right_thumbstick_touch, p.right_thumbstick_touch),
        xr::Binding::new(a.left_thumbstick_click, p.left_thumbstick_click),
        xr::Binding::new(a.right_thumbstick_click, p.right_thumbstick_click),
        xr::Binding::new(a.left_primary, p.left_x_click),
        xr::Binding::new(a.left_secondary, p.left_y_click),
        xr::Binding::new(a.right_primary, p.right_a_click),
        xr::Binding::new(a.right_secondary, p.right_b_click),
        xr::Binding::new(a.left_thumbrest_touch, p.left_thumbrest_touch),
        xr::Binding::new(a.right_thumbrest_touch, p.right_thumbrest_touch),
        xr::Binding::new(a.left_menu, p.left_menu_click),
    ]
}

/// Meta Touch Pro (`XR_FB_touch_controller_pro`); Touch superset on Quest Pro hardware.
/// We bind the standard Touch surface; pro-specific extras (trigger curl/slide, thumbrest force,
/// stylus force) are not currently exposed as host actions.
fn meta_touch_pro_bindings<'a>(a: &'a ActionRefs<'a>, p: &'a BindingPaths) -> Vec<xr::Binding<'a>> {
    oculus_touch_bindings(a, p)
}

/// Meta Touch Plus (`XR_META_touch_controller_plus`); Touch superset on Quest 3 hardware.
/// Same Touch surface as the standard Oculus profile.
fn meta_touch_plus_bindings<'a>(
    a: &'a ActionRefs<'a>,
    p: &'a BindingPaths,
) -> Vec<xr::Binding<'a>> {
    oculus_touch_bindings(a, p)
}

/// Applies all known interaction profile binding tables; succeeds if at least one profile accepted bindings.
///
/// Each profile suggestion is logged (info on accept, warn on reject) so runtime mismatches
/// — e.g. a profile rejected because the runtime does not recognise an extension path — are
/// diagnosable rather than silently swallowed by the previous "first error wins" pattern.
///
/// Profiles whose OpenXR extension was not enabled on the instance (see [`ProfileExtensionGates`])
/// are skipped so runtimes that do not know those profile paths do not log errors.
pub(super) fn apply_suggested_interaction_bindings(
    instance: &xr::Instance,
    profiles: &InteractionProfilePaths,
    paths: &BindingPaths,
    actions: &ActionRefs<'_>,
    gates: &ProfileExtensionGates,
) -> Result<(), xr::sys::Result> {
    let a = actions;
    let p = paths;
    let ip = profiles;

    let mut any_bindings = false;
    let mut last_binding_err = None;
    let mut suggest =
        |profile_name: &str, profile: xr::Path, bindings: &[xr::Binding<'_>]| match instance
            .suggest_interaction_profile_bindings(profile, bindings)
        {
            Ok(()) => {
                any_bindings = true;
                logger::info!("OpenXR binding suggestion accepted: {profile_name}");
            }
            Err(e) => {
                last_binding_err = Some(e);
                logger::warn!("OpenXR binding suggestion rejected: {profile_name}: {e:?}");
            }
        };

    let oculus_touch = oculus_touch_bindings(a, p);
    suggest(
        "/interaction_profiles/oculus/touch_controller",
        ip.oculus_touch,
        &oculus_touch,
    );
    if gates.bd_controller {
        let pico4 = pico4_controller_bindings(a, p);
        suggest(
            "/interaction_profiles/bytedance/pico4_controller",
            ip.pico4_controller,
            &pico4,
        );
        let pico_neo3 = pico_neo3_controller_bindings(a, p);
        suggest(
            "/interaction_profiles/bytedance/pico_neo3_controller",
            ip.pico_neo3_controller,
            &pico_neo3,
        );
    }
    let valve_index = valve_index_bindings(a, p);
    suggest(
        "/interaction_profiles/valve/index_controller",
        ip.valve_index,
        &valve_index,
    );
    let htc_vive = htc_vive_bindings(a, p);
    suggest(
        "/interaction_profiles/htc/vive_controller",
        ip.htc_vive,
        &htc_vive,
    );
    if gates.htc_vive_cosmos_controller_interaction {
        let cosmos = htc_vive_cosmos_bindings(a, p);
        suggest(
            "/interaction_profiles/htc/vive_cosmos_controller",
            ip.htc_vive_cosmos,
            &cosmos,
        );
    }
    if gates.htc_vive_focus3_controller_interaction {
        let focus3 = htc_vive_focus3_bindings(a, p);
        suggest(
            "/interaction_profiles/htc/vive_focus3_controller",
            ip.htc_vive_focus3,
            &focus3,
        );
    }
    let microsoft_motion = microsoft_motion_bindings(a, p);
    suggest(
        "/interaction_profiles/microsoft/motion_controller",
        ip.microsoft_motion,
        &microsoft_motion,
    );
    if gates.ext_hp_mixed_reality_controller {
        let reverb = hp_reverb_g2_bindings(a, p);
        suggest(
            "/interaction_profiles/hp/mixed_reality_controller",
            ip.hp_reverb_g2,
            &reverb,
        );
    }
    if gates.ext_samsung_odyssey_controller {
        let odyssey = samsung_odyssey_bindings(a, p);
        suggest(
            "/interaction_profiles/samsung/odyssey_controller",
            ip.samsung_odyssey,
            &odyssey,
        );
    }
    if gates.fb_touch_controller_pro {
        let touch_pro = meta_touch_pro_bindings(a, p);
        suggest(
            "/interaction_profiles/facebook/touch_controller_pro",
            ip.meta_touch_pro,
            &touch_pro,
        );
    }
    if gates.meta_touch_controller_plus {
        let touch_plus = meta_touch_plus_bindings(a, p);
        suggest(
            "/interaction_profiles/meta/touch_controller_plus",
            ip.meta_touch_plus,
            &touch_plus,
        );
    }
    if gates.khr_generic_controller {
        let generic_controller = generic_controller_bindings(a, p);
        suggest(
            "/interaction_profiles/khr/generic_controller",
            ip.generic_controller,
            &generic_controller,
        );
    }
    let simple_controller = simple_controller_bindings(a, p);
    suggest(
        "/interaction_profiles/khr/simple_controller",
        ip.simple_controller,
        &simple_controller,
    );

    if !any_bindings {
        return Err(last_binding_err.unwrap_or(xr::sys::Result::ERROR_PATH_UNSUPPORTED));
    }
    Ok(())
}
