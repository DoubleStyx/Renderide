//! Path resolution for [`super::openxr_actions::create_openxr_input_parts`].

use openxr as xr;

use super::bindings::BindingPaths;

/// Left/right grip pose and left/right aim pose [`xr::Path`] tuple.
type GripAimPathPack = (xr::Path, xr::Path, xr::Path, xr::Path);
/// Trigger value, touch (×2), and click (×2) paths.
type TriggerPathPack = (xr::Path, xr::Path, xr::Path, xr::Path, xr::Path, xr::Path);
/// Squeeze value and click paths for both hands.
type SqueezePathPack = (xr::Path, xr::Path, xr::Path, xr::Path);
/// Thumbstick vector, touch, and click paths for both hands.
type ThumbstickPathPack = (xr::Path, xr::Path, xr::Path, xr::Path, xr::Path, xr::Path);
/// Trackpad vector, touch, click, and force paths for both hands.
type TrackpadPathPack = (
    xr::Path,
    xr::Path,
    xr::Path,
    xr::Path,
    xr::Path,
    xr::Path,
    xr::Path,
    xr::Path,
);
/// Face button click/touch paths (XY left, AB both sides).
type FaceButtonPathPack = (
    xr::Path,
    xr::Path,
    xr::Path,
    xr::Path,
    xr::Path,
    xr::Path,
    xr::Path,
    xr::Path,
    xr::Path,
    xr::Path,
    xr::Path,
    xr::Path,
);

/// User hand paths and interaction profile [`xr::Path`] handles.
pub(super) struct UserAndProfilePaths {
    pub left_user_path: xr::Path,
    pub right_user_path: xr::Path,
    pub oculus_touch_profile: xr::Path,
    pub valve_index_profile: xr::Path,
    pub htc_vive_profile: xr::Path,
    pub microsoft_motion_profile: xr::Path,
    pub generic_controller_profile: xr::Path,
    pub simple_controller_profile: xr::Path,
    pub pico4_controller_profile: xr::Path,
    pub pico_neo3_controller_profile: xr::Path,
    pub hp_reverb_g2_profile: xr::Path,
    pub samsung_odyssey_profile: xr::Path,
    pub htc_vive_cosmos_profile: xr::Path,
    pub htc_vive_focus3_profile: xr::Path,
    pub meta_touch_pro_profile: xr::Path,
    pub meta_touch_plus_profile: xr::Path,
}

/// Resolves `/user/hand/*` and `/interaction_profiles/*` paths.
///
/// Every profile path is resolved unconditionally; whether the runtime actually recognises a
/// vendor profile depends on whether the corresponding extension was enabled at instance
/// creation. Paths that the runtime does not know still resolve successfully as
/// [`xr::Instance::string_to_path`] is a string-interning operation — they just never match in
/// [`super::openxr_input::OpenxrInput::detect_profile`].
pub(super) fn resolve_user_and_profile_paths(
    instance: &xr::Instance,
) -> Result<UserAndProfilePaths, xr::sys::Result> {
    Ok(UserAndProfilePaths {
        left_user_path: instance.string_to_path("/user/hand/left")?,
        right_user_path: instance.string_to_path("/user/hand/right")?,
        oculus_touch_profile: instance
            .string_to_path("/interaction_profiles/oculus/touch_controller")?,
        valve_index_profile: instance
            .string_to_path("/interaction_profiles/valve/index_controller")?,
        htc_vive_profile: instance.string_to_path("/interaction_profiles/htc/vive_controller")?,
        microsoft_motion_profile: instance
            .string_to_path("/interaction_profiles/microsoft/motion_controller")?,
        generic_controller_profile: instance
            .string_to_path("/interaction_profiles/khr/generic_controller")?,
        simple_controller_profile: instance
            .string_to_path("/interaction_profiles/khr/simple_controller")?,
        pico4_controller_profile: instance
            .string_to_path("/interaction_profiles/bytedance/pico4_controller")?,
        pico_neo3_controller_profile: instance
            .string_to_path("/interaction_profiles/bytedance/pico_neo3_controller")?,
        hp_reverb_g2_profile: instance
            .string_to_path("/interaction_profiles/hp/mixed_reality_controller")?,
        samsung_odyssey_profile: instance
            .string_to_path("/interaction_profiles/samsung/odyssey_controller")?,
        htc_vive_cosmos_profile: instance
            .string_to_path("/interaction_profiles/htc/vive_cosmos_controller")?,
        htc_vive_focus3_profile: instance
            .string_to_path("/interaction_profiles/htc/vive_focus3_controller")?,
        meta_touch_pro_profile: instance
            .string_to_path("/interaction_profiles/facebook/touch_controller_pro")?,
        meta_touch_plus_profile: instance
            .string_to_path("/interaction_profiles/meta/touch_controller_plus")?,
    })
}

/// Grip and aim pose paths used across interaction profiles.
fn resolve_grip_aim_subpaths(instance: &xr::Instance) -> Result<GripAimPathPack, xr::sys::Result> {
    Ok((
        instance.string_to_path("/user/hand/left/input/grip/pose")?,
        instance.string_to_path("/user/hand/right/input/grip/pose")?,
        instance.string_to_path("/user/hand/left/input/aim/pose")?,
        instance.string_to_path("/user/hand/right/input/aim/pose")?,
    ))
}

/// Trigger value, touch, and click subpaths (left then right for each category).
fn resolve_trigger_subpaths(instance: &xr::Instance) -> Result<TriggerPathPack, xr::sys::Result> {
    Ok((
        instance.string_to_path("/user/hand/left/input/trigger/value")?,
        instance.string_to_path("/user/hand/right/input/trigger/value")?,
        instance.string_to_path("/user/hand/left/input/trigger/touch")?,
        instance.string_to_path("/user/hand/right/input/trigger/touch")?,
        instance.string_to_path("/user/hand/left/input/trigger/click")?,
        instance.string_to_path("/user/hand/right/input/trigger/click")?,
    ))
}

/// Squeeze analog and digital subpaths.
fn resolve_squeeze_subpaths(instance: &xr::Instance) -> Result<SqueezePathPack, xr::sys::Result> {
    Ok((
        instance.string_to_path("/user/hand/left/input/squeeze/value")?,
        instance.string_to_path("/user/hand/right/input/squeeze/value")?,
        instance.string_to_path("/user/hand/left/input/squeeze/click")?,
        instance.string_to_path("/user/hand/right/input/squeeze/click")?,
    ))
}

/// Thumbstick vector, touch, and click subpaths.
fn resolve_thumbstick_subpaths(
    instance: &xr::Instance,
) -> Result<ThumbstickPathPack, xr::sys::Result> {
    Ok((
        instance.string_to_path("/user/hand/left/input/thumbstick")?,
        instance.string_to_path("/user/hand/right/input/thumbstick")?,
        instance.string_to_path("/user/hand/left/input/thumbstick/touch")?,
        instance.string_to_path("/user/hand/right/input/thumbstick/touch")?,
        instance.string_to_path("/user/hand/left/input/thumbstick/click")?,
        instance.string_to_path("/user/hand/right/input/thumbstick/click")?,
    ))
}

/// Trackpad vector, touch, click, and force subpaths.
fn resolve_trackpad_subpaths(instance: &xr::Instance) -> Result<TrackpadPathPack, xr::sys::Result> {
    Ok((
        instance.string_to_path("/user/hand/left/input/trackpad")?,
        instance.string_to_path("/user/hand/right/input/trackpad")?,
        instance.string_to_path("/user/hand/left/input/trackpad/touch")?,
        instance.string_to_path("/user/hand/right/input/trackpad/touch")?,
        instance.string_to_path("/user/hand/left/input/trackpad/click")?,
        instance.string_to_path("/user/hand/right/input/trackpad/click")?,
        instance.string_to_path("/user/hand/left/input/trackpad/force")?,
        instance.string_to_path("/user/hand/right/input/trackpad/force")?,
    ))
}

/// Face button (XY on left, AB on right) click and touch subpaths; some bindings omit touch on XY.
fn resolve_face_button_subpaths(
    instance: &xr::Instance,
) -> Result<FaceButtonPathPack, xr::sys::Result> {
    Ok((
        instance.string_to_path("/user/hand/left/input/x/click")?,
        instance.string_to_path("/user/hand/left/input/y/click")?,
        instance.string_to_path("/user/hand/left/input/x/touch")?,
        instance.string_to_path("/user/hand/left/input/y/touch")?,
        instance.string_to_path("/user/hand/left/input/a/click")?,
        instance.string_to_path("/user/hand/left/input/b/click")?,
        instance.string_to_path("/user/hand/left/input/a/touch")?,
        instance.string_to_path("/user/hand/left/input/b/touch")?,
        instance.string_to_path("/user/hand/right/input/a/click")?,
        instance.string_to_path("/user/hand/right/input/b/click")?,
        instance.string_to_path("/user/hand/right/input/a/touch")?,
        instance.string_to_path("/user/hand/right/input/b/touch")?,
    ))
}

/// Menu, thumbrest, and system-select click paths.
type MenuThumbrestSelectPathPack = (xr::Path, xr::Path, xr::Path, xr::Path, xr::Path, xr::Path);

fn resolve_menu_thumbrest_select_subpaths(
    instance: &xr::Instance,
) -> Result<MenuThumbrestSelectPathPack, xr::sys::Result> {
    Ok((
        instance.string_to_path("/user/hand/left/input/menu/click")?,
        instance.string_to_path("/user/hand/right/input/menu/click")?,
        instance.string_to_path("/user/hand/left/input/thumbrest/touch")?,
        instance.string_to_path("/user/hand/right/input/thumbrest/touch")?,
        instance.string_to_path("/user/hand/left/input/select/click")?,
        instance.string_to_path("/user/hand/right/input/select/click")?,
    ))
}

/// Intermediate [`xr::Path`] tuples from [`resolve_*_subpaths`] for [`binding_paths_from_resolved`].
struct ResolvedBindingPathPacks {
    grip: GripAimPathPack,
    trigger: TriggerPathPack,
    squeeze: SqueezePathPack,
    thumbstick: ThumbstickPathPack,
    trackpad: TrackpadPathPack,
    face: FaceButtonPathPack,
    menu_thumbrest_select: MenuThumbrestSelectPathPack,
}

/// Resolves every hand `/input/...` subpath used for default binding suggestions.
fn resolve_all_binding_path_packs(
    instance: &xr::Instance,
) -> Result<ResolvedBindingPathPacks, xr::sys::Result> {
    Ok(ResolvedBindingPathPacks {
        grip: resolve_grip_aim_subpaths(instance)?,
        trigger: resolve_trigger_subpaths(instance)?,
        squeeze: resolve_squeeze_subpaths(instance)?,
        thumbstick: resolve_thumbstick_subpaths(instance)?,
        trackpad: resolve_trackpad_subpaths(instance)?,
        face: resolve_face_button_subpaths(instance)?,
        menu_thumbrest_select: resolve_menu_thumbrest_select_subpaths(instance)?,
    })
}

/// Assembles [`BindingPaths`] from pre-resolved OpenXR path tuples.
fn binding_paths_from_resolved(p: ResolvedBindingPathPacks) -> BindingPaths {
    let (left_grip_pose, right_grip_pose, left_aim_pose, right_aim_pose) = p.grip;
    let (
        left_trigger_value,
        right_trigger_value,
        left_trigger_touch,
        right_trigger_touch,
        left_trigger_click,
        right_trigger_click,
    ) = p.trigger;
    let (left_squeeze_value, right_squeeze_value, left_squeeze_click, right_squeeze_click) =
        p.squeeze;
    let (
        left_thumbstick,
        right_thumbstick,
        left_thumbstick_touch,
        right_thumbstick_touch,
        left_thumbstick_click,
        right_thumbstick_click,
    ) = p.thumbstick;
    let (
        left_trackpad,
        right_trackpad,
        left_trackpad_touch,
        right_trackpad_touch,
        left_trackpad_click,
        right_trackpad_click,
        left_trackpad_force,
        right_trackpad_force,
    ) = p.trackpad;
    let (
        left_x_click,
        left_y_click,
        left_x_touch,
        left_y_touch,
        left_a_click,
        left_b_click,
        left_a_touch,
        left_b_touch,
        right_a_click,
        right_b_click,
        right_a_touch,
        right_b_touch,
    ) = p.face;
    let (
        left_menu_click,
        right_menu_click,
        left_thumbrest_touch,
        right_thumbrest_touch,
        left_select_click,
        right_select_click,
    ) = p.menu_thumbrest_select;

    BindingPaths {
        left_grip_pose,
        right_grip_pose,
        left_aim_pose,
        right_aim_pose,
        left_trigger_value,
        right_trigger_value,
        left_trigger_touch,
        right_trigger_touch,
        left_trigger_click,
        right_trigger_click,
        left_squeeze_value,
        right_squeeze_value,
        left_squeeze_click,
        right_squeeze_click,
        left_thumbstick,
        right_thumbstick,
        left_thumbstick_touch,
        right_thumbstick_touch,
        left_thumbstick_click,
        right_thumbstick_click,
        left_trackpad,
        right_trackpad,
        left_trackpad_touch,
        right_trackpad_touch,
        left_trackpad_click,
        right_trackpad_click,
        left_trackpad_force,
        right_trackpad_force,
        left_x_click,
        left_y_click,
        left_x_touch,
        left_y_touch,
        left_a_click,
        left_b_click,
        left_a_touch,
        left_b_touch,
        right_a_click,
        right_b_click,
        right_a_touch,
        right_b_touch,
        left_menu_click,
        right_menu_click,
        left_thumbrest_touch,
        right_thumbrest_touch,
        left_select_click,
        right_select_click,
    }
}

/// All `/user/hand/*/input/...` subpaths used when building [`BindingPaths`].
pub(super) fn resolve_binding_subpaths(
    instance: &xr::Instance,
) -> Result<BindingPaths, xr::sys::Result> {
    let packs = resolve_all_binding_path_packs(instance)?;
    Ok(binding_paths_from_resolved(packs))
}
