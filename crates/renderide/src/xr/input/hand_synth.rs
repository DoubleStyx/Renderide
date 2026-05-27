//! Synthesises per-finger [`HandState`] data from controller input.
//!
//! Without this, the host would receive no hand-tracking data and its `HandPoser` would reset
//! every finger to its `OriginalRotation`, leaving the avatar playing the desktop idle pose while
//! the user is in VR.
//!
//! The presets below match the host's `Idle` and `Fist` finger-pose presets.
//! Segment layout matches the host's unpack loop: 24 entries indexed by
//! `BodyNode - LeftThumbMetacarpal`, ordered Thumb(Met,Prox,Dist,Tip), Index(Met,Prox,Inter,Dist,Tip),
//! Middle(..), Ring(..), Pinky(..). Right-hand [`HandState`] values reuse the same indexing but hold
//! right-hand data; the host mirrors via `bodyNode.GetSide(chirality)`.
//!
//! The curl blend is deliberately conservative: we set [`HandState::tracks_metacarpals`] to
//! `false`, so the host overrides non-thumb metacarpals to the avatar's own rest pose. Thumb curl
//! follows thumb touch sensors, index curl follows the trigger analog, and middle/ring/pinky curl
//! follows the squeeze (grip) analog.

#[cfg(test)]
mod flat_presets;
mod presets;

use glam::{Quat, Vec3};

use crate::shared::{Chirality, HandState, VRControllerState};

use presets::{
    FIST_POS_LEFT, FIST_POS_RIGHT, FIST_ROT_LEFT, FIST_ROT_RIGHT, IDLE_POS_LEFT, IDLE_POS_RIGHT,
    IDLE_ROT_LEFT, IDLE_ROT_RIGHT, LEFT_HAND_ID, RIGHT_HAND_ID, SEGMENT_COUNTS,
    TOTAL_SEGMENT_COUNT,
};

/// Which finger a [`HandState`] segment index (0..24) belongs to.
#[derive(Clone, Copy, PartialEq, Eq)]
enum FingerKind {
    /// Thumb: segments 0..=3 (Metacarpal, Proximal, Distal, Tip -- no Intermediate).
    Thumb,
    /// Index finger: segments 4..=8.
    Index,
    /// Middle finger: segments 9..=13.
    Middle,
    /// Ring finger: segments 14..=18.
    Ring,
    /// Pinky: segments 19..=23.
    Pinky,
}

/// Finger order used by the packed `HandState` segment arrays.
const FINGER_KINDS: [FingerKind; 5] = [
    FingerKind::Thumb,
    FingerKind::Index,
    FingerKind::Middle,
    FingerKind::Ring,
    FingerKind::Pinky,
];

/// Controller-derived inputs used to drive the idle<->fist blend.
struct ControllerCurlInputs {
    /// Which hand this controller drives.
    side: Chirality,
    /// Tracking-space wrist position to report on [`HandState::wrist_position`]. When the runtime
    /// advertises a bound hand, this is the controller pose composed with the per-profile
    /// bound-hand offset (`controller.position + controller.rotation * controller.hand_position`),
    /// matching `TrackedDevicePositioner`'s own resolution of the
    /// `MappableTrackedObject.BodyNodePositionOffset` path in FrooxEngine. Otherwise it is the
    /// controller's tracking-space pose directly. `hand_position` / `hand_rotation` on
    /// [`VRControllerState`] are registration-time offsets (see
    /// [`crate::xr::input::pose::hand_pose_defaults`]), not tracking-space poses.
    wrist_position: Vec3,
    /// Tracking-space wrist rotation to report on [`HandState::wrist_rotation`]. Composed the same
    /// way as [`Self::wrist_position`] and normalised.
    wrist_rotation: Quat,
    /// Grip/squeeze analog in 0..=1. Drives middle, ring, and pinky curl.
    grip: f32,
    /// Trigger analog in 0..=1. Drives index finger curl.
    trigger: f32,
    /// Thumb boolean resting state. Drives thumb finger curl.
    thumb: bool,
}

/// Profile-agnostic pose + analog inputs feeding curl synthesis.
///
/// Each `VRControllerState` variant has the same set of pose fields and an analog trigger.
/// There are only a handful of profile-specific bits:
/// - `grip` is `f32` on Touch / Index but `bool` on Vive / WMR.
/// - `thumb` is conditioned on different sensors depending on controllers.
///
/// Match arms in [`extract_curl_inputs`] pre-coerce grip into `0.0` / `1.0` for the boolean
/// profiles, then call into [`curl_inputs_from_source`] uniformly.
struct ControllerCurlSource {
    is_tracking: bool,
    side: Chirality,
    position: Vec3,
    rotation: Quat,
    has_bound_hand: bool,
    hand_position: Vec3,
    hand_rotation: Quat,
    grip: f32,
    trigger: f32,
    thumb: bool,
}

/// Builds [`ControllerCurlSource`] from a profile-specific controller state with an explicit
/// grip expression. The other pose fields share the same names across every profile struct so
/// they forward 1:1.
macro_rules! curl_source {
    ($s:expr, grip = $grip:expr, thumb = $thumb:expr) => {
        ControllerCurlSource {
            is_tracking: $s.is_tracking,
            side: $s.side,
            position: $s.position,
            rotation: $s.rotation,
            has_bound_hand: $s.has_bound_hand,
            hand_position: $s.hand_position,
            hand_rotation: $s.hand_rotation,
            grip: $grip,
            trigger: $s.trigger,
            thumb: $thumb,
        }
    };
}

/// Builds [`ControllerCurlInputs`] from a profile-agnostic [`ControllerCurlSource`].
///
/// Returns `None` when the controller is not tracked (we do not want to feed the host random
/// hand poses). Grip and trigger are clamped to `0..=1` here so callers do not need to.
fn curl_inputs_from_source(src: ControllerCurlSource) -> Option<ControllerCurlInputs> {
    if !src.is_tracking {
        return None;
    }
    let (wrist_position, wrist_rotation) = if src.has_bound_hand {
        (
            src.position + src.rotation * src.hand_position,
            (src.rotation * src.hand_rotation).normalize(),
        )
    } else {
        (src.position, src.rotation)
    };
    Some(ControllerCurlInputs {
        side: src.side,
        wrist_position,
        wrist_rotation,
        grip: src.grip.clamp(0.0, 1.0),
        trigger: src.trigger.clamp(0.0, 1.0),
        thumb: src.thumb,
    })
}

/// Extracts the curl-driving inputs from a [`VRControllerState`] variant.
///
/// Returns `None` when the controller is not tracked or when the variant is not produced by
/// the current OpenXR dispatch. For controllers whose grip is a boolean (Vive wand, WMR), the
/// boolean is coerced to `0.0` / `1.0` before clamping.
fn extract_curl_inputs(controller: &VRControllerState) -> Option<ControllerCurlInputs> {
    match controller {
        VRControllerState::TouchControllerState(s) => curl_inputs_from_source(curl_source!(
            s,
            grip = s.grip,
            thumb = s.thumbrest_touch || s.button_xa_touch || s.button_yb_touch || s.joystick_touch
        )),
        VRControllerState::IndexControllerState(s) => curl_inputs_from_source(curl_source!(
            s,
            grip = s.grip,
            thumb = s.touchpad_touch || s.button_atouch || s.button_btouch || s.joystick_touch
        )),
        VRControllerState::ViveControllerState(s) => curl_inputs_from_source(curl_source!(
            s,
            grip = if s.grip { 1.0 } else { 0.0 },
            thumb = s.touchpad_touch
        )),
        VRControllerState::WindowsMRControllerState(s) => curl_inputs_from_source(curl_source!(
            s,
            grip = if s.grip { 1.0 } else { 0.0 },
            thumb = s.touchpad_touch
        )),
        VRControllerState::CosmosControllerState(_)
        | VRControllerState::GenericControllerState(_)
        | VRControllerState::HPReverbControllerState(_)
        | VRControllerState::PicoNeo2ControllerState(_) => {
            // These variants are not produced by the current OpenXR dispatch
            // (`crate::xr::input::state::dispatch_openxr_profile_to_host_state`). If they start
            // being emitted, add the analogous extractor here.
            None
        }
    }
}

/// Returns the idle<->fist blend factor for a given finger.
///
/// Non-thumb metacarpals are overridden on the host when [`HandState::tracks_metacarpals`] is
/// `false`, but blending them keeps the raw IPC pose internally coherent for consumers that read
/// the segment data directly.
/// - Index curl follows `trigger`.
/// - Middle, ring, and pinky curl follow `grip`.
/// - Thumb curl follows `thumb`.
fn blend_factor_for_segment(finger: FingerKind, grip: f32, trigger: f32, thumb: f32) -> f32 {
    match finger {
        FingerKind::Thumb => thumb,
        FingerKind::Index => trigger,
        FingerKind::Middle | FingerKind::Ring | FingerKind::Pinky => grip,
    }
}

/// Local idle/fist data used to append one blended finger.
struct LocalFingerBlend<'a> {
    /// Idle local segment positions.
    idle_positions: &'a [[f32; 3]; 5],
    /// Idle local segment rotations.
    idle_rotations: &'a [[f32; 4]; 5],
    /// Fist local segment positions.
    fist_positions: &'a [[f32; 3]; 5],
    /// Fist local segment rotations.
    fist_rotations: &'a [[f32; 4]; 5],
    /// Number of valid segment rows for this finger.
    segment_count: usize,
    /// Idle-to-fist blend factor.
    blend: f32,
}

/// Appends one blended finger to the flat hand-space segment arrays sent over IPC.
fn append_blended_finger_pose(
    segment_positions: &mut Vec<Vec3>,
    segment_rotations: &mut Vec<Quat>,
    finger: LocalFingerBlend<'_>,
) {
    let mut parent_position = Vec3::ZERO;
    let mut parent_rotation = Quat::IDENTITY;
    for segment_index in 0..finger.segment_count {
        let idle_position = Vec3::from_array(finger.idle_positions[segment_index]);
        let fist_position = Vec3::from_array(finger.fist_positions[segment_index]);
        let local_position = idle_position.lerp(fist_position, finger.blend);
        let flat_position = parent_position + parent_rotation * local_position;
        segment_positions.push(flat_position);

        let idle_rotation = Quat::from_array(finger.idle_rotations[segment_index]);
        let fist_rotation = Quat::from_array(finger.fist_rotations[segment_index]);
        let local_rotation = idle_rotation.slerp(fist_rotation, finger.blend).normalize();
        let flat_rotation = (parent_rotation * local_rotation).normalize();
        segment_rotations.push(flat_rotation);

        parent_position = flat_position;
        parent_rotation = flat_rotation;
    }
}

/// Builds a [`HandState`] for one controller by blending the idle and fist presets. Returns
/// `None` if the controller is untracked or not a variant we drive hands for.
fn synthesize_one_hand(controller: &VRControllerState) -> Option<HandState> {
    let inputs = extract_curl_inputs(controller)?;
    let thumb = if inputs.thumb { 1.0 } else { 0.0 };
    let (pos_idle, rot_idle, pos_fist, rot_fist, unique_id) = match inputs.side {
        Chirality::Left => (
            &IDLE_POS_LEFT,
            &IDLE_ROT_LEFT,
            &FIST_POS_LEFT,
            &FIST_ROT_LEFT,
            LEFT_HAND_ID,
        ),
        Chirality::Right => (
            &IDLE_POS_RIGHT,
            &IDLE_ROT_RIGHT,
            &FIST_POS_RIGHT,
            &FIST_ROT_RIGHT,
            RIGHT_HAND_ID,
        ),
    };
    let mut segment_positions = Vec::with_capacity(TOTAL_SEGMENT_COUNT);
    let mut segment_rotations = Vec::with_capacity(TOTAL_SEGMENT_COUNT);
    for (finger_index, &finger_segment_count) in SEGMENT_COUNTS.iter().enumerate() {
        let finger = FINGER_KINDS[finger_index];
        let t = blend_factor_for_segment(finger, inputs.grip, inputs.trigger, thumb);
        append_blended_finger_pose(
            &mut segment_positions,
            &mut segment_rotations,
            LocalFingerBlend {
                idle_positions: &pos_idle[finger_index],
                idle_rotations: &rot_idle[finger_index],
                fist_positions: &pos_fist[finger_index],
                fist_rotations: &rot_fist[finger_index],
                segment_count: finger_segment_count,
                blend: t,
            },
        );
    }
    Some(HandState {
        unique_id: Some(unique_id.to_string()),
        priority: 0,
        chirality: inputs.side,
        is_device_active: true,
        is_tracking: true,
        tracks_metacarpals: false,
        confidence: 1.0,
        wrist_position: inputs.wrist_position,
        wrist_rotation: inputs.wrist_rotation,
        segment_positions,
        segment_rotations,
    })
}

/// Produces one [`HandState`] per tracked VR controller in `controllers`, blending the idle and
/// fist presets using the controller's grip and trigger analogs.
///
/// Call this every XR frame after building the per-controller [`VRControllerState`] slice; the
/// returned vector belongs on [`crate::shared::VRInputsState::hands`].
pub fn synthesize_hand_states(controllers: &[VRControllerState]) -> Vec<HandState> {
    controllers.iter().filter_map(synthesize_one_hand).collect()
}

#[cfg(test)]
mod tests;
