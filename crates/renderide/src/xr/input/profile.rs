//! OpenXR interaction profile classification and human-readable device labels.

use std::sync::atomic::{AtomicU8, Ordering};

use crate::shared::Chirality;

/// Active interaction profile for a hand, derived from the OpenXR session's current interaction profile.
///
/// `Pico4`, `PicoNeo3`, `HpReverbG2`, `ViveCosmos`, and `ViveFocus3` all share the touch-class
/// host state shape (the host wire format has no dedicated variants for them) but are tracked
/// separately so logs and the host-facing device model string identify them correctly instead
/// of being flattened into Touch / WMR.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ActiveControllerProfile {
    Touch,
    Index,
    Vive,
    WindowsMr,
    Pico4,
    PicoNeo3,
    HpReverbG2,
    ViveCosmos,
    ViveFocus3,
    Generic,
    Simple,
}

pub(super) fn profile_code(profile: ActiveControllerProfile) -> u8 {
    match profile {
        ActiveControllerProfile::Touch => 1,
        ActiveControllerProfile::Index => 2,
        ActiveControllerProfile::Vive => 3,
        ActiveControllerProfile::WindowsMr => 4,
        ActiveControllerProfile::Generic => 5,
        ActiveControllerProfile::Simple => 6,
        ActiveControllerProfile::Pico4 => 7,
        ActiveControllerProfile::PicoNeo3 => 8,
        ActiveControllerProfile::HpReverbG2 => 9,
        ActiveControllerProfile::ViveCosmos => 10,
        ActiveControllerProfile::ViveFocus3 => 11,
    }
}

pub(super) fn decode_profile_code(code: u8) -> Option<ActiveControllerProfile> {
    match code {
        1 => Some(ActiveControllerProfile::Touch),
        2 => Some(ActiveControllerProfile::Index),
        3 => Some(ActiveControllerProfile::Vive),
        4 => Some(ActiveControllerProfile::WindowsMr),
        5 => Some(ActiveControllerProfile::Generic),
        6 => Some(ActiveControllerProfile::Simple),
        7 => Some(ActiveControllerProfile::Pico4),
        8 => Some(ActiveControllerProfile::PicoNeo3),
        9 => Some(ActiveControllerProfile::HpReverbG2),
        10 => Some(ActiveControllerProfile::ViveCosmos),
        11 => Some(ActiveControllerProfile::ViveFocus3),
        _ => None,
    }
}

pub(super) fn is_concrete_profile(profile: ActiveControllerProfile) -> bool {
    matches!(
        profile,
        ActiveControllerProfile::Touch
            | ActiveControllerProfile::Index
            | ActiveControllerProfile::Vive
            | ActiveControllerProfile::WindowsMr
            | ActiveControllerProfile::Pico4
            | ActiveControllerProfile::PicoNeo3
            | ActiveControllerProfile::HpReverbG2
            | ActiveControllerProfile::ViveCosmos
            | ActiveControllerProfile::ViveFocus3
    )
}

/// Logs when the resolved profile for a side changes (rate-limited to one log per transition).
pub(super) fn log_profile_transition(side: Chirality, profile: ActiveControllerProfile) {
    static LEFT: AtomicU8 = AtomicU8::new(0);
    static RIGHT: AtomicU8 = AtomicU8::new(0);
    let slot = match side {
        Chirality::Left => &LEFT,
        Chirality::Right => &RIGHT,
    };
    let code = profile_code(profile);
    let previous = slot.swap(code, Ordering::Relaxed);
    if previous != code {
        logger::info!("OpenXR {:?} controller profile: {:?}", side, profile);
    }
}

pub(super) fn device_label(profile: ActiveControllerProfile) -> &'static str {
    match profile {
        ActiveControllerProfile::Touch => "OpenXR Touch Controller",
        ActiveControllerProfile::Index => "OpenXR Index Controller",
        ActiveControllerProfile::Vive => "OpenXR Vive Controller",
        ActiveControllerProfile::WindowsMr => "OpenXR Windows MR Controller",
        ActiveControllerProfile::Pico4 => "OpenXR Pico 4 Controller",
        ActiveControllerProfile::PicoNeo3 => "OpenXR Pico Neo3 Controller",
        ActiveControllerProfile::HpReverbG2 => "OpenXR HP Reverb G2 Controller",
        ActiveControllerProfile::ViveCosmos => "OpenXR Vive Cosmos Controller",
        ActiveControllerProfile::ViveFocus3 => "OpenXR Vive Focus 3 Controller",
        ActiveControllerProfile::Generic => "OpenXR Generic Controller",
        ActiveControllerProfile::Simple => "OpenXR Simple Controller",
    }
}
