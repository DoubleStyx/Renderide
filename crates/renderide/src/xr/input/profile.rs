//! OpenXR interaction profile classification and human-readable device labels.

use std::sync::atomic::{AtomicU8, Ordering};

use crate::shared::Chirality;

/// Active interaction profile for a hand, derived from the OpenXR session's current interaction profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ActiveControllerProfile {
    Touch,
    Index,
    Vive,
    WindowsMr,
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
    )
}

/// Logs when the resolved profile for a side changes (rate-limited to one log per transition).
pub(super) fn log_profile_transition(side: Chirality, profile: ActiveControllerProfile) {
    static LEFT: AtomicU8 = AtomicU8::new(0);
    static RIGHT: AtomicU8 = AtomicU8::new(0);
    let slot = match side {
        Chirality::left => &LEFT,
        Chirality::right => &RIGHT,
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
        ActiveControllerProfile::Generic => "OpenXR Generic Controller",
        ActiveControllerProfile::Simple => "OpenXR Simple Controller",
    }
}
