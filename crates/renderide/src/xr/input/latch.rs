//! Stable host-facing controller profile selection for OpenXR input.

use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

use crate::shared::Chirality;

use super::profile::{ActiveControllerProfile, decode_profile_code, profile_code};

const HOST_PROFILE_LATCH_SAMPLES: u8 = 3;

/// Per-hand latch for the `VRControllerState` variant sent to FrooxEngine.
///
/// FrooxEngine caches controllers by `deviceID` and casts the cached controller to the incoming
/// state type. This latch waits for a small run of matching profile samples before the first
/// controller is emitted, then keeps that host wire type fixed for the session.
#[derive(Default)]
pub(super) struct HostProfileLatch {
    pending_profile: AtomicU8,
    pending_count: AtomicU8,
    latched_profile: AtomicU8,
    divergence_logged: AtomicBool,
}

impl HostProfileLatch {
    /// Returns the host profile to emit for this sample, or `None` while the startup profile is
    /// still settling.
    pub(super) fn profile_for_sample(
        &self,
        side: Chirality,
        profile: ActiveControllerProfile,
    ) -> Option<ActiveControllerProfile> {
        if let Some(latched) = self.latched_profile() {
            self.log_divergence_once(side, latched, profile);
            return Some(latched);
        }

        let count = self.record_sample(profile);
        if count < HOST_PROFILE_LATCH_SAMPLES {
            return None;
        }

        let code = profile_code(profile);
        let latched_code = match self.latched_profile.compare_exchange(
            0,
            code,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => {
                logger::info!("OpenXR {side:?} controller host profile latched: {profile:?}");
                code
            }
            Err(existing) => existing,
        };
        Some(decode_profile_code(latched_code).unwrap_or(profile))
    }

    fn latched_profile(&self) -> Option<ActiveControllerProfile> {
        decode_profile_code(self.latched_profile.load(Ordering::Relaxed))
    }

    fn record_sample(&self, profile: ActiveControllerProfile) -> u8 {
        let code = profile_code(profile);
        let previous = self.pending_profile.swap(code, Ordering::Relaxed);
        let count = if previous == code {
            self.pending_count
                .load(Ordering::Relaxed)
                .saturating_add(1)
                .min(HOST_PROFILE_LATCH_SAMPLES)
        } else {
            1
        };
        self.pending_count.store(count, Ordering::Relaxed);
        count
    }

    fn log_divergence_once(
        &self,
        side: Chirality,
        latched: ActiveControllerProfile,
        current: ActiveControllerProfile,
    ) {
        if latched == current || self.divergence_logged.swap(true, Ordering::Relaxed) {
            return;
        }
        logger::warn!(
            "OpenXR {side:?} controller profile changed after host latch: latched={latched:?} current={current:?}; keeping the latched host profile"
        );
    }
}

#[cfg(test)]
mod tests {
    use crate::shared::Chirality;

    use super::super::profile::ActiveControllerProfile;
    use super::HostProfileLatch;

    fn sample(
        latch: &HostProfileLatch,
        profile: ActiveControllerProfile,
    ) -> Option<ActiveControllerProfile> {
        latch.profile_for_sample(Chirality::Left, profile)
    }

    #[test]
    fn transient_startup_profile_does_not_latch() {
        let latch = HostProfileLatch::default();

        assert_eq!(sample(&latch, ActiveControllerProfile::Touch), None);
        assert_eq!(sample(&latch, ActiveControllerProfile::Index), None);
        assert_eq!(sample(&latch, ActiveControllerProfile::Index), None);
        assert_eq!(
            sample(&latch, ActiveControllerProfile::Index),
            Some(ActiveControllerProfile::Index)
        );
    }

    #[test]
    fn latched_profile_ignores_later_profile_changes() {
        let latch = HostProfileLatch::default();

        assert_eq!(sample(&latch, ActiveControllerProfile::Touch), None);
        assert_eq!(sample(&latch, ActiveControllerProfile::Touch), None);
        assert_eq!(
            sample(&latch, ActiveControllerProfile::Touch),
            Some(ActiveControllerProfile::Touch)
        );
        assert_eq!(
            sample(&latch, ActiveControllerProfile::Index),
            Some(ActiveControllerProfile::Touch)
        );
    }

    #[test]
    fn stable_fallback_profiles_can_latch() {
        for profile in [
            ActiveControllerProfile::Generic,
            ActiveControllerProfile::Simple,
        ] {
            let latch = HostProfileLatch::default();

            assert_eq!(sample(&latch, profile), None);
            assert_eq!(sample(&latch, profile), None);
            assert_eq!(sample(&latch, profile), Some(profile));
        }
    }
}
