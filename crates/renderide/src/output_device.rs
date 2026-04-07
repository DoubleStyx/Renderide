//! Maps [`HeadOutputDevice`](crate::shared::HeadOutputDevice) to VR / OpenXR presentation intent.

use crate::shared::HeadOutputDevice;

/// Returns `true` for SteamVR, Windows MR, Oculus, and Oculus Quest — aligned with FrooxEngine’s
/// `HeadOutputDeviceExtension.IsVR` (`(uint)(device - 6) <= 3`).
pub fn head_output_device_is_vr(device: HeadOutputDevice) -> bool {
    matches!(
        device,
        HeadOutputDevice::steam_vr
            | HeadOutputDevice::windows_mr
            | HeadOutputDevice::oculus
            | HeadOutputDevice::oculus_quest
    )
}

/// Whether to bootstrap the OpenXR Vulkan path when the `openxr` feature is enabled.
///
/// For this milestone this matches [`head_output_device_is_vr`]. Without `openxr`, always `false`.
#[cfg(feature = "openxr")]
pub fn head_output_device_wants_openxr(device: HeadOutputDevice) -> bool {
    head_output_device_is_vr(device)
}

/// Without the OpenXR feature, the renderer never selects the OpenXR GPU path.
#[cfg(not(feature = "openxr"))]
pub fn head_output_device_wants_openxr(_device: HeadOutputDevice) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_vr_matches_cs_table() {
        let cases = [
            (HeadOutputDevice::autodetect, false),
            (HeadOutputDevice::headless, false),
            (HeadOutputDevice::screen, false),
            (HeadOutputDevice::screen360, false),
            (HeadOutputDevice::static_camera, false),
            (HeadOutputDevice::static_camera360, false),
            (HeadOutputDevice::steam_vr, true),
            (HeadOutputDevice::windows_mr, true),
            (HeadOutputDevice::oculus, true),
            (HeadOutputDevice::oculus_quest, true),
            (HeadOutputDevice::unknown, false),
        ];
        for (d, want) in cases {
            assert_eq!(head_output_device_is_vr(d), want, "wrong IsVR for {d:?}");
        }
    }

    #[test]
    fn wants_openxr_tracks_feature_and_is_vr() {
        let vr = HeadOutputDevice::steam_vr;
        let non = HeadOutputDevice::screen;
        assert_eq!(
            head_output_device_wants_openxr(vr),
            cfg!(feature = "openxr")
        );
        assert!(!head_output_device_wants_openxr(non));
    }
}
