//! Bit-flag decoders for the host [`crate::shared::CameraState::flags`] word.

/// Returns `true` when [`crate::shared::CameraState::flags`] bit 0 is set.
#[inline]
pub fn camera_state_enabled(flags: u16) -> bool {
    flags & 1 != 0
}

/// Returns `true` when [`crate::shared::CameraState::flags`] bit 1 is set.
#[inline]
pub fn camera_state_use_transform_scale(flags: u16) -> bool {
    flags & (1 << 1) != 0
}

/// Returns `true` when [`crate::shared::CameraState::flags`] bit 6 is set.
#[inline]
pub fn camera_state_post_processing(flags: u16) -> bool {
    flags & (1 << 6) != 0
}

/// Returns `true` when [`crate::shared::CameraState::flags`] bit 7 is set.
#[inline]
pub fn camera_state_screen_space_reflections(flags: u16) -> bool {
    flags & (1 << 7) != 0
}

/// Returns `true` when [`crate::shared::CameraState::flags`] bit 8 is set.
#[inline]
pub fn camera_state_motion_blur(flags: u16) -> bool {
    flags & (1 << 8) != 0
}

/// Returns `true` when [`crate::shared::CameraState::flags`] bit 3 is set.
#[inline]
#[cfg(test)]
pub fn camera_state_render_private_ui(flags: u16) -> bool {
    flags & (1 << 3) != 0
}

/// Returns `true` when [`crate::shared::CameraState::flags`] bit 5 is set.
#[inline]
pub fn camera_state_render_shadows(flags: u16) -> bool {
    flags & (1 << 5) != 0
}

#[cfg(test)]
mod tests {
    use super::{
        camera_state_enabled, camera_state_motion_blur, camera_state_post_processing,
        camera_state_render_private_ui, camera_state_render_shadows,
        camera_state_screen_space_reflections, camera_state_use_transform_scale,
    };

    #[test]
    fn camera_state_enabled_reads_bit_zero() {
        assert!(!camera_state_enabled(0));
        assert!(camera_state_enabled(1));
        assert!(camera_state_enabled(0xffff));
        assert!(!camera_state_enabled(2));
    }

    #[test]
    fn camera_state_flags_decode_scale_and_private_ui_bits() {
        assert!(!camera_state_use_transform_scale(0));
        assert!(camera_state_use_transform_scale(1 << 1));
        assert!(!camera_state_render_private_ui(0));
        assert!(camera_state_render_private_ui(1 << 3));
    }

    #[test]
    fn camera_state_render_shadows_reads_bit_five() {
        assert!(!camera_state_render_shadows(0));
        assert!(camera_state_render_shadows(1 << 5));
        assert!(camera_state_render_shadows(0xffff));
        assert!(!camera_state_render_shadows(1 << 4));
    }

    #[test]
    fn camera_state_flags_decode_post_processing_bits() {
        assert!(!camera_state_post_processing(0));
        assert!(camera_state_post_processing(1 << 6));
        assert!(!camera_state_screen_space_reflections(0));
        assert!(camera_state_screen_space_reflections(1 << 7));
        assert!(!camera_state_motion_blur(0));
        assert!(camera_state_motion_blur(1 << 8));
    }
}
