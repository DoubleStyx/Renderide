//! Host-provided desktop-window icon conversion and application.

use glam::IVec2;
use thiserror::Error;
use winit::icon::{BadIcon, Icon, RgbaIcon};
use winit::window::Window;

/// Error produced while converting a host `SetWindowIcon` payload into a winit icon.
#[derive(Debug, Error)]
pub(super) enum HostWindowIconError {
    /// Host sent a non-positive icon extent.
    #[error("host window icon dimensions must be positive, got {width}x{height}")]
    InvalidDimensions {
        /// Host-provided width.
        width: i32,
        /// Host-provided height.
        height: i32,
    },
    /// Host icon dimensions overflowed the expected BGRA byte count.
    #[error("host window icon byte count overflow for {width}x{height}")]
    ByteCountOverflow {
        /// Validated icon width.
        width: u32,
        /// Validated icon height.
        height: u32,
    },
    /// Host shared-memory byte count did not match the icon extent.
    #[error("host window icon has {actual} byte(s), expected {expected} for {width}x{height}")]
    LengthMismatch {
        /// Validated icon width.
        width: u32,
        /// Validated icon height.
        height: u32,
        /// Expected BGRA byte count.
        expected: usize,
        /// Actual BGRA byte count.
        actual: usize,
    },
    /// Winit rejected the converted RGBA icon.
    #[error("winit rejected host window icon: {0}")]
    Winit(#[from] BadIcon),
}

/// Applies a host BGRA32 icon payload to `window`.
pub(super) fn apply_host_window_icon(
    window: &dyn Window,
    size: IVec2,
    bgra: &[u8],
) -> Result<(), HostWindowIconError> {
    let icon = host_window_icon_from_bgra(size, bgra)?;
    #[cfg(target_os = "windows")]
    {
        use winit::platform::windows::WindowExtWindows;

        window.set_taskbar_icon(Some(icon.clone()));
    }
    window.set_window_icon(Some(icon));
    Ok(())
}

fn host_window_icon_from_bgra(size: IVec2, bgra: &[u8]) -> Result<Icon, HostWindowIconError> {
    let (rgba, width, height) = bgra_to_rgba(size, bgra)?;
    Ok(RgbaIcon::new(rgba, width, height)?.into())
}

fn bgra_to_rgba(size: IVec2, bgra: &[u8]) -> Result<(Vec<u8>, u32, u32), HostWindowIconError> {
    let (width, height, expected) = validate_bgra_icon_shape(size, bgra.len())?;
    let mut rgba = Vec::with_capacity(expected);
    let row_stride = width as usize * 4;
    for row in bgra.chunks_exact(row_stride).rev() {
        for pixel in row.chunks_exact(4) {
            rgba.extend_from_slice(&[pixel[2], pixel[1], pixel[0], pixel[3]]);
        }
    }
    Ok((rgba, width, height))
}

fn validate_bgra_icon_shape(
    size: IVec2,
    actual_len: usize,
) -> Result<(u32, u32, usize), HostWindowIconError> {
    if size.x <= 0 || size.y <= 0 {
        return Err(HostWindowIconError::InvalidDimensions {
            width: size.x,
            height: size.y,
        });
    }
    let width = size.x as u32;
    let height = size.y as u32;
    let expected = checked_bgra_byte_len(width, height)?;
    if actual_len != expected {
        return Err(HostWindowIconError::LengthMismatch {
            width,
            height,
            expected,
            actual: actual_len,
        });
    }
    Ok((width, height, expected))
}

fn checked_bgra_byte_len(width: u32, height: u32) -> Result<usize, HostWindowIconError> {
    (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or(HostWindowIconError::ByteCountOverflow { width, height })
}

#[cfg(test)]
mod tests {
    use glam::IVec2;

    use super::{HostWindowIconError, bgra_to_rgba, checked_bgra_byte_len};

    #[test]
    fn bgra_payload_converts_to_rgba() {
        let bgra = [0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80];

        let (rgba, width, height) =
            bgra_to_rgba(IVec2::new(2, 1), &bgra).expect("valid BGRA icon should convert");

        assert_eq!(width, 2);
        assert_eq!(height, 1);
        assert_eq!(rgba, vec![0x30, 0x20, 0x10, 0x40, 0x70, 0x60, 0x50, 0x80,]);
    }

    #[test]
    fn bgra_payload_flips_rows_while_converting_to_rgba() {
        let bgra = [
            0x01, 0x02, 0x03, 0x04, 0x11, 0x12, 0x13, 0x14, 0x21, 0x22, 0x23, 0x24, 0x31, 0x32,
            0x33, 0x34,
        ];

        let (rgba, width, height) =
            bgra_to_rgba(IVec2::new(2, 2), &bgra).expect("valid BGRA icon should convert");

        assert_eq!(width, 2);
        assert_eq!(height, 2);
        assert_eq!(
            rgba,
            vec![
                0x23, 0x22, 0x21, 0x24, 0x33, 0x32, 0x31, 0x34, 0x03, 0x02, 0x01, 0x04, 0x13, 0x12,
                0x11, 0x14,
            ]
        );
    }

    #[test]
    fn rejects_non_positive_dimensions() {
        let err = bgra_to_rgba(IVec2::new(0, 1), &[]).expect_err("zero width is invalid");

        assert!(matches!(
            err,
            HostWindowIconError::InvalidDimensions {
                width: 0,
                height: 1
            }
        ));
    }

    #[test]
    fn rejects_byte_count_overflow() {
        let err = checked_bgra_byte_len(u32::MAX, u32::MAX)
            .expect_err("oversized icon byte count should overflow");

        assert!(matches!(err, HostWindowIconError::ByteCountOverflow { .. }));
    }

    #[test]
    fn rejects_length_mismatch() {
        let err = bgra_to_rgba(IVec2::new(2, 2), &[0; 4]).expect_err("short payload is invalid");

        assert!(matches!(
            err,
            HostWindowIconError::LengthMismatch {
                width: 2,
                height: 2,
                expected: 16,
                actual: 4
            }
        ));
    }
}
