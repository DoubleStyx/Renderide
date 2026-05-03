//! Embedded [`ResoniteLogo.png`](../../assets/ResoniteLogo.png) for [`winit::window::WindowAttributes::with_window_icon`].
//!
//! The logo is compiled into the binary with [`include_bytes!`]. On **Windows** and **X11**, the
//! resulting [`winit::window::Icon`] typically appears on the window and in task switching. On
//! **Wayland**, compositors may ignore runtime window icons until the stack supports the relevant
//! protocol and winit exposes it; a Freedesktop `.desktop` file with `Icon=` remains the reliable
//! way to supply a shell icon on many Linux desktops. On **macOS**, dock branding is primarily
//! driven by bundle assets (e.g. `.icns`), not the window icon API.

use winit::window::Icon;

/// PNG bytes for [`try_embedded_window_icon`], resolved from this crate's manifest directory.
const EMBEDDED_LOGO_PNG: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/assets/ResoniteLogo.png"
));

/// Builds a [`winit::window::Icon`] from the embedded PNG, or [`None`] if decoding or icon creation fails.
///
/// Failures are logged at **warn**; callers should treat a missing icon as non-fatal.
pub(crate) fn try_embedded_window_icon() -> Option<Icon> {
    let img = match image::load_from_memory(EMBEDDED_LOGO_PNG) {
        Ok(img) => img,
        Err(e) => {
            logger::warn!("embedded window icon: failed to decode PNG: {e}");
            return None;
        }
    };
    let rgba = img.to_rgba8();
    let (width, height) = rgba.dimensions();
    match Icon::from_rgba(rgba.into_raw(), width, height) {
        Ok(icon) => Some(icon),
        Err(e) => {
            logger::warn!("embedded window icon: winit rejected RGBA buffer: {e}");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::try_embedded_window_icon;

    #[test]
    fn embedded_logo_decodes_to_valid_icon() {
        assert!(
            try_embedded_window_icon().is_some(),
            "embedded ResoniteLogo.png must decode and produce a winit Icon"
        );
    }
}
