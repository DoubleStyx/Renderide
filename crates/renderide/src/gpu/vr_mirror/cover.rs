//! **Cover** UV mapping math for blitting the HMD eye / staging texture into the window (CSS `object-fit: cover`).

use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct SurfaceBlitUniform {
    pub(super) uv_scale: [f32; 2],
    pub(super) uv_offset: [f32; 2],
}

/// UV transform for [`cover_uv_params`].
///
/// Implements CSS **object-fit: cover**: the staging texture is scaled uniformly so the window is
/// fully covered; excess is cropped from the center (no black bars).
///
/// Compare window aspect `W_s/H_s` to staging aspect `W_t/H_t`: if the window is wider (larger
/// aspect ratio), crop top/bottom in texture space; if the window is taller (smaller aspect ratio),
/// crop left/right.
///
/// Shader: `tuv = uv * uv_scale + uv_offset` maps screen `uv` in [0, 1]^2 into texture UV in a centered
/// sub-rectangle of [0, 1]^2.
pub(super) fn cover_uv_params(
    eye_w: u32,
    eye_h: u32,
    surf_w: u32,
    surf_h: u32,
) -> SurfaceBlitUniform {
    let ew = eye_w.max(1) as f32;
    let eh = eye_h.max(1) as f32;
    let sw = surf_w.max(1) as f32;
    let sh = surf_h.max(1) as f32;
    let eye_aspect = ew / eh;
    let surf_aspect = sw / sh;
    if surf_aspect > eye_aspect {
        // Window is wider than the staging texture aspect (R_s > R_t): cover crops top/bottom.
        let frac = eye_aspect / surf_aspect;
        SurfaceBlitUniform {
            uv_scale: [1.0, frac],
            uv_offset: [0.0, (1.0 - frac) * 0.5],
        }
    } else {
        // Window is taller or narrower (R_s <= R_t): cover crops left/right.
        let frac = surf_aspect / eye_aspect;
        SurfaceBlitUniform {
            uv_scale: [frac, 1.0],
            uv_offset: [(1.0 - frac) * 0.5, 0.0],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::cover_uv_params;

    #[test]
    fn cover_uv_center_crop_when_surface_wider_than_eye() {
        // 2:1 window, 1:1 eye (R_s > R_t) -> crop texture top/bottom (cover).
        let u = cover_uv_params(100, 100, 200, 100);
        assert!((u.uv_scale[0] - 1.0).abs() < 1e-5);
        assert!((u.uv_scale[1] - 0.5).abs() < 1e-5);
        assert!((u.uv_offset[0] - 0.0).abs() < 1e-5);
        assert!((u.uv_offset[1] - 0.25).abs() < 1e-5);
    }

    #[test]
    fn cover_uv_center_crop_when_surface_taller_than_eye() {
        // 1:2 window, 1:1 eye (R_s < R_t) -> crop texture left/right (cover).
        let u = cover_uv_params(100, 100, 100, 200);
        assert!((u.uv_scale[0] - 0.5).abs() < 1e-5);
        assert!((u.uv_scale[1] - 1.0).abs() < 1e-5);
        assert!((u.uv_offset[0] - 0.25).abs() < 1e-5);
        assert!((u.uv_offset[1] - 0.0).abs() < 1e-5);
    }
}
