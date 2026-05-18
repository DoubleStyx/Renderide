//! Per-texture wrap-mode bit encoding consumed by material WGSL.

use crate::gpu_pools::SamplerState;
use crate::shared::TextureWrapMode;

/// Suffix convention for renderer-reserved texture wrap bitfields.
pub(super) const WRAP_MODE_BITS_SUFFIX: &str = "_WrapModeBits";

/// Bit marking MirrorOnce addressing on a texture's U axis.
const WRAP_MODE_MIRROR_ONCE_U: u32 = 1;
/// Bit marking MirrorOnce addressing on a texture's V axis.
const WRAP_MODE_MIRROR_ONCE_V: u32 = 2;
/// Bit marking MirrorOnce addressing on a texture's W axis.
const WRAP_MODE_MIRROR_ONCE_W: u32 = 4;

/// Encodes the axes that require shader-side MirrorOnce coordinate emulation.
pub(super) fn sampler_wrap_mode_bits(state: &SamplerState) -> u32 {
    wrap_axis_bit(state.wrap_u, WRAP_MODE_MIRROR_ONCE_U)
        | wrap_axis_bit(state.wrap_v, WRAP_MODE_MIRROR_ONCE_V)
        | wrap_axis_bit(state.wrap_w, WRAP_MODE_MIRROR_ONCE_W)
}

/// Returns `bit` when `mode` requires MirrorOnce emulation.
fn wrap_axis_bit(mode: TextureWrapMode, bit: u32) -> u32 {
    if matches!(mode, TextureWrapMode::MirrorOnce) {
        bit
    } else {
        0
    }
}
