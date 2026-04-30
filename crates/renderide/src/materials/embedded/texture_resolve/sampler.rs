//! Sampler descriptor and `wgpu::Sampler` construction shared across embedded `@group(1)` paths.
//!
//! All embedded textures (2D, 3D, cubemap) ultimately compile down to one
//! [`wgpu::SamplerDescriptor`] schema; the only thing that varies between shapes is which axes
//! consume which `SamplerState::wrap_*` fields and the descriptor label. [`sampler_descriptor`]
//! and [`create_sampler`] take a [`TextureBindKind`] discriminant so callers stop carrying three
//! near-identical helpers per shape.

use crate::gpu_pools::SamplerState;
use crate::shared::{TextureFilterMode, TextureWrapMode};

use super::super::bind_kind::TextureBindKind;

/// Wgpu filter triplet derived from the host texture filter mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ResolvedSamplerFilter {
    /// Magnification filter.
    pub(crate) mag_filter: wgpu::FilterMode,
    /// Minification filter.
    pub(crate) min_filter: wgpu::FilterMode,
    /// Mip-level selection filter.
    pub(crate) mipmap_filter: wgpu::MipmapFilterMode,
}

/// Texture address modes for a sampler descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ResolvedSamplerAddress {
    /// U/S address mode.
    pub(crate) u: wgpu::AddressMode,
    /// V/T address mode.
    pub(crate) v: wgpu::AddressMode,
    /// W/R address mode.
    pub(crate) w: wgpu::AddressMode,
}

impl TextureBindKind {
    /// Constant descriptor label for samplers built for this shape.
    fn sampler_label(self) -> &'static str {
        match self {
            TextureBindKind::Tex2D => "embedded_texture_sampler",
            TextureBindKind::Tex3D => "embedded_texture3d_sampler",
            TextureBindKind::Cube => "embedded_cubemap_sampler",
        }
    }

    /// Computes the address-mode triplet used by samplers of this shape from the host
    /// [`SamplerState`]. Tex2D and Cube mirror W onto U because shaders never sample on the third
    /// axis but `wgpu::SamplerDescriptor` still requires a value.
    fn sampler_address(self, state: &SamplerState) -> ResolvedSamplerAddress {
        let u = wrap_to_address(state.wrap_u);
        let v = wrap_to_address(state.wrap_v);
        let w = match self {
            TextureBindKind::Tex3D => wrap_to_address(state.wrap_w),
            TextureBindKind::Tex2D | TextureBindKind::Cube => u,
        };
        ResolvedSamplerAddress { u, v, w }
    }
}

/// Builds a sampler descriptor from arbitrary parts. Used by [`default_embedded_sampler`] which
/// does not have a [`SamplerState`] to pass in.
pub(crate) fn sampler_descriptor_from_parts(
    label: &'static str,
    address: ResolvedSamplerAddress,
    filter_mode: TextureFilterMode,
    aniso_level: i32,
    mip_levels_resident: u32,
) -> wgpu::SamplerDescriptor<'static> {
    let filter = filter_mode_to_wgpu(filter_mode);
    wgpu::SamplerDescriptor {
        label: Some(label),
        address_mode_u: address.u,
        address_mode_v: address.v,
        address_mode_w: address.w,
        mag_filter: filter.mag_filter,
        min_filter: filter.min_filter,
        mipmap_filter: filter.mipmap_filter,
        lod_min_clamp: 0.0,
        lod_max_clamp: mip_levels_resident.saturating_sub(1) as f32,
        anisotropy_clamp: anisotropy_clamp(filter_mode, aniso_level, filter),
        ..Default::default()
    }
}

/// Builds a sampler descriptor for a host texture binding of the given shape.
pub(crate) fn sampler_descriptor(
    state: &SamplerState,
    kind: TextureBindKind,
    mip_levels_resident: u32,
) -> wgpu::SamplerDescriptor<'static> {
    sampler_descriptor_from_parts(
        kind.sampler_label(),
        kind.sampler_address(state),
        state.filter_mode,
        state.aniso_level,
        mip_levels_resident,
    )
}

/// Builds a `wgpu::Sampler` for a host texture binding of the given shape.
pub(crate) fn create_sampler(
    device: &wgpu::Device,
    state: &SamplerState,
    kind: TextureBindKind,
    mip_levels_resident: u32,
) -> wgpu::Sampler {
    device.create_sampler(&sampler_descriptor(state, kind, mip_levels_resident))
}

/// Builds the fallback sampler used with the embedded white placeholder textures.
pub(crate) fn default_embedded_sampler(device: &wgpu::Device) -> wgpu::Sampler {
    let descriptor = sampler_descriptor_from_parts(
        "embedded_default_sampler",
        ResolvedSamplerAddress {
            u: wgpu::AddressMode::Repeat,
            v: wgpu::AddressMode::Repeat,
            w: wgpu::AddressMode::Repeat,
        },
        TextureFilterMode::Trilinear,
        1,
        1,
    );
    device.create_sampler(&descriptor)
}

/// Converts a host wrap mode to a wgpu address mode.
pub(crate) fn wrap_to_address(w: TextureWrapMode) -> wgpu::AddressMode {
    match w {
        TextureWrapMode::Repeat => wgpu::AddressMode::Repeat,
        TextureWrapMode::Clamp => wgpu::AddressMode::ClampToEdge,
        TextureWrapMode::Mirror => wgpu::AddressMode::MirrorRepeat,
        TextureWrapMode::MirrorOnce => wgpu::AddressMode::ClampToEdge,
    }
}

/// Converts a host filter mode to wgpu filter fields without changing host semantics.
pub(crate) fn filter_mode_to_wgpu(filter_mode: TextureFilterMode) -> ResolvedSamplerFilter {
    match filter_mode {
        TextureFilterMode::Point => ResolvedSamplerFilter {
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        },
        TextureFilterMode::Bilinear => ResolvedSamplerFilter {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        },
        TextureFilterMode::Trilinear => ResolvedSamplerFilter {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
        },
        TextureFilterMode::Anisotropic => ResolvedSamplerFilter {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
        },
    }
}

/// Returns the wgpu anisotropy clamp for a resolved host sampler mode.
fn anisotropy_clamp(
    filter_mode: TextureFilterMode,
    aniso_level: i32,
    filter: ResolvedSamplerFilter,
) -> u16 {
    if matches!(filter_mode, TextureFilterMode::Anisotropic)
        && filter.mag_filter == wgpu::FilterMode::Linear
        && filter.min_filter == wgpu::FilterMode::Linear
        && filter.mipmap_filter == wgpu::MipmapFilterMode::Linear
    {
        aniso_level.clamp(1, 16) as u16
    } else {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampler_filter_modes_preserve_host_semantics() {
        let address = ResolvedSamplerAddress {
            u: wgpu::AddressMode::Repeat,
            v: wgpu::AddressMode::Repeat,
            w: wgpu::AddressMode::Repeat,
        };

        let point = sampler_descriptor_from_parts("test", address, TextureFilterMode::Point, 16, 4);
        assert_eq!(point.mag_filter, wgpu::FilterMode::Nearest);
        assert_eq!(point.min_filter, wgpu::FilterMode::Nearest);
        assert_eq!(point.mipmap_filter, wgpu::MipmapFilterMode::Nearest);
        assert_eq!(point.anisotropy_clamp, 1);
        assert_eq!(point.lod_max_clamp, 3.0);

        let bilinear =
            sampler_descriptor_from_parts("test", address, TextureFilterMode::Bilinear, 16, 4);
        assert_eq!(bilinear.mag_filter, wgpu::FilterMode::Linear);
        assert_eq!(bilinear.min_filter, wgpu::FilterMode::Linear);
        assert_eq!(bilinear.mipmap_filter, wgpu::MipmapFilterMode::Nearest);
        assert_eq!(bilinear.anisotropy_clamp, 1);

        let trilinear =
            sampler_descriptor_from_parts("test", address, TextureFilterMode::Trilinear, 16, 4);
        assert_eq!(trilinear.mag_filter, wgpu::FilterMode::Linear);
        assert_eq!(trilinear.min_filter, wgpu::FilterMode::Linear);
        assert_eq!(trilinear.mipmap_filter, wgpu::MipmapFilterMode::Linear);
        assert_eq!(trilinear.anisotropy_clamp, 1);

        let anisotropic =
            sampler_descriptor_from_parts("test", address, TextureFilterMode::Anisotropic, 64, 4);
        assert_eq!(anisotropic.mag_filter, wgpu::FilterMode::Linear);
        assert_eq!(anisotropic.min_filter, wgpu::FilterMode::Linear);
        assert_eq!(anisotropic.mipmap_filter, wgpu::MipmapFilterMode::Linear);
        assert_eq!(anisotropic.anisotropy_clamp, 16);
    }

    #[test]
    fn sampler_descriptors_apply_wrap_anisotropy_and_lod_clamps_for_all_texture_kinds() {
        let texture2d = SamplerState {
            filter_mode: TextureFilterMode::Anisotropic,
            aniso_level: 8,
            wrap_u: TextureWrapMode::Mirror,
            wrap_v: TextureWrapMode::Clamp,
            wrap_w: TextureWrapMode::default(),
            mipmap_bias: 0.0,
        };
        let texture2d_desc = sampler_descriptor(&texture2d, TextureBindKind::Tex2D, 6);
        assert_eq!(
            texture2d_desc.address_mode_u,
            wgpu::AddressMode::MirrorRepeat
        );
        assert_eq!(
            texture2d_desc.address_mode_v,
            wgpu::AddressMode::ClampToEdge
        );
        assert_eq!(
            texture2d_desc.address_mode_w,
            wgpu::AddressMode::MirrorRepeat
        );
        assert_eq!(texture2d_desc.anisotropy_clamp, 8);
        assert_eq!(texture2d_desc.lod_max_clamp, 5.0);

        let texture3d = SamplerState {
            filter_mode: TextureFilterMode::Anisotropic,
            aniso_level: 12,
            wrap_u: TextureWrapMode::Repeat,
            wrap_v: TextureWrapMode::Mirror,
            wrap_w: TextureWrapMode::Clamp,
            mipmap_bias: 0.0,
        };
        let texture3d_desc = sampler_descriptor(&texture3d, TextureBindKind::Tex3D, 3);
        assert_eq!(texture3d_desc.address_mode_u, wgpu::AddressMode::Repeat);
        assert_eq!(
            texture3d_desc.address_mode_v,
            wgpu::AddressMode::MirrorRepeat
        );
        assert_eq!(
            texture3d_desc.address_mode_w,
            wgpu::AddressMode::ClampToEdge
        );
        assert_eq!(texture3d_desc.anisotropy_clamp, 12);
        assert_eq!(texture3d_desc.lod_max_clamp, 2.0);

        let cubemap = SamplerState {
            filter_mode: TextureFilterMode::Anisotropic,
            aniso_level: 4,
            wrap_u: TextureWrapMode::Repeat,
            wrap_v: TextureWrapMode::Clamp,
            wrap_w: TextureWrapMode::default(),
            mipmap_bias: 0.0,
        };
        let cubemap_desc = sampler_descriptor(&cubemap, TextureBindKind::Cube, 1);
        assert_eq!(cubemap_desc.address_mode_u, wgpu::AddressMode::Repeat);
        assert_eq!(cubemap_desc.address_mode_v, wgpu::AddressMode::ClampToEdge);
        assert_eq!(cubemap_desc.address_mode_w, wgpu::AddressMode::Repeat);
        assert_eq!(cubemap_desc.anisotropy_clamp, 4);
        assert_eq!(cubemap_desc.lod_max_clamp, 0.0);
    }
}
