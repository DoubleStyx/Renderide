//! Decodes packed texture handles from host material updates.
//!
//! Matches Renderite `IdPacker<TextureAssetType>` bit layout (type tag in high bits, asset id in low bits).

/// Host texture asset kind (same enum order as Renderite.Shared `TextureAssetType`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum HostTextureAssetKind {
    /// 2D texture asset (`Texture2D`).
    Texture2D = 0,
    /// 3D texture.
    Texture3D = 1,
    /// Cubemap.
    Cubemap = 2,
    /// Render texture.
    RenderTexture = 3,
    /// Video texture.
    VideoTexture = 4,
    /// Desktop capture texture.
    Desktop = 5,
}

const TEXTURE_ASSET_TYPE_COUNT: u32 = 6;

/// Matches `MathHelper.NecessaryBits((ulong)typeCount)` in Renderite.Shared.
fn necessary_bits(mut value: u32) -> u32 {
    let mut n = 0u32;
    while value != 0 {
        value >>= 1;
        n += 1;
    }
    n
}

/// Unpacks `packed` using `IdPacker<TextureAssetType>` layout (6 enum variants).
///
/// Returns `(asset_id, kind)` when the packed value is non-negative and the type field is valid.
pub fn unpack_host_texture_packed(packed: i32) -> Option<(i32, HostTextureAssetKind)> {
    if packed <= 0 {
        return None;
    }
    let type_bits = necessary_bits(TEXTURE_ASSET_TYPE_COUNT);
    let pack_type_shift = 32u32.saturating_sub(type_bits);
    let unpack_mask = (u32::MAX >> type_bits) as i32;
    let id = packed & unpack_mask;
    let type_val = (packed as u32).wrapping_shr(pack_type_shift);
    let kind = match type_val {
        0 => HostTextureAssetKind::Texture2D,
        1 => HostTextureAssetKind::Texture3D,
        2 => HostTextureAssetKind::Cubemap,
        3 => HostTextureAssetKind::RenderTexture,
        4 => HostTextureAssetKind::VideoTexture,
        5 => HostTextureAssetKind::Desktop,
        _ => return None,
    };
    Some((id, kind))
}

/// Resolves a packed `set_texture` value to a 2D texture asset id when the type is `Texture2D`.
pub fn texture2d_asset_id_from_packed(packed: i32) -> Option<i32> {
    let (id, k) = unpack_host_texture_packed(packed)?;
    (k == HostTextureAssetKind::Texture2D).then_some(id)
}

#[cfg(test)]
mod tests {
    use super::unpack_host_texture_packed;

    #[test]
    fn unpack_zero_is_none() {
        assert!(unpack_host_texture_packed(0).is_none());
    }

    #[test]
    fn unpack_negative_is_none() {
        assert!(unpack_host_texture_packed(-1).is_none());
    }
}
