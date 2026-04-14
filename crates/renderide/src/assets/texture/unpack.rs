//! Decodes packed texture handles from host [`crate::shared::MaterialsUpdateBatch`] `set_texture` ints.
//!
//! Matches the shared `IdPacker<T>` layout used on the host: a small type tag in the high bits and
//! the asset id in the low bits. [`SetTexture2DFormat::asset_id`](crate::shared::SetTexture2DFormat)
//! and [`crate::resources::TexturePool`] use the **unpacked** 2D asset id.

/// Host texture asset kind (same enum order as the shared `TextureAssetType` wire enum).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum HostTextureAssetKind {
    /// 2D texture asset (`Texture2D`).
    Texture2D = 0,
    /// 3D texture asset (`Texture3D`).
    Texture3D = 1,
    /// Cubemap texture asset.
    Cubemap = 2,
    /// Host render texture (`RenderTexture`).
    RenderTexture = 3,
    /// Video texture asset.
    VideoTexture = 4,
    /// Desktop-captured texture (`Desktop`).
    Desktop = 5,
}

const TEXTURE_ASSET_TYPE_COUNT: u32 = 6;

/// Matches `MathHelper.NecessaryBits((ulong)typeCount)` in the shared host packer.
fn necessary_bits(mut value: u32) -> u32 {
    let mut n = 0u32;
    while value != 0 {
        value >>= 1;
        n += 1;
    }
    n
}

/// Unpacks `packed` using the shared `IdPacker<TextureAssetType>` layout (six enum variants).
///
/// Returns `(asset_id, kind)` when `packed` is positive and the type field is valid.
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

/// Resolves a packed `set_texture` value to a 2D texture asset id when the type is [`HostTextureAssetKind::Texture2D`].
pub fn texture2d_asset_id_from_packed(packed: i32) -> Option<i32> {
    let (id, k) = unpack_host_texture_packed(packed)?;
    (k == HostTextureAssetKind::Texture2D).then_some(id)
}

#[cfg(test)]
mod tests {
    use super::{texture2d_asset_id_from_packed, unpack_host_texture_packed, HostTextureAssetKind};

    #[test]
    fn unpack_zero_is_none() {
        assert!(unpack_host_texture_packed(0).is_none());
    }

    #[test]
    fn unpack_negative_is_none() {
        assert!(unpack_host_texture_packed(-1).is_none());
    }

    #[test]
    fn texture2d_plain_id_matches_pool_key() {
        let id = 42i32;
        assert_eq!(texture2d_asset_id_from_packed(id), Some(id));
        assert_eq!(
            unpack_host_texture_packed(id),
            Some((id, HostTextureAssetKind::Texture2D))
        );
    }

    #[test]
    fn nonzero_type_bits_still_yields_correct_low_id() {
        let type_bits = super::necessary_bits(super::TEXTURE_ASSET_TYPE_COUNT);
        let pack_type_shift = 32u32.saturating_sub(type_bits);
        let packed = 5i32 | ((HostTextureAssetKind::Texture3D as i32) << pack_type_shift);
        assert_eq!(texture2d_asset_id_from_packed(packed), None);
        let (id, k) = unpack_host_texture_packed(packed).expect("unpack");
        assert_eq!(id, 5);
        assert_eq!(k, HostTextureAssetKind::Texture3D);
    }

    #[test]
    fn texture2d_with_type_tag_zero_matches_unpack() {
        let id = 0x00AB_CD01i32;
        assert_eq!(
            unpack_host_texture_packed(id),
            Some((id, HostTextureAssetKind::Texture2D))
        );
        assert_eq!(texture2d_asset_id_from_packed(id), Some(id));
    }

    #[test]
    fn render_texture_packed_id_unpacks() {
        let type_bits = super::necessary_bits(super::TEXTURE_ASSET_TYPE_COUNT);
        let pack_type_shift = 32u32.saturating_sub(type_bits);
        let asset_id = 7i32;
        let packed = asset_id | ((HostTextureAssetKind::RenderTexture as i32) << pack_type_shift);
        let (id, k) = unpack_host_texture_packed(packed).expect("unpack");
        assert_eq!(id, asset_id);
        assert_eq!(k, HostTextureAssetKind::RenderTexture);
        assert_eq!(texture2d_asset_id_from_packed(packed), None);
    }
}
