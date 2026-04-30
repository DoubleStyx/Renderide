//! Decodes packed texture handles from host [`crate::shared::MaterialsUpdateBatch`] `set_texture` ints.
//!
//! Matches the shared `IdPacker<T>` layout used on the host: a small type tag in the high bits and
//! the asset id in the low bits. [`SetTexture2DFormat::asset_id`](crate::shared::SetTexture2DFormat)
//! and [`crate::gpu_pools::TexturePool`] use the **unpacked** 2D asset id.

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
/// Returns `(asset_id, kind)` when the type field is valid.
pub fn unpack_host_texture_packed(packed: i32) -> Option<(i32, HostTextureAssetKind)> {
    let type_bits = necessary_bits(TEXTURE_ASSET_TYPE_COUNT);
    let pack_type_shift = 32u32 - type_bits;
    let unpack_mask = u32::MAX >> type_bits;
    let packed_bits = packed as u32;
    let id = (packed_bits & unpack_mask) as i32;
    let type_val = packed_bits >> pack_type_shift;
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
    use super::{HostTextureAssetKind, texture2d_asset_id_from_packed, unpack_host_texture_packed};

    fn pack_host_texture(asset_id: i32, kind: HostTextureAssetKind) -> i32 {
        let type_bits = super::necessary_bits(super::TEXTURE_ASSET_TYPE_COUNT);
        let pack_type_shift = 32u32 - type_bits;
        ((asset_id as u32) | ((kind as u32) << pack_type_shift)) as i32
    }

    #[test]
    fn unpack_zero_is_texture2d_asset_zero() {
        assert_eq!(
            unpack_host_texture_packed(0),
            Some((0, HostTextureAssetKind::Texture2D))
        );
    }

    #[test]
    fn unpack_null_sentinel_is_none() {
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
    fn all_host_texture_kinds_round_trip_from_host_bits() {
        let cases = [
            (0, HostTextureAssetKind::Texture2D),
            (5, HostTextureAssetKind::Texture3D),
            (6, HostTextureAssetKind::Cubemap),
            (7, HostTextureAssetKind::RenderTexture),
            (8, HostTextureAssetKind::VideoTexture),
            (9, HostTextureAssetKind::Desktop),
        ];

        for (asset_id, kind) in cases {
            assert_eq!(
                unpack_host_texture_packed(pack_host_texture(asset_id, kind)),
                Some((asset_id, kind))
            );
        }
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
    fn sign_bit_texture_kinds_still_unpack() {
        for kind in [
            HostTextureAssetKind::VideoTexture,
            HostTextureAssetKind::Desktop,
        ] {
            let packed = pack_host_texture(11, kind);
            assert!(packed < 0);
            assert_eq!(unpack_host_texture_packed(packed), Some((11, kind)));
        }
    }

    #[test]
    fn texture2d_asset_id_only_accepts_texture2d_kind() {
        assert_eq!(
            texture2d_asset_id_from_packed(pack_host_texture(12, HostTextureAssetKind::Texture2D)),
            Some(12)
        );

        for kind in [
            HostTextureAssetKind::Texture3D,
            HostTextureAssetKind::Cubemap,
            HostTextureAssetKind::RenderTexture,
            HostTextureAssetKind::VideoTexture,
            HostTextureAssetKind::Desktop,
        ] {
            assert_eq!(
                texture2d_asset_id_from_packed(pack_host_texture(12, kind)),
                None
            );
        }
    }
}
