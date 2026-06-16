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
const fn necessary_bits(mut value: u32) -> u32 {
    let mut n = 0u32;
    while value != 0 {
        value >>= 1;
        n += 1;
    }
    n
}

fn pack_type_shift() -> u32 {
    32u32 - necessary_bits(TEXTURE_ASSET_TYPE_COUNT)
}

fn unpack_mask() -> u32 {
    u32::MAX >> necessary_bits(TEXTURE_ASSET_TYPE_COUNT)
}

/// Packs a non-negative texture asset id using the shared host texture-handle layout.
///
/// Returns [`None`] when `asset_id` is negative or too large to fit beside the type tag.
#[cfg(test)]
fn pack_host_texture_id(asset_id: i32, kind: HostTextureAssetKind) -> Option<i32> {
    if asset_id < 0 {
        return None;
    }
    let id = asset_id as u32;
    if id & !unpack_mask() != 0 {
        return None;
    }
    Some((id | ((kind as u32) << pack_type_shift())) as i32)
}

/// Unpacks `packed` using the shared `IdPacker<TextureAssetType>` layout (six enum variants).
///
/// Returns `(asset_id, kind)` when the type field is valid.
pub fn unpack_host_texture_packed(packed: i32) -> Option<(i32, HostTextureAssetKind)> {
    let packed_bits = packed as u32;
    let id = (packed_bits & unpack_mask()) as i32;
    let type_val = packed_bits >> pack_type_shift();
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
    use super::{
        HostTextureAssetKind, pack_host_texture_id, texture2d_asset_id_from_packed,
        unpack_host_texture_packed,
    };

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
            let packed = pack_host_texture_id(asset_id, kind).expect("packable id");
            assert_eq!(unpack_host_texture_packed(packed), Some((asset_id, kind)));
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
            let packed = pack_host_texture_id(11, kind).expect("packable id");
            assert!(packed < 0);
            assert_eq!(unpack_host_texture_packed(packed), Some((11, kind)));
        }
    }

    #[test]
    fn texture2d_asset_id_only_accepts_texture2d_kind() {
        assert_eq!(
            texture2d_asset_id_from_packed(
                pack_host_texture_id(12, HostTextureAssetKind::Texture2D).expect("packable id")
            ),
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
                texture2d_asset_id_from_packed(
                    pack_host_texture_id(12, kind).expect("packable id")
                ),
                None
            );
        }
    }

    #[test]
    fn pack_rejects_negative_and_overwide_ids() {
        assert_eq!(
            pack_host_texture_id(-1, HostTextureAssetKind::Texture2D),
            None
        );
        assert_eq!(
            pack_host_texture_id(0x2000_0000, HostTextureAssetKind::Texture2D),
            None
        );
    }
}
