//! Host-side encoder for `TransformsUpdate.pose_updates` shared-memory rows.
//!
//! Each row is a fixed 44 bytes: `i32 transform_id` + [`crate::shared::RenderTransform`]
//! (40 bytes; pack order `position → scale → rotation` per the renderer-side parser at
//! `crates/renderide/src/scene/transforms_apply.rs`).

use crate::shared::{RenderTransform, RENDER_TRANSFORM_HOST_ROW_BYTES};

/// Number of bytes per [`TransformPoseRow`] on the wire (`i32 + RenderTransform`).
pub const TRANSFORM_POSE_ROW_BYTES: usize = 4 + RENDER_TRANSFORM_HOST_ROW_BYTES;

/// One row in [`crate::shared::TransformsUpdate::pose_updates`].
#[derive(Clone, Copy, Debug)]
pub struct TransformPoseRow {
    /// Dense transform index this row updates.
    pub transform_id: i32,
    /// Local pose for the node (position, scale, rotation).
    pub pose: RenderTransform,
}

/// Encodes a list of [`TransformPoseRow`] into the byte layout the renderer's
/// `apply_transforms_update` reads from shared memory.
pub fn encode_transform_pose_updates(rows: &[TransformPoseRow]) -> Vec<u8> {
    let mut out = vec![0u8; rows.len() * TRANSFORM_POSE_ROW_BYTES];
    for (i, row) in rows.iter().enumerate() {
        let base = i * TRANSFORM_POSE_ROW_BYTES;
        out[base..base + 4].copy_from_slice(&row.transform_id.to_le_bytes());
        // RenderTransform is `position(Vec3) → scale(Vec3) → rotation(Quat)` per shared.rs.
        let pos = row.pose.position;
        out[base + 4..base + 8].copy_from_slice(&pos.x.to_le_bytes());
        out[base + 8..base + 12].copy_from_slice(&pos.y.to_le_bytes());
        out[base + 12..base + 16].copy_from_slice(&pos.z.to_le_bytes());
        let scl = row.pose.scale;
        out[base + 16..base + 20].copy_from_slice(&scl.x.to_le_bytes());
        out[base + 20..base + 24].copy_from_slice(&scl.y.to_le_bytes());
        out[base + 24..base + 28].copy_from_slice(&scl.z.to_le_bytes());
        let rot = row.pose.rotation;
        out[base + 28..base + 32].copy_from_slice(&rot.x.to_le_bytes());
        out[base + 32..base + 36].copy_from_slice(&rot.y.to_le_bytes());
        out[base + 36..base + 40].copy_from_slice(&rot.z.to_le_bytes());
        out[base + 40..base + 44].copy_from_slice(&rot.w.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Quat, Vec3};

    #[test]
    fn row_size_is_44_bytes() {
        assert_eq!(TRANSFORM_POSE_ROW_BYTES, 44);
    }

    #[test]
    fn encodes_identity_row_with_correct_layout() {
        let rows = [TransformPoseRow {
            transform_id: 0,
            pose: RenderTransform {
                position: Vec3::new(0.0, 0.0, 0.0),
                scale: Vec3::new(1.0, 1.0, 1.0),
                rotation: Quat::IDENTITY,
            },
        }];
        let bytes = encode_transform_pose_updates(&rows);
        assert_eq!(bytes.len(), TRANSFORM_POSE_ROW_BYTES);
        assert_eq!(&bytes[0..4], &0i32.to_le_bytes());
        // position
        assert_eq!(&bytes[4..16], &[0u8; 12]);
        // scale = 1,1,1
        let one = 1.0f32.to_le_bytes();
        assert_eq!(&bytes[16..20], &one);
        assert_eq!(&bytes[20..24], &one);
        assert_eq!(&bytes[24..28], &one);
        // identity quaternion = 0,0,0,1 (xyzw)
        assert_eq!(&bytes[28..32], &0.0f32.to_le_bytes());
        assert_eq!(&bytes[32..36], &0.0f32.to_le_bytes());
        assert_eq!(&bytes[36..40], &0.0f32.to_le_bytes());
        assert_eq!(&bytes[40..44], &one);
    }

    #[test]
    fn encodes_multiple_rows_in_order() {
        let rows = [
            TransformPoseRow {
                transform_id: 7,
                pose: RenderTransform {
                    position: Vec3::new(1.0, 2.0, 3.0),
                    scale: Vec3::new(4.0, 5.0, 6.0),
                    rotation: Quat::from_xyzw(0.1, 0.2, 0.3, 0.9),
                },
            },
            TransformPoseRow {
                transform_id: 11,
                pose: RenderTransform {
                    position: Vec3::ZERO,
                    scale: Vec3::ONE,
                    rotation: Quat::IDENTITY,
                },
            },
        ];
        let bytes = encode_transform_pose_updates(&rows);
        assert_eq!(bytes.len(), 2 * TRANSFORM_POSE_ROW_BYTES);
        assert_eq!(i32::from_le_bytes(bytes[0..4].try_into().unwrap()), 7);
        assert_eq!(f32::from_le_bytes(bytes[4..8].try_into().unwrap()), 1.0);
        assert_eq!(
            i32::from_le_bytes(
                bytes[TRANSFORM_POSE_ROW_BYTES..TRANSFORM_POSE_ROW_BYTES + 4]
                    .try_into()
                    .unwrap()
            ),
            11
        );
    }
}
