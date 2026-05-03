//! SH2 projection readback adapter over the backend GPU job service.

use glam::Vec3;

use super::{MAX_PENDING_JOB_AGE_FRAMES, Sh2SourceKey};
use crate::backend::{GpuJobResources, GpuReadbackJobs, GpuReadbackOutcomes, SubmittedReadbackJob};
use crate::shared::RenderSH2;

/// GPU resources that must stay alive until an SH2 projection readback completes.
pub(super) struct SubmittedGpuSh2Job {
    /// Staging buffer copied from the compute output.
    pub(super) staging: wgpu::Buffer,
    /// Compute output buffer kept alive until readback finishes.
    pub(super) output: wgpu::Buffer,
    /// Bind group kept alive until the queued command has completed.
    pub(super) bind_group: wgpu::BindGroup,
    /// Uniform/parameter buffers kept alive until the queued command has completed.
    pub(super) buffers: Vec<wgpu::Buffer>,
}

impl From<SubmittedGpuSh2Job> for SubmittedReadbackJob {
    fn from(job: SubmittedGpuSh2Job) -> Self {
        Self {
            staging: job.staging,
            resources: GpuJobResources::new()
                .with_buffer(job.output)
                .with_buffers(job.buffers)
                .with_bind_group(job.bind_group),
        }
    }
}

/// Completed and failed SH2 readbacks drained during one maintenance tick.
pub(super) type Sh2ReadbackOutcomes = GpuReadbackOutcomes<Sh2SourceKey, RenderSH2>;

/// Owns all in-flight SH2 readback jobs plus their submit-done notification channel.
pub(super) struct Sh2ReadbackJobs {
    /// Shared keyed GPU readback lifecycle service.
    inner: GpuReadbackJobs<Sh2SourceKey, RenderSH2>,
}

impl Default for Sh2ReadbackJobs {
    fn default() -> Self {
        Self::new()
    }
}

impl Sh2ReadbackJobs {
    /// Creates an empty readback job owner.
    pub(super) fn new() -> Self {
        Self {
            inner: GpuReadbackJobs::new(MAX_PENDING_JOB_AGE_FRAMES, parse_sh2_bytes),
        }
    }

    /// Returns a sender that queue-submit callbacks can use to mark jobs done.
    pub(super) fn submit_done_sender(&self) -> crossbeam_channel::Sender<Sh2SourceKey> {
        self.inner.submit_done_sender()
    }

    /// Returns the number of currently pending readbacks.
    pub(super) fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns true when `key` is already pending.
    pub(super) fn contains_key(&self, key: &Sh2SourceKey) -> bool {
        self.inner.contains_key(key)
    }

    /// Inserts a newly submitted GPU readback job.
    pub(super) fn insert(&mut self, key: Sh2SourceKey, job: SubmittedGpuSh2Job) {
        self.inner.insert(key, job.into());
    }

    /// Advances submit notifications, mapping, completion, and age/failure handling.
    pub(super) fn maintain(&mut self) -> Sh2ReadbackOutcomes {
        self.inner.maintain()
    }
}

/// Parses nine packed `vec4<f32>` SH rows from GPU readback bytes.
fn parse_sh2_bytes(bytes: &[u8]) -> Option<RenderSH2> {
    const _: () = assert!(
        cfg!(target_endian = "little"),
        "renderide assumes a little-endian target for GPU readback unpacking",
    );
    const SH2_BYTES: usize = 9 * 16;
    let payload = bytes.get(..SH2_BYTES)?;
    let coeffs: [[f32; 4]; 9] = bytemuck::try_pod_read_unaligned(payload).ok()?;
    Some(RenderSH2 {
        sh0: Vec3::new(coeffs[0][0], coeffs[0][1], coeffs[0][2]),
        sh1: Vec3::new(coeffs[1][0], coeffs[1][1], coeffs[1][2]),
        sh2: Vec3::new(coeffs[2][0], coeffs[2][1], coeffs[2][2]),
        sh3: Vec3::new(coeffs[3][0], coeffs[3][1], coeffs[3][2]),
        sh4: Vec3::new(coeffs[4][0], coeffs[4][1], coeffs[4][2]),
        sh5: Vec3::new(coeffs[5][0], coeffs[5][1], coeffs[5][2]),
        sh6: Vec3::new(coeffs[6][0], coeffs[6][1], coeffs[6][2]),
        sh7: Vec3::new(coeffs[7][0], coeffs[7][1], coeffs[7][2]),
        sh8: Vec3::new(coeffs[8][0], coeffs[8][1], coeffs[8][2]),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies SH2 readback bytes are parsed as nine RGB coefficient rows.
    #[test]
    fn parse_sh2_bytes_reads_nine_rgb_rows() {
        let mut bytes = vec![0u8; 9 * 16];
        for row in 0..9 {
            for channel in 0..3 {
                let value = row as f32 + channel as f32 * 0.25;
                let offset = row * 16 + channel * 4;
                bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            }
        }

        let sh = parse_sh2_bytes(&bytes).unwrap();
        assert_eq!(sh.sh0, Vec3::new(0.0, 0.25, 0.5));
        assert_eq!(sh.sh8, Vec3::new(8.0, 8.25, 8.5));
    }

    /// A short payload should be rejected rather than reading out of bounds.
    #[test]
    fn parse_sh2_bytes_rejects_short_payload() {
        let bytes = vec![0u8; 9 * 16 - 1];
        assert!(parse_sh2_bytes(&bytes).is_none());
    }

    /// Parsing must work even when the slice does not start at an f32-aligned address.
    #[test]
    fn parse_sh2_bytes_handles_unaligned_input() {
        let mut padded = [0u8; 1 + 9 * 16];
        for row in 0..9 {
            for channel in 0..3 {
                let value = row as f32 - channel as f32 * 0.5;
                let offset = 1 + row * 16 + channel * 4;
                padded[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            }
        }
        let sh = parse_sh2_bytes(&padded[1..]).unwrap();
        assert_eq!(sh.sh3, Vec3::new(3.0, 2.5, 2.0));
    }
}
