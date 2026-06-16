//! Per-frame deferred [`wgpu::Queue::write_buffer`] routing.
//!
//! Record paths that run per-view push their uniform / storage uploads into a
//! [`FrameUploadBatch`] instead of invoking [`wgpu::Queue::write_buffer`] directly. The batch is
//! drained onto the main thread after all per-view recording finishes but before the single
//! [`crate::gpu::GpuContext::submit_frame_batch`] call. Writes are replayed by executor scope
//! `(frame-global before per-view, then view index, pass index, local call order)` so the result
//! is independent of which rayon worker won the upload-batch mutex first. All buffered writes
//! therefore land in the queue prior to submit and are visible to every command buffer in the
//! frame, identical to the direct-call serial path.
//!
//! This plumbing decouples queue ownership from parallel recording: a [`FrameUploadBatch`] can be
//! shared as a read-only reference across rayon workers, whereas [`wgpu::Queue`] access during
//! concurrent recording risks host-side ordering bugs on some backends.

mod batch;
mod scope;
mod sink;
mod stats;

pub use batch::FrameUploadBatch;
pub use sink::GraphUploadSink;
pub use stats::FrameUploadBatchStats;

pub(crate) use scope::FrameUploadScope;
