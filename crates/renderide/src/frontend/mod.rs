//! Host transport and session layer: IPC queues, shared memory, init handshake, lock-step gating.
//!
//! Layout: [`init_state`] ([`InitState`]), [`renderer_frontend`] ([`RendererFrontend`]),
//! [`frame_start_performance`] (optional [`crate::shared::PerformanceState`] on outgoing
//! [`crate::shared::FrameStartData`]), and [`input`] (winit → [`crate::shared::InputState`]).
//!
//! [`RendererFrontend`] owns [`DualQueueIpc`](crate::ipc::DualQueueIpc),
//! [`SharedMemoryAccessor`](crate::ipc::SharedMemoryAccessor), [`InitState`], and frame lock-step
//! fields (`last_frame_index`, when to send [`FrameStartData`](crate::shared::FrameStartData)). It
//! does **not** perform mesh/texture GPU uploads or mutate [`SceneCoordinator`](crate::scene::SceneCoordinator);
//! the runtime façade combines this layer with [`crate::backend::RenderBackend`] and scene.

mod begin_frame;
mod frame_start_performance;
mod init_state;
mod renderer_frontend;

/// Winit adapter and [`WindowInputAccumulator`](input::WindowInputAccumulator) for [`crate::shared::InputState`].
pub mod input;

pub use init_state::InitState;
pub use renderer_frontend::RendererFrontend;
