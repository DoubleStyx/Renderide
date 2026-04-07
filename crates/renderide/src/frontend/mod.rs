//! Host transport and session layer: IPC queues, shared memory, init handshake, lock-step gating.
//!
//! [`RendererFrontend`] owns [`DualQueueIpc`](crate::ipc::DualQueueIpc),
//! [`SharedMemoryAccessor`](crate::ipc::SharedMemoryAccessor),
//! [`InitState`], and frame lock-step fields (`last_frame_index`, when to send
//! [`FrameStartData`](crate::shared::FrameStartData)). It does **not** perform mesh/texture GPU
//! uploads or mutate [`SceneCoordinator`](crate::scene::SceneCoordinator); the runtime façade
//! combines this layer with [`crate::backend::RenderBackend`] and scene.

mod renderer_frontend;

/// Winit adapter and [`WindowInputAccumulator`](input::WindowInputAccumulator) for [`crate::shared::InputState`].
pub mod input;

pub use renderer_frontend::InitState;
pub use renderer_frontend::RendererFrontend;
