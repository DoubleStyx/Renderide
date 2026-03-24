//! Renderide library: Resonite renderer replacement.
//!
//! Provides the public API for the main binary, roundtrip test binary, and future extensions.

/// Application entry point: event loop, window lifecycle, and winit integration.
pub mod app;

/// Asset storage, registry, and mesh/texture/shader management.
pub mod assets;

/// Render configuration types (clip planes, FOV, display settings).
pub mod config;

/// Debug HUD (ImGui, feature `debug-hud`, default on) and log throttling/deduplication helpers.
mod diagnostics;

/// GPU state, pipelines, mesh buffers, and wgpu integration.
pub mod gpu;

/// Window input state and key mapping for host IPC.
pub mod input;

/// IPC: command receiver and shared memory access.
pub mod ipc;

/// Render loop, draw batching, and render graph.
pub mod render;

/// Scene graph and scene management.
pub mod scene;

/// Stencil state for GraphicsChunk masking (UI scroll rects, clipping).
pub mod stencil;

/// Session: orchestrates IPC, scene, assets, and frame flow.
pub mod session;

/// Shared types and memory packing for host–renderer IPC.
pub mod shared;

/// Shader modules: UnityShaderConverter output under `shaders/generated/`, re-exported from `shaders`.
pub mod shaders;

/// Runs the Renderide application. Entry point for the main binary.
pub fn run() -> Option<i32> {
    app::run()
}
