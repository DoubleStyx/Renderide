//! Core platonic types for the renderer framework.
//!
//! Engine-agnostic representations of scenes, views, drawables, and configuration
//! that are independent of the host protocol.

pub mod config;
pub mod math;
pub mod types;

pub use config::{RenderConfig, SessionConfig};
pub use math::render_transform_to_matrix;
pub use types::{
    Drawable, Frame, MeshHandle, MaterialHandle, NodeId, Projection, RenderTargetDesc, Scene,
    SceneId, Transform, TextureHandle, View, ViewId, Viewport,
};
