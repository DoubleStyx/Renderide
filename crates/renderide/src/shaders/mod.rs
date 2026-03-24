//! Shader modules: UnityShaderConverter output lives under [`generated`].
//! `pub use generated::*` keeps existing `crate::shaders::pbs_metallic` paths stable.

pub mod generated;

pub use generated::*;
