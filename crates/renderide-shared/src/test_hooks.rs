//! Test-only hooks shared between the harness and renderer. These do not affect production
//! behavior -- the production host never produces the prefix below, so the renderer's
//! AssetBundle resolution path is unchanged when running attached to a real host.
//!
//! When the renderer integration harness wants to drive a particular embedded WGSL stem
//! without supplying a Unity AssetBundle, it sets [`crate::shared::ShaderUpload::file`] to
//! `RENDERIDE_TEST_STEM:<stem>`. The renderer's shader route resolver checks for the prefix
//! and short-circuits to [`embedded_default_stem_for_shader_asset_name`] without touching the
//! filesystem.
//!
//! [`embedded_default_stem_for_shader_asset_name`]: ../../renderide/src/materials/index.html

/// Sentinel prefix that opts a [`crate::shared::ShaderUpload::file`] into direct embedded-stem
/// routing in tests. The production host never produces this prefix.
pub const RENDERIDE_TEST_STEM_PREFIX: &str = "RENDERIDE_TEST_STEM:";
