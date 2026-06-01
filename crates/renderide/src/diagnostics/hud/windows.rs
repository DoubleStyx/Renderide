//! Concrete HUD windows and the **Renderide debug** main panel's tab impls.
//!
//! Each module under `windows/` is a unit struct + [`super::view::HudWindow`] impl. Shared
//! formatting helpers and label utilities live alongside as `labels` / `table_helpers`.

pub mod feedback;
pub mod frame_timing;
pub mod labels;
pub mod main_debug;
pub mod renderer_config;
pub mod scene_transforms;
pub mod table_helpers;
pub mod texture_debug;
