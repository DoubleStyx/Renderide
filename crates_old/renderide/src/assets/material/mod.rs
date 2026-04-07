//! Material property contracts, batch wiring, and per-shader UI / world-unlit helpers.

pub mod batch_wire_metrics;
pub mod native_ui_blend;
pub mod properties;
pub mod property_host;
pub mod ui_contract;
pub mod update_batch;
pub mod world_unlit_contract;

pub use super::{
    AssetRegistry, EssentialShaderProgram, MaterialPropertyLookupIds, MaterialPropertyStore,
    MaterialPropertyValue, texture2d_asset_id_from_packed,
};
pub use crate::assets::shader_logical_name;
pub use batch_wire_metrics as material_batch_wire_metrics;
pub use properties as material_properties;
pub use property_host as material_property_host;
pub use ui_contract as ui_material_contract;
pub use world_unlit_contract as world_unlit_material_contract;
