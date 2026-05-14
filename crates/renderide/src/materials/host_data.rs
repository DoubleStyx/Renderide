//! Material property batches from IPC (`MaterialsUpdateBatch`, property id interning).

mod properties;
mod property_registry;
mod update_batch;

pub use properties::{
    MaterialDictionary, MaterialPropertyLookupIds, MaterialPropertyStore, MaterialPropertyValue,
};
pub use property_registry::PropertyIdRegistry;
pub use update_batch::{
    MaterialBatchParseReport, ParseMaterialBatchOptions,
    parse_materials_update_batch_into_store_with_instance_changed,
};
