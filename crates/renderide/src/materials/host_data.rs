//! Material property batches from IPC (`MaterialsUpdateBatch`, property id interning).

mod properties;
mod property_registry;
mod update_batch;

pub use properties::{
    MATERIAL_BATCH_MAX_FLOAT_ARRAY_LEN, MATERIAL_BATCH_MAX_FLOAT4_ARRAY_LEN, MaterialDictionary,
    MaterialPropertyLookupIds, MaterialPropertyStore, MaterialPropertyValue,
};
pub use property_registry::{MaterialPropertySemanticHook, PropertyIdRegistry};
pub use update_batch::{
    MaterialBatchBlobLoader, ParseMaterialBatchOptions, parse_materials_update_batch_into_store,
    parse_materials_update_batch_into_store_with_instance_changed,
};
