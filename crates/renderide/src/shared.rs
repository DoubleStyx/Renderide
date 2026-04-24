//! Re-export of [`renderide_shared`], the workspace shared types and packing crate.
//!
//! This thin shim preserves `crate::shared::*` paths used throughout the renderer source. New
//! renderer code may import directly from [`renderide_shared`] instead.

pub use renderide_shared::buffer;
pub use renderide_shared::packing;
pub use renderide_shared::packing_extras;
pub use renderide_shared::shader_upload_extras;

/// Generated Renderite shared types and decode helpers (re-exported from
/// [`renderide_shared::shared`]).
pub use renderide_shared::shared;

pub use renderide_shared::packing::polymorphic_decode_error::PolymorphicDecodeError;
pub use renderide_shared::packing::wire_decode_error::WireDecodeError;
pub use renderide_shared::packing::{
    default_entity_pool, enum_repr, memory_packable, memory_packer, memory_packer_entity_pool,
    memory_unpack_error, memory_unpacker, packed_bools, polymorphic_decode_error,
    polymorphic_memory_packable_entity, wire_decode_error,
};
pub use renderide_shared::shared::*;

#[cfg(test)]
mod reexport_tests {
    /// Compile-time smoke: the core generated enums and structs referenced throughout the renderer
    /// remain reachable via `crate::shared::*`. Losing any of these re-exports would break many
    /// call sites at once, so a focused compile check is cheaper than chasing the downstream noise.
    #[test]
    fn core_reexports_are_reachable() {
        use crate::shared::{
            ComputeResult, HeadOutputDevice, LightType, RenderTransform, RendererInitData,
            ShadowType,
        };
        let _ = HeadOutputDevice::Screen;
        let _ = LightType::Directional;
        let _ = ShadowType::None;
        let _ = ComputeResult::Failed;
        let _ = RenderTransform::default();
        let _ = RendererInitData::default();
    }

    /// The packing and decode-error re-exports are used by every IPC-touching module; reference
    /// them by path here so that a removed re-export fails at this test's compile rather than at
    /// the first IPC call site.
    #[test]
    fn packing_reexports_are_reachable() {
        let _ = std::mem::size_of::<crate::shared::PolymorphicDecodeError>();
        let _ = std::mem::size_of::<crate::shared::WireDecodeError>();
    }
}
