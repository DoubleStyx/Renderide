//! Naga type helpers for global resources (address space resolution, storage stride).

use naga::proc::Layouter;
use naga::{AddressSpace, ArraySize, TypeInner};
use naga::{Module, Type};

use super::types::ReflectError;

/// Resolves the address space and data type for a global resource (WGSL may use a plain struct/array
/// type with [`naga::GlobalVariable::space`], or a [`TypeInner::Pointer`] wrapper).
pub(super) fn resource_data_ty(
    module: &Module,
    gv: &naga::GlobalVariable,
) -> (AddressSpace, naga::Handle<Type>) {
    match &module.types[gv.ty].inner {
        TypeInner::Pointer { base, space } => (*space, *base),
        _ => (gv.space, gv.ty),
    }
}

/// Stride of one element in a runtime-sized storage array (e.g. `GpuLight`, `u32`, `atomic<u32>`).
pub(super) fn storage_array_element_stride(
    module: &Module,
    layouter: &Layouter,
    data_ty: naga::Handle<Type>,
    binding: u32,
) -> Result<u32, ReflectError> {
    match &module.types[data_ty].inner {
        TypeInner::Array { base: el, size, .. } => {
            let el_stride = layouter[*el].to_stride();
            match size {
                ArraySize::Pending(_) => Err(ReflectError::UnsupportedBinding {
                    group: 0,
                    binding,
                    reason: "pending array size".into(),
                }),
                ArraySize::Constant(_) | ArraySize::Dynamic => Ok(el_stride),
            }
        }
        _ => Err(ReflectError::UnsupportedBinding {
            group: 0,
            binding,
            reason: "expected runtime-sized array".into(),
        }),
    }
}
