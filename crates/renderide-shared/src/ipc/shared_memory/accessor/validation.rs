//! Descriptor validation helpers for [`super::SharedMemoryAccessor`].

use crate::buffer::SharedMemoryBufferDescriptor;

/// Validates the descriptor invariants shared by all `access_copy_*` paths: positive length and
/// length within the `max_bytes` ceiling.
pub(super) fn validate_access_copy_descriptor(
    descriptor: &SharedMemoryBufferDescriptor,
    max_bytes: i32,
    prefix_err: &impl Fn(&str) -> String,
) -> Result<(), String> {
    if descriptor.length <= 0 {
        return Err(prefix_err(&format!(
            "length<=0 (buffer_id={} offset={} length={})",
            descriptor.buffer_id, descriptor.offset, descriptor.length
        )));
    }
    if descriptor.length > max_bytes {
        return Err(prefix_err(&format!(
            "length {} exceeds max {} (buffer_id={})",
            descriptor.length, max_bytes, descriptor.buffer_id
        )));
    }
    Ok(())
}

/// Validates a host-row descriptor before opening a mapping for sentinel-aware row decoding.
pub(super) fn validate_memory_packable_row_descriptor(
    descriptor: &SharedMemoryBufferDescriptor,
    element_stride: usize,
    max_bytes: i32,
    prefix_err: &impl Fn(&str) -> String,
) -> Result<(), String> {
    if element_stride == 0 {
        return Err(prefix_err("element_stride must be nonzero"));
    }
    validate_access_copy_descriptor(descriptor, max_bytes, prefix_err)?;
    let length = descriptor.length as usize;
    let remainder = length % element_stride;
    if remainder != 0 {
        return Err(prefix_err(&format!(
            "length {length} is not a multiple of element_stride {element_stride} (remainder {remainder})"
        )));
    }
    Ok(())
}
