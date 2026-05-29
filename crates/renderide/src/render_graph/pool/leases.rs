//! Lease construction from `Pool` entries.

use super::policy::{BufferSlotValue, TextureSlotValue};
use super::{PooledBufferLease, PooledTextureLease, TransientPoolError};

pub(super) fn saturating_usize(value: u64) -> usize {
    if value > usize::MAX as u64 {
        usize::MAX
    } else {
        value as usize
    }
}

pub(super) fn texture_lease_from_entry(
    id: usize,
    slot: &TextureSlotValue,
) -> Result<PooledTextureLease, TransientPoolError> {
    let texture = slot
        .texture
        .clone()
        .ok_or(TransientPoolError::MissingTextureResources { pool_id: id })?;
    let view = slot
        .view
        .clone()
        .ok_or(TransientPoolError::MissingTextureResources { pool_id: id })?;
    Ok(PooledTextureLease {
        pool_id: id,
        texture,
        view,
        view_cache: slot.view_cache.clone(),
        resource_generation: slot.resource_generation,
    })
}

pub(super) fn buffer_lease_from_entry(
    id: usize,
    slot: &BufferSlotValue,
) -> Result<PooledBufferLease, TransientPoolError> {
    let buffer = slot
        .buffer
        .clone()
        .ok_or(TransientPoolError::MissingBuffer { pool_id: id })?;
    Ok(PooledBufferLease {
        pool_id: id,
        _buffer: buffer,
        _size: slot.size,
    })
}
