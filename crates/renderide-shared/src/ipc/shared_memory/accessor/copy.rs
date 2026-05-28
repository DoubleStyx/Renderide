//! POD copy helpers for [`super::SharedMemoryAccessor`].

use core::mem::{align_of, size_of};

use bytemuck::{Pod, Zeroable};

/// Copies a typed POD slice out of `bytes`, falling back to per-element unaligned reads when the
/// source pointer is not [`T`]-aligned.
pub(super) fn copy_pod_slice<T: Pod + Zeroable>(
    bytes: &[u8],
    length: usize,
    prefix_err: &impl Fn(&str) -> String,
) -> Result<Vec<T>, String> {
    let type_size = size_of::<T>();
    let remainder = length % type_size;
    if remainder != 0 {
        return Err(prefix_err(&format!(
            "length {length} is not a multiple of type size {type_size} (remainder {remainder})"
        )));
    }
    let count = length / type_size;
    if count == 0 {
        return Ok(Vec::new());
    }

    let align = align_of::<T>();
    let base = bytes.as_ptr() as usize;
    if base.is_multiple_of(align)
        && let Ok(slice) = bytemuck::try_cast_slice::<u8, T>(bytes)
        && slice.len() >= count
    {
        return Ok(slice[..count].to_vec());
    }

    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let start = i * type_size;
        let chunk = bytes
            .get(start..start + type_size)
            .ok_or_else(|| prefix_err("pod chunk subslice"))?;
        let value = bytemuck::try_pod_read_unaligned::<T>(chunk)
            .map_err(|e| prefix_err(&format!("pod_read_unaligned: {e:?}")))?;
        out.push(value);
    }
    Ok(out)
}
