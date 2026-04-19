//! Helpers for [`ShaderUpload`](super::ShaderUpload) without editing generated `shared.rs`.
//!
//! Stock IPC encodes [`ShaderUpload`] as `asset_id` plus optional `file` string only. A custom host may
//! append an extra length-prefixed UTF-16 string (same wire format as
//! [`MemoryUnpacker::read_str`](crate::packing::memory_unpacker::MemoryUnpacker::read_str))
//! after those fields; use [`unpack_appended_shader_logical_name`] on the trailing bytes when you decode
//! messages outside the generated decode path, then pass the result to the renderer's logical-name
//! resolver (`renderide::assets::shader::logical_name::resolve_logical_shader_name_from_upload_with_host_hint`).
//! For routing and embedded stem matching, prefer the renderer's
//! `renderide::assets::shader::resolve_shader_routing_name_from_upload` with the unpacked string as `host_hint`.

use crate::packing::default_entity_pool::DefaultEntityPool;
use crate::packing::memory_unpacker::MemoryUnpacker;

/// Decodes an optional logical Unity shader name from bytes that follow a fully unpacked stock [`ShaderUpload`].
///
/// Returns [`None`] when `trailing` is empty or when the slice does not contain a valid string prefix.
pub fn unpack_appended_shader_logical_name(trailing: &[u8]) -> Option<String> {
    if trailing.is_empty() {
        return None;
    }
    let mut pool = DefaultEntityPool;
    let mut unpacker = MemoryUnpacker::new(trailing, &mut pool);
    unpacker.read_str().ok().flatten()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packing::memory_packable::MemoryPackable;
    use crate::packing::memory_packer::MemoryPacker;
    use crate::packing::memory_unpacker::MemoryUnpacker;
    use crate::shared::ShaderUpload;

    #[test]
    fn appended_name_roundtrips_after_stock_shader_upload_pack() {
        let mut buf = vec![0u8; 512];
        let cap = buf.len();
        let mut upload = ShaderUpload {
            asset_id: 9,
            file: Some(r"C:\tmp\variant.wgsl".to_string()),
        };
        let mut packer = MemoryPacker::new(&mut buf[..]);
        upload.pack(&mut packer);
        let after_upload = cap - packer.remaining_len();
        let name = "Custom/AppendedShader";
        packer.write_str(Some(name));
        let payload_len = cap - packer.remaining_len();

        let mut pool = DefaultEntityPool;
        let mut unpacker = MemoryUnpacker::new(&buf[..after_upload], &mut pool);
        let mut decoded = ShaderUpload::default();
        decoded
            .unpack(&mut unpacker)
            .expect("unpack stock ShaderUpload");
        assert_eq!(decoded.asset_id, 9);
        assert_eq!(
            unpack_appended_shader_logical_name(&buf[after_upload..payload_len]).as_deref(),
            Some(name)
        );
    }
}
