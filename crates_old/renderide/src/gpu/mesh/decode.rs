//! Decode packed vertex attributes from mesh byte slices into `f32` component values.

use crate::shared::VertexAttributeFormat;

/// Read a vec3 (normal or tangent) from vertex data at base+offset, converting from the given format to f32.
pub(super) fn read_vec3(
    data: &[u8],
    base: usize,
    offset: usize,
    format: VertexAttributeFormat,
) -> Option<[f32; 3]> {
    match format {
        VertexAttributeFormat::float32 => {
            if base + offset + 12 <= data.len() {
                Some([
                    f32::from_le_bytes(data[base + offset..base + offset + 4].try_into().ok()?),
                    f32::from_le_bytes(data[base + offset + 4..base + offset + 8].try_into().ok()?),
                    f32::from_le_bytes(
                        data[base + offset + 8..base + offset + 12]
                            .try_into()
                            .ok()?,
                    ),
                ])
            } else {
                None
            }
        }
        VertexAttributeFormat::half16 => {
            if base + offset + 6 <= data.len() {
                Some([
                    half_to_f32(u16::from_le_bytes(
                        data[base + offset..base + offset + 2].try_into().ok()?,
                    )),
                    half_to_f32(u16::from_le_bytes(
                        data[base + offset + 2..base + offset + 4].try_into().ok()?,
                    )),
                    half_to_f32(u16::from_le_bytes(
                        data[base + offset + 4..base + offset + 6].try_into().ok()?,
                    )),
                ])
            } else {
                None
            }
        }
        VertexAttributeFormat::u_norm8 => {
            if base + offset + 3 <= data.len() {
                Some([
                    (data[base + offset] as f32 / 255.0) * 2.0 - 1.0,
                    (data[base + offset + 1] as f32 / 255.0) * 2.0 - 1.0,
                    (data[base + offset + 2] as f32 / 255.0) * 2.0 - 1.0,
                ])
            } else {
                None
            }
        }
        VertexAttributeFormat::u_norm16 => {
            if base + offset + 6 <= data.len() {
                Some([
                    (u16::from_le_bytes(data[base + offset..base + offset + 2].try_into().ok()?)
                        as f32
                        / 65535.0)
                        * 2.0
                        - 1.0,
                    (u16::from_le_bytes(data[base + offset + 2..base + offset + 4].try_into().ok()?)
                        as f32
                        / 65535.0)
                        * 2.0
                        - 1.0,
                    (u16::from_le_bytes(data[base + offset + 4..base + offset + 6].try_into().ok()?)
                        as f32
                        / 65535.0)
                        * 2.0
                        - 1.0,
                ])
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Read a vec2 UV from vertex data at base+offset, converting from the given format to f32.
pub(super) fn read_uv(
    data: &[u8],
    base: usize,
    offset: usize,
    format: VertexAttributeFormat,
) -> Option<[f32; 2]> {
    match format {
        VertexAttributeFormat::float32 => {
            if base + offset + 8 <= data.len() {
                Some([
                    f32::from_le_bytes(data[base + offset..base + offset + 4].try_into().ok()?),
                    f32::from_le_bytes(data[base + offset + 4..base + offset + 8].try_into().ok()?),
                ])
            } else {
                None
            }
        }
        VertexAttributeFormat::half16 => {
            if base + offset + 4 <= data.len() {
                Some([
                    half_to_f32(u16::from_le_bytes(
                        data[base + offset..base + offset + 2].try_into().ok()?,
                    )),
                    half_to_f32(u16::from_le_bytes(
                        data[base + offset + 2..base + offset + 4].try_into().ok()?,
                    )),
                ])
            } else {
                None
            }
        }
        VertexAttributeFormat::u_norm8 => {
            if base + offset + 2 <= data.len() {
                Some([
                    data[base + offset] as f32 / 255.0,
                    data[base + offset + 1] as f32 / 255.0,
                ])
            } else {
                None
            }
        }
        VertexAttributeFormat::u_norm16 => {
            if base + offset + 4 <= data.len() {
                Some([
                    u16::from_le_bytes(data[base + offset..base + offset + 2].try_into().ok()?)
                        as f32
                        / 65535.0,
                    u16::from_le_bytes(data[base + offset + 2..base + offset + 4].try_into().ok()?)
                        as f32
                        / 65535.0,
                ])
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Reads a linear RGBA color (float32×4) from vertex data.
pub(super) fn read_color_float4(
    data: &[u8],
    base: usize,
    offset: usize,
    format: VertexAttributeFormat,
) -> Option<[f32; 4]> {
    match format {
        VertexAttributeFormat::float32 => {
            if base + offset + 16 <= data.len() {
                Some([
                    f32::from_le_bytes(data[base + offset..base + offset + 4].try_into().ok()?),
                    f32::from_le_bytes(data[base + offset + 4..base + offset + 8].try_into().ok()?),
                    f32::from_le_bytes(
                        data[base + offset + 8..base + offset + 12]
                            .try_into()
                            .ok()?,
                    ),
                    f32::from_le_bytes(
                        data[base + offset + 12..base + offset + 16]
                            .try_into()
                            .ok()?,
                    ),
                ])
            } else {
                None
            }
        }
        VertexAttributeFormat::u_norm8 => {
            if base + offset + 4 <= data.len() {
                Some([
                    data[base + offset] as f32 / 255.0,
                    data[base + offset + 1] as f32 / 255.0,
                    data[base + offset + 2] as f32 / 255.0,
                    data[base + offset + 3] as f32 / 255.0,
                ])
            } else {
                None
            }
        }
        VertexAttributeFormat::half16 => {
            if base + offset + 8 <= data.len() {
                Some([
                    half_to_f32(u16::from_le_bytes(
                        data[base + offset..base + offset + 2].try_into().ok()?,
                    )),
                    half_to_f32(u16::from_le_bytes(
                        data[base + offset + 2..base + offset + 4].try_into().ok()?,
                    )),
                    half_to_f32(u16::from_le_bytes(
                        data[base + offset + 4..base + offset + 6].try_into().ok()?,
                    )),
                    half_to_f32(u16::from_le_bytes(
                        data[base + offset + 6..base + offset + 8].try_into().ok()?,
                    )),
                ])
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Reads a vec4 (tangent / packed data) from vertex data.
pub(super) fn read_vec4_f32(
    data: &[u8],
    base: usize,
    offset: usize,
    format: VertexAttributeFormat,
) -> Option<[f32; 4]> {
    match format {
        VertexAttributeFormat::float32 => {
            if base + offset + 16 <= data.len() {
                Some([
                    f32::from_le_bytes(data[base + offset..base + offset + 4].try_into().ok()?),
                    f32::from_le_bytes(data[base + offset + 4..base + offset + 8].try_into().ok()?),
                    f32::from_le_bytes(
                        data[base + offset + 8..base + offset + 12]
                            .try_into()
                            .ok()?,
                    ),
                    f32::from_le_bytes(
                        data[base + offset + 12..base + offset + 16]
                            .try_into()
                            .ok()?,
                    ),
                ])
            } else {
                None
            }
        }
        VertexAttributeFormat::u_norm8 => {
            if base + offset + 4 <= data.len() {
                Some([
                    data[base + offset] as f32 / 255.0,
                    data[base + offset + 1] as f32 / 255.0,
                    data[base + offset + 2] as f32 / 255.0,
                    data[base + offset + 3] as f32 / 255.0,
                ])
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Convert IEEE 754 half-precision (f16) to f32.
pub(super) fn half_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;
    if exp == 0 {
        let f = (sign << 31) | (mant << 13);
        f32::from_bits(f) * 5.960_464_5e-8
    } else if exp == 31 {
        let f = (sign << 31) | 0x7F800000 | (mant << 13);
        f32::from_bits(f)
    } else {
        let f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
        f32::from_bits(f)
    }
}
