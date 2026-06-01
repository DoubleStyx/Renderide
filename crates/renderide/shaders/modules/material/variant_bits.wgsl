//! Helpers for renderer-reserved shader variant bitfields.

#define_import_path renderide::material::variant_bits

override renderide_static_variant_bits_mode: u32 = 0u;
override renderide_static_variant_bits: u32 = 0u;

fn effective(bits: u32) -> u32 {
    if (renderide_static_variant_bits_mode != 0u) {
        return renderide_static_variant_bits;
    }
    return bits;
}

fn enabled(bits: u32, mask: u32) -> bool {
    return (effective(bits) & mask) != 0u;
}
