//! Build-time converted Unity/Resonite shaders and generated material descriptors.
//!
//! Regenerate outputs with:
//! `dotnet run --project UnityShaderConverter -- --skip-slang` (from the `Renderide/` directory),
//! or omit `--skip-slang` when `slangc` is on `PATH` / `SLANGC` is set.

pub mod generated;

#[cfg(test)]
mod wgsl_validate_tests {
    /// Validates that committed sample WGSL parses with the same `naga` revision `wgpu` uses.
    #[test]
    fn minimal_unlit_sample_wgsl_parses() {
        let src = include_str!("generated/wgsl/converter_minimal_unlit_pass0_v0.wgsl");
        let mut front = naga::front::wgsl::Frontend::new();
        let _ = front
            .parse(src)
            .expect("sample WGSL should parse; fix the file or regenerate with slangc");
    }
}
