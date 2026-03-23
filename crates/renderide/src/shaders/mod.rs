//! Build-time converted Unity/Resonite shaders and generated material descriptors.
//!
//! Output is produced by **UnityShaderConverter** into [`generated`](generated/).
//!
//! Regenerate from the `Renderide/` directory:
//! `dotnet run --project generators/UnityShaderConverter --` (with `slangc` on `PATH` / `SLANGC`), or `--skip-slang` if WGSL is already on disk.

pub mod generated;

#[cfg(test)]
mod wgsl_validate_tests {
    /// Validates that committed sample WGSL parses with the same `naga` revision `wgpu` uses.
    #[test]
    fn minimal_unlit_sample_wgsl_parses() {
        let src = super::generated::wgsl_sources::converter_minimal_unlit::PASS0_V0;
        let mut front = naga::front::wgsl::Frontend::new();
        let _ = front
            .parse(src)
            .expect("sample WGSL should parse; fix the file or regenerate with slangc");
    }
}
