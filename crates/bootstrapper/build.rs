//! Build script that embeds a Windows side-by-side manifest declaring a
//! dependency on Common Controls v6, so `rfd`'s `TaskDialogIndirect` import
//! resolves at process load time instead of failing with "Entry Point Not
//! Found in comctl32.dll".
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=build.rs");
    emit_release_metadata();
    if std::env::var_os("CARGO_CFG_WINDOWS").is_some() {
        use embed_manifest::{embed_manifest, new_manifest};
        embed_manifest(new_manifest("Renderide.Bootstrapper"))?;
    }
    Ok(())
}

/// Forwards CI release metadata into the compiled launcher when the release
/// workflow supplies it. Local builds leave these unset, which keeps update
/// checks disabled without needing fragile source-tree detection.
fn emit_release_metadata() {
    for key in [
        "RENDERIDE_RELEASE_CHANNEL",
        "RENDERIDE_RELEASE_TAG",
        "RENDERIDE_RELEASE_COMMIT",
        "RENDERIDE_RELEASE_PLATFORM",
    ] {
        println!("cargo:rerun-if-env-changed={key}");
        if let Ok(value) = std::env::var(key)
            && !value.trim().is_empty()
        {
            println!("cargo:rustc-env={key}={value}");
        }
    }
}
