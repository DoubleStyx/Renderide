fn main() {
    #[cfg(windows)]
    copy_openxr_loader();
}

#[cfg(windows)]
fn copy_openxr_loader() {
    use std::path::PathBuf;

    let out_dir = std::env::var("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            // Fall back: walk up from CARGO_MANIFEST_DIR to find target/
            let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
            manifest
                .ancestors()
                .find_map(|p| {
                    let t = p.join("target");
                    if t.is_dir() {
                        Some(t)
                    } else {
                        None
                    }
                })
                .expect("could not find target/ directory")
        });

    let profile = std::env::var("PROFILE").unwrap_or_else(|_| "debug".into());
    let dest_dir = out_dir.join(&profile);

    // Common SteamVR locations for openxr_loader.dll
    let candidates = [
        r"C:\Program Files (x86)\Steam\steamapps\common\SteamVR\bin\win64\openxr_loader.dll",
        r"C:\Program Files\Steam\steamapps\common\SteamVR\bin\win64\openxr_loader.dll",
    ];

    let src = candidates
        .iter()
        .map(std::path::Path::new)
        .find(|p| p.exists());

    if let Some(src) = src {
        let dest = dest_dir.join("openxr_loader.dll");
        if !dest.exists() {
            std::fs::copy(src, &dest)
                .unwrap_or_else(|e| panic!("failed to copy openxr_loader.dll: {e}"));
            println!("cargo:warning=Copied openxr_loader.dll from SteamVR to {dest:?}");
        }
    } else {
        println!("cargo:warning=openxr_loader.dll not found in SteamVR — VR will fall back to desktop. Install SteamVR or copy openxr_loader.dll manually to target/{profile}/");
    }

    println!("cargo:rerun-if-changed=build.rs");
}
