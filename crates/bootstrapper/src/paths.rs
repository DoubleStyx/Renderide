//! Resonite installation and `dotnet` discovery (Steam, env vars, registry on Windows).
//!
//! Split into focused submodules:
//! - [`resonite`]: Resonite install detection (env var, candidate ordering, install-dir check).
//! - [`steam`]: Steam-specific introspection (`libraryfolders.vdf`, default roots, registry).
//! - [`dotnet`]: `dotnet` resolution (bundled vs. system `PATH`).

mod dotnet;
mod resonite;
mod steam;

pub use dotnet::find_dotnet_for_host;
pub use resonite::{RENDERITE_HOST_DLL, resolve_resonite_dir};
