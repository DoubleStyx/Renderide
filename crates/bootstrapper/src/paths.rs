//! Resonite installation and `dotnet` discovery (Steam, env vars, registry on Windows).
//!
//! Split into focused submodules:
//! - [`resonite`]: Resonite install detection (env var, candidate ordering, install-dir check).
//! - [`steam`]: Steam-specific introspection (`libraryfolders.vdf`, default roots, registry).
//! - [`dotnet`]: `dotnet` resolution (bundled vs. system `PATH`).

mod dotnet;
mod resonite;
mod steam;

pub(crate) use dotnet::find_dotnet_for_host;
pub(crate) use resonite::{find_resonite_dir, RENDERITE_HOST_DLL};
