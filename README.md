A modern Rust + wgpu renderer for Resonite.

## Status

Experimental: Performance, stability, and platform support are still evolving.  
Visual bugs and missing features are expected. Not intended for general use yet.

## Quick Start

1. Ensure you have a Steam installation of [Resonite](https://store.steampowered.com/app/2519830/Resonite/).

1. Install [ResoniteModLoader](https://github.com/resonite-modding-group/ResoniteModLoader) using their [installation guide](https://github.com/resonite-modding-group/ResoniteModLoader/wiki/Installation).

1. Clone this repository and switch to the `Renderide/` directory:

   ```bash
   git clone https://github.com/DoubleStyx/Renderide.git
   cd Renderide
   ```

1. Build the renderer and the mod. Ensure you have the [.NET 10 SDK](https://dotnet.microsoft.com/download) and [Rustup](https://rustup.rs/) installed.

   ```bash
   cargo build --release

   dotnet build RenderidePatches/RenderidePatches.csproj -c Release
   ```

1. Run the bootstrapper:

   ```bash
   cargo run --release -p bootstrapper
   ```

The bootstrapper will launch the Resonite host and connect Renderide automatically.

Logs appear in the `logs/` folder (see [Debugging](#debugging) for details).

## Configuration

Optional `configuration.ini` can be placed next to the executable or in the working directory.  
See `crates/renderide/src/config.rs` for available keys. For example, `rendering.use_opengl` forces the GLES backend (useful when Vulkan is unavailable), and `rendering.use_dx12` forces the DirectX 12 backend (primarily useful on Windows).

## Repository Layout

### Rust Crates

| Crate          | Path                              | Purpose |
|----------------|-----------------------------------|---------|
| `bootstrapper` | `crates/bootstrapper/`            | Launches Resonite host and manages IPC |
| `renderide`    | `crates/renderide/`               | Main renderer (wgpu, shaders, scene) |
| `interprocess` | `crates/interprocess/`            | Shared-memory IPC queues |
| `logger`       | `crates/logger/`                  | Shared structured logging |

### .NET Projects

| Project               | Path                               | Purpose |
|-----------------------|------------------------------------|---------|
| `UnityShaderConverter` | `generators/UnityShaderConverter/` | Converts Unity `.shader` files → WGSL + Rust modules |
| `SharedTypeGenerator` | `generators/SharedTypeGenerator/`  | Generates Rust types from `Renderite.Shared.dll` |
| `RenderidePatches` | `RenderidePatches/RenderidePatches.csproj` | Modifies Resonite to use Renderide |

### Third-Party

| Project                 | Path                                 | Purpose |
|-------------------------|--------------------------------------|---------|
| `UnityShaderParser`     | `third_party/UnityShaderParser/`     | Vendored parser for ShaderLab |
| `Resonite.UnityShaders` | `third_party/Resonite.UnityShaders/` | Vendored Unity shaders |

## Debugging

All logs go to the `logs/` folder relative to the bootstrapper’s working directory.

| Log file                   | Created by           | Notes |
|----------------------------|----------------------|-------|
| `Bootstrapper.log`         | Bootstrapper         | Orchestration & IPC |
| `HostOutput.log`           | Bootstrapper         | Resonite host stdout/stderr |
| `Renderide.log`            | Renderide            | Main renderer logs |
| `UnityShaderConverter.log` | UnityShaderConverter | Shader conversion details |
| `SharedTypeGenerator.log`  | SharedTypeGenerator  | Type generation details |

**Verbosity**:  
`cargo run --release -p bootstrapper -- --log-level debug`

**GPU Validation**: Set `RENDERIDE_GPU_VALIDATION=1` before starting (performance-heavy).

## Goals

- Full Unity renderer parity
- Modern clustered-forward rendering
- Raytracing and other optional rendering features
- Excellent VR performance and correctness

## License

See [LICENSE](LICENSE).
