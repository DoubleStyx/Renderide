# Renderide

A modern Rust + wgpu renderer for [Resonite](https://store.steampowered.com/app/2519830/Resonite/).

## Status

Experimental: performance, stability, and platform support are still evolving.
Visual bugs and missing features are expected.

## What is Renderide

Resonite ships with a Unity-based renderer driven by the FrooxEngine host. Renderide is a drop-in replacement for that renderer, written in Rust on top of [wgpu](https://wgpu.rs/) and [OpenXR](https://www.khronos.org/openxr/). The host process is unchanged; Renderide attaches to it over shared-memory queues and takes over rendering, windowing, and XR.

The split lets the engine and renderer evolve independently and lets the renderer target Vulkan, Metal, and DirectX 12 from a single Rust codebase.

## Design goals

- **Cross-platform parity** - Linux, macOS, and Windows are all first-class. Mobile is a future direction; portability constraints are respected today.
- **Data-driven render graph** - Passes, materials, and resources route through shared systems rather than one-off code paths.
- **No per-frame allocations** - The hot path reuses pooled buffers and asset slots; allocation is restricted to init and asset integration.
- **OpenXR-first VR** - Stereo rendering and head-tracked input are part of the core path, not an afterthought.
- **Profiling-friendly** - Tracy CPU and GPU instrumentation is built in and zero-cost when disabled.
- **Safe by default** - `unsafe` is restricted to FFI and justified hot paths; library code avoids `unwrap`, `expect`, and `panic!`.

## Architecture

Renderide runs as a sibling process to the Resonite host. The bootstrapper launches both and wires up the IPC channels:

```
Bootstrapper  --shm queues-->  Host (.NET / Resonite)
                                   |
                              shm queues (Primary + Background)
                                   |
                                   v
                              Renderer (renderide)
```

Inside the renderer, work is organized into three layers:

1. **Frontend** - polls IPC queues, drives the winit event loop, and runs the lock-step protocol that gates frames against the host.
2. **Scene** - owns transforms, render spaces, mesh and skinned renderables, lights, and cameras. Pure data; does not touch wgpu.
3. **Backend** - owns the wgpu device, asset pools, the material system, and the compiled render graph. Produces command buffers and presents.

Each tick: poll IPC, integrate a budgeted slice of pending assets, run the optional OpenXR frame loop, complete the lock-step exchange with the host, render, then present.

## Repository layout

The workspace lives under `crates/`:

| Crate | Purpose |
| --- | --- |
| [`bootstrapper`](crates/bootstrapper) | Launches the Resonite host and the renderer; owns bootstrap IPC (heartbeats, clipboard, start signals). |
| [`renderide`](crates/renderide) | The renderer itself - winit, wgpu, OpenXR, scene, render graph, materials, assets. |
| [`renderide-shared`](crates/renderide-shared) | Generated IPC types and the hand-maintained wire-format helpers. |
| [`interprocess`](crates/interprocess) | Cloudtoid-compatible shared-memory ring queues used by every IPC channel. |
| [`logger`](crates/logger) | File-first logging used by the bootstrapper, host capture, and renderer. |
| [`renderide-test`](crates/renderide-test) | Integration test harness that drives the renderer end-to-end. |

A C# generator under [`generators/SharedTypeGenerator`](generators/SharedTypeGenerator) emits `crates/renderide-shared/src/shared.rs`. It is only needed when shared IPC types change.

## Building and Running

Prerequisites: a Vulkan-, Metal-, or DirectX 12-capable GPU and a Steam installation of [Resonite](https://store.steampowered.com/app/2519830/Resonite/).

1. Clone this repository and switch to the `Renderide/` directory:

   ```bash
   git clone https://github.com/DoubleStyx/Renderide.git
   cd Renderide
   ```

1. Install Rust with [Rustup](https://rustup.rs/) (if missing) and build the renderer:

   ```bash
   cargo build --release
   ```

1. Run the bootstrapper:

   ```bash
   ./target/release/bootstrapper
   ```

The bootstrapper will launch the Resonite host and connect Renderide automatically.

## Feature flags

The `renderide` crate exposes opt-in Cargo features for capabilities that depend on platform-specific system libraries or that are only useful in some workflows. Stock builds (`cargo build`) enable none of them.


```
Multiple features can be combined as a single space-separated argument:

```bash
cargo build --features "tracy video-textures"
```

### `tracy`

CPU and GPU profiling integration. Activates `profiling::scope!` zones, frame marks, and `wgpu-profiler` GPU timestamp queries that stream into the [Tracy](https://github.com/wolfpld/tracy) profiler GUI on port 8086. The Tracy client links statically, so this feature has no system-library prerequisites.

```bash
cargo build --features tracy
```

See [Profiling](#profiling) for adapter requirements and connection details.

### `video-textures`

GStreamer-backed video texture playback. With the feature off (the default), video texture IPC commands still allocate a GPU placeholder, but no decoding runs and the placeholder stays black.

System dependencies:

- **Linux**: `libgstreamer1.0-dev` and `libgstreamer-plugins-base1.0-dev` on Debian/Ubuntu, or the equivalent `gstreamer` packages on other distros.
- **macOS**: `brew install gstreamer`.
- **Windows**: the official GStreamer MSVC SDK plus a working `pkg-config` (`pkgconf` rather than `pkgconfiglite`).

```bash
cargo build --features video-textures
```

## Configuration

Renderide reads its settings from a TOML file discovered (or created) at startup. The runtime watches the file and applies most changes without a restart, and the in-renderer ImGui overlay edits the same settings.

The full schema lives next to the loader in [`crates/renderide/src/config`](crates/renderide/src/config).

## Debugging

1. Build the workspace in release mode:

   ```bash
   cargo build --release
   ```

1. Run the bootstrapper with Rust backtrace enabled:

   ```bash
   RUST_BACKTRACE=1 ./target/dev-fast/bootstrapper
   ```

1. Enable validation layers in the config hud to get more detailed error messages for GPU crashes. Requires a restart.

1. Inspect logs in the `logs/` folder for panics, crashes, backtraces, and validation errors.

## Profiling

Renderide integrates with [Tracy](https://github.com/wolfpld/tracy) for CPU and GPU profiling.
CPU spans come from the `profiling` crate; GPU timestamp queries come from `wgpu-profiler`.
GPU timing requires `TIMESTAMP_QUERY` and `TIMESTAMP_QUERY_INSIDE_ENCODERS` adapter support.
If either is missing, a warning is logged and only CPU spans are emitted.

### Building with profiling enabled

```bash
cargo build --profile dev-fast --features tracy
```

### Connecting Tracy

1. Download the Tracy profiler GUI from the [Tracy releases page](https://github.com/wolfpld/tracy/releases)
   and launch it.

1. Start Renderide normally (bootstrapper or renderer directly).

1. In the Tracy GUI, connect to `localhost` on port **8086**.

Renderide uses Tracy's `ondemand` mode: data is only streamed while the GUI is connected, so
profiled builds carry near-zero runtime cost when Tracy is not attached.

## Cross-platform support

Linux, macOS, and Windows are all tier-1 targets and exercised in CI ([`.github/workflows/`](.github/workflows)). iOS and Android are not yet supported, but the codebase avoids hard dependencies on desktop-only APIs where portable alternatives exist.

## Contributing

Contributions are welcome. The workspace builds with the standard Cargo commands listed above; lints (`cargo clippy --all-targets --all-features`) and formatting (`cargo fmt`, plus `taplo fmt` when editing `Cargo.toml`) are expected to be clean before opening a pull request, and CI runs the same checks across all three platforms.

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) to learn how to get started.

## License

MIT - see [`LICENSE`](LICENSE).
