# Renderide

A modern Rust + wgpu renderer for [Resonite](https://store.steampowered.com/app/2519830/Resonite/). Unofficial Renderide thread discussion [here](https://discord.com/channels/1040316820650991766/1156348246973751487) (in the Resonite Discord).

Also available as an [AUR package](https://aur.archlinux.org/packages/renderide-git).

If you're interested in supporting my work, please consider donating on [Ko-Fi](https://ko-fi.com/DoubleStyx) or [GitHub Sponsors](https://github.com/sponsors/DoubleStyx).

## PHOTOSENSITIVITY WARNING

Renderide is experimental and may have visual bugs that are more severe or unexpected than Resonite's default Unity renderer, including flicker, flashing frames, incorrect brightness or contrast, broken post-processing, and rapidly changing patterns. These renderer artifacts, as well as user-created content, can trigger seizures or other symptoms in people with photosensitive epilepsy or related sensitivities. Stop using Renderide immediately and move away from the display if you feel dizzy, disoriented, nauseated, experience eye discomfort, or notice involuntary movement or vision changes.

## Feedback

Please report crashes, bugs, missing features, performance problems, and other feedback through whichever channel fits best:

- [GitHub Issues](https://github.com/DoubleStyx/Renderide/issues) for tracked bugs, crashes, feature requests, and reproducible performance issues.
- [Resonite Rendering Discussion Discord](https://discord.com/channels/1040316820650991766/1156348246973751487) for discussion and quick triage.
- [Telegram](https://t.me/DoubleStyx) for direct contact.

## Status

Experimental: performance, stability, and platform support are still evolving.
*Visual bugs and missing features are expected.*

## What is Renderide

Resonite ships with a Unity-based renderer driven by the FrooxEngine host. Renderide is a drop-in replacement for that renderer, written in Rust on top of [wgpu](https://wgpu.rs/) and [OpenXR](https://www.khronos.org/openxr/). The host process is unchanged; Renderide attaches to it over shared-memory queues and takes over rendering, windowing, and XR.

The split lets the engine and renderer evolve independently and lets the renderer target Vulkan, Metal, and DirectX 12 from a single Rust codebase. OpenGL can also be selected as a fallback backend in the renderer config.

## Building and Running

Prerequisites: a GPU with a wgpu-supported desktop backend (Vulkan, Metal, DirectX 12, or OpenGL fallback) and a Steam installation of [Resonite](https://store.steampowered.com/app/2519830/Resonite/). OpenXR VR startup currently uses Vulkan regardless of the configured desktop graphics API.

1. Clone this repository and switch to the `Renderide/` directory:

   ```bash
   git clone https://github.com/DoubleStyx/Renderide.git
   cd Renderide
   ```

1. Install Rust with [Rustup](https://rustup.rs/) (if missing) and build the renderer:

   ```bash
   cargo build --release
   ```

1. Run the launcher:

   ```bash
   ./target/release/renderide
   ```

The launcher will start the Resonite host and connect Renderide automatically.

- Enable GPU validation layers in the config HUD to get more detailed error messages for GPU crashes. Requires a restart.

- Logs are timestamped files under a selected logs root. Source builds normally resolve the active repository and write renderer logs to `logs/renderer/`. Installed release binaries fall back to the current user's platform log root: `$XDG_STATE_HOME/renderide/logs` or `~/.local/state/renderide/logs` on Linux, `~/Library/Logs/Renderide` on macOS, and `%LOCALAPPDATA%\Renderide\logs` on Windows. Set `RENDERIDE_LOGS_ROOT` to choose the root explicitly; component logs then live under `renderer/`, `bootstrapper/`, `host/`, `renderer-test/`, and `SharedTypeGenerator/`. The Renderer config HUD also shows the selected log folder and includes an "Open log folder" button.

- You can add Steam-style launch arguments after the launcher to enable mods: `<path-to-renderide> -LoadAssembly Libraries/ResoniteModLoader.dll`

### macOS

Renderide runs the Resonite Host from the Windows depot and renders natively through Metal. The launcher accepts an explicit Resonite install path, so local builds and release zips should not need hand-written symlinks or `dev-fast` path edits.

1. Install the Windows Resonite depot with SteamCMD:

   ```bash
   steamcmd +@sSteamCmdForcePlatformType windows \
     +force_install_dir "$HOME/Games/ResoniteWindows" \
     +login anonymous \
     +app_update 2519830 validate \
     +quit
   ```

   If anonymous access is unavailable, use the Steam login that owns Resonite.

1. Install the .NET runtime requested by the Resonite Host so `dotnet` is available on `PATH`.

1. Run the launcher with the Windows depot path:

   ```bash
   ./target/release/renderide --resonite-dir "$HOME/Games/ResoniteWindows"
   ```

   Release zips use the same argument from the extracted folder:

   ```bash
   ./renderide --resonite-dir "$HOME/Games/ResoniteWindows"
   ```

The macOS release zip contains `renderide`, `renderide-renderer`, `xr`, and the bundled OpenXR loader. If macOS quarantine blocks a downloaded zip, remove the quarantine attribute from the extracted Renderide folder:

```bash
xattr -dr com.apple.quarantine /path/to/renderide-folder
```

Leave `RENDERIDE_INTERPROCESS_DIR` unset unless every participating Renderide and Host process is configured to the same path. For host-limited framerates, Resonite's forced separation setting can improve throughput, but it is not required for startup.

## Design goals

- **Cross-platform parity** - Linux, macOS, and Windows are all first-class. Mobile is a future direction; portability constraints are respected today.
- **Data-driven render graph** - Passes, materials, and resources route through shared systems rather than one-off code paths.
- **Allocation-conscious hot paths** - The frame loop leans on pooled buffers, persistent scratch, cached graph resources, and reusable asset slots so steady-state rendering avoids avoidable churn.
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
                              Renderer (renderide-renderer)
```

Inside the renderer, work is organized by layer:

1. **App** - owns process bootstrap, logging, config loading, shutdown handling, the winit event loop, frame clock, window target, and OpenXR target selection.
2. **Frontend** - owns Host transport: IPC queues, shared memory, init handshake, input conversion, output-device policy, and lock-step state.
3. **Scene** - owns the host world mirror: transforms, render spaces, mesh and skinned renderables, lights, cameras, and overrides. Pure data; does not touch wgpu.
4. **Backend** - owns GPU-facing renderer state: asset pools, material and shader systems, frame resources, draw preparation, render graph assembly, and command recording.
5. **Runtime** - coordinates frontend, scene, and backend in the fixed per-tick order used by the app driver.

Each tick: poll IPC, integrate a budgeted slice of pending assets, drain offscreen camera/reflection-probe tasks, run the optional OpenXR begin step, complete the lock-step exchange with the host, schedule and render views, present or submit the HMD frame, then update HUD and diagnostics state.

## Repository layout

The workspace lives under `crates/`:

| Crate | Purpose |
| --- | --- |
| [`bootstrapper`](crates/bootstrapper) | Builds the `renderide` launcher. Launches the Resonite host and renderer, owns bootstrap IPC (heartbeats, clipboard, renderer argv), and ties child process lifetimes together. |
| [`renderide`](crates/renderide) | Builds the `renderide-renderer` process and `roundtrip` helper. Owns winit, wgpu, OpenXR, scene mirroring, render graph, materials, assets, diagnostics, and presentation. |
| [`renderide-shared`](crates/renderide-shared) | Generated IPC types, binary packing helpers, shared-memory accessors/writers, and dual-queue wrappers. |
| [`interprocess`](crates/interprocess) | Cloudtoid-compatible shared-memory ring queues used by every IPC channel. |
| [`logger`](crates/logger) | File-first logging used by the bootstrapper, host capture, and renderer. |
| [`renderide-test`](crates/renderide-test) | Integration test harness that drives the renderer end-to-end. |

A C# generator under [`generators/SharedTypeGenerator`](generators/SharedTypeGenerator) emits `crates/renderide-shared/src/shared.rs`. Its test project lives under [`generators/SharedTypeGenerator.Tests`](generators/SharedTypeGenerator.Tests) and uses the `roundtrip` binary to compare C# and Rust packing. [`RenderideMod`](RenderideMod) contains the host-side Resonite mod, and [`third_party/openxr_loader`](third_party/openxr_loader) contains vendored OpenXR loader binaries used by release artifacts on Windows and macOS.

## Feature flags

The `renderide` crate exposes opt-in Cargo features for capabilities that depend on platform-specific system libraries or that are only useful in some workflows. Stock builds (`cargo build`) enable none of them.

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

- **Linux**: `libgstreamer1.0-dev`, `libgstreamer-plugins-base1.0-dev`, and `gstreamer1.0-plugins-good` on Debian/Ubuntu, or the equivalent GStreamer core/base development packages plus Good Plugins package on other distros. The Good Plugins package provides `videoflip`, which Renderide uses to match the renderer's texture orientation.
- **macOS**: `brew install gstreamer`.
- **Windows**: the official GStreamer MSVC SDK plus a working `pkg-config` (`pkgconf` rather than `pkgconfiglite`).

```bash
cargo build --features video-textures
```

## Configuration

Renderide reads its settings from a TOML file discovered (or created) at startup. Set `RENDERIDE_CONFIG` to point at a specific file; otherwise Renderide uses the current user's platform config directory and writes `Renderide/config.toml` there on first launch when possible.

The in-renderer ImGui config HUD edits the shared in-memory settings and persists them back to the same TOML file. Manual file edits are not watched or hot-reloaded while the process is running. Some settings, including GPU validation layers, graphics API, adapter power preference, and watchdog settings, are startup-only and require a renderer restart after changing. Explicit desktop graphics API choices are used for screen and headless startup; OpenXR startup logs a warning for non-Vulkan choices and uses its Vulkan path.

The full schema lives next to the loader in [`crates/renderide/src/config`](crates/renderide/src/config).

## Profiling

Renderide integrates with [Tracy](https://github.com/wolfpld/tracy) for CPU and GPU profiling.
CPU spans come from the `profiling` crate; GPU timestamp queries come from `wgpu-profiler`.
CPU profiling only requires the `tracy` feature. Pass-level GPU profiling requires `TIMESTAMP_QUERY` adapter support; frame-bracket and encoder-level GPU timing also require `TIMESTAMP_QUERY_INSIDE_ENCODERS`.
If timestamp queries are unavailable, a warning is logged and Tracy still receives CPU spans.

### Building with profiling enabled

```bash
cargo build --profile dev-fast --features tracy
```

### Connecting Tracy

1. Download the Tracy profiler GUI from the [Tracy releases page](https://github.com/wolfpld/tracy/releases)
   and launch it.

1. Start Renderide normally (launcher or renderer directly).

1. In the Tracy GUI, connect to `localhost` on port **8086**.

Renderide uses Tracy's `ondemand` mode: data is only streamed while the GUI is connected, so
profiled builds carry near-zero runtime cost when Tracy is not attached.

## Cross-platform support

Linux, macOS, and Windows are all tier-1 targets and exercised in CI ([`.github/workflows/`](.github/workflows)). iOS and Android are not yet supported, but the codebase avoids hard dependencies on desktop-only APIs where portable alternatives exist.

## Contributing

Contributions are welcome. The workspace builds with the standard Cargo commands listed above; lints (`cargo clippy --all-targets --all-features`) and formatting (`cargo fmt`, plus `taplo fmt` when editing `Cargo.toml`) are expected to be clean before opening a pull request, and CI runs the same checks across all three platforms.

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) to learn how to get started.

## Whitepaper

For a longer overview of the renderer architecture and motivation, see the [Renderide whitepaper](renderide_whitepaper/renderide_whitepaper.pdf).

## AI Policy

Renderide does not accept AI-generated or AI-assisted contributions. Source code, shaders, documentation, tests, issues, pull requests, and review comments submitted to this repository must be authored by the human contributor without generative AI tools. Contributors found submitting AI-generated material or using AI to participate in the project may be blocked from future contribution.

Renderide depends on upstream projects with their own contribution rules. For example, [`wgpu` explicitly allows LLM/AI-generated code](https://github.com/gfx-rs/wgpu/blob/trunk/CONTRIBUTING.md#llms-ai) when the pull request author accepts full ownership of the change. Renderide cannot impose this policy on upstream projects or dependencies; using those dependencies does not change the policy for contributions to this repository.

## License

MIT - see [`LICENSE`](LICENSE).
