# Contributing to Renderide

This document is a long-form guide for people who want to work on the Renderide renderer. It assumes you can build the workspace already (if not, [`README.md`](README.md) covers prerequisites, build commands, and how to run the renderer end-to-end). The point of this guide is to give you the conceptual map you need to make changes that fit the codebase, whether you are fixing a small bug, porting a shader, adding a render pass, or rebuilding a subsystem.

The guide is structured as a difficulty ramp. Part 1 is for everyone. Part 2 is for people who plan to send a pull request. Part 3 is the architecture deep dive aimed at graphics programmers. Part 4 walks through the specific rendering techniques the codebase uses today. Each part has more, narrower subsections than the one before it, so you can stop wherever you have what you need.

You do not have to read straight through. The table of contents below is the map.

## Table of contents

- [Part 1: The basics](#part-1-the-basics)
  - [1.1 What Renderide is](#11-what-renderide-is)
  - [1.2 The mental model](#12-the-mental-model)
  - [1.3 Repository layout](#13-repository-layout)
- [Part 2: Working in the repo](#part-2-working-in-the-repo)
  - [2.1 The Rust workspace](#21-the-rust-workspace)
  - [2.2 The C# projects](#22-the-c-projects)
  - [2.3 Top-level files and folders](#23-top-level-files-and-folders)
  - [2.4 Build profiles and feature flags](#24-build-profiles-and-feature-flags)
  - [2.5 Running tests](#25-running-tests)
  - [2.6 Lints and formatting](#26-lints-and-formatting)
  - [2.7 Code conventions](#27-code-conventions)
  - [2.8 Logging and the diagnostics overlay](#28-logging-and-the-diagnostics-overlay)
  - [2.9 Continuous integration](#29-continuous-integration)
- [Part 3: The renderer architecture](#part-3-the-renderer-architecture)
  - [3.1 The three-process model](#31-the-three-process-model)
  - [3.2 IPC: queues and shared memory](#32-ipc-queues-and-shared-memory)
  - [3.3 The shared types crate](#33-the-shared-types-crate)
  - [3.4 The renderer's internal layers](#34-the-renderers-internal-layers)
  - [3.5 The frame lifecycle](#35-the-frame-lifecycle)
  - [3.6 The scene model](#36-the-scene-model)
  - [3.7 The asset pipeline](#37-the-asset-pipeline)
  - [3.8 The material system](#38-the-material-system)
  - [3.9 The render graph](#39-the-render-graph)
  - [3.10 Pass implementations](#310-pass-implementations)
  - [3.11 Planned views and presentation](#311-planned-views-and-presentation)
  - [3.12 OpenXR integration](#312-openxr-integration)
  - [3.13 The headless test harness](#313-the-headless-test-harness)
- [Part 4: Graphics deep dive](#part-4-graphics-deep-dive)
  - [4.1 Reverse-Z depth and projection](#41-reverse-z-depth-and-projection)
  - [4.2 Multiview stereo rendering](#42-multiview-stereo-rendering)
  - [4.3 The VR mirror invariant](#43-the-vr-mirror-invariant)
  - [4.4 Mesh skinning and blendshape compute](#44-mesh-skinning-and-blendshape-compute)
  - [4.5 Visibility and Hi-Z occlusion](#45-visibility-and-hi-z-occlusion)
  - [4.6 Clustered forward lighting](#46-clustered-forward-lighting)
  - [4.7 Image-based lighting](#47-image-based-lighting)
  - [4.8 Reflection probes and SH2 projection](#48-reflection-probes-and-sh2-projection)
  - [4.9 Skybox families](#49-skybox-families)
  - [4.10 The HDR scene-color chain](#410-the-hdr-scene-color-chain)
  - [4.11 ACES tone mapping](#411-aces-tone-mapping)
  - [4.12 Bloom](#412-bloom)
  - [4.13 GTAO ambient occlusion](#413-gtao-ambient-occlusion)
  - [4.14 The shader source tree](#414-the-shader-source-tree)
  - [4.15 Naga-oil composition](#415-naga-oil-composition)
  - [4.16 Bind group conventions](#416-bind-group-conventions)
  - [4.17 Pipeline state vs shader uniforms](#417-pipeline-state-vs-shader-uniforms)
  - [4.18 The pass directive system](#418-the-pass-directive-system)
  - [4.19 GPU resource pools and budgeting](#419-gpu-resource-pools-and-budgeting)
  - [4.20 Frame resource management](#420-frame-resource-management)
  - [4.21 The driver thread and queue access gate](#421-the-driver-thread-and-queue-access-gate)
  - [4.22 Profiling with Tracy](#422-profiling-with-tracy)
- [License](#license)

---

## Part 1: The basics

### 1.1 What Renderide is

Renderide is a Rust renderer built on [wgpu](https://wgpu.rs/) that replaces the default Unity-based renderer used by [Resonite](https://store.steampowered.com/app/2519830/Resonite/). Resonite's game engine, FrooxEngine, runs in a separate .NET process and tells Renderide what to draw over a shared-memory IPC channel. Renderide owns the window, the GPU device, the OpenXR session, and the present loop. FrooxEngine owns everything else.

The project's goals shape almost every design decision:

- Cross-platform parity. Linux, macOS, and Windows are first-class targets.
- A data-driven render graph. Passes, materials, and resources route through shared systems instead of one-off code paths.
- No per-frame allocations on the hot path. Pools and frame-resource managers absorb the churn.
- OpenXR-first VR. Stereo rendering and head-tracked input are part of the core path, not an add-on.
- Profiling-friendly. Tracy CPU and GPU instrumentation is built in and zero cost when disabled.
- Safe by default. Library code avoids `unwrap`, `expect`, and `panic!`. `unsafe` is restricted to FFI shims and a small number of justified hot paths.

What Renderide is not: it is not a game engine, it does not own world state, it does not load asset bundles, and it does not have a scene editor. The host owns those concerns. The renderer is a comparatively narrow piece of software that takes a stream of commands describing a scene and produces pixels.

### 1.2 The mental model

Three processes cooperate to render a frame.

```mermaid
flowchart LR
    BS[Bootstrapper] -- launches --> Host[Host process<br/>FrooxEngine, .NET]
    BS -- launches --> R[Renderer process<br/>renderide]
    Host == primary IPC ==> R
    Host -. background IPC .-> R
    R == primary IPC ==> Host
    R -. background IPC .-> Host
    R --> Display[(Window or HMD)]
```

The bootstrapper launches the other two and ties their lifetimes together. The host runs simulation and tells the renderer what should appear on screen. The renderer mirrors the host's request into local state, decides what views need to be drawn, runs a compiled render graph, and presents the result. Input flows back the other direction so the host can advance simulation for the next frame.

Two ideas are worth holding onto from the start:

1. The host is authoritative. If the renderer is doing something "smart" about world state, ask whether the host should be telling it that instead.
2. The host and renderer agree, once per frame, that they are now in frame N. This handshake is called lock-step. It is the heartbeat of the system and the natural place to anchor everything per-tick.

### 1.3 Repository layout

The repository is a workspace that mixes a Rust workspace, a .NET solution, a body of WGSL shader source, vendored native libraries, and a small set of runtime assets. Conceptually it splits into five regions.

```mermaid
flowchart TB
    subgraph rust[Rust workspace]
        crates[crates/]
    end
    subgraph dotnet[.NET solution]
        gen[generators/]
        mod[RenderideMod/]
        sln[Generators.sln]
    end
    subgraph shaders[Shader source]
        s[crates/renderide/shaders/]
    end
    subgraph assets[Runtime assets]
        a[crates/renderide/assets/]
    end
    subgraph vendored[Vendored native libs]
        tp[third_party/]
    end
    crates --> shaders
    crates --> assets
    crates --> vendored
    gen -. generates Rust .-> crates
```

- The Rust workspace under `crates/` is where the renderer, the launcher, the IPC transport, the logger, the shared-types crate, and the headless test harness live.
- The .NET solution under `generators/` produces a Rust source file that both the renderer and any host-side tooling depend on. `RenderideMod/` is a separate C# project that injects renderer-aware behavior into the live host. `Generators.sln` is the solution file that ties the .NET projects together.
- The shader source tree under `crates/renderide/shaders/` holds all WGSL. It is large enough and important enough to deserve its own region.
- The runtime asset tree under `crates/renderide/assets/` holds files the renderer ships with at run time, including the window icon and OpenXR controller binding profiles.
- `third_party/` holds vendored native libraries, currently the OpenXR loader.

Everything above the workspace root that isn't one of these regions is configuration: `Cargo.toml`, `.taplo.toml`, `clippy.toml`, `.gitignore`, `.gitattributes`, the `LICENSE`, the `README.md`, and the GitHub Actions workflows under `.github/workflows/`.

---

## Part 2: Working in the repo

### 2.1 The Rust workspace

The workspace lists six member crates in `Cargo.toml`. Each has a single, focused job.

| Crate | Kind | Purpose |
| --- | --- | --- |
| [`bootstrapper`](crates/bootstrapper) | binary plus library | Launches the host and the renderer, runs the bootstrap IPC loop, bridges process services such as clipboard access and the desktop-versus-VR launch dialog, and ties child process lifetimes together. |
| [`interprocess`](crates/interprocess) | library | Cloudtoid-compatible shared-memory ring queues and semaphores. The transport every IPC channel rides on. Cross-platform mmap on Unix, named file mappings on Windows. |
| [`logger`](crates/logger) | library | File-first logger shared by every process. Writes to `logs/<component>/<timestamp>.log` by default, with a `RENDERIDE_LOGS_ROOT` override. |
| [`renderide-shared`](crates/renderide-shared) | library | The host-renderer wire-format crate. Holds the generated shared types, the binary packing helpers, the dual-queue IPC wrappers (one for the host side, one for the renderer side), and the shared-memory accessor and writer. |
| [`renderide`](crates/renderide) | two binaries plus library | The renderer itself. Owns winit, wgpu, OpenXR, the scene model, the render graph, materials, assets, profiling, and diagnostics. Builds the `renderide` binary (the renderer process) and the `roundtrip` binary (a small CLI used by the .NET generator's roundtrip tests to validate that Rust packing and C# packing agree on the bytes). |
| [`renderide-test`](crates/renderide-test) | binary plus library | Headless integration harness. Acts as a minimal host, drives the real IPC protocol, spawns the renderer, captures its output, and validates the result against golden images and golden state machines. |

`bootstrapper`, `interprocess`, and `logger` know nothing about graphics. `renderide-shared` knows nothing about graphics either. The graphics knowledge is concentrated in `renderide`. `renderide-test` is the only crate that depends on `renderide` for graphics-aware testing, and it does so through the same IPC contract a real host would use.

```mermaid
flowchart TB
    bootstrapper --> interprocess
    bootstrapper --> logger
    bootstrapper --> renderide_shared[renderide-shared]
    renderide --> interprocess
    renderide --> logger
    renderide --> renderide_shared
    renderide_test[renderide-test] --> interprocess
    renderide_test --> logger
    renderide_test --> renderide_shared
    renderide_test -. spawns and inspects .-> renderide
    renderide_shared --> interprocess
    renderide_shared --> logger
```

### 2.2 The C# projects

Two .NET projects sit next to the Rust workspace, joined by `Generators.sln` at the repo root.

- [`generators/SharedTypeGenerator`](generators/SharedTypeGenerator) is a code generator. Its job is to read the canonical C# definitions of the host-renderer wire types and emit `crates/renderide-shared/src/shared.rs`, which both sides of the IPC then agree on. Internally it is structured like a small compiler: an `Analysis` stage that parses the inputs, an `IR` stage that holds the typed intermediate representation, an `Emission` stage that writes Rust, and an `Options` stage that handles CLI configuration. The entry point is `Program.cs`.
- [`generators/SharedTypeGenerator.Tests`](generators/SharedTypeGenerator.Tests) is the test project for the generator. It splits into `Unit/` tests for the generator's internal stages and `Roundtrip/` tests that pack a value with the generated C# packing code, then unpack it with the Rust `roundtrip` binary, then re-pack and re-check, asserting that both sides agree on every byte for every shape of every type the generator emits.
- [`RenderideMod`](RenderideMod) is a separate Resonite mod that hooks into the live host using HarmonyLib and ResoniteModLoader. It contains a `Patches/` folder for Harmony patches, an `Ipc/` folder for the host-side IPC plumbing the patches use, and a `Properties/` folder for assembly metadata. It is not a renderer dependency; it is the host-side counterpart that knows about Renderide and prepares the host to talk to it.

The .NET solution is built and tested by its own CI workflow (see [2.9](#29-continuous-integration)). You do not need a .NET SDK installed to build or run the renderer itself, only when you change the generator or the mod.

### 2.3 Top-level files and folders

| Path | What it is |
| --- | --- |
| `Cargo.toml` | Rust workspace manifest. Lists the member crates, defines the `dev-fast` and `release` profiles, and centralizes the project's clippy and rustc lint configuration. |
| `Cargo.lock` | Resolved dependency lockfile. Checked in so all builds (and CI runners) agree. |
| `Generators.sln` | .NET solution file for the C# projects under `generators/`. |
| `clippy.toml` | Per-crate clippy tuning. Notably allows `unwrap`, `expect`, `panic`, `dbg`, `print`, and indexing in tests, and raises the by-value pass size limit so a deliberately Copy host camera struct can be threaded through hot paths by value. |
| `.taplo.toml` | TOML formatter configuration, scoped to manifests so build output under `target/` is not formatted. |
| `.gitignore`, `.gitattributes` | Standard git configuration. |
| `LICENSE` | Project license. |
| `README.md` | User-facing build, run, feature, and profiling guide. |
| `crates/` | The Rust workspace member crates. |
| `generators/` | The C# code generator and its tests. |
| `RenderideMod/` | The C# Resonite mod. |
| `third_party/` | Vendored native libraries. Currently holds the OpenXR loader, which the renderer's build script copies onto Windows targets so the loader is available next to the binary. |
| `.github/workflows/` | Continuous integration pipelines. |

Inside the renderer crate at `crates/renderide/`, three top-level companions live next to `src/`:

- `assets/` holds runtime assets that ship with the renderer: the window icon and the OpenXR controller binding profiles for every supported headset and controller family. The build script copies these into the artifact directory so the binary can find them at run time.
- `shaders/` holds every WGSL source file the renderer compiles. It is divided into `materials/` (one shader per host material program), `modules/` (shared logic composed via naga-oil), and `passes/` with subdirectories for `backend/`, `compute/`, `post/`, and `present/` shaders.
- `build.rs` and `build_support/` together compose the WGSL source tree at build time, generate an embedded shader registry that the renderer links in, copy XR assets into the artifact directory, and copy the vendored OpenXR loader on Windows. The `build_support/shader/` subdirectory is where the shader composition logic lives, broken into `source`, `modules`, `compose`, `directives`, `validation`, `parallel`, `emit`, `model`, and `error`.

### 2.4 Build profiles and feature flags

The workspace defines two non-default profiles in `Cargo.toml`:

- `dev-fast` inherits `dev` (so debug symbols, unwind, and assertions are on) but raises the optimization level to 2. Use this for everyday dev cycles when stock `cargo build` is too slow but you still want assertions.
- `release` raises optimization to 3, enables LTO, disables debug assertions, sets `panic = "abort"`, and keeps debug symbols on for crash report symbolication.

The `renderide` crate declares two opt-in Cargo features. Both are off by default to keep stock builds and CI lean.

- `tracy` enables Tracy profiling. CPU spans come from the `profiling` crate. GPU timestamp queries come from `wgpu-profiler`. The Tracy client links statically and runs in on-demand mode, so a profiled build idles near zero cost when no GUI is connected.
- `video-textures` enables GStreamer-backed video texture decoding. Without this feature, the renderer still accepts video texture IPC commands and allocates GPU placeholders, but no decoding runs and the placeholder stays black.

See `README.md` for the exact build commands and platform-specific dependencies for each feature.

### 2.5 Running tests

Tests live in three places.

- Unit tests live in `mod tests` blocks at the bottom of the file they test.
- Per-crate integration tests live in `crates/<crate>/tests/`. Each file in a `tests/` directory is its own integration test binary linked against the crate's library API.
- Cross-process integration tests live in `crates/renderide-test/`, which builds a full host emulator and drives the real IPC contract end to end.

The renderer crate carries a curated set of integration tests under `crates/renderide/tests/` that cover non-GPU concerns: shader composition, configuration loading, the IPC decoupling state machine, native stdio forwarding on Unix, error mirror routing, the render-graph compiler, shader module audits, instancing batch shapes, and packing roundtrips. None of them require a GPU adapter.

The supporting crates each have their own integration tests:

- `crates/bootstrapper/tests/` covers the public CLI surface and the IPC queue tempdir lifecycle.
- `crates/interprocess/tests/` exercises the queue end-to-end across publisher and subscriber.
- `crates/logger/tests/` is a fairly large suite that covers initialization, append behavior, per-component layouts, malicious-timestamp sanitization, mirror writers, and concurrency.
- `crates/renderide-shared/tests/` covers the wire packing for both polymorphic types and primitives, plus the singleton claim that prevents two renderers from racing on the same IPC name.
- `crates/renderide-test/tests/` covers the harness itself: argument parsing, golden-image diff writing, golden round-trips, log folder routing, the PNG stability state machine, the spawn argument table, and an end-to-end sphere pipeline.

GPU-driven testing is deliberately out of scope. Integration tests for the renderer are non-GPU by policy: GPU paths are validated at run time and through the headless harness in `renderide-test`, not in CI integration tests.

### 2.6 Lints and formatting

The workspace puts a heavy lint configuration at the workspace level so every crate inherits it. The intent is that code which lands in this repo holds itself to a consistent standard regardless of which crate it lives in.

The Rust lint set includes (among many others):

- `missing_docs` warns on undocumented public items.
- `unsafe_op_in_unsafe_fn` and `missing_unsafe_on_extern` enforce explicit unsafe scoping.
- `keyword_idents_2024` and several Rust 2024 hygiene lints keep the codebase aligned with the current edition.

The clippy set includes:

- `unwrap_used`, `expect_used`, `panic`, `todo`, `unimplemented`, `print_stdout`, `print_stderr`, `dbg_macro`, `mem_forget`. The expectation is that runtime and library paths do not panic, do not leak through `mem::forget`, and route output through the logger.
- `mod_module_files` enforces the `module_name.rs` plus `module_name/` layout instead of the older `mod.rs` style.
- `undocumented_unsafe_blocks` requires a `// SAFETY:` comment on every `unsafe` block.
- A long list of style and correctness lints (`uninlined_format_args`, `needless_pass_by_ref_mut`, `redundant_clone`, `large_stack_arrays`, `await_holding_lock`, `significant_drop_in_scrutinee`, `manual_clamp`, and so on) that catch the kind of paper cuts that compound across a large codebase.

`clippy.toml` allows the panic-and-unwrap family of lints inside test code so tests can assert directly without ceremony.

Format with `cargo fmt --all` for Rust, `taplo fmt` for `Cargo.toml` and other manifests, and `dotnet format` for the C# projects.

### 2.7 Code conventions

A few conventions are worth spelling out because the lint configuration assumes them.

- The project targets the Rust 2024 edition.
- Modules use `module_name.rs` next to a `module_name/` directory, never `mod.rs`.
- Errors use explicit `thiserror` enums. There is no `anyhow` dependency.
- Library and runtime code does not use `unwrap`, `expect`, or `panic!`. Tests, build scripts, and one-shot startup paths can use them when failure is unrecoverable by design.
- `unsafe` is restricted to FFI shims and explicitly justified hot paths. Every `unsafe` block carries a `// SAFETY:` comment that names the invariant it depends on.
- Public items carry `///` doc comments. Inline `//` comments are reserved for the non-obvious why.
- Output goes through the `logger` crate. `println!` and `eprintln!` are clippy-warned everywhere.
- Collections default to `hashbrown::HashMap`. Locks default to `parking_lot::Mutex`.
- Dependency versions are pinned to major and minor only (for example `thiserror = "2.1"`), and the latest stable release is preferred.

For C# code in the generator and the mod: throw specific exception types instead of catching `Exception` at internal boundaries, and keep public types documented.

### 2.8 Logging and the diagnostics overlay

The renderer has two complementary visibility systems.

The first is the file-first logger from the `logger` crate. Every process initializes it on startup and writes to its own subdirectory under `logs/`. The supported component names are `bootstrapper`, `host` (captured host stdout and stderr), `renderer`, `renderer-test`, and `SharedTypeGenerator`. Log files are named with a UTC timestamp so they sort and can be compared across runs. The location can be redirected with `RENDERIDE_LOGS_ROOT`. The recommended levels are `error` for unrecoverable failures, `warn` for recoverable anomalies, `info` for lifecycle events, `debug` for per-frame and per-asset control flow (the default), and `trace` for tight loops and high-frequency paths.

The second is the in-renderer diagnostics overlay built with Dear ImGui. It surfaces per-frame timings, per-view information, scene and asset inspection, host process metrics, encoder errors, and a watchdog. The overlay reads from snapshots captured at layer boundaries rather than borrowing live state from the renderer, which keeps the overlay safe to run alongside the per-frame work it is observing.

### 2.9 Continuous integration

Two GitHub Actions workflows live under `.github/workflows/`.

- `rust-ci.yml` builds and tests the Rust workspace on Ubuntu, Windows, and macOS in parallel. Linux is the only matrix entry that uses `--all-features`, because GStreamer dev packages are reliably installable from the system package manager only on Linux. Windows and macOS still build the `tracy` feature so it stays warning-free on those platforms. The Linux job also installs Vulkan tooling so the `materials::registry` smoke test can find an adapter.
- `dotnet-ci.yml` builds the .NET solution on the same three OSes, runs the generator's unit and roundtrip tests, and verifies formatting on Linux only (Windows checkouts can disagree with the in-repo encoding because of `core.autocrlf`, so format checks would fail spuriously there).

Both workflows trigger on push to `main` or `master`, on pull requests, and on manual dispatch.

---

## Part 3: The renderer architecture

### 3.1 The three-process model

Renderide is the renderer process in a three-process system. The bootstrapper owns lifetimes; the host owns the world; the renderer owns the GPU and the window.

```mermaid
flowchart TB
    BS[Bootstrapper] -- spawn --> Host[Host process]
    BS -- spawn --> R[Renderer process]
    BS <-. bootstrap IPC<br/>heartbeats, clipboard,<br/>start signals .-> Host
    BS <-. bootstrap IPC .-> R
    Host == primary IPC ==> R
    Host -. background IPC .-> R
    R == primary IPC ==> Host
    R -. background IPC .-> Host
```

A few invariants are worth knowing up front:

- The bootstrapper is the only process that knows about both ends. The host and the renderer never spawn each other directly.
- The bootstrapper bridges OS-level services that should not live inside a renderer: the desktop-versus-VR launch dialog, the clipboard, Wine detection on Linux, panic hook installation. It also detects when the renderer has died first (for example, because the user closed the window) and tears the host down with it.
- The renderer can also run without the host, either standalone (no IPC) or under the headless test harness. All three modes share the same architectural paths.

### 3.2 IPC: queues and shared memory

Two queues plus one shared-memory region make up the contract between host and renderer.

```mermaid
flowchart LR
    Host -- "FrameStart, scene submits,<br/>materials, lights, input ack" --> PQ[(Primary queue<br/>lock-step)]
    PQ --> R[Renderer]
    R -- "FrameStartData, input,<br/>begin-frame events" --> PQ2[(Primary queue<br/>lock-step)]
    PQ2 --> Host

    Host -- "asset uploads,<br/>completion ack,<br/>non-frame-critical traffic" --> BQ[(Background queue<br/>asynchronous)]
    BQ --> R
    R -- "completion ack,<br/>readback results" --> BQ2[(Background queue<br/>asynchronous)]
    BQ2 --> Host

    Host -. writes payloads to .-> SHM[(Shared memory:<br/>vertex / index / pixel /<br/>transform / material batches)]
    R -. reads payloads from .-> SHM
    PQ -. references shm IDs .-> SHM
    BQ -. references shm IDs .-> SHM
```

The primary queue carries per-frame control flow: frame begin and end, scene submits, lights, materials, input acknowledgements. Both sides drain it as part of the lock-step exchange that gates frame cadence. The background queue carries asynchronous traffic: large asset uploads, completion acknowledgements, readback results. The renderer integrates background traffic on its own clock, with a budget per tick so the frame stays responsive while the host catches up.

Queue messages are small and reference shared-memory regions by ID. Shared memory is where the bulk lives: vertex and index data, texture pixels, transform batches, material property batches. The host writes; the renderer reads. Lifetime is governed by the IPC protocol so the renderer never reads a region the host has freed.

The transport layer is the `interprocess` crate, which implements Cloudtoid-compatible shared-memory ring queues. On Unix it backs queues with file mappings under a configurable directory (defaulting to `/dev/shm/.cloudtoid` on Linux). On Windows it uses named file mappings and named semaphores. Queue parameters are passed by CLI; both sides agree on the same names by sharing the same configuration.

### 3.3 The shared types crate

Wire compatibility lives in `renderide-shared`. The crate is structured around five concerns:

- `shared` is the generated module containing every type that crosses the host-renderer boundary. It is generated by the C# `SharedTypeGenerator` and should never be edited by hand. To change the wire format, change the generator's input or the generator itself, then regenerate.
- `packing` is the binary contract: a `MemoryPacker` and `MemoryUnpacker` plus the supporting traits that turn typed values into bytes and back. There is also a small `extras` submodule with hand-rolled packing for types whose layout the auto-classifier cannot derive.
- `buffer` describes shared-memory regions in a way both sides understand.
- `wire_writer` is the host-side helper for emitting bytes onto a shared-memory region.
- `ipc` carries the dual-queue wrapper used by the renderer (`DualQueueIpc`), the matching host-side wrapper (`HostDualQueueIpc`), the read-only shared-memory accessor used by the renderer, and the shared-memory writer used by the host.

Because anything that crosses the process boundary lives here, neither the renderer nor the host has to depend on the other's heavy dependencies (wgpu, naga, OpenXR, winit, imgui on the renderer side; the .NET runtime on the host side).

```mermaid
flowchart LR
    csharp[C# definitions] --> gen[SharedTypeGenerator]
    gen --> shared[crates/renderide-shared/src/shared.rs]
    shared --> rshared[renderide-shared]
    rshared --> renderer[renderide]
    rshared --> testharness[renderide-test]
    rshared --> rtools[Future host-side Rust tools]
```

### 3.4 The renderer's internal layers

The `renderide` crate splits into five clearly named layers. The split is the contract that lets the renderer run with or without a host, with or without a window, and under the test harness.

```mermaid
flowchart TB
    App["App<br/>process boundary, winit,<br/>headless, frame clock, exit codes"]
    Frontend["Frontend<br/>IPC queues, lock-step,<br/>input conversion"]
    Scene["Scene<br/>host world mirror,<br/>transforms, renderables, lights"]
    Backend["Backend<br/>wgpu device, pools,<br/>materials, render graph"]
    Runtime["Runtime<br/>per-tick orchestration facade"]

    App --> Runtime
    Runtime --> Frontend
    Runtime --> Scene
    Runtime --> Backend
    Frontend -. command stream .-> Scene
    Scene -. snapshots .-> Backend
    Frontend -. shm references .-> Backend
```

Each layer corresponds to a top-level module in `crates/renderide/src/`:

- The `app` module owns the process boundary. It contains the `bootstrap` for IPC wiring, the `driver` abstraction (winit and headless), the `frame_clock`, the window icon, and the run-exit code that decides what the process returns to its parent.
- The `frontend` module owns transport. It contains the `transport` itself, the `init_state` and handshake, the `lockstep_state`, the `decoupling` state machine that lets rendering stay responsive while the host catches up, the `dispatch` of incoming `RendererCommand`s, the begin-frame logic, the `input` conversion from winit and OpenXR events into shared `InputState` structures, and the `output_policy` that picks where the next frame should land.
- The `scene` module owns the host world mirror. It contains `coordinator`, `world`, `render_space`, `dense_update`, `transforms_apply`, `mesh_apply`, `mesh_renderable`, `lights`, `render_overrides`, `reflection_probe`, `pose`, and `math`. There is no wgpu in here.
- The `backend` module owns GPU state. It contains the `facade` that the runtime calls into, the `frame_gpu` that builds frame-global GPU state, the `frame_resource_manager`, the `gpu_jobs` for nonblocking GPU work such as SH2 projection, the `light_gpu` packer, `cluster_gpu`, `per_draw_resources`, the `view_resource_registry`, the `per_view_resource_map`, the `history_registry`, and the `material_property_reader`. Beneath the backend module live the closely related top-level modules `gpu` (device-facing primitives), `gpu_pools` (resident asset pools), `materials` (material registry and pipelines), `assets` (asset integration queues), `mesh_deform` (skinning and blendshape compute), `world_mesh` (visibility planning), `occlusion`, `skybox`, `reflection_probes`, `camera`, `passes`, and `render_graph`.
- The `runtime` module is a thin facade that wires the other layers together one tick at a time. It does not own IPC queues, scene tables, or GPU resources. Each per-tick concern lives in its own submodule: `tick`, `ipc_entry`, `asset_integration`, `gpu_services`, `view_planning`, `frame_view_plan`, `frame_extract`, `frame_render`, `debug_hud_frame`, `lockstep`, and `xr_glue`.

The arrows in the diagram are the allowed dependency directions. If you find yourself wanting an upward dependency (scene reaching into IPC, backend mutating frontend state) it is almost always a sign the data should move differently, usually as a snapshot taken at a layer boundary.

### 3.5 The frame lifecycle

Every tick walks through the same conceptual phases in the same order. The order is not arbitrary; it is what makes the per-tick contract between IPC, scene, GPU, and presentation work.

```mermaid
flowchart TB
    P1[1. Wall-clock prologue<br/>reset per-tick flags, start frame clock]
    P2[2. Poll IPC<br/>drain async resolutions,<br/>dispatch RendererCommands]
    P3[3. Integrate assets<br/>budgeted mesh, texture,<br/>material, shader work]
    P4[4. Maintain GPU services<br/>occlusion readbacks,<br/>async jobs, transient eviction]
    P5[5. Begin XR when active<br/>frame wait, locate views]
    P6[6. Send begin-frame<br/>FrameStartData with input + perf]
    P7[7. Plan views<br/>HMD eyes, secondaries, desktop<br/>into one logical list]
    P8[8. Collect draws<br/>cull, resolve materials,<br/>batch, sort per view]
    P9[9. Execute graph<br/>frame-global once,<br/>then per-view, single submit]
    P10[10. Present and diagnose<br/>present or mirror, HUD,<br/>timing, watchdog]

    P1 --> P2 --> P3 --> P4 --> P5 --> P6 --> P7 --> P8 --> P9 --> P10
```

A few things about why this order matters:

- IPC poll and asset integration happen before any GPU work so the rest of the tick sees the latest state.
- GPU service maintenance runs after asset integration so transient eviction and async jobs see the new uploads.
- XR pose acquisition runs before begin-frame so headset pose, input, and views all describe the same frame.
- Scene mutation is fully settled before draw collection. Collection reads from snapshots, not from live transport state.
- Graph execution consumes prepared per-view plans rather than reaching back into transport for live data during the per-view loop.

If you add new per-tick work, place it in the correct phase rather than carving out a new one.

### 3.6 The scene model

The scene layer is a typed projection of the IPC command stream. It mirrors the host's logical world state and exposes it to the backend through stable APIs.

What lives there:

- Render spaces, one per host-visible world partition. Each owns its own transforms, renderables, lights, skybox, and reflection probe state.
- Dense transform arenas keyed by host index, including the host's removal ordering. The dense layout is part of the wire contract: random reorderings would break alignment with what the host is sending.
- Mesh and skinned renderables, each carrying the mesh, material, transform, and per-instance overrides needed to draw it.
- Lights merged from frame submits and from light-buffer submissions.
- Render overrides and material property blocks for per-camera filtering and per-instance tweaks.
- Skybox state, with the actual filtered cube cache living in the backend.
- Reflection probe task rows tracking host requests for SH2 projection.
- Cached world matrices.

Two strict properties keep this layer honest:

1. There is no wgpu. The scene module compiles without the GPU dependency. If a GPU resource handle wants to be in scene state, that is the cue to instead pass the handle to the backend at the right boundary.
2. World matrices are scene-owned cached data. The backend asks for resolved matrices; it does not rebuild scene hierarchy logic inside GPU code.

### 3.7 The asset pipeline

Asset integration is cooperative, queue-backed, and budgeted. The host can send formats before data, data before GPU attach, multiple uploads for the same ID, and cancellations, and the renderer must keep up without dropping frames.

```mermaid
flowchart LR
    Host["Host IPC commands<br/>(mesh / texture / cubemap /<br/>material / shader)"]
    AT[Asset transfer queue]
    Mesh[Mesh integration]
    Tex[Texture integration]
    Mat[Material integration]
    Sh[Shader resolution]
    Pool[Resident pools]
    Reg[Material registry]
    Cache[Pipeline cache]
    Frame[Frame draw collection]

    Host --> AT
    AT --> Mesh --> Pool
    AT --> Tex --> Pool
    AT --> Mat --> Reg
    AT --> Sh --> Reg
    Reg --> Cache
    Pool --> Frame
    Cache --> Frame
```

The relevant code lives in `crates/renderide/src/assets/`, structured around an `asset_transfer_queue` plus per-type integration subdirectories (`mesh`, `texture`, `shader`, `video`, plus a small `util`). The integration phase of the runtime tick (`asset_integration.rs`) drains these tasks under a time budget that shrinks when the renderer is decoupled from the host. Uploaded resources land in resident pools managed by `gpu_pools/` (see [4.19](#419-gpu-resource-pools-and-budgeting)) and become visible to draw collection on the same tick.

Each asset type has its own integration loop because each has different ordering invariants (formats before data, palettes before frames, mip levels independently uploadable, video decoding running on a worker), but they all share one property: the runtime tick decides when integration runs and how long it gets, not the host.

### 3.8 The material system

The material system is the most concept-dense part of the renderer because it sits at the crossroads of asset integration, shader compilation, pipeline state, and per-frame draw resolution.

The host writes material properties as a flat key-value store. Each property lands in exactly one of four places:

| Property kind | Examples | Resolved by | Lives in |
| --- | --- | --- | --- |
| Pipeline state | `_SrcBlend`, `_DstBlend`, `_ZWrite`, `_ZTest`, `_Cull`, `_Stencil*`, `_ColorMask`, `_OffsetFactor`, `_OffsetUnits` | Material blend mode and render state | Pipeline cache key (used to build `wgpu::RenderPipeline`) |
| Shader uniform value | `_Color`, `_Tint`, `_Cutoff`, `_Glossiness`, `*_ST` | Property store, packed by reflection | Material struct uniform at `@group(1) @binding(0)` |
| Shader uniform keyword | `_NORMALMAP`, `_ALPHATEST_ON`, `_ALPHABLEND_ON` | Property store or inferred from material | Material struct uniform at `@group(1) @binding(0)` |
| Texture | `_MainTex`, `_NormalMap`, ... | Texture pools, bound by reflection | Material bindings at `@group(1) @binding(N)` |

The split is enforced. Pipeline-state property names must never appear in a shader's group 1 material uniform. They are dead weight there: shaders never read them, and the host writes them. The build script rejects any material WGSL that violates this contract. Two materials sharing a shader but differing in pipeline state correctly resolve to distinct cached pipelines because the cache key includes the resolved blend mode and render state.

The relevant code lives in `crates/renderide/src/materials/`. It is large enough to be worth a quick map: the `registry` is the central table; `host_data` holds the property store; `cache` holds the pipeline cache; `router` maps host shader asset IDs to pipeline families or embedded WGSL stems; `wgsl_reflect` does naga-based reflection; `pipeline_*.rs` files build pipelines and resolve their properties; `material_passes.rs` and `material_pass_tables.rs` carry the per-shader pass declarations parsed at build time; `shader_permutation.rs` handles keyword permutations; `embedded.rs` holds the static table of built-in shaders compiled into the binary; `render_state.rs` holds blend, depth, stencil, cull, and color mask state; `render_queue.rs` carries the queue ordering for opaque, alpha test, transparent, and overlay draws.

### 3.9 The render graph

The render graph is the renderer's per-frame compiler. Passes declare typed access to resources up front (imported frame targets, transient resources, persistent history resources), and the graph turns those declarations into a scheduled, recordable sequence of GPU work.

```mermaid
flowchart TB
    Setup[Setup<br/>passes declare resources]
    Compile[Compile<br/>schedule, transient aliasing,<br/>access ordering]
    Cache[Cache by frame shape]
    Prewarm[Pre-warm transients,<br/>per-view state,<br/>pipeline cache]
    Global[Frame-global encoder<br/>FrameGlobal passes]
    Per[Per-view encoders<br/>PerView passes]
    Submit[Single submit]
    Present[Present or mirror]

    Setup --> Compile
    Compile --> Cache
    Cache --> Prewarm
    Prewarm --> Global
    Global --> Per
    Per --> Submit
    Submit --> Present
```

The relevant code lives in `crates/renderide/src/render_graph/`. The pieces:

- `builder` is the setup-time API. Passes register their resource access and any per-pass parameters.
- `compiled` is the immutable result: a flattened pass list in dependency order, transient usage unions, lifetime-based alias slots, and the entry points (`execute` and `execute_multi_view`) that record encoders and submit them.
- `cache` memoizes a compiled graph by a `GraphCacheKey` that captures the durable frame-shape inputs: surface extent, MSAA, multiview, surface format, scene HDR format. The backend rebuilds only when one of those inputs changes.
- `pass` defines the typed pass traits that concrete passes implement: `RasterPass`, `ComputePass`, `CopyPass`, and `CallbackPass`.
- `pool` manages the transient resource pool.
- `resources` and `frame_params` carry the typed resource handles and per-frame parameter packs.
- `record_parallel` is the parallel record path that makes the per-view loop scale across cores.
- `frame_upload_batch` coalesces deferred buffer writes into a single drain before submit.
- `gpu_cache`, `blackboard`, `context`, `ids`, `error`, `schedule`, `swapchain_scope`, `secondary_camera`, `main_graph`, `post_processing`, and `test_fixtures` provide the supporting machinery.

Two structural ideas matter most:

- Pass phase. The graph distinguishes `FrameGlobal` passes from `PerView` passes. Frame-global work runs once per tick (mesh deformation, light prep, Hi-Z that depends on previous-frame depth). Per-view work runs once per planned view (world rendering, view-dependent post-processing, scene-color compose).
- Encoder topology. The executor records frame-global passes in a dedicated encoder, then one encoder per planned view for per-view passes. Deferred buffer writes are drained before the single submit. The per-view loop pre-warms transients, per-view per-draw resources, and the pipeline cache once across all views before recording, so the recording loop never pays lazy allocation costs (a structural prerequisite for the parallel record path).

When you find yourself tempted to record a pass on a borrowed encoder outside the graph, register a graph node instead. The graph is what keeps frame resource lifetimes correct.

### 3.10 Pass implementations

Concrete passes live in `crates/renderide/src/passes/`. Each implements one of the four pass traits and registers against the graph builder.

```mermaid
flowchart TB
    swc[swapchain_clear]
    md[mesh_deform<br/>skinning + blendshapes]
    hiz[hi_z_build<br/>from previous-frame depth]
    cl[clustered_light<br/>build cluster light list]
    wmf[world_mesh_forward<br/>opaque + alpha + transparent +<br/>depth resolve + color resolve]
    pp[post_processing<br/>GTAO, bloom, tonemap]
    sc[scene_color_compose]

    swc --> hiz
    hiz --> cl
    md --> wmf
    cl --> wmf
    wmf --> pp
    pp --> sc

    classDef global fill:#fff0e8,stroke:#333
    classDef perview fill:#e8f0ff,stroke:#333
    class swc,hiz,cl,md global
    class wmf,pp,sc perview
```

The currently implemented passes:

- `swapchain_clear` clears the swapchain target.
- `hi_z_build` builds the hierarchical Z pyramid from the previous frame's depth, used for occlusion testing this frame (see [4.5](#45-visibility-and-hi-z-occlusion)).
- `clustered_light` runs the compute pass that bins lights into clusters for this frame's camera (see [4.6](#46-clustered-forward-lighting)).
- `mesh_deform` runs mesh skinning and blendshape compute (see [4.4](#44-mesh-skinning-and-blendshape-compute)).
- `world_mesh_forward` is the workhorse forward pass. It splits into a prepare step, an opaque pass, an intersect pass, a transparent pass, depth and color resolve passes, and depth and color snapshot passes that feed downstream effects.
- `post_processing` is a family of effects, each of which is its own graph node: GTAO, bloom, ACES tone mapping. See sections [4.11](#411-aces-tone-mapping) through [4.13](#413-gtao-ambient-occlusion).
- `scene_color_compose` copies the HDR scene color into the swapchain, the XR target, or an offscreen output, depending on the planned view.

### 3.11 Planned views and presentation

Each tick produces one logical list of views to render: the desktop target when no HMD is active, the HMD eye targets when one is, and any number of secondary cameras the host has requested (typically render textures driven by world cameras).

```mermaid
flowchart TB
    Tick[Tick] --> Plan[View planning]
    Plan --> HMD{HMD active?}
    HMD -- yes --> Eyes[HMD stereo views<br/>multiview when supported]
    HMD -- no --> Desktop[Desktop view]
    Plan --> Sec[Secondary render-texture cameras]
    Eyes --> Mirror[Mirror blit one eye<br/>to desktop window]
    Eyes --> Submit[GPU submit]
    Desktop --> Submit
    Sec --> Submit
    Submit --> Present[Present]
```

The relevant code lives in `crates/renderide/src/runtime/view_planning.rs` and `frame_view_plan.rs`, supported by `crates/renderide/src/camera/` (which holds the `state`, `frame`, `view`, `projection`, `projection_plan`, `secondary`, `stereo`, and `geometry` modules) and `crates/renderide/src/gpu/vr_mirror/` for the mirror blit.

Two invariants are easy to violate and worth restating:

- The VR mirror is a blit, not a re-render. When the HMD path renders successfully, the desktop window shows a mirror copy of one HMD eye. Adding a separate desktop scene that runs alongside the HMD path is a bug, not a feature.
- Secondary cameras and render textures are first-class planned views. They go in the same planned-view list as desktop and HMD, render through the same graph, and (in a VR session) render alongside the HMD workflow.

### 3.12 OpenXR integration

OpenXR support lives in `crates/renderide/src/xr/`. It is structured around several concerns:

- `bootstrap` brings up the OpenXR session.
- `app_integration` integrates the XR session with the app driver and the per-tick frame lifecycle.
- `session` carries the typed session state.
- `swapchain` owns the XR swapchain images and view geometry.
- `input` converts OpenXR action state into the same `InputState` shape that winit input is converted into.
- `host_camera_sync` keeps the host's notion of camera in sync with the headset pose.
- `output_device` adapts the host's `HeadOutputDevice` concept to the OpenXR present path.
- `openxr_loader_paths` and the vendored loader under `third_party/openxr_loader/` ensure the loader can be found at run time, especially on Windows where the build script copies the vendored DLL next to the binary.
- `debug_utils` wires up OpenXR debug callbacks for diagnostics.

Controller bindings ship as TOML files under `crates/renderide/assets/xr/bindings/`. Each TOML maps an OpenXR interaction profile (Oculus Touch, Valve Index, HTC Vive, Vive Cosmos, Vive Focus 3, Pico Neo 3, Pico 4, HP Reverb, Microsoft Motion, Samsung Odyssey, Meta Touch Plus, Quest Touch Pro, KHR Simple, KHR Generic) to the action set the renderer requests. `actions.toml` defines the action set itself.

If OpenXR initialization or a per-frame acquire fails, the renderer should degrade through the desktop and secondary camera paths and keep the failure visible in diagnostics. The system does not crash when the headset disappears.

### 3.13 The headless test harness

`renderide-test` is a minimal host that drives the real IPC contract end to end without a real FrooxEngine. It is structured around:

- `cli` for command-line parsing.
- `host` for the host-side IPC plumbing.
- `scene` for the scenes the harness can stage.
- `golden` for golden-image comparison and PNG diff writing.
- `logging` for the harness's own log routing.

The harness is what stands between "the renderer compiles" and "the renderer behaves." It exercises the same queues, the same shared-memory protocol, and the same renderer entry points a real launch would, then captures the rendered frames for comparison against golden references. New end-to-end integration tests for the renderer as a whole belong in this harness, not in `crates/renderide/tests/`.

---

## Part 4: Graphics deep dive

This part assumes you have read at least Part 3 or are comfortable diving in cold. Each subsection covers one technique or system at a conceptual level. The point is to give you the vocabulary and the mental model, not to walk you through specific code.

### 4.1 Reverse-Z depth and projection

Renderide uses a reverse-Z depth convention: the near plane maps to depth 1.0 and the far plane to depth 0.0. The projection math, depth comparison direction, and clear value all follow.

The motivation is precision. Floating-point depth values are dense near 0 and sparse near 1. With a conventional forward-Z projection, the geometric distortion of a perspective transform pushes most of the precision toward the near plane where you do not need it. With reverse-Z, the dense region of the float distribution lands on the far plane where the perspective transform is simultaneously squeezing depth values together. The two effects roughly cancel and you get usable precision across the whole frustum, which matters a lot for the kind of mixed near-and-far scenes Resonite worlds tend to contain.

The relevant math lives in `crates/renderide/src/camera/projection.rs` and is consumed everywhere a depth comparison or a projection matrix is needed.

### 4.2 Multiview stereo rendering

When the HMD path is active and the GPU supports it, the renderer issues a single set of draws and lets the GPU broadcast them to two eye-specific render targets, indexing per-eye state by `view_index`. This is wgpu's multiview feature, and it cuts the CPU side of stereo rendering roughly in half compared to issuing two passes.

Multiview is a property of the planned view, not a separate rendering path. Shaders that participate in multiview use a single source file with naga-oil conditional compilation to select the multiview-capable code, rather than a parallel non-multiview source file. The render graph carries multiview as part of its cache key so a session that toggles between multiview and non-multiview rebuilds the graph cleanly.

Multiview only kicks in when both the adapter supports it and the planned view set is the HMD pair. Secondary cameras and the desktop fall back to single-view rendering through the same passes.

### 4.3 The VR mirror invariant

When the HMD path renders successfully, the desktop window shows a mirror copy of one HMD eye. It does not run a second world render to populate the desktop. The mirror is a small set of blit shaders under `shaders/passes/present/` (`vr_mirror_eye_to_staging.wgsl` and `vr_mirror_surface.wgsl`) plus a small backend module under `crates/renderide/src/gpu/vr_mirror/`.

The reason this is an invariant rather than an implementation detail: a second world render would double GPU cost, would draw the world twice with possibly different camera state, and would cause secondary cameras to be evaluated twice per tick. None of those are acceptable. If you are tempted to draw the desktop "again" in a VR session, you are about to break the invariant.

### 4.4 Mesh skinning and blendshape compute

Skinning and blendshape work runs as a frame-global compute pass before per-view rendering. Two compute shaders under `shaders/passes/compute/` drive it: `mesh_skinning.wgsl` and `mesh_blendshape.wgsl`.

The CPU-side machinery lives in `crates/renderide/src/mesh_deform/`:

- `mesh_preprocess` is the entry point that the graph pass calls into.
- `skinning_palette` holds the bone matrix palettes that the skinning shader reads.
- `skin_cache` caches skinning results between frames where possible.
- `blendshape_bind_chunks` packs blendshape inputs into bind groups large enough to amortize binding cost without overflowing the GPU's per-bind-group limits.
- `range_alloc` and `scratch` manage the scratch buffers the compute passes write into.
- `per_draw_uniforms` carries the per-draw uniform packing that the forward pass consumes.

The output is a set of deformed vertex buffers (or buffer ranges) plus per-draw uniform data, all of which become the inputs to `world_mesh_forward`.

### 4.5 Visibility and Hi-Z occlusion

The renderer runs both CPU frustum culling and GPU Hi-Z occlusion culling against a hierarchical Z pyramid built from the previous frame's depth. The combination is the classical "depth from N-1 to cull N" trick: occlusion is approximate (objects that became visible between frames will pop in for one frame), but the result is dramatically less overdraw and a roughly bounded GPU cost per scene.

The CPU side lives in `crates/renderide/src/occlusion/cpu/` and the world-mesh visibility planner under `crates/renderide/src/world_mesh/culling/`. The GPU side lives in `crates/renderide/src/occlusion/gpu/` plus the `hi_z_build` graph pass and two compute shaders under `shaders/passes/compute/`: `hi_z_mip0.wgsl` (build the base mip from the resolved depth target) and `hi_z_downsample_max.wgsl` (build each subsequent mip by taking the max of the four corresponding texels in the previous mip).

`hi_z_downsample_max` uses the max because of reverse-Z: a max in reverse-Z space is a min in world-space depth, which is what an occluder query needs to be conservative.

The whole thing is a frame-global pass: the pyramid is built once per tick from the previous tick's depth, then read by the forward pass during per-view culling.

### 4.6 Clustered forward lighting

Renderide is a clustered forward renderer. Lights are binned into a 3D grid of clusters in view space (`x` by `y` by `z` slices, with `z` typically distributed exponentially to follow perspective), and each cluster carries a list of lights that affect it. The forward shader fetches the cluster for the current fragment and iterates only those lights.

Two pieces drive it:

- A compute pass `clustered_light` reads the scene's light list and writes the per-cluster light tables. The shader is `shaders/passes/compute/clustered_light.wgsl`. The CPU-side lives in `crates/renderide/src/passes/clustered_light/`.
- The forward pass reads the cluster tables at fragment-shader time. The shader-side helpers live in `shaders/modules/cluster_math.wgsl` and `shaders/modules/pbs_cluster.wgsl`.

The cluster geometry and light packing lives in `crates/renderide/src/world_mesh/cluster/` and `crates/renderide/src/backend/cluster_gpu.rs`. The clustered approach scales to many lights without paying the per-fragment cost of a brute-force forward loop, while keeping the bandwidth advantages of forward rendering over deferred for transparent surfaces, MSAA, and stylized shading.

### 4.7 Image-based lighting

The skybox feeds image-based lighting through a small pipeline of compute passes that produce a filtered cube map and the diffuse irradiance the shader needs.

The relevant code lives in `crates/renderide/src/skybox/` (with `params`, `prepared`, `specular`, and `ibl_cache`) and the relevant shaders live under `shaders/passes/compute/`:

- `skybox_mip0_cube_params.wgsl` and `skybox_mip0_equirect_params.wgsl` build the base mip of the environment cube depending on the source format.
- `skybox_bake_params.wgsl` and `skybox_ibl_convolve_params.wgsl` produce the prefiltered specular mip chain using a GGX prefilter (see `shaders/modules/ggx_prefilter.wgsl`).

The forward shader samples the prefiltered cube at a mip level chosen by perceptual roughness, which is the standard split-sum approximation for image-based specular. Diffuse IBL is handled separately by spherical harmonics (see [4.8](#48-reflection-probes-and-sh2-projection)).

### 4.8 Reflection probes and SH2 projection

The host can ask the renderer to project the environment around a point into spherical-harmonic coefficients. The renderer runs this as a nonblocking GPU job: a compute pass projects the environment cube (or equirect) into 9 SH2 coefficients per channel, and a readback job copies the coefficients back to host-visible memory and answers the host's request via the IPC.

The CPU-side lives in `crates/renderide/src/reflection_probes/` and on the scene side in `crates/renderide/src/scene/reflection_probe.rs`. The shaders live under `shaders/passes/compute/`: `sh2_project_cubemap.wgsl`, `sh2_project_equirect.wgsl`, and `sh2_project_sky_params.wgsl` for the parameter packing.

SH2 ambient is what the forward shader uses for low-frequency indirect lighting, with the reconstruction in `shaders/modules/sh2_ambient.wgsl`. The forward shader reads pre-baked coefficients for the active probe, evaluates the SH2 basis in the surface normal direction, and adds the result as ambient lighting.

### 4.9 Skybox families

Several skybox families are supported, each as its own pass shader under `shaders/passes/backend/`:

- `skybox_solid_color.wgsl` is the constant-color case.
- `skybox_gradientskybox.wgsl` is a vertical gradient between two colors (with an optional ground tint).
- `skybox_proceduralskybox.wgsl` is the procedural Rayleigh-and-Mie case used by the Unity asset of the same name.
- `skybox_projection360.wgsl` is the equirectangular case used by 360 photos and HDRIs.

The shared evaluator helpers live in `shaders/modules/skybox_common.wgsl` and `shaders/modules/skybox_evaluator.wgsl`. The skybox is rendered both as the visible background and as the input to the IBL pipeline; both consumers use the same evaluator code.

### 4.10 The HDR scene-color chain

The renderer runs in HDR end to end. The world is rendered into an HDR scene-color target (16-bit-per-channel float by default), every post-processing effect reads from and writes to that HDR domain, and the final tone mapper folds the HDR signal down into the swapchain's SDR (or wide-gamut) format right before present.

The chain is:

```mermaid
flowchart LR
    Forward[world_mesh_forward<br/>writes HDR scene color]
    Snap[depth + color snapshots]
    Eff[Post-process effects:<br/>GTAO, bloom, tonemap]
    Compose[scene_color_compose<br/>writes to swapchain / XR / RT]

    Forward --> Snap
    Snap --> Eff
    Eff --> Compose
```

Two structural points:

- Effects do not hijack the swapchain. They register graph work that consumes and produces HDR-chain resources, and the final compose pass is the only thing that touches the swapchain target.
- The depth and color snapshots produced by the forward pass are first-class graph resources. Effects that need them (notably GTAO and any depth-aware bloom) read from those resources rather than poking the live forward target.

### 4.11 ACES tone mapping

The tone mapper is an ACES (Academy Color Encoding System) approximation. It folds the HDR signal into the swapchain's color space, applies the ACES RRT-and-ODT compositing curve, and outputs gamma-corrected color to the next stage of the chain.

The implementation is one shader (`shaders/passes/post/aces_tonemap.wgsl`) plus the corresponding pass code under `crates/renderide/src/passes/post_processing/`. The pass is registered as a per-view post-processing effect.

### 4.12 Bloom

Bloom is implemented as a downsample-and-upsample pyramid: the bright parts of the HDR scene color are extracted, downsampled through a chain of progressively smaller mips, then upsampled back and added to the original. The result is a soft halo around bright pixels that approximates the optical bloom a real camera produces.

The shader is `shaders/passes/post/bloom.wgsl` and the pass code lives under `crates/renderide/src/passes/post_processing/`. Bloom runs in the HDR domain before tone mapping so the tone mapper can compress the bloomed signal correctly.

### 4.13 GTAO ambient occlusion

GTAO (Ground Truth Ambient Occlusion) is a screen-space ambient occlusion technique that estimates how much of the hemisphere above a surface is occluded by nearby geometry. The result is multiplied into ambient lighting to give crevices and contact points the darkening they would have if proper indirect light transport were modeled.

GTAO splits into three passes:

- `gtao_main.wgsl` does the main hemisphere sampling and produces a noisy AO texture.
- `gtao_denoise.wgsl` filters the noise spatially using depth-aware weights.
- `gtao_apply.wgsl` modulates the lighting by the denoised AO.

The shared filter math lives in `shaders/modules/gtao_filter.wgsl`. The pass code lives under `crates/renderide/src/passes/post_processing/`. GTAO runs in the post-processing chain after the forward pass and before tone mapping, reading from the depth and color snapshots produced by the forward pass.

### 4.14 The shader source tree

The shader tree under `crates/renderide/shaders/` is organized into four regions.

```mermaid
flowchart TB
    root["shaders/"] --> mat["materials/<br/>~117 host-routed material shaders"]
    root --> mod["modules/<br/>shared logic, naga-oil composable"]
    root --> pas["passes/<br/>renderer-internal passes"]
    pas --> bk["backend/<br/>skybox, depth blits"]
    pas --> cmp["compute/<br/>skinning, hi-z, lights, sh2, ibl"]
    pas --> post["post/<br/>tonemap, bloom, gtao,<br/>scene compose, msaa resolve"]
    pas --> pres["present/<br/>VR mirror"]
    mod --> sub1["material/<br/>color, alpha, sample, fresnel"]
    mod --> sub2["mesh/<br/>vertex, billboard"]
    mod --> sub3["pbs/<br/>BRDF, normal, displace, cluster"]
    mod --> sub4["ui/<br/>UI helpers"]
```

The four roles:

- `materials/` contains one shader per host material program. The filenames mirror the original Unity shader names, lowercased. There are over a hundred of them; the major families are PBS (physically based, with many variants for shadow, displacement, intersect, alpha, and so on), unlit (basic, overlay, billboard), Xiexe Toon (a stylized BRDF), CAD (line work), and a long list of effect shaders (blur, fresnel, gradient, lut, gamma, hsv, invert, grayscale, channelmatrix, paint, matcap, fresnel-lerp, displacement variants).
- `modules/` contains shared logic: math helpers, normal decoding, UV utilities, sky evaluation, BRDFs, GGX prefiltering, SH2 reconstruction, GTAO filter math, voronoi noise, text SDF math, billboard math, vertex transforms, and so on. Modules are composed into materials and passes via naga-oil (see [4.15](#415-naga-oil-composition)).
- `passes/backend/` contains shaders the renderer uses for backend tasks that are not material-driven: depth blits, skybox families.
- `passes/compute/` contains all compute shaders: skinning, blendshape, Hi-Z mip 0 and downsample, clustered light binning, MSAA depth resolve, SH2 projection, IBL convolution.
- `passes/post/` contains the post-processing shaders: GTAO, bloom, ACES tone mapping, MSAA resolve for HDR, scene color compose.
- `passes/present/` contains the VR mirror shaders.

The build script under `build.rs` and `build_support/shader/` discovers every WGSL file at build time, composes them, validates them, and emits an embedded shader registry the renderer links in. Material shaders are validated against the pipeline-state-versus-uniform contract (see [4.17](#417-pipeline-state-vs-shader-uniforms)). Shader composition runs in parallel across CPUs.

### 4.15 Naga-oil composition

The renderer uses [naga-oil](https://github.com/bevyengine/naga_oil) for shader composition. Naga-oil is a small layer on top of naga that adds module imports and conditional compilation to WGSL.

A shader file declares the modules it imports at the top, and the build script resolves those imports against `shaders/modules/` and the other shaders in the tree. Conditional compilation lets a single source file produce multiple variants (the most common is multiview versus non-multiview, but keyword permutations work the same way).

This composition system is what lets the renderer share BRDF, lighting, normal-mapping, and skybox-evaluation code across many materials without hand-rolling a copy in each one. It is also what keeps multiview a single-source-file concern instead of a code-duplication problem.

The Rust side of composition lives in `crates/renderide/build_support/shader/`. The layout is:

- `source` discovers shader source files.
- `modules` discovers and registers naga-oil composable modules.
- `compose` runs the composition.
- `directives` parses the comment directives (the most important being the `//#pass` directive, see [4.18](#418-the-pass-directive-system)).
- `validation` enforces the cross-cutting contracts (notably the pipeline-state-versus-uniform separation).
- `parallel` runs composition jobs across cores.
- `emit` writes the embedded shader registry that the renderer links in.
- `model` and `error` carry the data model and the error type.

### 4.16 Bind group conventions

The renderer fixes the role of each bind group so reflection can be applied uniformly across all material shaders.

| Group | Role |
| --- | --- |
| `@group(0)` | Per-frame data: time, camera, view matrices, lighting environment, cluster tables, IBL handles. |
| `@group(1)` | Per-material data. Binding 0 is always the material struct uniform. Subsequent bindings are textures and samplers. |
| `@group(2)` | Per-draw slab: per-draw uniforms produced by mesh deform and forward prepare, indexed by draw. |

Two consequences worth knowing:

- A material shader does not need to know anything about per-frame layout or per-draw layout; both are imported from modules and bound by the renderer. The shader only declares its own group 1 layout.
- The build script can validate bind group usage by inspecting the shader's reflection data and rejecting violations of the convention.

### 4.17 Pipeline state vs shader uniforms

This is the single most important rule in the material system, and the build script enforces it.

Pipeline state (blend factors, depth test direction, depth write enable, cull mode, color mask, stencil state, polygon offset) is part of the `wgpu::RenderPipeline` and is keyed in the pipeline cache. Shader uniforms (color tints, smoothness, cutoffs, UV scale and offset) are part of the material struct uniform at `@group(1) @binding(0)`. Keywords (`_NORMALMAP`, `_ALPHATEST_ON`, `_ALPHABLEND_ON`) are also part of the material struct uniform.

If a pipeline-state property name appears in a material shader's group 1 uniform, the build fails. The reason is correctness, not aesthetics: pipeline-state properties never affect the shader (the pipeline is what consumes them), but if they live in the uniform struct, reflection will allocate uniform space for them and the host will write values into that space that nothing reads. Worse, two materials that share the shader but differ only in pipeline state would correctly produce different cached pipelines but would also incorrectly differ in their uniform contents, which is the opposite of what the cache key encodes.

Adding a new shader uniform is uneventful. Adding a new pipeline-state property requires updating the canonical list of pipeline property IDs in the materials code and confirming that the build-time validator is teaching the new property correctly.

### 4.18 The pass directive system

Every material WGSL file under `shaders/materials/` declares one or more `//#pass <kind>` directives, each sitting directly above an `@fragment` entry point. The build script parses these into a static pass description table per shader stem, and the materials system uses that table to build one `wgpu::RenderPipeline` per declared pass.

The forward encode loop dispatches all pipelines for every draw that binds the material, in declared order. This is how a material that needs depth-only and color passes, or an opaque-and-shadow split, expresses itself: not by writing two shaders but by declaring two passes in the same file.

### 4.19 GPU resource pools and budgeting

The renderer is built around the assumption that the per-frame hot path does not allocate. The supporting machinery is the resident pool family under `crates/renderide/src/gpu_pools/`.

The pools cover:

- Meshes (vertex and index buffers, plus deformation outputs).
- 2D textures (`Texture2D` in the host's vocabulary).
- 3D textures (`Texture3D`).
- Cube maps.
- Render textures.
- Video textures (with the `video-textures` Cargo feature).

Each pool participates in a shared VRAM budget. When a new asset arrives and the pool needs to grow, the budget logic decides what (if anything) to evict to make room. Eviction is bounded; the budget never silently leaks across ticks.

The pool code is split into `pools` (the registry of pool kinds), `resource_pool` (the shared base behavior), `budget` (the VRAM accounting), `sampler_state` (the sampler pool), and `texture_allocation` (the texture-specific allocation logic).

### 4.20 Frame resource management

Above the per-asset pools sits the frame resource manager (`crates/renderide/src/backend/frame_resource_manager.rs`). It owns the per-frame state that is too short-lived to be a permanent pool entry but too expensive to reallocate every frame: per-view per-draw resources, frame-global GPU state, the light packing buffers, the cluster GPU buffers, and the bookkeeping that ties them to the planned views.

The render graph pre-warms this state once before the per-view loop begins, which is a structural prerequisite for the parallel record path: if the per-view loop had to lazily allocate frame state, parallel recording would race on the allocation.

### 4.21 The driver thread and queue access gate

GPU work is submitted from a dedicated driver thread that owns the wgpu device and queue. The runtime tick orchestrates work onto that thread through a queue access gate (`crates/renderide/src/gpu/queue_access_gate.rs`).

The gate exists to prevent two kinds of bugs at once. First, it makes the boundary between "code that can submit GPU work" and "code that cannot" explicit, so it is harder to accidentally submit from somewhere that should only be reading or planning. Second, it gives the runtime a place to coordinate submission with lock-step state, so the renderer never submits a frame that the host has not begun.

The driver thread itself lives in `crates/renderide/src/gpu/driver_thread/`. It is supported by the rest of `crates/renderide/src/gpu/` (the `adapter`, `context`, `instance_limits`, `limits`, `present`, `submission_state`, `frame_globals`, `frame_cpu_gpu_timing`, `bind_layout`, `depth`, `output_depth_mode`, `msaa_depth_resolve`, and `vr_mirror` modules).

### 4.22 Profiling with Tracy

When the `tracy` feature is enabled, the renderer streams CPU spans and GPU timestamps to a Tracy GUI on port 8086. The integration is on-demand: data is only streamed while a GUI is connected, so a profiled build idles near zero cost when nothing is attached.

The CPU side comes from the `profiling` crate, which expands to no-ops when no backend feature is active. The GPU side comes from `wgpu-profiler`, which inserts timestamp queries around the render-graph execution sub-phases. GPU timing requires the adapter to support `TIMESTAMP_QUERY` and `TIMESTAMP_QUERY_INSIDE_ENCODERS`. If either is missing, the renderer logs a warning at startup and falls back to CPU spans only.

When you add a new hot path or a long-running per-tick phase, instrument it. Match the granularity of nearby code: too coarse and you cannot see what is slow; too fine and you flood the trace.

The profiling glue lives in `crates/renderide/src/profiling.rs`. See `README.md` for build-and-connect instructions.

---

## License

Renderide is MIT licensed; see [`LICENSE`](LICENSE). For build instructions and how to launch the renderer, see [`README.md`](README.md). For day-to-day code conventions, see [Part 2 of this document](#part-2-working-in-the-repo).
