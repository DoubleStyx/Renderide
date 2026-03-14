# Renderide

A Rust renderer for Resonite, replacing the Unity-based default with wgpu/winit. Early-stage project with AAA renderer ambitions.

## Overview

Resonite (formerly Neos VR) is a FrooxEngine-based VR and social platform. Renderite is its renderer abstraction layer (Host, Shared, Unity). Renderide acts as a drop-in renderer replacement, cross-platform with a focus on Linux via native dotnet host.

```mermaid
flowchart TB
    subgraph Bootstrapper [Bootstrapper Rust]
        CreateQueues[Create IPC queues]
        SpawnHost[Spawn Renderite.Host]
        QueueLoop[Queue command loop]
        SpawnRenderer[Spawn Renderide]
    end
    subgraph Host [Renderite.Host C#]
        GameEngine[Game engine init]
    end
    subgraph Renderide [Renderide Rust]
        Renderer[wgpu/winit renderer]
    end
    CreateQueues --> SpawnHost
    SpawnHost --> QueueLoop
    QueueLoop -->|StartRenderer| SpawnRenderer
    Host <-->|IPC queues| Bootstrapper
    Renderide <-->|IPC queues| Host
```
 
## Project Status

Early development.

### Critical

1. Improve performance.
2. Fix skinned mesh renderers.

### Low-Priority

1. Add mirror support.
2. Add camera support.
3. Upload textures for meshes.
4. Use correct shaders for meshes.
5. Remove useless logging.
6. Make the logging tiered.
7. Add reflection probes.
8. Add post-processing.
9. Add light probes.
10. Add RTAO.
11. Add RT reflections.
12. Add path tracing.
13. Add skyboxes.
14. Implement a UI renderer.
15. Add video player support.
16. Refactor code to remove dead modules and improve organization.
17. Make code more robust and safe.
18. Abstract out original IPC commands more.
19. Add VR support.

## Architecture

Bootstrapper creates IPC queues, spawns Renderite.Host, runs the queue loop, and spawns Renderide when the Host sends StartRenderer. Host and Renderide communicate via shared-memory queues.

| Crate | Path | Purpose |
|-------|------|---------|
| **interprocess** | `crates/interprocess/` | Shared-memory queues (Publisher/Subscriber), circular buffers. Used by bootstrapper and renderide for IPC. |
| **bootstrapper** | `crates/bootstrapper/` | Orchestrator: creates `bootstrapper_in`/`bootstrapper_out` queues, spawns Renderite.Host from Resonite install, runs queue loop (HEARTBEAT, SHUTDOWN, GETTEXT, SETTEXT, StartRenderer), spawns Renderide when Host requests it. Supports Wine on Linux. |
| **renderide** | `crates/renderide/` | Main renderer: wgpu, winit, session/IPC receiver, shared types + packing, scene graph, assets, GPU meshes. Binaries: `renderide`, `roundtrip`. |

## SharedTypeGenerator

**Location:** `SharedTypeGenerator/` (C# .NET 10)

Converts `Renderite.Shared.dll` into `crates/renderide/src/shared/shared.rs`. Pipeline: TypeAnalyzer (Mono.Cecil) → PackMethodParser (IL → SerializationStep) → RustTypeMapper → RustEmitter + PackEmitter. Outputs Rust types (POD structs, packable structs, polymorphic entities, enums) with `MemoryPackable` impls matching the C# wire format.

```bash
dotnet run --project SharedTypeGenerator -- -i /path/to/Renderite.Shared.dll [-o output.rs]
```

Default output: `crates/renderide/src/shared/shared.rs`

## Tests

**SharedTypeGenerator.Tests/** — xUnit C# tests. Cross-language round-trip: C# packs a random instance → bytes A; Rust `roundtrip` binary unpacks and packs → bytes B; assert A == B.

**Prerequisite:** `Renderite.Shared.dll` in `SharedTypeGenerator.Tests/lib/` or set `RENDERITE_SHARED_DLL`.

```bash
cargo build --bin roundtrip
dotnet test SharedTypeGenerator.Tests/
```

## Logging

All logs under `logs/` (relative to bootstrapper CWD or repo root). Truncated at each run.

| Log | Path | Created By |
|-----|------|------------|
| Bootstrapper.log | `logs/Bootstrapper.log` | Bootstrapper crate — orchestration, queue commands, errors |
| HostOutput.log | `logs/HostOutput.log` | Bootstrapper (redirects C# host stdout/stderr with [Host stdout]/[Host stderr] prefixes) |
| Renderide.log | `logs/Renderide.log` | Renderide crate — renderer diagnostics (path: repo root via CARGO_MANIFEST_DIR) |

## Building and Running

**Rust:**

```bash
cargo build
cargo build --release
```

**Generator:**

```bash
dotnet run --project SharedTypeGenerator -- -i /path/to/Renderite.Shared.dll
```

**Resonite discovery:** `RESONITE_DIR` or Steam (`~/.steam/steam/steamapps/common/Resonite`, `~/.local/share/Steam`, libraryfolders.vdf).

**Bootstrapper:** Run from repo root; expects `target/debug/renderide` (or `Renderite.Renderer` symlink on Linux).

**Wine:** Bootstrapper detects Wine and uses `LinuxBootstrap.sh` in the Resonite directory.

## Goals

- AAA-quality renderer (path tracing, RTAO, RT reflections, PBR, etc.)
- Cross-platform (Linux native via dotnet host)
- Type-safe IPC via generated shared types
- Performance and correctness (skinned meshes, proper shaders, textures)
