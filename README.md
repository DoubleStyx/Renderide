A modern Rust + wgpu renderer for Resonite.

## Status

Experimental: Performance, stability, and platform support are still evolving.  
Visual bugs and missing features are expected.

## Building and Running

1. Ensure you have a Steam installation of [Resonite](https://store.steampowered.com/app/2519830/Resonite/).

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

## Debugging

1. Build the workspace in dev mode:

   ```bash
   cargo build --profile dev
   ```

1. Run the bootstrapper in dev mode:

   ```bash
   RUST_BACKTRACE=1 ./target/dev/bootstrapper
   ```

1. Enable validation layers in the config hud to get more detailed error messages for GPU crashes. Requires a restart.

1. Inspect logs in the `logs/` folder for panics, crashes, backtraces, and validation errors.
