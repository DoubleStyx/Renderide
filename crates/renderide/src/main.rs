//! Renderer binary entry point.

fn main() {
    if let Some(code) = renderide::run() {
        std::process::exit(code);
    }
}
