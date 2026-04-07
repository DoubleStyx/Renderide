//! Renderide main binary: thin entry point that delegates to the library.

fn main() {
    if let Some(code) = renderide::run() {
        std::process::exit(code);
    }
}
