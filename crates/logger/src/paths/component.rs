//! Log component identifiers and component directory names.

/// Which part of the system produces a log stream under [`crate::logs_root`] / `<component>/`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LogComponent {
    /// Bootstrapper process (Rust).
    Bootstrapper,
    /// Host process output captured by the bootstrapper (stdout/stderr into one file).
    Host,
    /// Renderer process (Rust).
    Renderer,
    /// Renderer integration-test harness process (Rust).
    RendererTest,
}

impl LogComponent {
    /// Subdirectory name under `logs/` for this component.
    pub const fn subdir(self) -> &'static str {
        match self {
            Self::Bootstrapper => "bootstrapper",
            Self::Host => "host",
            Self::Renderer => "renderer",
            Self::RendererTest => "renderer-test",
        }
    }
}

impl std::fmt::Display for LogComponent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.subdir())
    }
}
