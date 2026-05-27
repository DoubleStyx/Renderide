//! Parsing Host messages into bootstrapper commands.

/// Command sent from the Host over `bootstrapper_in`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostCommand {
    /// Extends the IPC watchdog deadline.
    Heartbeat,
    /// Clean shutdown request.
    Shutdown,
    /// Clipboard read request.
    GetText,
    /// Clipboard write (payload after `SETTEXT` prefix).
    SetText(String),
    /// Spawn renderer with argv-style tokens from the message (whitespace-separated).
    StartRenderer(Vec<String>),
}

/// Parses a UTF-8 message from the Host into a [`HostCommand`].
///
/// Recognized prefixes: `HEARTBEAT`, `SHUTDOWN`, `GETTEXT`, `SETTEXT<payload>`.
/// Any other input is treated as whitespace-separated argv for [`HostCommand::StartRenderer`];
/// this catch-all is how `BootstrapperManager` requests a renderer launch.
pub fn parse_host_command(s: &str) -> HostCommand {
    match s {
        "HEARTBEAT" => HostCommand::Heartbeat,
        "SHUTDOWN" => HostCommand::Shutdown,
        "GETTEXT" => HostCommand::GetText,
        _ if s.starts_with("SETTEXT") => HostCommand::SetText(
            s.strip_prefix("SETTEXT")
                .map(str::to_string)
                .unwrap_or_default(),
        ),
        _ => {
            let argv: Vec<String> = s.split_whitespace().map(String::from).collect();
            let first = argv.first().map_or("<empty>", String::as_str);
            logger::debug!(
                "Bootstrap message did not match a known command; treating as renderer argv (first token: {first})"
            );
            HostCommand::StartRenderer(argv)
        }
    }
}
