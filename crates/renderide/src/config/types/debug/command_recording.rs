//! Command-recording strategy override for render-graph diagnostics.

use crate::labeled_enum;

labeled_enum! {
    /// Render-graph command-recording mode used for profiling and diagnostics.
    pub enum CommandRecordingMode: "command recording mode" {
        default => Auto;

        /// Use the renderer's conservative automatic command-recording policy.
        Auto => {
            persist: "auto",
            label: "Auto",
        },
        /// Record graph commands serially, preferring the coarsest encoder path.
        Serial => {
            persist: "serial",
            label: "Serial",
        },
        /// Record independent views across Rayon workers when there is enough draw work.
        AcrossViews => {
            persist: "across_views",
            label: "Across views",
            aliases: ["per_view", "per_view_parallel"],
        },
        /// Record scheduler-admitted pass units inside one view across Rayon workers.
        InView => {
            persist: "in_view",
            label: "In-view",
            aliases: ["in_view_parallel"],
        },
    }
}

impl CommandRecordingMode {
    /// Numeric value used by Tracy plots and compact diagnostics.
    pub const fn as_plot_value(self) -> u64 {
        match self {
            Self::Auto => 0,
            Self::Serial => 1,
            Self::AcrossViews => 2,
            Self::InView => 3,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::CommandRecordingMode;

    #[test]
    fn command_recording_mode_parses_aliases() {
        assert_eq!(
            CommandRecordingMode::parse_persist("auto"),
            Some(CommandRecordingMode::Auto)
        );
        assert_eq!(
            CommandRecordingMode::parse_persist("per_view_parallel"),
            Some(CommandRecordingMode::AcrossViews)
        );
        assert_eq!(
            CommandRecordingMode::parse_persist("in_view_parallel"),
            Some(CommandRecordingMode::InView)
        );
        assert_eq!(CommandRecordingMode::parse_persist(""), None);
    }
}
