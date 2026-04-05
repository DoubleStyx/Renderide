namespace SharedTypeGenerator.Logging;

/// <summary>Logical channel for generator log messages.</summary>
public enum LogCategory
{
    /// <summary>Bootstrap, CLI, and assembly loading.</summary>
    Startup,

    /// <summary>Type graph and IL analysis.</summary>
    Analysis,

    /// <summary>High-level generation milestones (e.g. engine version).</summary>
    Generator,

    /// <summary>Rust source emission.</summary>
    Emission,

    /// <summary>Filesystem output (delete/write paths).</summary>
    Output,

    /// <summary>Incomplete emission, FIXME comments in Rust, or analyzer/emitter assumptions that may be wrong.</summary>
    Fixme,

    /// <summary>Fatal pipeline failures and uncaught exceptions (logged before exit).</summary>
    Bug,
}
