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

    /// <summary>Places where generated Rust may contain FIXME comments.</summary>
    Fixme,

    /// <summary>Unexpected conditions worth investigating.</summary>
    Bug,
}
