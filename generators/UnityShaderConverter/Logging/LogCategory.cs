namespace UnityShaderConverter.Logging;

/// <summary>Logical channel for converter log messages.</summary>
public enum LogCategory
{
    /// <summary>CLI, configuration, and startup.</summary>
    Startup,

    /// <summary>ShaderLab discovery and parsing.</summary>
    Parse,

    /// <summary>Slang source emission.</summary>
    Slang,

    /// <summary>External <c>slangc</c> invocation.</summary>
    SlangCompile,

    /// <summary>Rust source emission.</summary>
    Rust,

    /// <summary>Filesystem writes and cleanup.</summary>
    Output,

    /// <summary>Shader variant expansion.</summary>
    Variants,

    /// <summary>End-of-run summary of shaders that did not fully generate.</summary>
    FailureReport,
}
