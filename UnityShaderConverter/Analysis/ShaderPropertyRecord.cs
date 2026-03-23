using UnityShaderParser.ShaderLab;

namespace UnityShaderConverter.Analysis;

/// <summary>One <c>Properties</c> block entry mapped for Rust emission.</summary>
public sealed class ShaderPropertyRecord
{
    /// <summary>Uniform name including leading underscore when present in source.</summary>
    public required string Name { get; init; }

    /// <summary>Inspector display string.</summary>
    public required string DisplayLabel { get; init; }

    /// <summary>ShaderLab property kind.</summary>
    public ShaderPropertyKind Kind { get; init; }

    /// <summary>Range min/max when <see cref="Kind"/> is range.</summary>
    public (float Min, float Max)? Range { get; init; }

    /// <summary>Human-readable default for generated Rust comments.</summary>
    public required string DefaultSummary { get; init; }
}
