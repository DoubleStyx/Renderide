namespace SharedTypeGenerator.IR;

/// <summary>A single variant in a C# enum.</summary>
public sealed class EnumMember
{
    /// <summary>Enum member name.</summary>
    public required string Name { get; init; }

    /// <summary>Underlying numeric value.</summary>
    public required object Value { get; init; }

    /// <summary>Whether this is the first emitted variant (Rust #[default]).</summary>
    public required bool IsDefault { get; init; }
}
