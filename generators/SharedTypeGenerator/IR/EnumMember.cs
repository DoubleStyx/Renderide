namespace SharedTypeGenerator.IR;

/// <summary>A single variant in a C# enum.</summary>
public sealed class EnumMember
{
    public required string Name { get; init; }
    public required object Value { get; init; }
    public required bool IsDefault { get; init; }
}
