namespace SharedTypeGenerator.IR;

/// <summary>Describes a single field on a type, with both the C# metadata
/// and the pre-computed Rust representation.</summary>
public sealed class FieldDescriptor
{
    public required string CSharpName { get; init; }
    public required string RustName { get; init; }
    public required string RustType { get; init; }
    public required FieldKind Kind { get; init; }

    /// <summary>Only set for ExplicitLayout (PodStruct) fields.</summary>
    public int? ExplicitOffset { get; init; }
}
