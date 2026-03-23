namespace SharedTypeGenerator.IR;

/// <summary>Describes a single field on a type, with both the C# metadata
/// and the pre-computed Rust representation.</summary>
public sealed class FieldDescriptor
{
    /// <summary>Original C# field name.</summary>
    public required string CSharpName { get; init; }

    /// <summary>Rust field identifier (typically snake_case).</summary>
    public required string RustName { get; init; }

    /// <summary>Rust type string as emitted in struct definitions.</summary>
    public required string RustType { get; init; }

    /// <summary>Serialization classification for pack/unpack emission.</summary>
    public required FieldKind Kind { get; init; }

    /// <summary>Byte offset for ExplicitLayout (PodStruct) fields; otherwise null.</summary>
    public int? ExplicitOffset { get; init; }
}
