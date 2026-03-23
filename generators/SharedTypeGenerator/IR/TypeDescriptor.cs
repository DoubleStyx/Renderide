namespace SharedTypeGenerator.IR;

/// <summary>Complete intermediate representation of a single type,
/// containing everything needed to emit its Rust definition and trait impls.</summary>
public sealed class TypeDescriptor
{
    public required string CSharpName { get; init; }
    public required string RustName { get; init; }
    public required TypeShape Shape { get; init; }
    public required List<FieldDescriptor> Fields { get; init; }

    /// <summary>Ordered serialization operations derived from the Pack method IL.
    /// Empty for enums and types without Pack methods.</summary>
    public List<SerializationStep> PackSteps { get; init; } = [];

    /// <summary>Steps that run only during unpack (e.g. decodedTime = UtcNow).
    /// Emitted in unpack but not in pack.</summary>
    public List<SerializationStep> UnpackOnlySteps { get; init; } = [];

    /// <summary>For PackableStruct with inheritance (e.g., AssetCommand -> RendererCommand).</summary>
    public string? BaseTypeName { get; init; }

    /// <summary>For ValueEnum/FlagsEnum: the C# underlying type (int, byte, etc.).</summary>
    public Type? UnderlyingEnumType { get; init; }

    /// <summary>For ValueEnum/FlagsEnum: the enum variants.</summary>
    public List<EnumMember>? EnumMembers { get; init; }

    /// <summary>For PolymorphicBase: the registered subtype variants.</summary>
    public List<PolymorphicVariant>? Variants { get; init; }

    /// <summary>Whether the struct can derive Pod + Zeroable + Clone + Copy.</summary>
    public bool IsPod { get; init; }

    /// <summary>For ExplicitLayout structs: the declared size from StructLayoutAttribute.</summary>
    public int? ExplicitSize { get; init; }

    /// <summary>Computed padding bytes needed to match the declared ExplicitLayout size.</summary>
    public int PaddingBytes { get; init; }

    /// <summary>The Rust type string for the enum's underlying type (e.g., "i32", "u8").</summary>
    public string? RustUnderlyingType { get; init; }
}
