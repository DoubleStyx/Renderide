namespace SharedTypeGenerator.IR;

/// <summary>One registered subtype in a polymorphic type registry
/// (extracted from the static constructor's InitTypes call).</summary>
public sealed class PolymorphicVariant
{
    /// <summary>C# type name of the variant.</summary>
    public required string CSharpName { get; init; }

    /// <summary>Rust identifier for the variant.</summary>
    public required string RustName { get; init; }

    /// <summary>Loaded CLR type for enqueueing referenced types.</summary>
    public required Type RuntimeType { get; init; }
}
