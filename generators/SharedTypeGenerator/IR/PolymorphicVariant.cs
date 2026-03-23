namespace SharedTypeGenerator.IR;

/// <summary>One registered subtype in a polymorphic type registry
/// (extracted from the static constructor's InitTypes call).</summary>
public sealed class PolymorphicVariant
{
    public required string CSharpName { get; init; }
    public required string RustName { get; init; }
    public required Type RuntimeType { get; init; }
}
