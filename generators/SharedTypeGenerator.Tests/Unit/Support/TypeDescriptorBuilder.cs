using SharedTypeGenerator.IR;

namespace SharedTypeGenerator.Tests.Unit.Support;

/// <summary>Concise factories for IR records used by unit tests.
/// Centralizing construction here keeps test arrange blocks resilient to new
/// optional <see cref="TypeDescriptor"/> / <see cref="FieldDescriptor"/> properties.</summary>
internal static class Ir
{
    /// <summary>Builds a <see cref="TypeShape.PodStruct"/> <see cref="TypeDescriptor"/> with sensible defaults.</summary>
    /// <param name="name">Name shared between <see cref="TypeDescriptor.CSharpName"/> and <see cref="TypeDescriptor.RustName"/>.</param>
    /// <param name="isPod">Whether the struct can derive Pod in Rust.</param>
    /// <param name="fields">Optional explicit field list (defaults to empty).</param>
    /// <param name="packSteps">Optional pack steps (defaults to empty).</param>
    public static TypeDescriptor PodStruct(
        string name,
        bool isPod = true,
        IEnumerable<FieldDescriptor>? fields = null,
        IEnumerable<SerializationStep>? packSteps = null) =>
        new()
        {
            CSharpName = name,
            RustName = name,
            Shape = TypeShape.PodStruct,
            Fields = fields?.ToList() ?? [],
            IsPod = isPod,
            PackSteps = packSteps?.ToList() ?? [],
        };

    /// <summary>Builds a <see cref="FieldKind.Pod"/> <see cref="FieldDescriptor"/>.</summary>
    /// <param name="name">Name shared between <see cref="FieldDescriptor.CSharpName"/> and <see cref="FieldDescriptor.RustName"/>.</param>
    /// <param name="rustType">Rust type string.</param>
    /// <param name="kind">Override the <see cref="FieldKind"/>; defaults to <see cref="FieldKind.Pod"/>.</param>
    public static FieldDescriptor PodField(string name, string rustType, FieldKind kind = FieldKind.Pod) =>
        new()
        {
            CSharpName = name,
            RustName = name,
            RustType = rustType,
            Kind = kind,
        };

    /// <summary>Builds a <see cref="FieldKind.Object"/> <see cref="FieldDescriptor"/>.</summary>
    /// <param name="name">Name shared between <see cref="FieldDescriptor.CSharpName"/> and <see cref="FieldDescriptor.RustName"/>.</param>
    /// <param name="rustType">Rust type string (may include <c>Option&lt;...&gt;</c>).</param>
    public static FieldDescriptor ObjectField(string name, string rustType) =>
        new()
        {
            CSharpName = name,
            RustName = name,
            RustType = rustType,
            Kind = FieldKind.Object,
        };
}
