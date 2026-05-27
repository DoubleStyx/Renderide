using NotEnoughLogs;
using SharedTypeGenerator.IR;
using SharedTypeGenerator.Logging;

namespace SharedTypeGenerator.Emission;

/// <summary>Backend orchestrator: takes a list of TypeDescriptors and emits the complete
/// shared.rs file. Dispatches each type to shape-specific emission methods.
/// No reflection, no Cecil -- purely data-driven from the IR.</summary>
internal sealed partial class RustEmitter
{
    /// <summary>Stable order for <c>use glam::{{ ... }};</c>; only names referenced in emitted Rust types are imported.</summary>
    private static readonly string[] GlamImportOrder =
    [
        "IVec2", "IVec3", "IVec4",
        "Mat4", "Quat",
        "Vec2", "Vec3", "Vec4",
    ];

    private readonly RustWriter _w;
    private readonly Logger _logger;
    private readonly string _engineVersion;
    private readonly bool _ilVerbose;

    /// <summary>Creates an emitter targeting <paramref name="writer"/>.</summary>
    /// <param name="logger">Receives warnings when emission is incomplete or emits FIXME-equivalent Rust.</param>
    /// <param name="ilVerbose">When true, emits leading comments naming each C# type before its Rust definition.</param>
    public RustEmitter(RustWriter writer, Logger logger, string engineVersion, bool ilVerbose = false)
    {
        _w = writer;
        _logger = logger;
        _engineVersion = engineVersion;
        _ilVerbose = ilVerbose;
    }

    /// <summary>Emits the complete shared.rs file from the analyzed type list.</summary>
    public void Emit(List<TypeDescriptor> types)
    {
        EmitHeader(types);

        foreach (TypeDescriptor type in types)
        {
            EmitType(type);
            _w.BlankLine();
        }

        EmitRoundtripDispatch(types);
    }

    private void EmitType(TypeDescriptor type)
    {
        if (_ilVerbose)
            _w.Comment($"IL-verbose: C# {type.CSharpName} ({type.Shape})");

        switch (type.Shape)
        {
            case TypeShape.PolymorphicBase:
                EmitPolymorphic(type);
                break;
            case TypeShape.ValueEnum:
                EmitValueEnum(type);
                break;
            case TypeShape.FlagsEnum:
                EmitFlagsEnum(type);
                break;
            case TypeShape.PodStruct:
                EmitPodStruct(type);
                break;
            case TypeShape.PackableStruct:
                EmitPackableStruct(type);
                break;
            case TypeShape.GeneralStruct:
                EmitGeneralStruct(type);
                break;
        }
    }
}
