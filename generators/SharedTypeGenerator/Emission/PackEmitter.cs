using NotEnoughLogs;
using SharedTypeGenerator.Analysis;
using SharedTypeGenerator.IR;
using SharedTypeGenerator.Logging;

namespace SharedTypeGenerator.Emission;

/// <summary>Emits pack/unpack method bodies from <see cref="SerializationStep"/> lists and explicit-layout fields.</summary>
public static partial class PackEmitter
{
    /// <summary>Buffer size (bytes) for generated <c>roundtrip_dispatch</c> output and test harness packing.</summary>
    public const int RoundtripBufferBytes = 1024 * 1024;

    /// <summary>Emits the full pack method body for a list of steps.</summary>
    /// <param name="csharpTypeName">C# type name (for log context).</param>
    public static void EmitPack(RustWriter w, Logger logger, string csharpTypeName, List<SerializationStep> steps,
        List<FieldDescriptor> fields)
    {
        if (steps.Count == 0)
        {
            w.Line("let _ = self;");
            w.Line("let _ = packer;");
            return;
        }

        foreach (SerializationStep step in steps)
            EmitPackStep(w, logger, csharpTypeName, step, fields);
    }

    /// <summary>Emits the full unpack method body for a list of steps.
    /// <paramref name="unpackOnlySteps"/> (e.g. decodedTime = UtcNow) are emitted only in unpack, not in pack.</summary>
    /// <param name="csharpTypeName">C# type name (for log context).</param>
    public static void EmitUnpack(RustWriter w, Logger logger, string csharpTypeName, List<SerializationStep> steps,
        List<FieldDescriptor> fields,
        List<SerializationStep>? unpackOnlySteps = null)
    {
        if (steps.Count == 0 && (unpackOnlySteps == null || unpackOnlySteps.Count == 0))
        {
            w.Line("let _ = self;");
            w.Line("let _ = unpacker;");
            w.Line("Ok(())");
            return;
        }

        foreach (SerializationStep step in steps)
            EmitUnpackStep(w, logger, csharpTypeName, step, fields);

        foreach (SerializationStep step in unpackOnlySteps ?? [])
            EmitUnpackStep(w, logger, csharpTypeName, step, fields);

        w.Line("Ok(())");
    }
}
