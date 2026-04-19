using NotEnoughLogs;
using SharedTypeGenerator.IR;
using SharedTypeGenerator.Logging;

namespace SharedTypeGenerator.Emission;

/// <summary>Explicit-layout struct emission: field-by-field writes without IL step lists.</summary>
public static partial class PackEmitter
{
    /// <summary>Emits pack body for ExplicitLayout structs (field-by-field with offsets).</summary>
    /// <param name="csharpTypeName">C# type name (for log context).</param>
    public static void EmitExplicitPack(RustWriter w, Logger logger, string csharpTypeName, List<FieldDescriptor> fields,
        int paddingBytes)
    {
        if (fields.Count == 0 && paddingBytes == 0)
        {
            w.Line("let _ = self;");
            w.Line("let _ = packer;");
            return;
        }

        foreach (FieldDescriptor field in fields)
        {
            if (field.Kind == FieldKind.Bool)
                w.Line($"packer.write_bool(self.{field.RustName} != 0);");
            else if (field.Kind is FieldKind.Enum or FieldKind.FlagsEnum)
                w.Line($"packer.write_object_required(&mut self.{field.RustName});");
            else if (field.Kind == FieldKind.ObjectRequired)
                w.Line($"packer.write_object_required(&mut self.{field.RustName});");
            else if (field.Kind == FieldKind.Pod)
                w.Line($"packer.write(&self.{field.RustName});");
            else
            {
                logger.LogWarning(
                    LogCategory.Fixme,
                    $"[{csharpTypeName}] ExplicitLayout pack: field '{field.RustName}' is FieldKind.{field.Kind}; treating as Pod (may not match C# MemoryPack).");
                w.Line($"packer.write(&self.{field.RustName});");
            }
        }

        if (paddingBytes > 0)
            w.Line("packer.write(&self._padding);");
    }

    /// <summary>Emits unpack body for ExplicitLayout structs.</summary>
    /// <param name="csharpTypeName">C# type name (for log context).</param>
    public static void EmitExplicitUnpack(RustWriter w, Logger logger, string csharpTypeName,
        List<FieldDescriptor> fields, int paddingBytes)
    {
        if (fields.Count == 0 && paddingBytes == 0)
        {
            w.Line("let _ = self;");
            w.Line("let _ = unpacker;");
            w.Line("Ok(())");
            return;
        }

        foreach (FieldDescriptor field in fields)
        {
            if (field.Kind == FieldKind.Bool)
                w.Line($"self.{field.RustName} = unpacker.read_bool()? as u8;");
            else if (field.Kind is FieldKind.Enum or FieldKind.FlagsEnum)
            {
                string rustType = field.RustType;
                w.Line($"self.{field.RustName} = {{ let mut x = {rustType}::default(); unpacker.read_object_required(&mut x)?; x }};");
            }
            else if (field.Kind == FieldKind.ObjectRequired)
                w.Line($"unpacker.read_object_required(&mut self.{field.RustName})?;");
            else if (field.Kind == FieldKind.Pod)
                w.Line($"self.{field.RustName} = unpacker.read()?;");
            else
            {
                logger.LogWarning(
                    LogCategory.Fixme,
                    $"[{csharpTypeName}] ExplicitLayout unpack: field '{field.RustName}' is FieldKind.{field.Kind}; treating as Pod read (may not match C# MemoryPack).");
                w.Line($"self.{field.RustName} = unpacker.read()?;");
            }
        }

        if (paddingBytes > 0)
            w.Line($"self._padding.copy_from_slice(&unpacker.access::<u8>({paddingBytes})?);");

        w.Line("Ok(())");
    }
}
