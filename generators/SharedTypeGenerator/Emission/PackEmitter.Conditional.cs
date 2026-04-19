using NotEnoughLogs;
using SharedTypeGenerator.Analysis;
using SharedTypeGenerator.IR;
using SharedTypeGenerator.Logging;

namespace SharedTypeGenerator.Emission;

/// <summary>
/// Pack/unpack emission for <see cref="SerializationStep"/> lists: conditional blocks, packed bools, and per-field lines.
/// </summary>
public static partial class PackEmitter
{
    private static void EmitPackStep(RustWriter w, Logger logger, string csharpTypeName, SerializationStep step,
        List<FieldDescriptor> fields)
    {
        switch (step)
        {
            case WriteField wf:
                if (wf.Kind == FieldKind.StringList)
                    EmitStringListPack(w, wf.FieldName);
                else
                    w.Line(PackLine(logger, csharpTypeName, wf.FieldName, wf.Kind));
                break;

            case PackedBools pb:
                {
                    var args = new List<string>();
                    foreach (string name in pb.FieldNames)
                        args.Add($"self.{name}");
                    while (args.Count < 8)
                        args.Add("false");
                    w.Line($"packer.write_packed_bools_array([{string.Join(", ", args)}]);");
                    break;
                }

            case CallBase:
                logger.LogWarning(
                    LogCategory.Fixme,
                    $"[{csharpTypeName}] Pack: CallBase step was not inlined during analysis (FIXME emitted in Rust).");
                w.Fixme("CallBase should have been inlined during analysis");
                break;

            case TimestampNow:
                break;

            case ConditionalBlock cb:
                {
                    using (w.BeginIf($"self.{cb.ConditionField}"))
                    {
                        foreach (SerializationStep inner in cb.Steps)
                            EmitPackStep(w, logger, csharpTypeName, inner, fields);
                    }

                    break;
                }
        }
    }

    private static void EmitStringListPack(RustWriter w, string name)
    {
        w.Line($"let __strs: Vec<Option<&str>> = self.{name}.iter().map(|s| s.as_deref()).collect();");
        w.Line("packer.write_string_list(Some(&__strs));");
    }

    private static void EmitUnpackStep(RustWriter w, Logger logger, string csharpTypeName, SerializationStep step,
        List<FieldDescriptor> fields)
    {
        switch (step)
        {
            case WriteField wf:
                w.Line(UnpackLine(logger, csharpTypeName, wf.FieldName, wf.Kind, fields));
                break;

            case PackedBools pb:
                {
                    var fieldNames = pb.FieldNames.ToList();
                    while (fieldNames.Count < 8)
                        fieldNames.Add("_");

                    w.Line("let __p = unpacker.read_packed_bools()?;");
                    for (int i = 0; i < 8; i++)
                    {
                        if (fieldNames[i] != "_")
                            w.Line($"self.{fieldNames[i]} = __p.bit{i};");
                    }

                    break;
                }

            case CallBase:
                logger.LogWarning(
                    LogCategory.Fixme,
                    $"[{csharpTypeName}] Unpack: CallBase step was not inlined during analysis (FIXME emitted in Rust).");
                w.Fixme("CallBase should have been inlined during analysis");
                break;

            case ConditionalBlock cb:
                {
                    using (w.BeginIf($"self.{cb.ConditionField}"))
                    {
                        foreach (SerializationStep inner in cb.Steps)
                            EmitUnpackStep(w, logger, csharpTypeName, inner, fields);
                    }

                    break;
                }

            case TimestampNow ts:
                w.Line($"self.{ts.FieldName} = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos() as i128;");
                break;
        }
    }

    private static string PackLine(Logger logger, string csharpTypeName, string name, FieldKind kind) => kind switch
    {
        FieldKind.Pod => $"packer.write(&self.{name});",
        FieldKind.Bool => $"packer.write_bool(self.{name});",
        FieldKind.String => $"packer.write_str(self.{name}.as_deref());",
        FieldKind.Enum => $"packer.write_object_required(&mut self.{name});",
        FieldKind.FlagsEnum => $"packer.write_object_required(&mut self.{name});",
        FieldKind.Nullable => $"packer.write_option(self.{name}.as_ref());",
        FieldKind.Object => $"packer.write_object(self.{name}.as_mut());",
        FieldKind.ObjectRequired => $"packer.write_object_required(&mut self.{name});",
        FieldKind.ValueList => $"packer.write_value_list(Some(&self.{name}));",
        FieldKind.EnumValueList => $"packer.write_enum_value_list(Some(&self.{name}));",
        FieldKind.ObjectList => $"packer.write_object_list(Some(&mut self.{name}[..]));",
        FieldKind.PolymorphicList => $"packer.write_polymorphic_list(Some(&mut self.{name}[..]));",
        FieldKind.StringList => throw new InvalidOperationException("StringList must use EmitStringListPack."),
        FieldKind.NestedValueList => $"packer.write_nested_value_list(Some(&self.{name}));",
        _ => UnknownFieldKindPackLine(logger, csharpTypeName, name, kind),
    };

    private static string UnknownFieldKindPackLine(Logger logger, string csharpTypeName, string name, FieldKind kind)
    {
        logger.LogWarning(
            LogCategory.Fixme,
            $"[{csharpTypeName}] Pack: unhandled FieldKind {kind} for field '{name}' (FIXME comment emitted in Rust).");
        return $"// FIXME: Unknown FieldKind {kind} for {name}";
    }

    private static string UnpackLine(Logger logger, string csharpTypeName, string name, FieldKind kind,
        List<FieldDescriptor> fields) => kind switch
        {
            FieldKind.Pod => $"self.{name} = unpacker.read()?;",
            FieldKind.Bool => $"self.{name} = unpacker.read_bool()?;",
            FieldKind.String => $"self.{name} = unpacker.read_str()?;",
            FieldKind.Enum => $"unpacker.read_object_required(&mut self.{name})?;",
            FieldKind.FlagsEnum => $"unpacker.read_object_required(&mut self.{name})?;",
            FieldKind.Nullable => $"self.{name} = unpacker.read_option()?;",
            FieldKind.Object => UnpackObjectLine(logger, csharpTypeName, name, fields),
            FieldKind.ObjectRequired => $"unpacker.read_object_required(&mut self.{name})?;",
            FieldKind.ValueList => $"self.{name} = unpacker.read_value_list()?;",
            FieldKind.EnumValueList => $"self.{name} = unpacker.read_enum_value_list()?;",
            FieldKind.ObjectList => $"self.{name} = unpacker.read_object_list()?;",
            FieldKind.PolymorphicList => UnpackPolymorphicListLine(logger, csharpTypeName, name, fields),
            FieldKind.StringList => $"self.{name} = unpacker.read_string_list()?;",
            FieldKind.NestedValueList => $"self.{name} = unpacker.read_nested_value_list()?;",
            _ => UnknownFieldKindUnpackLine(logger, csharpTypeName, name, kind),
        };

    private static string UnknownFieldKindUnpackLine(Logger logger, string csharpTypeName, string name, FieldKind kind)
    {
        logger.LogWarning(
            LogCategory.Fixme,
            $"[{csharpTypeName}] Unpack: unhandled FieldKind {kind} for field '{name}' (FIXME comment emitted in Rust).");
        return $"// FIXME: Unknown FieldKind {kind} for {name}";
    }

    private static string UnpackObjectLine(Logger logger, string csharpTypeName, string name, List<FieldDescriptor> fields)
    {
        FieldDescriptor? field = fields.FirstOrDefault(f => f.RustName == name);
        if (field == null)
        {
            logger.LogWarning(
                LogCategory.Fixme,
                $"[{csharpTypeName}] Unpack: object field '{name}' has no FieldDescriptor; emitted read_object::<_>().");
            return $"self.{name} = unpacker.read_object::<_>()?;";
        }

        string rustType = RustTypeMapper.NormalizeRustTypeName(field.RustType);
        return $"self.{name} = unpacker.read_object::<{rustType}>()?;";
    }

    private static string UnpackPolymorphicListLine(Logger logger, string csharpTypeName, string name,
        List<FieldDescriptor> fields)
    {
        FieldDescriptor? field = fields.FirstOrDefault(f => f.RustName == name);
        if (field == null)
        {
            logger.LogWarning(
                LogCategory.Fixme,
                $"[{csharpTypeName}] Unpack: polymorphic list field '{name}' has no FieldDescriptor; emitted read_polymorphic_list(unimplemented_decode).");
            return $"self.{name} = unpacker.read_polymorphic_list(unimplemented_decode)?;";
        }

        string rustType = RustTypeMapper.StripVecElementType(field.RustType);
        string decodeFn = "decode_" + rustType.HumanizeField();
        return $"self.{name} = unpacker.read_polymorphic_list({decodeFn})?;";
    }
}
