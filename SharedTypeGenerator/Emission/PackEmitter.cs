using SharedTypeGenerator.Analysis;
using SharedTypeGenerator.IR;

namespace SharedTypeGenerator.Emission;

/// <summary>Emits pack/unpack method bodies from SerializationSteps.
/// Each step maps to exactly one pack line and one unpack line -- no duplication.</summary>
public static class PackEmitter
{
    /// <summary>Emits the full pack method body for a list of steps.</summary>
    public static void EmitPack(RustWriter w, List<SerializationStep> steps, List<FieldDescriptor> fields)
    {
        if (steps.Count == 0)
        {
            w.Line("let _ = self;");
            w.Line("let _ = packer;");
            return;
        }

        foreach (SerializationStep step in steps)
            EmitPackStep(w, step, fields);
    }

    /// <summary>Emits the full unpack method body for a list of steps.
    /// unpackOnlySteps (e.g. decodedTime = UtcNow) are emitted only in unpack, not in pack.</summary>
    public static void EmitUnpack(RustWriter w, List<SerializationStep> steps, List<FieldDescriptor> fields,
        List<SerializationStep>? unpackOnlySteps = null)
    {
        if (steps.Count == 0 && (unpackOnlySteps == null || unpackOnlySteps.Count == 0))
        {
            w.Line("let _ = self;");
            w.Line("let _ = unpacker; // FIXME: Type not generating any members");
            return;
        }

        foreach (SerializationStep step in steps)
            EmitUnpackStep(w, step, fields);

        foreach (SerializationStep step in unpackOnlySteps ?? [])
            EmitUnpackStep(w, step, fields);
    }

    private static void EmitPackStep(RustWriter w, SerializationStep step, List<FieldDescriptor> fields)
    {
        switch (step)
        {
            case WriteField wf:
                w.Line(PackLine(wf.FieldName, wf.Kind));
                break;

            case PackedBools pb:
            {
                var args = new List<string>();
                foreach (string name in pb.FieldNames)
                    args.Add($"self.{name}");
                while (args.Count < 8)
                    args.Add("false");
                w.Line($"packer.write_packed_bools({string.Join(", ", args)});");
                break;
            }

            case CallBase:
                // Base steps are inlined during analysis, so this shouldn't normally appear
                // in the final step list. If it does, emit a FIXME.
                w.Fixme("CallBase should have been inlined during analysis");
                break;

            case TimestampNow:
                // TimestampNow (e.g. decodedTime = UtcNow) runs only in unpack, not in pack.
                break;

            case ConditionalBlock cb:
            {
                using (w.BeginIf($"self.{cb.ConditionField}"))
                {
                    foreach (SerializationStep inner in cb.Steps)
                        EmitPackStep(w, inner, fields);
                }
                break;
            }
        }
    }

    private static void EmitUnpackStep(RustWriter w, SerializationStep step, List<FieldDescriptor> fields)
    {
        switch (step)
        {
            case WriteField wf:
                w.Line(UnpackLine(wf.FieldName, wf.Kind, fields));
                break;

            case PackedBools pb:
            {
                var fieldNames = pb.FieldNames.ToList();
                while (fieldNames.Count < 8)
                    fieldNames.Add("_");

                w.Line("let __p = unpacker.read_packed_bools();");
                for (int i = 0; i < 8; i++)
                {
                    if (fieldNames[i] != "_")
                        w.Line($"self.{fieldNames[i]} = __p.bit{i};");
                }
                break;
            }

            case CallBase:
                w.Fixme("CallBase should have been inlined during analysis");
                break;

            case ConditionalBlock cb:
            {
                using (w.BeginIf($"self.{cb.ConditionField}"))
                {
                    foreach (SerializationStep inner in cb.Steps)
                        EmitUnpackStep(w, inner, fields);
                }
                break;
            }

            case TimestampNow ts:
                w.Line($"self.{ts.FieldName} = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos() as i128;");
                break;
        }
    }

    private static string PackLine(string name, FieldKind kind) => kind switch
    {
        FieldKind.Pod => $"packer.write(&self.{name});",
        FieldKind.Bool => $"packer.write_bool(self.{name});",
        FieldKind.String => $"packer.write_str(self.{name}.as_ref().map(|s| s.as_str()));",
        FieldKind.Enum => $"packer.write_object_required(&mut self.{name});",
        FieldKind.FlagsEnum => $"packer.write_object_required(&mut self.{name});",
        FieldKind.Nullable => $"packer.write_option(self.{name}.as_ref());",
        FieldKind.Object => $"packer.write_object(self.{name}.as_mut());",
        FieldKind.ObjectRequired => $"packer.write_object_required(&mut self.{name});",
        FieldKind.ValueList => $"packer.write_value_list(Some(&self.{name}));",
        FieldKind.EnumValueList => $"packer.write_enum_value_list(Some(&self.{name}));",
        FieldKind.ObjectList => $"packer.write_object_list(Some(&mut self.{name}[..]));",
        FieldKind.PolymorphicList => $"packer.write_polymorphic_list(Some(&mut self.{name}[..]));",
        FieldKind.StringList => WriteStringListPack(name),
        FieldKind.NestedValueList => $"packer.write_nested_value_list(Some(&self.{name}));",
        _ => $"// FIXME: Unknown FieldKind {kind} for {name}",
    };

    private static string UnpackLine(string name, FieldKind kind, List<FieldDescriptor> fields) => kind switch
    {
        FieldKind.Pod => $"self.{name} = unpacker.read();",
        FieldKind.Bool => $"self.{name} = unpacker.read_bool();",
        FieldKind.String => $"self.{name} = unpacker.read_str();",
        FieldKind.Enum => $"unpacker.read_object_required(&mut self.{name});",
        FieldKind.FlagsEnum => $"unpacker.read_object_required(&mut self.{name});",
        FieldKind.Nullable => $"self.{name} = unpacker.read_option();",
        FieldKind.Object => UnpackObjectLine(name, fields),
        FieldKind.ObjectRequired => $"unpacker.read_object_required(&mut self.{name});",
        FieldKind.ValueList => $"self.{name} = unpacker.read_value_list();",
        FieldKind.EnumValueList => $"self.{name} = unpacker.read_enum_value_list();",
        FieldKind.ObjectList => $"self.{name} = unpacker.read_object_list();",
        FieldKind.PolymorphicList => UnpackPolymorphicListLine(name, fields),
        FieldKind.StringList => $"self.{name} = unpacker.read_string_list();",
        FieldKind.NestedValueList => $"self.{name} = unpacker.read_nested_value_list();",
        _ => $"// FIXME: Unknown FieldKind {kind} for {name}",
    };

    private static string WriteStringListPack(string name)
    {
        // String lists need a temporary conversion to Vec<Option<&str>>
        return $"let __strs: Vec<Option<&str>> = self.{name}.iter().map(|s| s.as_deref()).collect();\n" +
               $"        packer.write_string_list(Some(&__strs));";
    }

    private static string UnpackObjectLine(string name, List<FieldDescriptor> fields)
    {
        FieldDescriptor? field = fields.FirstOrDefault(f => f.RustName == name);
        if (field == null)
            return $"self.{name} = unpacker.read_object::<_>();";

        // Extract the inner type name from Option<TypeName>
        string rustType = field.RustType;
        if (rustType.StartsWith("Option<") && rustType.EndsWith(">"))
            rustType = rustType[7..^1];

        return $"self.{name} = unpacker.read_object::<{rustType}>();";
    }

    private static string UnpackPolymorphicListLine(string name, List<FieldDescriptor> fields)
    {
        FieldDescriptor? field = fields.FirstOrDefault(f => f.RustName == name);
        if (field == null)
            return $"self.{name} = unpacker.read_polymorphic_list(unimplemented_decode);";

        // Extract the element type from Vec<TypeName>
        string rustType = field.RustType;
        if (rustType.StartsWith("Vec<") && rustType.EndsWith(">"))
            rustType = rustType[4..^1];

        string decodeFn = "decode_" + rustType.HumanizeField();
        return $"self.{name} = unpacker.read_polymorphic_list({decodeFn});";
    }

    // ── ExplicitLayout (PodStruct) pack/unpack ───────────────────

    /// <summary>Emits pack body for ExplicitLayout structs (field-by-field with offsets).</summary>
    public static void EmitExplicitPack(RustWriter w, List<FieldDescriptor> fields, int paddingBytes)
    {
        foreach (FieldDescriptor field in fields)
        {
            if (field.Kind == FieldKind.Bool)
                w.Line($"packer.write_bool(self.{field.RustName} != 0);");
            else if (field.Kind is FieldKind.Enum or FieldKind.FlagsEnum)
                w.Line($"packer.write_object_required(&mut self.{field.RustName});");
            else
                w.Line($"packer.write(&self.{field.RustName});");
        }

        if (paddingBytes > 0)
            w.Line($"packer.write(&self._padding);");
    }

    /// <summary>Emits unpack body for ExplicitLayout structs.</summary>
    public static void EmitExplicitUnpack(RustWriter w, List<FieldDescriptor> fields, int paddingBytes)
    {
        foreach (FieldDescriptor field in fields)
        {
            if (field.Kind == FieldKind.Bool)
                w.Line($"self.{field.RustName} = unpacker.read_bool() as u8;");
            else if (field.Kind is FieldKind.Enum or FieldKind.FlagsEnum)
            {
                string rustType = field.RustType;
                w.Line($"self.{field.RustName} = {{ let mut x = {rustType}::default(); unpacker.read_object_required(&mut x); x }};");
            }
            else
                w.Line($"self.{field.RustName} = unpacker.read();");
        }

        if (paddingBytes > 0)
            w.Line($"self._padding.copy_from_slice(&unpacker.access::<u8>({paddingBytes}));");
    }
}
