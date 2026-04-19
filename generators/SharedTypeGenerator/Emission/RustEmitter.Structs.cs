using System.Linq;
using SharedTypeGenerator.Analysis;
using SharedTypeGenerator.IR;

namespace SharedTypeGenerator.Emission;

public partial class RustEmitter
{
    private void EmitPodStruct(TypeDescriptor type)
    {
        string name = type.RustName;

        // Emit fields in offset order for ExplicitLayout so repr(C) matches C# layout
        var orderedFields = type.Fields
            .OrderBy(f => f.ExplicitOffset ?? int.MaxValue)
            .ToList();

        // Padding may be in orderedFields (synthetic _padding at correct offsets) or trailing
        bool hasPaddingFields = orderedFields.Any(f => f.RustName.StartsWith("_padding", StringComparison.Ordinal));
        bool needsTrailingPadding = type.PaddingBytes > 0 && !hasPaddingFields;

        // Struct definition (Clone needed when used in Option/Vec)
        if (type.IsPod)
        {
            using (_w.BeginExternStruct(name, "Clone", "Copy", "Pod", "Zeroable"))
            {
                foreach (FieldDescriptor field in orderedFields)
                    _w.StructMember(field.RustName, field.RustType);
                if (needsTrailingPadding)
                    _w.StructMember("_padding", $"[u8; {type.PaddingBytes}]");
            }
        }
        else
        {
            using (_w.BeginExternStruct(name, "Clone", "Copy"))
            {
                foreach (FieldDescriptor field in orderedFields)
                    _w.StructMember(field.RustName, field.RustType);
                if (needsTrailingPadding)
                    _w.StructMember("_padding", $"[u8; {type.PaddingBytes}]");
            }
        }

        if (type.HostInteropSizeBytes is int hostBytes)
        {
            string prefix = type.RustName.ToScreamingSnakeTypeName();
            _w.DocLine($"Host interop size from C# `Marshal.SizeOf` for `{type.CSharpName}` (SHM row stride).");
            _w.Line($"pub const {prefix}_HOST_ROW_BYTES: usize = {hostBytes};");
            _w.BlankLine();
        }

        // MemoryPackable impl (use same field order as struct for correct wire format)
        _w.BlankLine();
        using (_w.BeginTraitImpl("MemoryPackable", name))
        {
            using (_w.BeginMethod("pack", "", null, ["&mut self", "packer: &mut MemoryPacker<'_>"], isPublic: false))
                PackEmitter.EmitExplicitPack(_w, _logger, type.CSharpName, orderedFields, needsTrailingPadding ? type.PaddingBytes : 0);
            using (_w.BeginMethod("unpack", "Result<(), WireDecodeError>", ["P: MemoryPackerEntityPool"], ["&mut self", "unpacker: &mut MemoryUnpacker<'_, '_, P>"], isPublic: false))
                PackEmitter.EmitExplicitUnpack(_w, _logger, type.CSharpName, orderedFields, needsTrailingPadding ? type.PaddingBytes : 0);
        }

        if (!type.IsPod && type.HostInteropSizeBytes.HasValue)
        {
            string prefix = type.RustName.ToScreamingSnakeTypeName();
            string testFn = $"verify_{type.RustName.HumanizeField()}_host_row_bytes_contract";
            _w.BlankLine();
            _w.Line("#[cfg(test)]");
            _w.Line("#[test]");
            _w.Line($"fn {testFn}() {{");
            _w.Indent();
            _w.Line($"let mut buf = vec![0u8; {prefix}_HOST_ROW_BYTES];");
            _w.Line("let mut packer = MemoryPacker::new(&mut buf);");
            _w.Line($"let mut v = {name}::default();");
            _w.Line("v.pack(&mut packer);");
            _w.Line("assert_eq!(packer.remaining_len(), 0, \"pack must fill host row\");");
            _w.Dedent();
            _w.Line("}");
        }
    }

    private void EmitPackableStruct(TypeDescriptor type)
    {
        string name = type.RustName;

        // Struct definition (Clone needed when used in Option/Vec)
        using (_w.BeginStruct(name, "Clone"))
        {
            foreach (FieldDescriptor field in type.Fields)
                _w.StructMember(field.RustName, field.RustType);
        }

        // MemoryPackable impl
        _w.BlankLine();
        using (_w.BeginTraitImpl("MemoryPackable", name))
        {
            using (_w.BeginMethod("pack", "", null, ["&mut self", "packer: &mut MemoryPacker<'_>"], isPublic: false))
                PackEmitter.EmitPack(_w, _logger, type.CSharpName, type.PackSteps, type.Fields);
            using (_w.BeginMethod("unpack", "Result<(), WireDecodeError>", ["P: MemoryPackerEntityPool"], ["&mut self", "unpacker: &mut MemoryUnpacker<'_, '_, P>"], isPublic: false))
                PackEmitter.EmitUnpack(_w, _logger, type.CSharpName, type.PackSteps, type.Fields, type.UnpackOnlySteps);
        }
    }

    private void EmitGeneralStruct(TypeDescriptor type)
    {
        string name = type.RustName;

        if (type.IsPod)
        {
            using (_w.BeginExternStruct(name, "Clone", "Copy", "Pod", "Zeroable"))
            {
                foreach (FieldDescriptor field in type.Fields)
                    _w.StructMember(field.RustName, field.RustType);
            }
        }
        else
        {
            using (_w.BeginExternStruct(name))
            {
                foreach (FieldDescriptor field in type.Fields)
                    _w.StructMember(field.RustName, field.RustType);
            }
        }

        // Only emit MemoryPackable if the type has pack steps (e.g., Guid)
        if (type.PackSteps.Count > 0)
        {
            _w.BlankLine();
            using (_w.BeginTraitImpl("MemoryPackable", name))
            {
                using (_w.BeginMethod("pack", "", null, ["&mut self", "packer: &mut MemoryPacker<'_>"], isPublic: false))
                    PackEmitter.EmitPack(_w, _logger, type.CSharpName, type.PackSteps, type.Fields);
                using (_w.BeginMethod("unpack", "Result<(), WireDecodeError>", ["P: MemoryPackerEntityPool"], ["&mut self", "unpacker: &mut MemoryUnpacker<'_, '_, P>"], isPublic: false))
                    PackEmitter.EmitUnpack(_w, _logger, type.CSharpName, type.PackSteps, type.Fields, type.UnpackOnlySteps);
            }
        }
    }
}
