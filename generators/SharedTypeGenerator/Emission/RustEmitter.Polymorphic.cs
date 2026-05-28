using System.Linq;
using SharedTypeGenerator.Analysis;
using SharedTypeGenerator.IR;
using SharedTypeGenerator.Logging;

namespace SharedTypeGenerator.Emission;

internal sealed partial class RustEmitter
{
    private void EmitPolymorphic(TypeDescriptor type)
    {
        if (type.Variants == null || type.Variants.Count == 0)
        {
            _logger.LogWarning(
                LogCategory.Fixme,
                $"Skipping polymorphic base {type.CSharpName}: no variants in IR (check PolymorphicAnalyzer / inheritance graph).");
            return;
        }

        string unionName = type.RustName;
        var typeNames = type.Variants.Select(v => v.CSharpName).ToList();

        // Discriminant enum: {Name}Types
        using (_w.BeginEnum($"{type.CSharpName}Types", "i32"))
        {
            bool first = true;
            foreach (string name in typeNames)
            {
                _w.EnumMember(name, isDefault: first);
                first = false;
            }
        }

        // Tagged union enum (Clone needed when used in Option/Vec; no Default - variants have payloads)
        using (_w.BeginUnion(type.CSharpName, "Clone"))
        {
            foreach (string name in typeNames)
                _w.EnumVariantWithPayload(name, name);
        }

        // PolymorphicEncode impl
        _w.BlankLine();
        using (_w.BeginTraitImpl("PolymorphicEncode", unionName))
        {
            using (_w.BeginMethod("encode", "", null, ["&mut self", "packer: &mut MemoryPacker<'_>"], isPublic: false))
            {
                _w.Line("match self {");
                for (int i = 0; i < typeNames.Count; i++)
                {
                    string variant = typeNames[i].HumanizeVariant();
                    _w.Line($"    {unionName}::{variant}(x) => {{ packer.write(&{i}i32); x.pack(packer); }}");
                }
                _w.Line("}");
            }
        }

        // decode function
        string decodeFnName = "decode_" + unionName.HumanizeField();
        _w.BlankLine();
        using (_w.BeginMethod(decodeFnName, $"Result<{unionName}, WireDecodeError>", ["P: MemoryPackerEntityPool"], ["unpacker: &mut MemoryUnpacker<'_, '_, P>"], isPublic: true))
        {
            _w.Line("let tag = unpacker.read::<i32>()?;");
            _w.Line("match tag {");
            for (int i = 0; i < typeNames.Count; i++)
            {
                string variant = typeNames[i].HumanizeVariant();
                string payloadType = typeNames[i].HumanizeType();
                _w.Line($"    {i} => Ok({unionName}::{variant}({{ let mut x = {payloadType}::default(); x.unpack(unpacker)?; x }})),");
            }
            _w.Line($"    _ => Err(PolymorphicDecodeError {{ discriminator: tag, union: \"{unionName}\" }}.into()),");
            _w.Line("}");
        }
    }
}
