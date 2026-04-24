using System.Globalization;
using System.Linq;
using NotEnoughLogs;
using SharedTypeGenerator.Analysis;
using SharedTypeGenerator.IR;
using SharedTypeGenerator.Logging;

namespace SharedTypeGenerator.Emission;

public partial class RustEmitter
{
    private void EmitValueEnum(TypeDescriptor type)
    {
        if (type.EnumMembers == null || type.RustUnderlyingType == null)
        {
            _logger.LogWarning(
                LogCategory.Fixme,
                $"Skipping value enum {type.CSharpName}: missing EnumMembers or RustUnderlyingType in IR.");
            return;
        }

        string rustType = type.RustUnderlyingType;
        string name = type.RustName;

        // Enums used as keys or in comparisons need PartialEq
        using (_w.BeginEnum(name, rustType, "PartialEq"))
        {
            foreach (EnumMember member in type.EnumMembers)
                _w.EnumMemberWithValue(member.Name, member.Value.ToString()!, isDefault: member.IsDefault);
        }

        // MemoryPackable impl — decode without transmute: invalid host values must not panic (Rust 1.94+
        // treats invalid `repr` enum bit patterns as immediate UB/panic on transmute).
        _w.BlankLine();
        using (_w.BeginTraitImpl("MemoryPackable", name))
        {
            using (_w.BeginMethod("pack", "", null, ["&mut self", "packer: &mut MemoryPacker<'_>"], isPublic: false))
                _w.Line($"packer.write(&(*self as {rustType}));");
            using (_w.BeginMethod("unpack", "Result<(), WireDecodeError>", ["P: MemoryPackerEntityPool"], ["&mut self", "unpacker: &mut MemoryUnpacker<'_, '_, P>"], isPublic: false))
                EmitValueEnumUnpackMatch(name, rustType, type.EnumMembers);
        }

        // EnumRepr impl
        _w.BlankLine();
        using (_w.BeginTraitImpl("EnumRepr", name))
        {
            using (_w.BeginMethod("as_i32", "i32", null, ["self"], isPublic: false))
                _w.Line("self as i32");
            using (_w.BeginMethod("from_i32", "Self", null, ["i: i32"], isPublic: false))
                EmitValueEnumFromI32Match(name, type.EnumMembers);
        }

        // No bytemuck `Pod` / `Zeroable` impls: restricted-variant enums have invalid bit patterns
        // (e.g. byte 4 for a 0..=3 `ShadowCastMode`). Reading those via `pod_read_unaligned` is UB,
        // and on Rust 1.94+ the resulting enum-validity trap aborts the process. The wire path goes
        // through `MemoryPackable::unpack`, which validates bytes via the `match` arms above.
    }

    /// <summary>Emits <c>*self = match raw { ... }</c> for value enum wire decode.</summary>
    private void EmitValueEnumUnpackMatch(string enumRustName, string rustType, List<EnumMember> members)
    {
        EnumMember defaultMember = members.First(static m => m.IsDefault);
        string defaultVariant = defaultMember.Name.HumanizeVariant();

        _w.Line($"let raw = unpacker.read::<{rustType}>()?;");
        _w.Line("*self = match raw {");
        foreach (EnumMember member in members)
        {
            string lit = FormatRustPatternLiteralForUnderlying(member.Value, rustType);
            _w.Line($"    {lit} => Self::{member.Name.HumanizeVariant()},");
        }

        _w.Line("    _ => {");
        _w.Line(
            $"        trace!(\"invalid {enumRustName} wire value {{}}; using default\", raw);");
        _w.Line($"        Self::{defaultVariant}");
        _w.Line("    }");
        _w.Line("};");
        _w.Line("Ok(())");
    }

    /// <summary>Emits <c>match i { ... }</c> for <see cref="EnumRepr"/> without transmute.</summary>
    private void EmitValueEnumFromI32Match(string enumRustName, List<EnumMember> members)
    {
        EnumMember defaultMember = members.First(static m => m.IsDefault);
        string defaultVariant = defaultMember.Name.HumanizeVariant();

        _w.Line("match i {");
        foreach (EnumMember member in members)
        {
            long v = Convert.ToInt64(member.Value, CultureInfo.InvariantCulture);
            if (v < int.MinValue || v > int.MaxValue)
            {
                _logger.LogWarning(
                    LogCategory.Fixme,
                    $"Enum {enumRustName} member {member.Name} value {v} is outside i32; skipping from_i32 arm.");
                continue;
            }

            int arm = (int)v;
            _w.Line($"    {arm} => Self::{member.Name.HumanizeVariant()},");
        }

        _w.Line("    _ => {");
        _w.Line(
            $"        trace!(\"invalid {enumRustName} discriminant {{}}; using default\", i);");
        _w.Line($"        Self::{defaultVariant}");
        _w.Line("    }");
        _w.Line("}");
    }

    /// <summary>Rust pattern literal matching <paramref name="rustType"/> (underlying storage).</summary>
    private static string FormatRustPatternLiteralForUnderlying(object value, string rustType)
    {
        IFormatProvider inv = CultureInfo.InvariantCulture;
        return rustType switch
        {
            "u8" => Convert.ToByte(value, inv).ToString(inv),
            "i8" => Convert.ToSByte(value, inv).ToString(inv),
            "u16" => $"{Convert.ToUInt16(value, inv)}u16",
            "i16" => $"{Convert.ToInt16(value, inv)}",
            "u32" => $"{Convert.ToUInt32(value, inv)}u32",
            "i32" => $"{Convert.ToInt32(value, inv)}",
            "u64" => $"{Convert.ToUInt64(value, inv)}u64",
            "i64" => $"{Convert.ToInt64(value, inv)}i64",
            _ => Convert.ToInt64(value, inv).ToString(inv),
        };
    }

    private void EmitFlagsEnum(TypeDescriptor type)
    {
        if (type.EnumMembers == null || type.RustUnderlyingType == null)
        {
            _logger.LogWarning(
                LogCategory.Fixme,
                $"Skipping flags enum {type.CSharpName}: missing EnumMembers or RustUnderlyingType in IR.");
            return;
        }

        string rustType = type.RustUnderlyingType;
        string name = type.RustName;

        // repr(transparent) struct
        _w.TransparentStruct(name, rustType);

        _w.BlankLine();
        using (_w.BeginImpl(name))
        {
            foreach (EnumMember member in type.EnumMembers)
            {
                int val = Convert.ToInt32(member.Value, CultureInfo.InvariantCulture);
                if (val == 0) continue;
                string constName = member.Name.HumanizeField().ToUpperInvariant();
                _w.Line($"pub const {constName}: {rustType} = {val};");
            }
            foreach (EnumMember member in type.EnumMembers)
            {
                int val = Convert.ToInt32(member.Value, CultureInfo.InvariantCulture);
                if (val == 0) continue;
                string methodName = member.Name.HumanizeField();
                string constName = member.Name.HumanizeField().ToUpperInvariant();
                _w.Line($"pub fn {methodName}(&self) -> bool {{ self.0 & Self::{constName} != 0 }}");
            }
        }

        // MemoryPackable impl
        _w.BlankLine();
        using (_w.BeginTraitImpl("MemoryPackable", name))
        {
            using (_w.BeginMethod("pack", "", null, ["&mut self", "packer: &mut MemoryPacker<'_>"], isPublic: false))
                _w.Line("packer.write(&self.0);");
            using (_w.BeginMethod("unpack", "Result<(), WireDecodeError>", ["P: MemoryPackerEntityPool"], ["&mut self", "unpacker: &mut MemoryUnpacker<'_, '_, P>"], isPublic: false))
            {
                _w.Line("self.0 = unpacker.read()?;");
                _w.Line("Ok(())");
            }
        }
    }
}
