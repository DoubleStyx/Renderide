using System.Globalization;
using System.Text;
using UnityShaderConverter.Analysis;
using UnityShaderParser.ShaderLab;

namespace UnityShaderConverter.Emission;

/// <summary>
/// Describes one member of the generated <c>MaterialUniform</c> block in both WGSL and Rust, including
/// explicit <c>u32</c> padding rows required for OpenGL/WebGPU std140 alignment.
/// </summary>
public sealed class Std140UniformField
{
    /// <summary>Initializes a non-padding uniform member mapped from a Unity property.</summary>
    /// <param name="wgslFieldName">Lower-cased WGSL struct field identifier.</param>
    /// <param name="rustFieldName">Rust field identifier (may be prefixed for digits).</param>
    /// <param name="wgslType">WGSL type spelling (e.g. <c>vec4&lt;f32&gt;</c>).</param>
    /// <param name="rustType">Rust type spelling (e.g. <c>Vec4</c>, <c>f32</c>).</param>
    /// <param name="byteOffset">Byte offset from the start of the uniform struct.</param>
    public Std140UniformField(string wgslFieldName, string rustFieldName, string wgslType, string rustType, uint byteOffset)
    {
        WgslFieldName = wgslFieldName;
        RustFieldName = rustFieldName;
        WgslType = wgslType;
        RustType = rustType;
        ByteOffset = byteOffset;
        IsPadding = false;
    }

    /// <summary>Initializes a std140 padding slot as <c>u32</c> in both WGSL and Rust.</summary>
    /// <param name="wgslFieldName">Generated padding name (e.g. <c>_pad0</c>).</param>
    /// <param name="rustFieldName">Same as <paramref name="wgslFieldName"/> for struct parity.</param>
    /// <param name="byteOffset">Byte offset of this padding word.</param>
    /// <param name="u32SlotIndex">Monotonic index used only for stable naming.</param>
    public Std140UniformField(string wgslFieldName, string rustFieldName, uint byteOffset, int u32SlotIndex)
    {
        WgslFieldName = wgslFieldName;
        RustFieldName = rustFieldName;
        WgslType = "u32";
        RustType = "u32";
        ByteOffset = byteOffset;
        IsPadding = true;
        U32SlotIndex = u32SlotIndex;
    }

    /// <summary>WGSL struct member name.</summary>
    public string WgslFieldName { get; }

    /// <summary>Rust struct field name.</summary>
    public string RustFieldName { get; }

    /// <summary>WGSL type spelling.</summary>
    public string WgslType { get; }

    /// <summary>Rust type spelling.</summary>
    public string RustType { get; }

    /// <summary>Byte offset from the start of the uniform block.</summary>
    public uint ByteOffset { get; }

    /// <summary>True when this exists only for std140 alignment.</summary>
    public bool IsPadding { get; }

    /// <summary>Monotonic padding index for stable naming.</summary>
    public int U32SlotIndex { get; }
}

/// <summary>Builds matching std140 layout for WGSL <c>MaterialUniform</c> and Rust <c>#[repr(C, align(16))]</c>.</summary>
public static class MaterialUniformStd140Layout
{
    /// <summary>True when the property is a texture slot (excluded from the uniform buffer).</summary>
    public static bool IsTexturePropertyKind(ShaderPropertyKind k) =>
        k is ShaderPropertyKind.Texture2D or ShaderPropertyKind.Texture3D or ShaderPropertyKind.TextureCube
            or ShaderPropertyKind.TextureAny or ShaderPropertyKind.Texture2DArray or ShaderPropertyKind.Texture3DArray
            or ShaderPropertyKind.TextureCubeArray;

    /// <summary>Computes ordered fields and total std140 size (multiple of 16).</summary>
    public static (IReadOnlyList<Std140UniformField> Fields, uint TotalSizeBytes) Build(IReadOnlyList<ShaderPropertyRecord> properties)
    {
        var fields = new List<Std140UniformField>();
        uint offset = 0;
        int padIdx = 0;
        foreach (ShaderPropertyRecord prop in properties)
        {
            if (IsTexturePropertyKind(prop.Kind))
                continue;
            (uint align, uint size) = Std140AlignAndSize(prop.Kind);
            uint aligned = AlignUp(offset, align);
            uint padBytes = aligned - offset;
            AppendPaddingU32(fields, ref padBytes, ref offset, ref padIdx);
            string wgslName = WgslFieldName(prop.Name);
            string rustName = RustFieldName(prop.Name);
            fields.Add(new Std140UniformField(wgslName, rustName, WgslScalarType(prop.Kind), RustScalarType(prop.Kind), offset));
            offset += size;
        }

        uint tailPad = AlignUp(offset, 16) - offset;
        AppendPaddingU32(fields, ref tailPad, ref offset, ref padIdx);
        return (fields, AlignUp(offset, 16));
    }

    /// <summary>Emits WGSL <c>struct MaterialUniform</c> body lines (no surrounding struct keyword).</summary>
    public static void AppendWgslStructBody(StringBuilder sb, IReadOnlyList<Std140UniformField> fields)
    {
        foreach (Std140UniformField f in fields)
            sb.Append("    ").Append(f.WgslFieldName).Append(": ").Append(f.WgslType).AppendLine(",");
    }

    private static void AppendPaddingU32(List<Std140UniformField> fields, ref uint padBytes, ref uint offset, ref int padIdx)
    {
        while (padBytes >= 4)
        {
            string name = "_pad" + padIdx.ToString(CultureInfo.InvariantCulture);
            fields.Add(new Std140UniformField(name, name, offset, padIdx));
            offset += 4;
            padBytes -= 4;
            padIdx++;
        }

        if (padBytes != 0)
            throw new InvalidOperationException("std140 padding must be a multiple of 4 bytes.");
    }

    private static (uint align, uint size) Std140AlignAndSize(ShaderPropertyKind k) =>
        k switch
        {
            ShaderPropertyKind.Float or ShaderPropertyKind.Range => (4u, 4u),
            ShaderPropertyKind.Integer or ShaderPropertyKind.Int => (4u, 4u),
            ShaderPropertyKind.Color or ShaderPropertyKind.Vector => (16u, 16u),
            _ => (4u, 4u),
        };

    private static uint AlignUp(uint value, uint alignment)
    {
        uint m = value % alignment;
        return m == 0 ? value : value + (alignment - m);
    }

    private static string WgslScalarType(ShaderPropertyKind k) =>
        k switch
        {
            ShaderPropertyKind.Float or ShaderPropertyKind.Range => "f32",
            ShaderPropertyKind.Integer or ShaderPropertyKind.Int => "i32",
            ShaderPropertyKind.Color or ShaderPropertyKind.Vector => "vec4<f32>",
            _ => "u32",
        };

    private static string RustScalarType(ShaderPropertyKind k) =>
        k switch
        {
            ShaderPropertyKind.Float or ShaderPropertyKind.Range => "f32",
            ShaderPropertyKind.Integer or ShaderPropertyKind.Int => "i32",
            ShaderPropertyKind.Color or ShaderPropertyKind.Vector => "Vec4",
            _ => "u32",
        };

    private static string WgslFieldName(string uniformName)
    {
        string n = uniformName.TrimStart('_');
        if (n.Length == 0)
            return "unnamed";
        return char.ToLowerInvariant(n[0]) + n[1..].Replace(' ', '_');
    }

    private static string RustFieldName(string uniformName)
    {
        string n = uniformName.TrimStart('_');
        if (n.Length == 0)
            n = "unnamed";
        string lowered = $"{char.ToLowerInvariant(n[0])}{n.AsSpan(1)}";
        if (lowered.Length > 0 && char.IsDigit(lowered[0]))
            return "u_" + lowered;
        return lowered;
    }
}
