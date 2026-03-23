using System.Globalization;
using System.Text;
using UnityShaderConverter.Analysis;
using UnityShaderParser.ShaderLab;

namespace UnityShaderConverter.Emission;

/// <summary>Prepends a <c>MaterialUniform</c> struct and <c>@group(1)</c> bindings so WGSL matches generated Rust layout.</summary>
public static class WgslMaterialUniformInjector
{
    /// <summary>Inserts the material block at the top of merged WGSL when there are properties.</summary>
    public static string PrependMaterialBlock(string wgslBody, IReadOnlyList<ShaderPropertyRecord> properties)
    {
        if (properties.Count == 0)
            return wgslBody;

        var sb = new StringBuilder();
        sb.AppendLine("// --- Material block (UnityShaderConverter) @group(1) ---");
        sb.AppendLine("struct MaterialUniform {");
        (IReadOnlyList<Std140UniformField> uniformFields, _) = MaterialUniformStd140Layout.Build(properties);
        MaterialUniformStd140Layout.AppendWgslStructBody(sb, uniformFields);
        sb.AppendLine("}");
        sb.AppendLine();
        sb.AppendLine("@group(1) @binding(0) var<uniform> material: MaterialUniform;");
        uint binding = 1;
        foreach (ShaderPropertyRecord p in properties)
        {
            if (!MaterialUniformStd140Layout.IsTexturePropertyKind(p.Kind))
                continue;
            string stem = WgslTextureVarStem(p.Name);
            string texDecl = WgslTextureDeclaration(p.Kind, stem);
            sb.Append("@group(1) @binding(").Append(binding.ToString(CultureInfo.InvariantCulture))
                .Append(") var ").Append(texDecl).AppendLine(";");
            binding++;
            sb.Append("@group(1) @binding(").Append(binding.ToString(CultureInfo.InvariantCulture))
                .Append(") var ").Append(stem).AppendLine("_sampler: sampler;");
            binding++;
        }

        sb.AppendLine("// --- End material block ---");
        sb.AppendLine();
        sb.Append(wgslBody);
        return sb.ToString();
    }

    /// <summary>WGSL texture variable declaration without <c>@group</c> / <c>@binding</c>.</summary>
    public static string WgslTextureDeclaration(ShaderPropertyKind kind, string stem) =>
        kind switch
        {
            ShaderPropertyKind.TextureCube => $"{stem}: texture_cube<f32>",
            ShaderPropertyKind.Texture2DArray => $"{stem}: texture_2d_array<f32>",
            ShaderPropertyKind.TextureCubeArray => $"{stem}: texture_cube_array<f32>",
            ShaderPropertyKind.Texture3D or ShaderPropertyKind.Texture3DArray => $"{stem}: texture_3d<f32>",
            ShaderPropertyKind.Texture2D or ShaderPropertyKind.TextureAny => $"{stem}: texture_2d<f32>",
            _ => $"{stem}: texture_2d<f32>",
        };

    private static string WgslTextureVarStem(string uniformName)
    {
        string f = WgslFieldName(uniformName);
        foreach (char c in Path.GetInvalidFileNameChars())
            f = f.Replace(c, '_');
        if (f.Length > 0 && char.IsDigit(f[0]))
            f = "u_" + f;
        return f;
    }

    private static string WgslFieldName(string uniformName)
    {
        string n = uniformName.TrimStart('_');
        if (n.Length == 0)
            return "unnamed";
        return char.ToLowerInvariant(n[0]) + n[1..].Replace(' ', '_');
    }
}
