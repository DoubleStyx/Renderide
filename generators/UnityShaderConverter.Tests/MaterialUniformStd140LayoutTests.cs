using UnityShaderConverter.Analysis;
using UnityShaderConverter.Emission;
using UnityShaderParser.ShaderLab;

namespace UnityShaderConverter.Tests;

/// <summary>Tests for <see cref="MaterialUniformStd140Layout"/>.</summary>
public sealed class MaterialUniformStd140LayoutTests
{
    /// <summary>A scalar before a <c>vec4</c> inserts u32 padding to 16-byte alignment.</summary>
    [Fact]
    public void Build_FloatBeforeVec4_InsertsPadding()
    {
        var props = new List<ShaderPropertyRecord>
        {
            new()
            {
                Name = "_Roughness",
                DisplayLabel = "",
                Kind = ShaderPropertyKind.Float,
                DefaultSummary = "0.5",
            },
            new()
            {
                Name = "_Color",
                DisplayLabel = "",
                Kind = ShaderPropertyKind.Color,
                DefaultSummary = "(1,1,1,1)",
            },
        };

        (IReadOnlyList<Std140UniformField> fields, uint total) = MaterialUniformStd140Layout.Build(props);
        Assert.True(total >= 32);
        Assert.Contains(fields, f => f.IsPadding && f.RustFieldName.StartsWith("_pad", StringComparison.Ordinal));
        Assert.Equal(2, fields.Count(f => !f.IsPadding));
    }
}
