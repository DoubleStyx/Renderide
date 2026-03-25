using UnityShaderConverter.Analysis;
using UnityShaderConverter.Emission;
using UnityShaderConverter.Variants;
using UnityShaderParser.ShaderLab;

namespace UnityShaderConverter.Tests;

/// <summary>Tests that generated <c>VariantKey::default</c> matches <see cref="SpecializationAxis.DefaultConstantValue"/>.</summary>
public sealed class RustEmitterVariantKeyDefaultsTests
{
    [Fact]
    public void EmitShaderMaterialRs_VariantKeyDefault_MirrorsSpecializationAxisDefaults()
    {
        var pass = new ShaderPassDocument
        {
            PassName = null,
            PassIndex = 0,
            ProgramSource = "#pragma vertex vert\n#pragma fragment frag\n",
            Pragmas = Array.Empty<string>(),
            VertexEntry = "vert",
            FragmentEntry = "frag",
            RenderStateSummary = "",
            FixedFunctionState = RenderStateExtractor.Extract(Array.Empty<ShaderLabCommandNode>(), new Dictionary<string, string>()),
        };
        var doc = new ShaderFileDocument
        {
            SourcePath = "t.shader",
            ShaderName = "Test/KeyDefault",
            Properties = Array.Empty<ShaderPropertyRecord>(),
            SubShaderTags = new Dictionary<string, string>(),
            Passes = new[] { pass },
            MultiCompilePragmas = Array.Empty<string>(),
            AnalyzerWarnings = Array.Empty<string>(),
            TotalSubShaderCount = 1,
        };
        var axes = new[]
        {
            new SpecializationAxis(0, "A", "USC_A", "a", true),
            new SpecializationAxis(1, "B", "USC_B", "b", false),
        };
        string rs = RustEmitter.EmitShaderMaterialRs(
            doc,
            "t.shader",
            1,
            axes,
            new[] { PassVertexLayout.Empty });

        Assert.Contains("a: true,", rs, StringComparison.Ordinal);
        Assert.Contains("b: false,", rs, StringComparison.Ordinal);
        Assert.Contains("WGSL override initializer defaults", rs, StringComparison.Ordinal);
    }
}
