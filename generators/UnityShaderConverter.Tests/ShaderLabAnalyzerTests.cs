using UnityShaderConverter.Analysis;

namespace UnityShaderConverter.Tests;

/// <summary>Tests for <see cref="ShaderLabAnalyzer"/>.</summary>
public sealed class ShaderLabAnalyzerTests
{
    /// <summary>Ensures the sample MinimalUnlit shader parses and exposes one pass.</summary>
    [Fact]
    public void TryAnalyze_MinimalUnlit_Succeeds()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TestData", "MinimalUnlit.shader");
        Assert.True(File.Exists(path), $"Missing test file: {path}");
        bool ok = ShaderLabAnalyzer.TryAnalyze(path, out var doc, out var diags, out var errors);
        Assert.True(ok, string.Join("; ", errors) + string.Join("; ", diags));
        Assert.NotNull(doc);
        Assert.Equal("Converter/MinimalUnlit", doc!.ShaderName);
        Assert.Single(doc.Passes);
        Assert.Equal("vert", doc.Passes[0].VertexEntry);
        Assert.Equal("frag", doc.Passes[0].FragmentEntry);
        Assert.Single(doc.Properties);
        Assert.Equal("_Color", doc.Properties[0].Name);
    }

    /// <summary>
    /// Resonite shaders include <c>UnityCG.cginc</c>; the analyzer must resolve vendored CGIncludes from the UnityShaderParser test pack.
    /// </summary>
    [Fact]
    public void TryAnalyze_ResoniteUnlit_WithUnityCgIncludes_Succeeds()
    {
        string renderideRoot = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", ".."));
        string path = Path.Combine(
            renderideRoot,
            "third_party",
            "Resonite.UnityShaders",
            "Assets",
            "Shaders",
            "Common",
            "Unlit.shader");
        Assert.True(File.Exists(path), $"Expected Resonite Unlit shader at {path}");

        bool ok = ShaderLabAnalyzer.TryAnalyze(path, out var doc, out var diags, out var errors);
        Assert.True(ok, string.Join("; ", errors) + string.Join("; ", diags));
        Assert.NotNull(doc);
        Assert.NotEmpty(doc!.Passes);
    }
}
