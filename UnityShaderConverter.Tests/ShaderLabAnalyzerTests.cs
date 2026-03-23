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
}
