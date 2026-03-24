namespace UnityShaderConverter.Tests;

/// <summary>Content checks for <c>runtime_slang</c> Unity compatibility files.</summary>
public sealed class UnityCompatSlangTests
{
    /// <summary>Ensures platform defines and Slang workarounds remain present in source.</summary>
    [Fact]
    public void UnityCompatSource_ContainsPlatformAndSamplerWorkarounds()
    {
        string path = Path.Combine(
            AppContext.BaseDirectory,
            "..",
            "..",
            "..",
            "..",
            "UnityShaderConverter",
            "runtime_slang",
            "UnityCompat.slang");
        path = Path.GetFullPath(path);
        Assert.True(File.Exists(path), $"Expected {path}");
        string text = File.ReadAllText(path);
        Assert.Contains("SHADER_API_D3D11", text, StringComparison.Ordinal);
        Assert.Contains("SHADER_TARGET", text, StringComparison.Ordinal);
        Assert.Contains("UNITY_SC_JOIN2", text, StringComparison.Ordinal);
        Assert.Contains("samplermainTex", text, StringComparison.Ordinal);
        Assert.Contains("UNITY_PBS_USE_BRDF1", text, StringComparison.Ordinal);
        Assert.Contains("#define clip", text, StringComparison.Ordinal);
    }

    /// <summary>Post-Unity header override must redefine texture macros for Slang token pasting.</summary>
    [Fact]
    public void UnityCompatPostUnitySource_ContainsDeclareOverride()
    {
        string path = Path.Combine(
            AppContext.BaseDirectory,
            "..",
            "..",
            "..",
            "..",
            "UnityShaderConverter",
            "runtime_slang",
            "UnityCompatPostUnity.slang");
        path = Path.GetFullPath(path);
        Assert.True(File.Exists(path), $"Expected {path}");
        string text = File.ReadAllText(path);
        Assert.Contains("UNITY_DECLARE_TEX2D", text, StringComparison.Ordinal);
        Assert.Contains("UNITY_SC_SAMPLER_FOR", text, StringComparison.Ordinal);
        Assert.Contains("#ifdef UNITY_DECLARE_TEX2D", text, StringComparison.Ordinal);
    }
}
