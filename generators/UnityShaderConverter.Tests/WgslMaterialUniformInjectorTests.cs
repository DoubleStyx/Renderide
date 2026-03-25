using UnityShaderConverter.Emission;

namespace UnityShaderConverter.Tests;

/// <summary>Tests for <see cref="WgslMaterialUniformInjector"/> stripping of prepended material blocks.</summary>
public sealed class WgslMaterialUniformInjectorTests
{
    [Fact]
    public void StripInjectedMaterialBlock_RemovesGroup1Header()
    {
        const string wgsl = """
            // --- Material block (UnityShaderConverter) @group(1) ---
            struct MaterialUniform { x: f32, }
            // --- End material block ---

            // Generated
            fn main() {}
            """;
        string stripped = WgslMaterialUniformInjector.StripInjectedMaterialBlock(wgsl);
        Assert.DoesNotContain("Material block", stripped, StringComparison.Ordinal);
        Assert.Contains("// Generated", stripped, StringComparison.Ordinal);
    }

    [Fact]
    public void StripInjectedMaterialBlock_RemovesGroup2Header()
    {
        const string wgsl = """
            // --- Material block (UnityShaderConverter) @group(2) ---
            struct MaterialUniform { x: f32, }
            // --- End material block ---

            enable f16;
            """;
        string stripped = WgslMaterialUniformInjector.StripInjectedMaterialBlock(wgsl);
        Assert.StartsWith("enable f16;", stripped, StringComparison.Ordinal);
    }

    [Fact]
    public void StripInjectedMaterialBlock_NoHeader_Unchanged()
    {
        const string wgsl = "// Generated\nfn vert() {}\n";
        Assert.Equal(wgsl, WgslMaterialUniformInjector.StripInjectedMaterialBlock(wgsl));
    }
}
