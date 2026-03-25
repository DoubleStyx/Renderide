using UnityShaderConverter.Emission;

namespace UnityShaderConverter.Tests;

/// <summary>Tests for <see cref="WgslNagaCompatibilityTransforms"/>.</summary>
public sealed class WgslNagaCompatibilityTransformsTests
{
    [Fact]
    public void NormalizeHostSharableArraySizes_ReplacesI32ArrayCount()
    {
        const string src = """
            struct S { x : array<vec4<f32>, i32(4)>, }
            """;
        string o = WgslNagaCompatibilityTransforms.NormalizeHostSharableArraySizes(src);
        Assert.Contains("array<vec4<f32>, 4>", o, StringComparison.Ordinal);
        Assert.DoesNotContain("i32(4)", o, StringComparison.Ordinal);
    }

    [Fact]
    public void NormalizeHostSharableArraySizes_ReplacesNestedArrayI32()
    {
        const string src = "a : array<array<vec4<f32>, i32(4)>, i32(2)>";
        string o = WgslNagaCompatibilityTransforms.NormalizeHostSharableArraySizes(src);
        Assert.Contains("array<array<vec4<f32>, 4>, 2>", o, StringComparison.Ordinal);
    }

    [Fact]
    public void NormalizeHostSharableArraySizes_ReplacesU32ArrayCount()
    {
        const string src = "x : array<u32, u32(8)>";
        string o = WgslNagaCompatibilityTransforms.NormalizeHostSharableArraySizes(src);
        Assert.Contains("array<u32, 8>", o, StringComparison.Ordinal);
    }
}
