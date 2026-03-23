using UnityShaderConverter.Emission;

namespace UnityShaderConverter.Tests;

/// <summary>Tests for <see cref="ShaderNaming"/>.</summary>
public sealed class ShaderNamingTests
{
    /// <summary>Path segments become snake_case with word boundaries preserved.</summary>
    [Fact]
    public void FileStem_Converter_MinimalUnlit_IsSnake()
    {
        Assert.Equal("converter_minimal_unlit", ShaderNaming.FileStem("Converter/MinimalUnlit"));
    }
}
