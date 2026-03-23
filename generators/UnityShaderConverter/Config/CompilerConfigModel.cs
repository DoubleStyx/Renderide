using System.Text.Json.Serialization;

namespace UnityShaderConverter.Config;

/// <summary>JSON model for <c>--compiler-config</c> merged with <see cref="DefaultCompilerConfig"/> defaults.</summary>
public sealed class CompilerConfigModel
{
    /// <summary>Glob patterns (paths relative to Renderide root; default <c>**/*.shader</c>) selecting shaders that may invoke <c>slangc</c>.</summary>
    [JsonPropertyName("slangEligibleGlobPatterns")]
    public List<string> SlangEligibleGlobPatterns { get; set; } = new();

    /// <summary>Maximum Cartesian variant count per shader before the converter fails.</summary>
    [JsonPropertyName("maxVariantCombinationsPerShader")]
    public int MaxVariantCombinationsPerShader { get; set; } = 32;
}
