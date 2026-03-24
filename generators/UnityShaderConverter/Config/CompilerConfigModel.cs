using System.Text.Json.Serialization;

namespace UnityShaderConverter.Config;

/// <summary>JSON model for <c>--compiler-config</c> merged with <see cref="DefaultCompilerConfig"/> defaults.</summary>
public sealed class CompilerConfigModel
{
    /// <summary>Glob patterns (paths relative to Renderide root; default <c>**/*.shader</c>) selecting shaders that may invoke <c>slangc</c>.</summary>
    [JsonPropertyName("slangEligibleGlobPatterns")]
    public List<string> SlangEligibleGlobPatterns { get; set; } = new();

    /// <summary>
    /// Glob patterns excluding shaders from <see cref="SlangEligibleGlobPatterns"/> even when they match (parser failures, geometry-only assets, etc.).
    /// </summary>
    [JsonPropertyName("slangExcludeGlobPatterns")]
    public List<string> SlangExcludeGlobPatterns { get; set; } = new();

    /// <summary>Maximum Cartesian variant count per shader before the converter fails.</summary>
    [JsonPropertyName("maxVariantCombinationsPerShader")]
    public int MaxVariantCombinationsPerShader { get; set; } = 512;

    /// <summary>When true, maps <c>multi_compile</c> keywords to <c>[vk::constant_id]</c> bools for a single WGSL per pass.</summary>
    [JsonPropertyName("enableSlangSpecialization")]
    public bool EnableSlangSpecialization { get; set; } = true;

    /// <summary>Maximum <c>vk::constant_id</c> specialization bools emitted per shader.</summary>
    [JsonPropertyName("maxSpecializationConstants")]
    public int MaxSpecializationConstants { get; set; } = 8;

    /// <summary>
    /// When true, passes <c>-warnings-disable</c> for common noisy Slang diagnostics and strips <c>warning[E…]</c> lines from
    /// logged <c>slangc</c> stderr; errors are never stripped.
    /// </summary>
    [JsonPropertyName("suppressSlangWarnings")]
    public bool SuppressSlangWarnings { get; set; } = true;
}
