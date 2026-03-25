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
    /// <remarks>
    /// Shaders matching here still go through the rest of generation unless they also match <see cref="ShaderGenerationExcludeGlobPatterns"/>:
    /// without <c>--skip-slang</c>, the converter expects existing WGSL on disk for each pass (see converter runner pass loop).
    /// </remarks>
    [JsonPropertyName("slangExcludeGlobPatterns")]
    public List<string> SlangExcludeGlobPatterns { get; set; } = new();

    /// <summary>
    /// Glob patterns (relative to Renderide root) for <c>.shader</c> files that skip WGSL/Rust emission entirely (no failures, no output folder).
    /// </summary>
    /// <remarks>
    /// Use for asset families that are out of scope (e.g. experimental fur stacks) without removing them from shared input trees.
    /// This differs from <see cref="SlangExcludeGlobPatterns"/>, which only disables <c>slangc</c> while still requiring WGSL files.
    /// </remarks>
    [JsonPropertyName("shaderGenerationExcludeGlobPatterns")]
    public List<string> ShaderGenerationExcludeGlobPatterns { get; set; } = new();

    /// <summary>Maximum Cartesian variant count per shader before the converter fails.</summary>
    [JsonPropertyName("maxVariantCombinationsPerShader")]
    public int MaxVariantCombinationsPerShader { get; set; } = 512;

    /// <summary>When true, maps <c>multi_compile</c> keywords to <c>[vk::constant_id]</c> bools for a single WGSL per pass.</summary>
    [JsonPropertyName("enableSlangSpecialization")]
    public bool EnableSlangSpecialization { get; set; } = true;

    /// <summary>
    /// Glob patterns (relative to Renderide root) for shaders that skip specialization (no <c>USC_*</c> / runtime <c>#if</c> rewrite).
    /// Use for ShaderLab whose <c>#if</c> nests resources or varyings in ways that break a single merged TU (e.g. cube vs 2D paths).
    /// </summary>
    [JsonPropertyName("slangSpecializationExcludeGlobPatterns")]
    public List<string> SlangSpecializationExcludeGlobPatterns { get; set; } = new();

    /// <summary>Maximum <c>vk::constant_id</c> specialization bools emitted per shader.</summary>
    [JsonPropertyName("maxSpecializationConstants")]
    public int MaxSpecializationConstants { get; set; } = 8;

    /// <summary>
    /// When true, passes <c>-warnings-disable</c> for common noisy Slang diagnostics (including implicit global shader parameters
    /// <c>39019</c> and unreachable-code <c>41000</c> in Unity shadow helpers) and strips <c>warning[E…]</c> lines from logged
    /// <c>slangc</c> stderr; errors are never stripped.
    /// </summary>
    [JsonPropertyName("suppressSlangWarnings")]
    public bool SuppressSlangWarnings { get; set; } = true;

    /// <summary>Additional <c>-I</c> directories for <c>slangc</c> after Unity CGIncludes roots (patches, version-specific trees).</summary>
    [JsonPropertyName("extraSlangIncludeDirectories")]
    public List<string> ExtraSlangIncludeDirectories { get; set; } = new();

    /// <summary>Rust / WGSL <c>@group</c> index for clustered scene bindings (must match <c>RenderideClusterForward.slang</c>).</summary>
    [JsonPropertyName("sceneBindGroupIndex")]
    public uint SceneBindGroupIndex { get; set; } = 1;

    /// <summary>Rust constant and WGSL <c>@group</c> for <c>MaterialUniform</c> and property textures.</summary>
    [JsonPropertyName("materialBindGroupIndex")]
    public uint MaterialBindGroupIndex { get; set; } = 2;

    /// <summary>
    /// When true, prepends a generated <c>MaterialUniform</c> + texture block to merged WGSL so it matches
    /// <see cref="MaterialBindGroupIndex"/> and Rust <c>create_material_bind_group_layout</c>. When false (default),
    /// WGSL is left as Slang emitted it (Unity globals typically stay at <c>@group(0)</c>); re-runs still strip any prior injected block.
    /// </summary>
    /// <remarks>
    /// Disabling avoids duplicate/conflicting bind groups when shaders already declare uniforms in Slang output.
    /// Aligning host bind groups with WGSL when this is false may require runtime_slang or follow-up Rust emission changes.
    /// </remarks>
    [JsonPropertyName("injectMaterialUniformBlockWgsl")]
    public bool InjectMaterialUniformBlockWgsl { get; set; }
}
