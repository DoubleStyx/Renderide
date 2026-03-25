namespace UnityShaderConverter.Emission;

/// <summary>
/// Rewrites HLSL preprocessor branches for <c>multi_compile</c> / <c>shader_feature</c> specialization axes
/// so pipeline <c>[vk::constant_id]</c> bools drive shading. Delegates to
/// <see cref="SpecializationConditionalToRuntimeRewriter"/> (runtime <c>if (USC_*)</c> instead of
/// <c>defined(USC_*)</c>, which is always false for the preprocessor).
/// </summary>
public static class HlslSpecializationPreprocessor
{
    /// <summary>
    /// Rewrites axis-only <c>#if defined(KEYWORD)</c> chains into HLSL <c>if (USC_*)</c> / <c>else</c> for the entire
    /// <paramref name="source"/> (used by tests and snippets). Full pass programs should use the overload that takes
    /// vertex and fragment entry names so only function bodies are rewritten (HLSL forbids <c>if</c> at file scope).
    /// </summary>
    /// <param name="source">Pass program source (after <c>#pragma</c> stripping is not required).</param>
    /// <param name="keywordToSlangId">Unity keyword → Slang specialization bool name (e.g. <c>NORMALMAP</c> → <c>USC_NORMALMAP</c>).</param>
    public static string Rewrite(string source, IReadOnlyDictionary<string, string> keywordToSlangId) =>
        SpecializationConditionalToRuntimeRewriter.Rewrite(source, keywordToSlangId);

    /// <summary>
    /// Rewrites axis-only preprocessor branches only inside the named entry function bodies.
    /// </summary>
    /// <param name="vertexEntry">Vertex shader entry name from <c>#pragma vertex</c>, or null/empty to skip.</param>
    /// <param name="fragmentEntry">Fragment shader entry name from <c>#pragma fragment</c>, or null/empty to skip.</param>
    public static string Rewrite(
        string source,
        IReadOnlyDictionary<string, string> keywordToSlangId,
        string? vertexEntry,
        string? fragmentEntry) =>
        SpecializationConditionalToRuntimeRewriter.RewriteInsideShaderEntryFunctions(
            source,
            keywordToSlangId,
            vertexEntry,
            fragmentEntry);
}
