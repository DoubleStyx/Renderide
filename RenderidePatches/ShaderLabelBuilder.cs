using FrooxEngine;

namespace RenderidePatches;

/// <summary>
/// Builds a single stable logical name for <see cref="Renderite.Shared.ShaderUpload.file" /> from shader metadata.
/// The value is the base shader stem only (no shader keywords / variants) so the native renderer can map it to WGSL.
/// </summary>
internal static class ShaderLabelBuilder
{
    /// <summary>
    /// Produces a label from the source file leaf name (e.g. <c>UI_Unlit</c> from <c>UI_Unlit.shader</c>).
    /// This replaces the host bundle path in <see cref="Renderite.Shared.ShaderUpload.file" />; Renderide matches the stem to native shaders.
    /// </summary>
    /// <param name="shader">The shader whose metadata is read.</param>
    /// <returns>A non-empty stem, or <see langword="null" /> to keep the original bundle path.</returns>
    internal static string? TryBuildShaderLabel(Shader shader)
    {
        if (shader == null)
        {
            return null;
        }

        var metadata = shader.Metadata;
        if (metadata?.SourceFile?.FileName == null)
        {
            return null;
        }

        return StemFromShaderFileName(metadata.SourceFile.FileName);
    }

    /// <summary>
    /// Strips directory segments and a trailing <c>.shader</c> extension when present.
    /// </summary>
    /// <param name="fileName">File name or path from shader source metadata.</param>
    /// <returns>A stem suitable for matching converted shader assets.</returns>
    private static string StemFromShaderFileName(string fileName)
    {
        var leaf = Path.GetFileName(fileName);
        if (leaf.EndsWith(".shader", StringComparison.OrdinalIgnoreCase))
        {
            return leaf[..^".shader".Length];
        }

        return Path.GetFileNameWithoutExtension(leaf);
    }
}
