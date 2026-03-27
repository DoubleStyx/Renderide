using Elements.Assets;
using FrooxEngine;

namespace RenderidePatches;

/// <summary>
/// Builds a single stable string for <see cref="Renderite.Shared.ShaderUpload.file" /> from shader metadata and variant state.
/// </summary>
internal static class ShaderLabelBuilder
{
    /// <summary>
    /// Produces a label of the form <c>Stem kw1 kw2 ...</c> where keywords are those enabled in the current variant bitmask.
    /// </summary>
    /// <param name="shader">The shader whose metadata and variant index are read.</param>
    /// <returns>A non-empty label, or <see langword="null" /> to keep the original bundle path.</returns>
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

        var stem = StemFromShaderFileName(metadata.SourceFile.FileName);
        var keywords = metadata.UniqueKeywords;
        if (keywords == null || keywords.Count == 0)
        {
            return stem;
        }

        var variantBits = shader.VariantIndex;
        if (variantBits == null)
        {
            return null;
        }

        var active = new List<string>();
        var index = 0;
        foreach (var keyword in keywords)
        {
            var mask = 1u << index;
            if ((variantBits.Value & mask) != 0)
            {
                active.Add(keyword);
            }

            index++;
        }

        if (active.Count == 0)
        {
            return stem;
        }

        return $"{stem} {string.Join(" ", active)}";
    }

    /// <summary>
    /// Strips directory segments and a trailing <c>.shader</c> extension when present.
    /// </summary>
    /// <param name="fileName">File name or path from <see cref="ShaderSourceFile.FileName" />.</param>
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
