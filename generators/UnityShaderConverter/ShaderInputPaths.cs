namespace UnityShaderConverter;

/// <summary>Maps each discovered <c>.shader</c> file to its scan root and optional nested path segments (e.g. for WGSL layout beside a shader module).</summary>
internal static class ShaderInputPaths
{
    /// <summary>
    /// Finds the longest matching input root directory for <paramref name="shaderPathFull"/> so relative paths mirror the source tree.
    /// </summary>
    /// <returns>The normalized root directory, or <c>null</c> if no input root contains the file.</returns>
    public static string? FindContainingInputRoot(string shaderPathFull, IReadOnlyList<string> inputRoots)
    {
        string full = Path.GetFullPath(shaderPathFull);
        string? best = null;
        int bestLen = -1;
        foreach (string root in inputRoots)
        {
            string r = Path.GetFullPath(root);
            string rel;
            try
            {
                rel = Path.GetRelativePath(r, full);
            }
            catch (ArgumentException)
            {
                continue;
            }

            if (rel.Equals("..", StringComparison.Ordinal) || rel.StartsWith(".." + Path.DirectorySeparatorChar, StringComparison.Ordinal))
                continue;
            if (Path.IsPathRooted(rel))
                continue;
            if (r.Length > bestLen)
            {
                bestLen = r.Length;
                best = r;
            }
        }

        return best;
    }

    /// <summary>
    /// Directory segments (sanitized) under <c>wgsl/</c> for this shader, without the trailing WGSL file name.
    /// </summary>
    public static string GetWgslNestedDirectoryRelativeToWgslRoot(string shaderPathFull, string inputRootFull)
    {
        string rel = Path.GetRelativePath(Path.GetFullPath(inputRootFull), Path.GetFullPath(shaderPathFull));
        rel = rel.Replace('\\', '/');
        string? dir = Path.GetDirectoryName(rel.Replace('/', Path.DirectorySeparatorChar));
        if (string.IsNullOrEmpty(dir))
            return string.Empty;

        var parts = dir.Split(new[] { Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar },
            StringSplitOptions.RemoveEmptyEntries);
        return string.Join('/',
            parts.Select(SanitizePathSegment));
    }

    /// <summary>Replaces characters that are unsafe in directory or file names.</summary>
    public static string SanitizePathSegment(string segment)
    {
        if (segment.Length == 0)
            return "_";
        foreach (char c in Path.GetInvalidFileNameChars())
            segment = segment.Replace(c, '_');
        return segment;
    }
}
