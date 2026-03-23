namespace UnityShaderConverter.Analysis;

/// <summary>Enumerates <c>.shader</c> files under input roots.</summary>
public static class ShaderDiscovery
{
    /// <summary>Returns distinct shader paths in stable sort order.</summary>
    public static IReadOnlyList<string> Enumerate(IEnumerable<string> inputRoots)
    {
        var set = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (string root in inputRoots)
        {
            if (!Directory.Exists(root))
                continue;
            foreach (string file in Directory.EnumerateFiles(root, "*.shader", SearchOption.AllDirectories))
                set.Add(Path.GetFullPath(file));
        }

        var list = set.ToList();
        list.Sort(StringComparer.Ordinal);
        return list;
    }
}
