using System.Text.RegularExpressions;
using UnityShaderConverter.Analysis;
using UnityShaderConverter.Config;
using Linq = System.Linq;

namespace UnityShaderConverter.Variants;

/// <summary>Builds preprocessor define lists per shader from <c>#pragma multi_compile</c> and JSON overrides.</summary>
public static partial class VariantExpander
{
    [GeneratedRegex(@"^\s*#\s*pragma\s+multi_compile(?:_local)?\s+(.+)$", RegexOptions.IgnoreCase)]
    private static partial Regex MultiCompileRegex();

    [GeneratedRegex(@"^\s*#\s*pragma\s+shader_feature(?:_local)?\s+(.+)$", RegexOptions.IgnoreCase)]
    private static partial Regex ShaderFeatureRegex();

    /// <summary>Grouped <c>multi_compile</c> keyword options and their Cartesian product size (for diagnostics).</summary>
    /// <param name="Groups">Each inner list is one pragma’s keyword options (empty string = <c>_</c> slot).</param>
    /// <param name="Product">Product of group sizes; 1 when there are no groups.</param>
    public sealed record MultiCompileAnalysis(IReadOnlyList<IReadOnlyList<string>> Groups, long Product);

    /// <summary>Collects multi_compile groups and product without allocating every combination.</summary>
    public static MultiCompileAnalysis AnalyzeMultiCompileGroups(ShaderFileDocument document)
    {
        List<List<string>> groups = CollectGroups(document);
        if (groups.Count == 0)
            return new MultiCompileAnalysis(Array.Empty<IReadOnlyList<string>>(), 1);

        long product = 1;
        foreach (List<string> g in groups)
        {
            checked
            {
                product *= g.Count;
            }
        }

        return new MultiCompileAnalysis(
            groups.Select(g => (IReadOnlyList<string>)g).ToList(),
            product);
    }

    /// <summary>True when <c>--variant-config</c> supplies explicit variants for this shader name.</summary>
    public static bool HasForcedVariantConfig(ShaderFileDocument document, VariantConfigModel? variantOverrides) =>
        variantOverrides?.VariantsByShaderName is not null &&
        variantOverrides.VariantsByShaderName.TryGetValue(document.ShaderName, out List<VariantDefines>? forced) &&
        forced is not null &&
        forced.Count > 0;

    /// <summary>
    /// First Cartesian combination (index 0 from each group): used when retrying Slang without specialization.
    /// Does not allocate the full cross product. Throws if the product exceeds <see cref="CompilerConfigModel.MaxVariantCombinationsPerShader"/>.
    /// </summary>
    public static IReadOnlyList<string> GetFirstCartesianVariantDefines(
        ShaderFileDocument document,
        CompilerConfigModel compilerConfig,
        VariantConfigModel? variantOverrides)
    {
        if (HasForcedVariantConfig(document, variantOverrides))
        {
            List<VariantDefines> forced = variantOverrides!.VariantsByShaderName![document.ShaderName];
            return forced[0].Defines;
        }

        List<List<string>> groups = CollectGroups(document);
        if (groups.Count == 0)
            return Array.Empty<string>();

        long product = 1;
        foreach (List<string> g in groups)
        {
            checked
            {
                product *= g.Count;
            }
        }

        if (product > compilerConfig.MaxVariantCombinationsPerShader)
        {
            throw new InvalidOperationException(
                $"Shader '{document.ShaderName}' would expand to {product} variant combinations (limit {compilerConfig.MaxVariantCombinationsPerShader}). " +
                "Add a --variant-config override or raise maxVariantCombinationsPerShader.");
        }

        var combo = new List<string>();
        foreach (List<string> g in groups)
        {
            string kw = g[0];
            if (kw.Length > 0)
                combo.Add(kw);
        }

        return combo;
    }

    /// <summary>
    /// First Cartesian combination of <c>multi_compile</c> keywords without enforcing <see cref="CompilerConfigModel.MaxVariantCombinationsPerShader"/>.
    /// Used when retrying <c>slangc</c> without specialization so shaders like Fresnel still get a sensible default keyword set.
    /// </summary>
    public static IReadOnlyList<string> GetFirstCartesianVariantDefinesIgnoringProductLimit(
        ShaderFileDocument document,
        VariantConfigModel? variantOverrides)
    {
        if (HasForcedVariantConfig(document, variantOverrides))
        {
            List<VariantDefines> forced = variantOverrides!.VariantsByShaderName![document.ShaderName];
            return forced[0].Defines;
        }

        List<List<string>> groups = CollectGroups(document);
        if (groups.Count == 0)
            return Array.Empty<string>();

        var combo = new List<string>();
        foreach (List<string> g in groups)
        {
            string kw = g[0];
            if (kw.Length > 0)
                combo.Add(kw);
        }

        return combo;
    }

    /// <summary>Computes variant define lists for a parsed shader document.</summary>
    public static IReadOnlyList<IReadOnlyList<string>> Expand(
        ShaderFileDocument document,
        CompilerConfigModel compilerConfig,
        VariantConfigModel? variantOverrides)
    {
        if (HasForcedVariantConfig(document, variantOverrides))
        {
            return Linq.Enumerable.ToList(
                Linq.Enumerable.Select(variantOverrides!.VariantsByShaderName![document.ShaderName], v => (IReadOnlyList<string>)v.Defines));
        }

        List<List<string>> groups = CollectGroups(document);

        if (groups.Count == 0)
            return new List<IReadOnlyList<string>> { Array.Empty<string>() };

        if (compilerConfig.EnableSlangSpecialization)
        {
            return new List<IReadOnlyList<string>> { Array.Empty<string>() };
        }

        long product = 1;
        foreach (List<string> g in groups)
        {
            checked
            {
                product *= g.Count;
            }
        }

        if (product > compilerConfig.MaxVariantCombinationsPerShader)
        {
            throw new InvalidOperationException(
                $"Shader '{document.ShaderName}' would expand to {product} variant combinations (limit {compilerConfig.MaxVariantCombinationsPerShader}). " +
                "Add a --variant-config override or raise maxVariantCombinationsPerShader.");
        }

        var result = new List<IReadOnlyList<string>>();
        var indices = new int[groups.Count];
        while (true)
        {
            var combo = new List<string>();
            for (int i = 0; i < groups.Count; i++)
            {
                string kw = groups[i][indices[i]];
                if (kw.Length > 0)
                    combo.Add(kw);
            }

            result.Add(combo);
            int carry = 1;
            for (int i = groups.Count - 1; i >= 0; i--)
            {
                indices[i] += carry;
                if (indices[i] < groups[i].Count)
                {
                    carry = 0;
                    break;
                }

                indices[i] = 0;
            }

            if (carry != 0)
                break;
        }

        return result;
    }

    /// <summary>True for Unity macro pragmas we do not expand into specialization axes.</summary>
    public static bool IsIgnoredMultiCompilePragmaLine(string line) => ShouldIgnoreMultiCompileLine(line);

    /// <summary>Parses keyword tokens from a <c>#pragma multi_compile</c> or <c>shader_feature</c> line.</summary>
    public static List<string>? ParseMultiCompileKeywords(string line) => TryParseKeywordList(line);

    private static List<List<string>> CollectGroups(ShaderFileDocument document)
    {
        var groups = new List<List<string>>();
        foreach (string line in document.MultiCompilePragmas)
        {
            if (ShouldIgnoreMultiCompileLine(line))
                continue;

            List<string>? opts = TryParseKeywordList(line);
            if (opts is null || opts.Count == 0)
                continue;
            groups.Add(opts);
        }

        return groups;
    }

    private static bool ShouldIgnoreMultiCompileLine(string line)
    {
        string t = line.TrimStart();
        return t.Contains("multi_compile_fog", StringComparison.OrdinalIgnoreCase) ||
               t.Contains("multi_compile_instancing", StringComparison.OrdinalIgnoreCase) ||
               t.Contains("multi_compile_fwdbase", StringComparison.OrdinalIgnoreCase) ||
               t.Contains("multi_compile_fwdadd", StringComparison.OrdinalIgnoreCase) ||
               t.Contains("multi_compile_shadowcaster", StringComparison.OrdinalIgnoreCase) ||
               t.Contains("multi_compile_vertex", StringComparison.OrdinalIgnoreCase) ||
               t.Contains("multi_compile_fragment", StringComparison.OrdinalIgnoreCase);
    }

    private static List<string>? TryParseKeywordList(string line)
    {
        Match m = MultiCompileRegex().Match(line);
        if (!m.Success)
            m = ShaderFeatureRegex().Match(line);
        if (!m.Success)
            return null;

        string rest = m.Groups[1].Value.Trim();
        string[] parts = rest.Split((char[]?)null!, StringSplitOptions.RemoveEmptyEntries);
        var list = new List<string>();
        foreach (string p in parts)
        {
            if (p == "_")
                list.Add("");
            else
                list.Add(p);
        }

        return list;
    }
}
