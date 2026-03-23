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

    /// <summary>Computes variant define lists for a parsed shader document.</summary>
    public static IReadOnlyList<IReadOnlyList<string>> Expand(
        ShaderFileDocument document,
        CompilerConfigModel compilerConfig,
        VariantConfigModel? variantOverrides)
    {
        if (variantOverrides?.VariantsByShaderName is not null &&
            variantOverrides.VariantsByShaderName.TryGetValue(document.ShaderName, out List<VariantDefines>? forced) &&
            forced is not null &&
            forced.Count > 0)
        {
            return Linq.Enumerable.ToList(
                Linq.Enumerable.Select(forced, v => (IReadOnlyList<string>)v.Defines));
        }

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

        if (groups.Count == 0)
            return new List<IReadOnlyList<string>> { Array.Empty<string>() };

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
