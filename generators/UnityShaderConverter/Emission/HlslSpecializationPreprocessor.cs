using System.Text;
using System.Text.RegularExpressions;

namespace UnityShaderConverter.Emission;

/// <summary>
/// Rewrites HLSL preprocessor directives so <c>multi_compile</c> / <c>shader_feature</c> keywords
/// refer to the same identifiers as Slang <c>[vk::constant_id]</c> bools (<c>USC_*</c>), enabling
/// a single WGSL module with WGSL <c>override</c> specialization.
/// </summary>
public static class HlslSpecializationPreprocessor
{
    /// <summary>
    /// Replaces <c>#ifdef</c> / <c>#ifndef</c> / <c>defined(...)</c> uses of axis keywords with
    /// <paramref name="keywordToSlangId"/> values. Only whole preprocessor tokens matching a key are rewritten.
    /// </summary>
    /// <param name="source">Pass program source (after <c>#pragma</c> stripping is not required).</param>
    /// <param name="keywordToSlangId">Unity keyword → Slang bool name (e.g. <c>NORMALMAP</c> → <c>USC_NORMALMAP</c>).</param>
    /// <returns>Rewritten source; unchanged if the map is empty.</returns>
    public static string Rewrite(string source, IReadOnlyDictionary<string, string> keywordToSlangId)
    {
        if (keywordToSlangId.Count == 0)
            return source;

        var keys = new List<KeyValuePair<string, string>>(keywordToSlangId);
        keys.Sort(static (a, b) =>
        {
            int len = b.Key.Length.CompareTo(a.Key.Length);
            return len != 0 ? len : string.CompareOrdinal(a.Key, b.Key);
        });

        var sb = new StringBuilder(source);
        foreach (KeyValuePair<string, string> kv in keys)
        {
            string kw = kv.Key;
            string slang = kv.Value;
            if (kw.Length == 0)
                continue;

            ReplaceDefinedCalls(sb, kw, slang);
            ReplaceIfdefLine(sb, "ifdef", kw, slang);
            ReplaceIfdefLine(sb, "ifndef", kw, slang);
        }

        return sb.ToString();
    }

    private static void ReplaceDefinedCalls(StringBuilder sb, string keyword, string slangId)
    {
        string escaped = Regex.Escape(keyword);
        var rx = new Regex(@"\bdefined\s*\(\s*" + escaped + @"\s*\)", RegexOptions.CultureInvariant);
        string replacement = "defined(" + slangId + ")";
        string s = sb.ToString();
        s = rx.Replace(s, replacement);
        sb.Clear();
        sb.Append(s);
    }

    private static void ReplaceIfdefLine(StringBuilder sb, string directive, string keyword, string slangId)
    {
        string escaped = Regex.Escape(keyword);
        var rx = new Regex(
            @"^([ \t]*#\s*" + directive + @"[ \t]+)" + escaped + @"(\s*)$",
            RegexOptions.Multiline | RegexOptions.CultureInvariant);
        string s = sb.ToString();
        s = rx.Replace(s, m => m.Groups[1].Value + slangId + m.Groups[2].Value);
        sb.Clear();
        sb.Append(s);
    }
}
