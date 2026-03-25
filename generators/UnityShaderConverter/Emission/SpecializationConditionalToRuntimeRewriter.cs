using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;

namespace UnityShaderConverter.Emission;

/// <summary>
/// Converts preprocessor <c>#if</c> / <c>#elif</c> chains that only test <c>defined(KEYWORD)</c> for
/// specialization axes into HLSL <c>if</c> / <c>else if</c> / <c>else</c> using the same
/// <c>[vk::constant_id] const bool USC_*</c> symbols. Classic <c>defined(USC_*)</c> is always false
/// because <c>USC_*</c> are not macros; runtime <c>if (USC_*)</c> keeps bodies alive for WGSL <c>override</c>.
/// </summary>
public static class SpecializationConditionalToRuntimeRewriter
{
    private static readonly Regex IfDefLine = new(
        @"^([ \t]*#\s*)ifdef(\s+)(\S+)(\s*)$",
        RegexOptions.Multiline | RegexOptions.CultureInvariant,
        TimeSpan.FromSeconds(1));

    private static readonly Regex IfNDefLine = new(
        @"^([ \t]*#\s*)ifndef(\s+)(\S+)(\s*)$",
        RegexOptions.Multiline | RegexOptions.CultureInvariant,
        TimeSpan.FromSeconds(1));

    private static readonly Regex IfLine = new(
        @"^([ \t]*#\s*if[ \t]+)(.+?)(\s*)$",
        RegexOptions.Multiline | RegexOptions.CultureInvariant,
        TimeSpan.FromSeconds(1));

    private static readonly Regex ElifLine = new(
        @"^([ \t]*#\s*elif[ \t]+)(.+?)(\s*)$",
        RegexOptions.Multiline | RegexOptions.CultureInvariant,
        TimeSpan.FromSeconds(1));

    private static readonly Regex ElseLine = new(
        @"^([ \t]*#\s*else\b)(.*?)$",
        RegexOptions.Multiline | RegexOptions.CultureInvariant,
        TimeSpan.FromSeconds(1));

    private static readonly Regex EndifLine = new(
        @"^([ \t]*#\s*endif\b)(.*?)$",
        RegexOptions.Multiline | RegexOptions.CultureInvariant,
        TimeSpan.FromSeconds(1));

    private static readonly Regex AnyIfFamily = new(
        @"^([ \t]*#\s*(?:if|ifdef|ifndef)\b)",
        RegexOptions.Multiline | RegexOptions.CultureInvariant,
        TimeSpan.FromSeconds(1));

    /// <summary>
    /// Applies <see cref="Rewrite"/> only inside the bodies of the vertex and fragment entry functions
    /// named by <paramref name="vertexEntry"/> and <paramref name="fragmentEntry"/> (non-empty names only).
    /// HLSL does not allow <c>if</c> statements at file scope; rewriting top-level <c>#if</c> produced invalid Slang.
    /// </summary>
    public static string RewriteInsideShaderEntryFunctions(
        string source,
        IReadOnlyDictionary<string, string> keywordToSlangId,
        string? vertexEntry,
        string? fragmentEntry)
    {
        if (keywordToSlangId.Count == 0 || string.IsNullOrEmpty(source))
            return source;

        var spans = new List<(int InnerStart, int InnerEnd)>();
        void addSpan(string? name)
        {
            if (string.IsNullOrWhiteSpace(name))
                return;
            if (TryGetFunctionBodyInnerSpan(source, name.Trim(), out int start, out int end))
                spans.Add((start, end));
        }

        addSpan(vertexEntry);
        addSpan(fragmentEntry);
        if (spans.Count == 0)
            return source;

        spans.Sort((a, b) => b.InnerStart.CompareTo(a.InnerStart));
        string result = source;
        foreach ((int innerStart, int innerEnd) in spans)
        {
            if (innerStart < 0 || innerEnd <= innerStart || innerEnd > result.Length)
                continue;
            string inner = result.Substring(innerStart, innerEnd - innerStart);
            string rewritten = Rewrite(inner, keywordToSlangId);
            result = result.Substring(0, innerStart) + rewritten + result.Substring(innerEnd);
        }

        return result;
    }

    /// <summary>
    /// Locates the inner span of the function body (after <c>{</c>, before the matching <c>}</c>) for
    /// <paramref name="entryName"/> (first match). Returns false if the entry or braces cannot be found.
    /// </summary>
    internal static bool TryGetFunctionBodyInnerSpan(string source, string entryName, out int innerStart, out int innerEnd)
    {
        innerStart = innerEnd = 0;
        if (string.IsNullOrEmpty(source) || string.IsNullOrWhiteSpace(entryName))
            return false;

        string pattern = @"\b" + Regex.Escape(entryName.Trim()) + @"\s*\(";
        Match m = Regex.Match(source, pattern, RegexOptions.CultureInvariant, TimeSpan.FromSeconds(1));
        if (!m.Success)
            return false;

        int openParen = m.Index + m.Length - 1;
        if (openParen < 0 || source[openParen] != '(')
            return false;

        int closeParen = FindMatchingCloseParen(source, openParen);
        if (closeParen < 0)
            return false;

        int braceOpen = SkipToOpeningBraceAfterParams(source, closeParen + 1);
        if (braceOpen < 0)
            return false;

        int braceClose = FindMatchingBrace(source, braceOpen);
        if (braceClose < 0)
            return false;

        innerStart = braceOpen + 1;
        innerEnd = braceClose;
        return innerEnd > innerStart;
    }

    private static int FindMatchingCloseParen(string s, int openParenIndex)
    {
        int depth = 0;
        for (int i = openParenIndex; i < s.Length; i++)
        {
            char c = s[i];
            if (c == '(')
                depth++;
            else if (c == ')')
            {
                depth--;
                if (depth == 0)
                    return i;
            }
        }

        return -1;
    }

    private static int FindMatchingBrace(string s, int openBraceIndex)
    {
        int depth = 0;
        for (int i = openBraceIndex; i < s.Length; i++)
        {
            char c = s[i];
            if (c == '{')
                depth++;
            else if (c == '}')
            {
                depth--;
                if (depth == 0)
                    return i;
            }
        }

        return -1;
    }

    /// <summary>
    /// After the closing <c>)</c> of a parameter list, skips whitespace, optional semantics (<c>: SV_Target</c>),
    /// and attributes until the function body <c>{</c>.
    /// </summary>
    private static int SkipToOpeningBraceAfterParams(string s, int index)
    {
        int i = index;
        while (i < s.Length)
        {
            while (i < s.Length && char.IsWhiteSpace(s[i]))
                i++;
            if (i >= s.Length)
                return -1;
            if (s[i] == '{')
                return i;
            if (s[i] == ':')
            {
                i++;
                while (i < s.Length && s[i] != '{' && s[i] != '\n' && s[i] != '\r')
                    i++;
                continue;
            }

            if (s[i] == '[')
            {
                int close = FindMatchingBracket(s, i, '[', ']');
                if (close < 0)
                    return -1;
                i = close + 1;
                continue;
            }

            return -1;
        }

        return -1;
    }

    private static int FindMatchingBracket(string s, int openIndex, char open, char close)
    {
        int depth = 0;
        for (int i = openIndex; i < s.Length; i++)
        {
            if (s[i] == open)
                depth++;
            else if (s[i] == close)
            {
                depth--;
                if (depth == 0)
                    return i;
            }
        }

        return -1;
    }

    /// <summary>
    /// Rewrites specialization-axis preprocessor conditionals to HLSL control flow and recursively
    /// processes nested spec-safe groups inside converted bodies.
    /// </summary>
    public static string Rewrite(string source, IReadOnlyDictionary<string, string> keywordToSlangId)
    {
        if (keywordToSlangId.Count == 0 || string.IsNullOrEmpty(source))
            return source;

        string normalized = NormalizeIfdefDirectives(source, keywordToSlangId);
        List<string> lines = SplitLines(normalized);
        var sb = new StringBuilder();
        int i = 0;
        while (i < lines.Count)
        {
            if (!TryParseIfLine(lines[i], out string? cond) || cond is null)
            {
                sb.Append(lines[i]).Append('\n');
                i++;
                continue;
            }

            if (!IsSpecSafeCondition(cond, keywordToSlangId))
            {
                int end = FindMatchingEndif(lines, i);
                if (end < 0)
                {
                    sb.Append(lines[i]).Append('\n');
                    i++;
                    continue;
                }

                for (int k = i; k <= end; k++)
                    sb.Append(lines[k]).Append('\n');
                i = end + 1;
                continue;
            }

            if (!TryParseGroup(lines, i, keywordToSlangId, out List<Branch>? branches, out int endifIndex))
            {
                int end = FindMatchingEndif(lines, i);
                if (end < 0)
                {
                    sb.Append(lines[i]).Append('\n');
                    i++;
                    continue;
                }

                for (int k = i; k <= end; k++)
                    sb.Append(lines[k]).Append('\n');
                i = end + 1;
                continue;
            }

            string emitted = EmitRuntimeChain(branches!, keywordToSlangId);
            sb.Append(emitted);
            if (!emitted.EndsWith("\n", StringComparison.Ordinal))
                sb.Append('\n');
            i = endifIndex + 1;
        }

        return sb.ToString();
    }

    /// <summary>
    /// Maps <c>#ifdef KW</c> / <c>#ifndef KW</c> to <c>#if defined(KW)</c> / <c>#if !defined(KW)</c> when
    /// <paramref name="kw"/> is a specialization axis keyword.
    /// </summary>
    internal static string NormalizeIfdefDirectives(string source, IReadOnlyDictionary<string, string> keywordToSlangId)
    {
        string s = IfDefLine.Replace(
            source,
            m =>
            {
                string kw = m.Groups[3].Value;
                return keywordToSlangId.ContainsKey(kw)
                    ? $"{m.Groups[1].Value}if defined({kw}){m.Groups[4].Value}"
                    : m.Value;
            });
        s = IfNDefLine.Replace(
            s,
            m =>
            {
                string kw = m.Groups[3].Value;
                return keywordToSlangId.ContainsKey(kw)
                    ? $"{m.Groups[1].Value}if !defined({kw}){m.Groups[4].Value}"
                    : m.Value;
            });
        return s;
    }

    internal sealed class Branch
    {
        /// <summary>Null means <c>#else</c> (always-taken final branch).</summary>
        public string? ConditionToHlsl { get; init; }

        public List<string> BodyLines { get; init; } = new();
    }

    private static List<string> SplitLines(string s)
    {
        var lines = new List<string>();
        using var reader = new StringReader(s);
        while (reader.ReadLine() is { } line)
            lines.Add(line);
        return lines;
    }

    private static bool TryParseIfLine(string line, out string? condition)
    {
        condition = null;
        Match m = IfLine.Match(line);
        if (!m.Success)
            return false;
        condition = m.Groups[2].Value.Trim();
        return true;
    }

    private static bool TryParseElifLine(string line, out string? condition)
    {
        condition = null;
        Match m = ElifLine.Match(line);
        if (!m.Success)
            return false;
        condition = m.Groups[2].Value.Trim();
        return true;
    }

    private static bool IsElseLine(string line) => ElseLine.IsMatch(line);

    private static bool IsEndifLine(string line) => EndifLine.IsMatch(line);

    private static bool IsIfFamilyLine(string line) => AnyIfFamily.IsMatch(line);

    /// <summary>
    /// True when <paramref name="condition"/> uses only <c>defined(axis_kw)</c>, <c>||</c>, <c>&amp;&amp;</c>, <c>!</c>, and parentheses.
    /// </summary>
    internal static bool IsSpecSafeCondition(string condition, IReadOnlyDictionary<string, string> keywordToSlangId)
    {
        string s = condition.Trim();
        var keys = keywordToSlangId.Keys.OrderByDescending(k => k.Length).ToList();
        foreach (string key in keys)
        {
            string pat = @"\bdefined\s*\(\s*" + Regex.Escape(key) + @"\s*\)";
            s = Regex.Replace(s, pat, " 1 ", RegexOptions.CultureInvariant);
        }

        if (Regex.IsMatch(s, @"\bdefined\b", RegexOptions.CultureInvariant))
            return false;

        s = Regex.Replace(s, @"\s+", "", RegexOptions.CultureInvariant);
        return Regex.IsMatch(s, @"^[01|&!()]+$", RegexOptions.CultureInvariant);
    }

    /// <summary>
    /// Builds an HLSL expression using <c>(USC_*)</c> specialization bools.
    /// </summary>
    internal static string ConditionToHlsl(string condition, IReadOnlyDictionary<string, string> keywordToSlangId)
    {
        string s = condition.Trim();
        var keys = keywordToSlangId.Keys.OrderByDescending(k => k.Length).ToList();
        foreach (string key in keys)
        {
            if (!keywordToSlangId.TryGetValue(key, out string? slang))
                continue;
            string pat = @"\bdefined\s*\(\s*" + Regex.Escape(key) + @"\s*\)";
            s = Regex.Replace(s, pat, "(" + slang + ")", RegexOptions.CultureInvariant);
        }

        return s;
    }

    /// <summary>
    /// Finds the <c>#endif</c> that closes the <c>#if</c> starting at <paramref name="ifLineIndex"/>.</summary>
    internal static int FindMatchingEndif(IReadOnlyList<string> lines, int ifLineIndex)
    {
        int depth = 0;
        for (int k = ifLineIndex; k < lines.Count; k++)
        {
            if (IsIfFamilyLine(lines[k]))
                depth++;
            else if (IsEndifLine(lines[k]))
            {
                depth--;
                if (depth == 0)
                    return k;
            }
        }

        return -1;
    }

    /// <summary>
    /// Parses <c>#if</c> … <c>#elif*</c> … <c>#else?</c> … <c>#endif</c> when the opening <c>#if</c> is at
    /// <paramref name="ifLineIndex"/> and every branch condition is spec-safe.
    /// </summary>
    private static bool TryParseGroup(
        IReadOnlyList<string> lines,
        int ifLineIndex,
        IReadOnlyDictionary<string, string> keywordToSlangId,
        out List<Branch>? branches,
        out int endifIndex)
    {
        branches = null;
        endifIndex = -1;
        if (!TryParseIfLine(lines[ifLineIndex], out string? firstCond) || firstCond is null)
            return false;

        if (!IsSpecSafeCondition(firstCond, keywordToSlangId))
            return false;

        int end = FindMatchingEndif(lines, ifLineIndex);
        if (end < 0)
            return false;

        var list = new List<Branch>();
        string? currentHlsl = ConditionToHlsl(firstCond, keywordToSlangId);
        var body = new List<string>();
        int depth = 1;
        for (int k = ifLineIndex + 1; k < end; k++)
        {
            string line = lines[k];
            if (IsIfFamilyLine(line))
            {
                depth++;
                body.Add(line);
                continue;
            }

            if (IsEndifLine(line))
            {
                depth--;
                body.Add(line);
                continue;
            }

            if (TryParseElifLine(line, out string? elifCond) && depth == 1)
            {
                if (elifCond is null || !IsSpecSafeCondition(elifCond, keywordToSlangId))
                    return false;
                list.Add(new Branch { ConditionToHlsl = currentHlsl, BodyLines = body });
                currentHlsl = ConditionToHlsl(elifCond, keywordToSlangId);
                body = new List<string>();
                continue;
            }

            if (IsElseLine(line) && depth == 1)
            {
                list.Add(new Branch { ConditionToHlsl = currentHlsl, BodyLines = body });
                currentHlsl = null;
                body = new List<string>();
                continue;
            }

            body.Add(line);
        }

        list.Add(new Branch { ConditionToHlsl = currentHlsl, BodyLines = body });
        foreach (Branch br in list)
        {
            if (BranchBodyHasForbiddenContent(br.BodyLines))
                return false;
        }

        branches = list;
        endifIndex = end;
        return true;
    }

    /// <summary>
    /// Runtime <c>if</c> cannot wrap preprocessor <c>#define</c> / includes (macros leak globally) or global
    /// resource declarations (invalid inside blocks). Such groups stay preprocessor <c>#if</c>.
    /// </summary>
    internal static bool BranchBodyHasForbiddenContent(IReadOnlyList<string> bodyLines)
    {
        foreach (string line in bodyLines)
        {
            string t = line.TrimStart();
            if (t.StartsWith("#", StringComparison.Ordinal))
            {
                if (Regex.IsMatch(
                        t,
                        @"^\#\s*(define|include|undef|pragma)\b",
                        RegexOptions.CultureInvariant))
                    return true;
            }

            if (Regex.IsMatch(
                    line,
                    @"^\s*(sampler2D|sampler3D|samplerCUBE|Texture2D|Texture3D|TextureCube|cbuffer)\b",
                    RegexOptions.CultureInvariant))
                return true;
            if (Regex.IsMatch(line, @"^\s*UNITY_DECLARE_", RegexOptions.CultureInvariant))
                return true;
        }

        return false;
    }

    private static string EmitRuntimeChain(List<Branch> branches, IReadOnlyDictionary<string, string> keywordToSlangId)
    {
        var sb = new StringBuilder();
        for (int b = 0; b < branches.Count; b++)
        {
            Branch br = branches[b];
            string bodyText = string.Join("\n", br.BodyLines);
            string rewrittenBody = Rewrite(bodyText, keywordToSlangId).TrimEnd();

            if (br.ConditionToHlsl is not null)
            {
                if (b == 0)
                    sb.Append("if (").Append(br.ConditionToHlsl).AppendLine(") {");
                else
                    sb.Append("else if (").Append(br.ConditionToHlsl).AppendLine(") {");
            }
            else
            {
                sb.AppendLine("else {");
            }

            if (rewrittenBody.Length > 0)
            {
                foreach (string line in SplitLines(rewrittenBody))
                    sb.AppendLine(line);
            }

            sb.AppendLine("}");
        }

        return sb.ToString();
    }
}
