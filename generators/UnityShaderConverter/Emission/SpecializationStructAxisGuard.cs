using System.Text.RegularExpressions;

namespace UnityShaderConverter.Emission;

/// <summary>
/// Detects specialization axis keywords used under preprocessor directives inside <c>struct</c> bodies.
/// Runtime <c>#if defined(AXIS)</c> conversion in function bodies is unsafe if <c>AXIS</c> still strips struct
/// members at preprocessor time—normalize shaders (always declare fields) or disable specialization for that asset.
/// </summary>
public static class SpecializationStructAxisGuard
{
    private static readonly Regex StructLine = new(
        @"^\s*struct\s+\w+",
        RegexOptions.CultureInvariant,
        TimeSpan.FromSeconds(1));

    /// <summary>
    /// Returns axis keywords from <paramref name="specializationAxisKeywords"/> that appear on a preprocessor
    /// line (<c>#if</c> / <c>#ifdef</c> / <c>#ifndef</c> / <c>#elif</c>) while inside a struct body.
    /// </summary>
    public static HashSet<string> FindAxisKeywordsInStructConditionalBlocks(
        string source,
        IReadOnlyCollection<string> specializationAxisKeywords)
    {
        var axisSet = new HashSet<string>(specializationAxisKeywords, StringComparer.Ordinal);
        var found = new HashSet<string>(StringComparer.Ordinal);
        if (axisSet.Count == 0 || string.IsNullOrEmpty(source))
            return found;

        string[] lines = source.Split(['\n'], StringSplitOptions.None);
        bool awaitingStructBrace = false;
        bool inStruct = false;
        int structBraceDepth = 0;

        foreach (string raw in lines)
        {
            string line = raw.TrimEnd('\r');
            string trim = line.TrimStart();

            if (!inStruct && !awaitingStructBrace && StructLine.IsMatch(trim))
            {
                awaitingStructBrace = true;
                int delta = CountChar(line, '{') - CountChar(line, '}');
                if (delta > 0)
                {
                    inStruct = true;
                    structBraceDepth = delta;
                    awaitingStructBrace = false;
                }

                continue;
            }

            if (awaitingStructBrace)
            {
                int delta = CountChar(line, '{') - CountChar(line, '}');
                if (delta > 0)
                {
                    inStruct = true;
                    structBraceDepth = delta;
                    awaitingStructBrace = false;
                }

                continue;
            }

            if (inStruct)
            {
                if (trim.StartsWith('#'))
                    CollectAxisFromDirective(trim, axisSet, found);

                structBraceDepth += CountChar(line, '{') - CountChar(line, '}');
                if (structBraceDepth <= 0)
                {
                    inStruct = false;
                    structBraceDepth = 0;
                }
            }
        }

        return found;
    }

    private static int CountChar(string s, char c)
    {
        int n = 0;
        foreach (char ch in s)
        {
            if (ch == c)
                n++;
        }

        return n;
    }

    private static void CollectAxisFromDirective(string trim, HashSet<string> axisSet, HashSet<string> found)
    {
        if (trim.StartsWith("#ifdef ", StringComparison.Ordinal) ||
            trim.StartsWith("#ifndef ", StringComparison.Ordinal))
        {
            string rest = trim.Contains(' ', StringComparison.Ordinal)
                ? trim[(trim.IndexOf(' ', StringComparison.Ordinal) + 1)..].Trim()
                : string.Empty;
            int end = 0;
            while (end < rest.Length && !char.IsWhiteSpace(rest[end]))
                end++;
            string kw = rest[..end].Trim();
            if (axisSet.Contains(kw))
                found.Add(kw);
            return;
        }

        if (trim.StartsWith("#elif ", StringComparison.Ordinal))
        {
            string cond = trim["#elif ".Length..].Trim();
            foreach (string axis in axisSet)
            {
                if (DirectiveConditionMentionsKeyword(cond, axis))
                    found.Add(axis);
            }

            return;
        }

        if (!trim.StartsWith("#if ", StringComparison.Ordinal))
            return;

        string expr = trim["#if ".Length..].Trim();
        foreach (string axis in axisSet)
        {
            if (DirectiveConditionMentionsKeyword(expr, axis))
                found.Add(axis);
        }
    }

    private static bool DirectiveConditionMentionsKeyword(string expr, string keyword)
    {
        return expr.Contains("defined(" + keyword + ")", StringComparison.Ordinal) ||
               expr.Contains("defined( " + keyword + " )", StringComparison.Ordinal) ||
               expr.Contains("defined( " + keyword + ")", StringComparison.Ordinal) ||
               expr.Contains("defined(" + keyword + " )", StringComparison.Ordinal);
    }
}
