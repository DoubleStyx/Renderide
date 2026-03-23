using System.Text;
using System.Text.RegularExpressions;

namespace UnityShaderConverter.Analysis;

/// <summary>Minimal glob matching (<c>**</c>, <c>*</c>, <c>?</c>) for slash-normalized relative paths.</summary>
public static class GlobMatcher
{
    /// <summary>Returns true if <paramref name="relativePath"/> matches any include pattern.</summary>
    /// <param name="relativePath">Path using forward slashes, relative to the Renderide root.</param>
    public static bool MatchesAny(string relativePath, IEnumerable<string> patterns)
    {
        string normalized = relativePath.Replace('\\', '/').TrimStart('/');
        foreach (string pattern in patterns)
        {
            if (string.IsNullOrWhiteSpace(pattern))
                continue;
            string p = pattern.Replace('\\', '/').TrimStart('/');
            if (PatternToRegex(p).IsMatch(normalized))
                return true;
        }

        return false;
    }

    private static Regex PatternToRegex(string glob)
    {
        var sb = new StringBuilder();
        sb.Append('^');
        for (int i = 0; i < glob.Length; i++)
        {
            char c = glob[i];
            if (c == '*' && i + 1 < glob.Length && glob[i + 1] == '*')
            {
                i++;
                if (i + 1 < glob.Length && glob[i + 1] == '/')
                {
                    i++;
                    sb.Append("(?:.*/)?");
                }
                else
                {
                    sb.Append(".*");
                }
            }
            else if (c == '*')
            {
                sb.Append("[^/]*");
            }
            else if (c == '?')
            {
                sb.Append("[^/]");
            }
            else if (".^$()[]{}|+\\".IndexOf(c) >= 0)
            {
                sb.Append('\\').Append(c);
            }
            else
            {
                sb.Append(c);
            }
        }

        sb.Append('$');
        return new Regex(sb.ToString(), RegexOptions.CultureInvariant | RegexOptions.IgnoreCase);
    }
}
