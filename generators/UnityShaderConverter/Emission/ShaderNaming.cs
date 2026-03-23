using System.Text;

namespace UnityShaderConverter.Emission;

/// <summary>Stable file and Rust module names derived from Unity <c>Shader "A/B"</c> names.</summary>
public static class ShaderNaming
{
    /// <summary>Snake_case stem used for <c>.slang</c>, <c>.wgsl</c>, and Rust module names.</summary>
    public static string FileStem(string shaderName)
    {
        string[] segments = shaderName.Split(new[] { '/' }, StringSplitOptions.RemoveEmptyEntries);
        var parts = new List<string>();
        foreach (string seg in segments)
        {
            string snake = PascalOrMixedToSnake(seg.Trim());
            foreach (char c in Path.GetInvalidFileNameChars())
                snake = snake.Replace(c, '_');
            if (snake.Length > 0)
                parts.Add(snake);
        }

        string joined = parts.Count > 0 ? string.Join("_", parts) : "unnamed_shader";
        return RustSafeIdent(joined);
    }

    /// <summary>Rust module name (snake_case, safe ident).</summary>
    public static string ModuleName(string shaderName) => FileStem(shaderName);

    /// <summary>Snake_case stem for a pass WGSL file from <c>Name "ForwardBase"</c> or <c>pass{index}</c>.</summary>
    public static string WgslPassStem(string? passName, int fallbackIndex)
    {
        if (string.IsNullOrWhiteSpace(passName))
            return $"pass{fallbackIndex}";
        string snake = PascalOrMixedToSnake(passName.Trim());
        foreach (char c in Path.GetInvalidFileNameChars())
            snake = snake.Replace(c, '_');
        if (snake.Length == 0)
            return $"pass{fallbackIndex}";
        return RustSafeIdent(snake);
    }

    private static string PascalOrMixedToSnake(string s)
    {
        if (s.Length == 0)
            return "";
        var sb = new StringBuilder();
        for (int i = 0; i < s.Length; i++)
        {
            char c = s[i];
            if (char.IsLetterOrDigit(c))
            {
                if (i > 0 && char.IsUpper(c) && (char.IsLower(s[i - 1]) || (i + 1 < s.Length && char.IsLower(s[i + 1]))))
                    sb.Append('_');
                sb.Append(char.ToLowerInvariant(c));
            }
            else if (c == '_' || c == '-')
                sb.Append('_');
        }

        return sb.ToString().Trim('_');
    }

    private static string RustSafeIdent(string s)
    {
        if (s.Length == 0)
            return "unnamed_shader";
        if (char.IsDigit(s[0]))
            return "s_" + s;
        return s;
    }
}
