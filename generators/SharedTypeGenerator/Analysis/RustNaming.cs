using Humanizer;

namespace SharedTypeGenerator.Analysis;

/// <summary>Pure static methods for converting C# names to Rust naming conventions.
/// HumanizeField produces snake_case, HumanizeType produces PascalCase.
/// Both handle Rust keyword escaping and special abbreviation patterns.</summary>
public static class RustNaming
{
    private static readonly HashSet<string> RustKeywords =
    [
        "as", "async", "await", "break", "const", "continue", "crate",
        "dyn", "else", "enum", "extern", "false", "fn", "for", "if",
        "impl", "in", "let", "loop", "match", "mod", "move", "mut",
        "pub", "ref", "return", "self", "Self", "static", "struct",
        "super", "trait", "true", "type", "union", "unsafe", "use",
        "where", "while", "yield"
    ];

    private static string EscapeKeyword(string name)
    {
        return RustKeywords.Contains(name) ? $"r#{name}" : name;
    }

    /// <summary>Converts a C# field name (PascalCase/camelCase) to Rust snake_case,
    /// escaping Rust keywords with r# prefix.</summary>
    public static string HumanizeField(this string name)
    {
        name = name.Replace("2D", "_2D_", StringComparison.OrdinalIgnoreCase);
        name = name.Replace("3D", "_3D_", StringComparison.OrdinalIgnoreCase);

        name = name.Underscore().Trim('_').ToLowerInvariant();

        while (name.Contains("__"))
            name = name.Replace("__", "_");

        name = EscapeKeyword(name);

        name = name.Replace("2_d", "2d");
        name = name.Replace("3_d", "3d");
        name = name.Replace("i_ds", "ids");

        return name;
    }

    /// <summary>Converts a C# type name to Rust PascalCase, preserving
    /// Rust primitive types, generic syntax, and array notation.</summary>
    public static string HumanizeType(this string name)
    {
        if (name.Length > 1 && name[0] is 'i' or 'f' or 'u' && name[1..].All(char.IsNumber))
            return name;

        if (name == "bool")
            return name;

        if (name.Contains('<'))
            return name;

        int idx = name.LastIndexOf('.');
        if (idx != -1)
            return $"{name[..idx]}.{name[(idx + 1)..].HumanizeType()}";

        idx = name.LastIndexOf("[]const", StringComparison.Ordinal);
        if (idx != -1)
            return $"{name[..idx]} []const {name[(idx + "[]const".Length)..].HumanizeType()}";

        idx = name.LastIndexOf("[]", StringComparison.Ordinal);
        if (idx != -1)
            return $"{name[..idx]} []{name[(idx + "[]".Length)..].HumanizeType()}";

        return name.Replace("_", "").Pascalize();
    }

    /// <summary>SCREAMING_SNAKE for const names from a Rust PascalCase struct name (e.g. <c>LightData</c> → <c>LIGHT_DATA</c>).</summary>
    public static string ToScreamingSnakeTypeName(this string rustPascalTypeName)
    {
        return rustPascalTypeName.Underscore().Replace("__", "_", StringComparison.Ordinal).ToUpperInvariant();
    }
}
