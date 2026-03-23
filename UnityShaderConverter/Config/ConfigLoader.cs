using System.Text.Json;

namespace UnityShaderConverter.Config;

/// <summary>Loads optional JSON config files.</summary>
public static class ConfigLoader
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        ReadCommentHandling = JsonCommentHandling.Skip,
        AllowTrailingCommas = true,
    };

    /// <summary>Merges user compiler JSON over <paramref name="defaults"/>.</summary>
    public static CompilerConfigModel MergeCompilerConfig(CompilerConfigModel defaults, string? userPath)
    {
        var merged = new CompilerConfigModel
        {
            MaxVariantCombinationsPerShader = defaults.MaxVariantCombinationsPerShader,
            SlangEligibleGlobPatterns = new List<string>(defaults.SlangEligibleGlobPatterns),
        };
        if (string.IsNullOrWhiteSpace(userPath) || !File.Exists(userPath))
            return merged;
        string json = File.ReadAllText(userPath);
        var user = JsonSerializer.Deserialize<CompilerConfigModel>(json, JsonOptions);
        if (user is null)
            return merged;
        if (user.SlangEligibleGlobPatterns.Count > 0)
            merged.SlangEligibleGlobPatterns = user.SlangEligibleGlobPatterns;
        if (user.MaxVariantCombinationsPerShader > 0)
            merged.MaxVariantCombinationsPerShader = user.MaxVariantCombinationsPerShader;
        return merged;
    }

    /// <summary>Loads variant overrides or returns null.</summary>
    public static VariantConfigModel? LoadVariantConfig(string? path)
    {
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
            return null;
        string json = File.ReadAllText(path);
        return JsonSerializer.Deserialize<VariantConfigModel>(json, JsonOptions);
    }
}
