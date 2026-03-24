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

    /// <summary>Deep-copies compiler defaults for merging.</summary>
    private static CompilerConfigModel CloneCompilerDefaults(CompilerConfigModel d) =>
        new()
        {
            SlangEligibleGlobPatterns = new List<string>(d.SlangEligibleGlobPatterns),
            SlangExcludeGlobPatterns = new List<string>(d.SlangExcludeGlobPatterns),
            MaxVariantCombinationsPerShader = d.MaxVariantCombinationsPerShader,
            EnableSlangSpecialization = d.EnableSlangSpecialization,
            MaxSpecializationConstants = d.MaxSpecializationConstants,
            SuppressSlangWarnings = d.SuppressSlangWarnings,
        };

    /// <summary>Merges user compiler JSON over <paramref name="defaults"/>; only keys present in the user file override defaults.</summary>
    public static CompilerConfigModel MergeCompilerConfig(CompilerConfigModel defaults, string? userPath)
    {
        CompilerConfigModel merged = CloneCompilerDefaults(defaults);
        if (string.IsNullOrWhiteSpace(userPath) || !File.Exists(userPath))
            return merged;

        string json = File.ReadAllText(userPath);
        using JsonDocument doc = JsonDocument.Parse(json, new JsonDocumentOptions { CommentHandling = JsonCommentHandling.Skip });
        JsonElement root = doc.RootElement;

        if (root.TryGetProperty("slangEligibleGlobPatterns", out JsonElement globEl) &&
            globEl.ValueKind == JsonValueKind.Array &&
            globEl.GetArrayLength() > 0)
        {
            var list = new List<string>();
            foreach (JsonElement item in globEl.EnumerateArray())
            {
                if (item.ValueKind == JsonValueKind.String)
                {
                    string? s = item.GetString();
                    if (!string.IsNullOrWhiteSpace(s))
                        list.Add(s);
                }
            }

            if (list.Count > 0)
                merged.SlangEligibleGlobPatterns = list;
        }

        if (root.TryGetProperty("slangExcludeGlobPatterns", out JsonElement exclEl) &&
            exclEl.ValueKind == JsonValueKind.Array)
        {
            var excl = new List<string>();
            foreach (JsonElement item in exclEl.EnumerateArray())
            {
                if (item.ValueKind == JsonValueKind.String)
                {
                    string? s = item.GetString();
                    if (!string.IsNullOrWhiteSpace(s))
                        excl.Add(s);
                }
            }

            merged.SlangExcludeGlobPatterns = excl;
        }

        if (root.TryGetProperty("maxVariantCombinationsPerShader", out JsonElement mvcEl) &&
            mvcEl.ValueKind == JsonValueKind.Number &&
            mvcEl.TryGetInt32(out int mvc) &&
            mvc > 0)
        {
            merged.MaxVariantCombinationsPerShader = mvc;
        }

        if (root.TryGetProperty("enableSlangSpecialization", out JsonElement specEl) &&
            specEl.ValueKind is JsonValueKind.True or JsonValueKind.False)
        {
            merged.EnableSlangSpecialization = specEl.GetBoolean();
        }

        if (root.TryGetProperty("maxSpecializationConstants", out JsonElement mscEl) &&
            mscEl.ValueKind == JsonValueKind.Number &&
            mscEl.TryGetInt32(out int msc) &&
            msc > 0)
        {
            merged.MaxSpecializationConstants = msc;
        }

        if (root.TryGetProperty("suppressSlangWarnings", out JsonElement swEl) &&
            swEl.ValueKind is JsonValueKind.True or JsonValueKind.False)
        {
            merged.SuppressSlangWarnings = swEl.GetBoolean();
        }

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
