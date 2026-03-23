using System.Text.Json;

namespace UnityShaderConverter.Config;

/// <summary>Loads <c>DefaultCompilerConfig.json</c> copied next to the executable.</summary>
public static class DefaultCompilerConfig
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        ReadCommentHandling = JsonCommentHandling.Skip,
        AllowTrailingCommas = true,
    };

    /// <summary>Reads default compiler settings from the build output directory.</summary>
    public static CompilerConfigModel LoadFromOutputDirectory(string baseDirectory)
    {
        string path = Path.Combine(baseDirectory, "DefaultCompilerConfig.json");
        if (!File.Exists(path))
            return new CompilerConfigModel();
        string json = File.ReadAllText(path);
        return JsonSerializer.Deserialize<CompilerConfigModel>(json, JsonOptions) ?? new CompilerConfigModel();
    }
}
