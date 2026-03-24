using UnityShaderConverter.Config;

namespace UnityShaderConverter.Tests;

/// <summary>Tests for <see cref="ConfigLoader.MergeCompilerConfig"/> partial JSON merge semantics.</summary>
public sealed class ConfigLoaderTests
{
    /// <summary>Only keys present in the user JSON override defaults; other fields stay from defaults.</summary>
    [Fact]
    public void MergeCompilerConfig_PartialJson_PreservesUnspecifiedDefaults()
    {
        var defaults = new CompilerConfigModel
        {
            MaxVariantCombinationsPerShader = 512,
            EnableSlangSpecialization = true,
            SuppressSlangWarnings = true,
            MaxSpecializationConstants = 8,
            SlangEligibleGlobPatterns = new List<string> { "**/*.shader" },
        };
        string tmp = Path.Combine(Path.GetTempPath(), "usc_cfg_" + Guid.NewGuid().ToString("N") + ".json");
        try
        {
            File.WriteAllText(tmp, "{\"suppressSlangWarnings\": false}");
            CompilerConfigModel merged = ConfigLoader.MergeCompilerConfig(defaults, tmp);
            Assert.False(merged.SuppressSlangWarnings);
            Assert.True(merged.EnableSlangSpecialization);
            Assert.Equal(512, merged.MaxVariantCombinationsPerShader);
            Assert.Equal(8, merged.MaxSpecializationConstants);
            Assert.Single(merged.SlangEligibleGlobPatterns);
            Assert.Equal("**/*.shader", merged.SlangEligibleGlobPatterns[0]);
        }
        finally
        {
            try
            {
                File.Delete(tmp);
            }
            catch
            {
                // ignored
            }
        }
    }

    /// <summary><c>slangExcludeGlobPatterns</c> in user JSON replaces the default exclude list when provided.</summary>
    [Fact]
    public void MergeCompilerConfig_SlangExcludeGlobPatterns_OverridesDefaults()
    {
        var defaults = new CompilerConfigModel
        {
            SlangEligibleGlobPatterns = new List<string> { "**/*.shader" },
            SlangExcludeGlobPatterns = new List<string> { "**/Nosamplers.shader" },
        };
        string tmp = Path.Combine(Path.GetTempPath(), "usc_cfg_ex_" + Guid.NewGuid().ToString("N") + ".json");
        try
        {
            File.WriteAllText(tmp, "{\"slangExcludeGlobPatterns\":[\"**/OnlyThis.shader\"]}");
            CompilerConfigModel merged = ConfigLoader.MergeCompilerConfig(defaults, tmp);
            Assert.Single(merged.SlangExcludeGlobPatterns);
            Assert.Equal("**/OnlyThis.shader", merged.SlangExcludeGlobPatterns[0]);
        }
        finally
        {
            try
            {
                File.Delete(tmp);
            }
            catch
            {
                // ignored
            }
        }
    }
}
