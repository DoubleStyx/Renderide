using SharedTypeGenerator.Analysis;
using Xunit;

namespace SharedTypeGenerator.Tests.Unit;

/// <summary>Unit tests for narrow generated Rust field type overrides.</summary>
public sealed class RustFieldTypeOverridesTests
{
    /// <summary><c>RendererInitResult</c> renderer-owned strings use borrowed-or-owned storage.</summary>
    [Theory]
    [InlineData("rendererIdentifier")]
    [InlineData("stereoRenderingMode")]
    public void Apply_uses_cow_for_renderer_init_result_strings(string fieldName)
    {
        Assert.Equal(
            RustFieldTypeOverrides.StaticStringCowOption,
            RustFieldTypeOverrides.Apply("RendererInitResult", fieldName, "Option<String>"));
    }

    /// <summary>Other string fields keep owned storage.</summary>
    [Theory]
    [InlineData("RendererInitData", "windowTitle")]
    [InlineData("RendererInitProgressUpdate", "phase")]
    [InlineData("RendererInitResult", "other")]
    public void Apply_leaves_other_strings_owned(string typeName, string fieldName)
    {
        Assert.Equal(
            "Option<String>",
            RustFieldTypeOverrides.Apply(typeName, fieldName, "Option<String>"));
    }
}
