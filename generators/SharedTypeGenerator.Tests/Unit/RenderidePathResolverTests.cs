using SharedTypeGenerator.Logging;
using Xunit;

namespace SharedTypeGenerator.Tests.Unit;

/// <summary>Unit tests for <see cref="RenderidePathResolver"/>.</summary>
public sealed class RenderidePathResolverTests
{
    /// <summary>Git root containing <c>crates/renderide</c> resolves directly.</summary>
    [Fact]
    public void ResolveRenderideRoot_direct_layout()
    {
        string temp = Path.Combine(Path.GetTempPath(), "rr-" + Guid.NewGuid().ToString("N"));
        string crates = Path.Combine(temp, "crates", "renderide");
        Directory.CreateDirectory(crates);
        try
        {
            string root = RenderidePathResolver.ResolveRenderideRoot(temp);
            Assert.Equal(Path.GetFullPath(temp), root);
        }
        finally
        {
            Directory.Delete(temp, recursive: true);
        }
    }

    /// <summary>Nested <c>Renderide/crates/renderide</c> layout maps to the inner repo root.</summary>
    [Fact]
    public void ResolveRenderideRoot_nested_renderide_folder()
    {
        string temp = Path.Combine(Path.GetTempPath(), "rr-n-" + Guid.NewGuid().ToString("N"));
        string nested = Path.Combine(temp, "Renderide", "crates", "renderide");
        Directory.CreateDirectory(nested);
        try
        {
            string root = RenderidePathResolver.ResolveRenderideRoot(temp);
            Assert.Equal(Path.GetFullPath(Path.Combine(temp, "Renderide")), root);
        }
        finally
        {
            Directory.Delete(temp, recursive: true);
        }
    }

    /// <summary>Null git root falls back to walking the current working directory.</summary>
    [Fact]
    public void ResolveRenderideRoot_null_git_uses_fallback()
    {
        string root = RenderidePathResolver.ResolveRenderideRoot(null);
        Assert.False(string.IsNullOrWhiteSpace(root));
    }
}
