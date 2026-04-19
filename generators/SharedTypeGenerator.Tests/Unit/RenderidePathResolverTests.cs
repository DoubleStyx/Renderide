using SharedTypeGenerator.Logging;
using SharedTypeGenerator.Tests.Unit.Support;
using Xunit;

namespace SharedTypeGenerator.Tests.Unit;

/// <summary>Unit tests for <see cref="RenderidePathResolver"/>.</summary>
public sealed class RenderidePathResolverTests
{
    /// <summary>Git root containing <c>crates/renderide</c> resolves directly.</summary>
    [Fact]
    public void ResolveRenderideRoot_direct_layout()
    {
        using var temp = new TempDirectory();
        string crates = Path.Combine(temp.DirectoryPath, "crates", "renderide");
        Directory.CreateDirectory(crates);
        string root = RenderidePathResolver.ResolveRenderideRoot(temp.DirectoryPath);
        Assert.Equal(Path.GetFullPath(temp.DirectoryPath), root);
    }

    /// <summary>Nested <c>Renderide/crates/renderide</c> layout maps to the inner repo root.</summary>
    [Fact]
    public void ResolveRenderideRoot_nested_renderide_folder()
    {
        using var temp = new TempDirectory();
        string nested = Path.Combine(temp.DirectoryPath, "Renderide", "crates", "renderide");
        Directory.CreateDirectory(nested);
        string root = RenderidePathResolver.ResolveRenderideRoot(temp.DirectoryPath);
        Assert.Equal(Path.GetFullPath(Path.Combine(temp.DirectoryPath, "Renderide")), root);
    }

    /// <summary>Null git root falls back to walking the current working directory.</summary>
    [Fact]
    public void ResolveRenderideRoot_null_git_uses_fallback()
    {
        string root = RenderidePathResolver.ResolveRenderideRoot(null);
        Assert.False(string.IsNullOrWhiteSpace(root));
    }
}
