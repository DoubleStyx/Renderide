using System.Reflection;
using Mono.Cecil;
using SharedTypeGenerator.Analysis;
using Xunit;

namespace SharedTypeGenerator.Tests.Unit;

/// <summary>Unit tests for <see cref="CecilTypeResolver"/>.</summary>
public sealed class CecilTypeResolverTests
{
    /// <summary>Nested reflection type names convert to Cecil slash-separated nested names.</summary>
    [Fact]
    public void Resolve_handles_nested_types()
    {
        const string source = @"
namespace ResolverAsm {
  public sealed class Outer {
    public sealed class Inner { }
  }
}";
        (Assembly reflection, AssemblyDefinition cecil) = TestCompilation.Compile(source);
        Type inner = reflection.GetType("ResolverAsm.Outer+Inner", throwOnError: true)!;

        TypeDefinition? resolved = CecilTypeResolver.Resolve(cecil, inner);

        Assert.NotNull(resolved);
        Assert.Equal("ResolverAsm.Outer/Inner", resolved.FullName);
    }
}
