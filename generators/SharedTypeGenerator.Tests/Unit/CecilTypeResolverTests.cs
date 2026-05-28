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

    /// <summary>Constructed generic types resolve through their open generic Cecil definitions.</summary>
    [Fact]
    public void Resolve_handles_constructed_generic_types()
    {
        const string source = @"
namespace ResolverAsm {
  public class GenericBase<T> { }
  public sealed class Derived : GenericBase<int> { }
}";
        (Assembly reflection, AssemblyDefinition cecil) = TestCompilation.Compile(source);
        Type derived = reflection.GetType("ResolverAsm.Derived", throwOnError: true)!;
        Type genericBase = derived.BaseType!;

        TypeDefinition? resolved = CecilTypeResolver.Resolve(cecil, genericBase);

        Assert.NotNull(resolved);
        Assert.Equal("ResolverAsm.GenericBase`1", resolved.FullName);
    }
}
