using System.Reflection;
using Mono.Cecil;
using SharedTypeGenerator.Analysis;
using Xunit;

namespace SharedTypeGenerator.Tests.Unit;

/// <summary>Unit tests for <see cref="PolymorphicAnalyzer"/> using in-memory assemblies.</summary>
public sealed class PolymorphicAnalyzerTests
{
    /// <summary>Static constructor <c>Ldtoken</c> registrations are surfaced as ordered variants.</summary>
    [Fact]
    public void ExtractVariants_reads_ldtoken_targets()
    {
        const string source = @"
namespace PolyAsm {
  public sealed class VariantA { }
  public sealed class VariantB { }
  public sealed class Registry {
    private static readonly System.Type Ta = typeof(VariantA);
    private static readonly System.Type Tb = typeof(VariantB);
  }
}";
        (Assembly reflection, AssemblyDefinition cecil) = TestCompilation.Compile(source);
        Type registry = reflection.GetType("PolyAsm.Registry", throwOnError: true)!;
        var analyzer = new PolymorphicAnalyzer(cecil, reflection);
        var variants = analyzer.ExtractVariants(registry);
        Assert.Equal(2, variants.Count);
        Assert.Equal("VariantA", variants[0].CSharpName);
        Assert.Equal("VariantB", variants[1].CSharpName);
    }

    /// <summary>Types without a static constructor yield an empty variant list.</summary>
    [Fact]
    public void ExtractVariants_empty_when_no_cctor()
    {
        const string source = @"
namespace PolyAsm {
  public sealed class NoStaticCtor { }
}";
        (Assembly reflection, AssemblyDefinition cecil) = TestCompilation.Compile(source);
        Type t = reflection.GetType("PolyAsm.NoStaticCtor", throwOnError: true)!;
        var analyzer = new PolymorphicAnalyzer(cecil, reflection);
        Assert.Empty(analyzer.ExtractVariants(t));
    }

    /// <summary><see cref="PolymorphicAnalyzer.GetReferencedTypes"/> maps variants to runtime types.</summary>
    [Fact]
    public void GetReferencedTypes_selects_runtime_types()
    {
        const string source = @"
namespace PolyAsm {
  public sealed class Only { }
  public sealed class Registry {
    private static readonly System.Type T1 = typeof(Only);
  }
}";
        (Assembly reflection, AssemblyDefinition cecil) = TestCompilation.Compile(source);
        Type registry = reflection.GetType("PolyAsm.Registry", throwOnError: true)!;
        var analyzer = new PolymorphicAnalyzer(cecil, reflection);
        var variants = analyzer.ExtractVariants(registry);
        List<Type> refs = PolymorphicAnalyzer.GetReferencedTypes(variants);
        Assert.Single(refs);
        Assert.Equal("Only", refs[0].Name);
    }
}
