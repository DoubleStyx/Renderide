using System.Reflection;
using NotEnoughLogs;
using NotEnoughLogs.Behaviour;
using SharedTypeGenerator.Analysis;
using SharedTypeGenerator.IR;
using SharedTypeGenerator.Tests.Unit.Support;
using Xunit;

namespace SharedTypeGenerator.Tests.Unit;

/// <summary>
/// Validates <see cref="TypeAnalyzer.ClassifyShape"/> across every <see cref="TypeShape"/>.
/// Top-level dispatch was previously only exercised indirectly via the cross-language
/// roundtrip suite, where a misclassification surfaces as an opaque pack failure.
/// </summary>
public sealed class TypeAnalyzerShapeDispatchTests
{
    private const string Source = @"
namespace ShapeAsm {
  public interface IMemoryPackable { }
  public abstract class PolymorphicMemoryPackableEntity<T> : IMemoryPackable { }

  public enum MyEnum { A, B }

  [System.Flags]
  public enum MyFlags { None = 0, A = 1, B = 2 }

  [System.Runtime.InteropServices.StructLayout(
      System.Runtime.InteropServices.LayoutKind.Explicit, Size = 8)]
  public struct MyPod {
    [System.Runtime.InteropServices.FieldOffset(0)] public int A;
    [System.Runtime.InteropServices.FieldOffset(4)] public int B;
  }

  public sealed class MyPackable : IMemoryPackable { }

  public abstract class MyPolyBase : PolymorphicMemoryPackableEntity<MyPolyBase> { }

  public struct MyGeneral { public int X; }
}";

    /// <summary>One assertion per <see cref="TypeShape"/> covers the dispatch surface.</summary>
    [Fact]
    public void ClassifyShape_covers_every_type_shape()
    {
        (Assembly asm, _, string path) = TestCompilation.CompileToFile(Source, assemblyName: "ShapeAsm");
        using var logger = new Logger(
            [new CollectingSink()],
            new LoggerConfiguration { Behaviour = new DirectLoggingBehaviour(), MaxLevel = LogLevel.Trace });
        var analyzer = new TypeAnalyzer(logger, path);

        Assert.Equal(TypeShape.ValueEnum, analyzer.ClassifyShape(GetType(asm, "ShapeAsm.MyEnum")));
        Assert.Equal(TypeShape.FlagsEnum, analyzer.ClassifyShape(GetType(asm, "ShapeAsm.MyFlags")));
        Assert.Equal(TypeShape.PodStruct, analyzer.ClassifyShape(GetType(asm, "ShapeAsm.MyPod")));
        Assert.Equal(TypeShape.PackableStruct, analyzer.ClassifyShape(GetType(asm, "ShapeAsm.MyPackable")));
        Assert.Equal(TypeShape.PolymorphicBase, analyzer.ClassifyShape(GetType(asm, "ShapeAsm.MyPolyBase")));
        Assert.Equal(TypeShape.GeneralStruct, analyzer.ClassifyShape(GetType(asm, "ShapeAsm.MyGeneral")));
    }

    private static Type GetType(Assembly asm, string fullName) =>
        asm.GetType(fullName, throwOnError: true)!;
}
