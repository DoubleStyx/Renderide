using System.Reflection;
using Mono.Cecil;
using SharedTypeGenerator.Analysis;
using SharedTypeGenerator.IR;
using Xunit;

namespace SharedTypeGenerator.Tests.Unit;

/// <summary>Unit tests for <see cref="PackMethodParser"/> using in-memory assemblies.</summary>
public sealed class PackMethodParserTests
{
    private static PackMethodParser CreateParser(AssemblyDefinition cecil) =>
        new(cecil, new FieldClassifier(new WellKnownTypes(typeof(PackMethodParserTests).Assembly.GetTypes())));

    /// <summary>A single <c>Write&lt;T&gt;</c> call produces a <see cref="WriteField"/> with <see cref="FieldKind.Pod"/>.</summary>
    [Fact]
    public void ParseWithConditionals_single_write_field()
    {
        const string source = @"
namespace PackAsm {
  public struct MemoryPacker {
    public void Write<T>(ref T value) where T : unmanaged { }
  }
  public sealed class S {
    public int field;
    public void Pack(ref MemoryPacker p) {
      p.Write(ref field);
    }
  }
}";
        (Assembly reflection, AssemblyDefinition cecil) = TestCompilation.Compile(source);
        Type s = reflection.GetType("PackAsm.S", throwOnError: true)!;
        FieldInfo[] fields = s.GetFields(BindingFlags.Public | BindingFlags.Instance);
        PackMethodParser parser = CreateParser(cecil);
        List<SerializationStep> steps = parser.ParseWithConditionals(s, fields);
        WriteField? wf = Assert.Single(steps) as WriteField;
        Assert.NotNull(wf);
        Assert.Equal("field", wf.FieldName);
        Assert.Equal(FieldKind.Pod, wf.Kind);
    }

    /// <summary>Conditional <c>if</c> around writes becomes a <see cref="ConditionalBlock"/>.</summary>
    [Fact]
    public void ParseWithConditionals_conditional_block()
    {
        const string source = @"
namespace PackAsm {
  public struct MemoryPacker {
    public void Write<T>(ref T value) where T : unmanaged { }
  }
  public sealed class C {
    public bool flag;
    public int inner;
    public void Pack(ref MemoryPacker p) {
      if (flag) {
        p.Write(ref inner);
      }
    }
  }
}";
        (Assembly reflection, AssemblyDefinition cecil) = TestCompilation.Compile(source);
        Type t = reflection.GetType("PackAsm.C", throwOnError: true)!;
        FieldInfo[] fields = t.GetFields(BindingFlags.Public | BindingFlags.Instance);
        PackMethodParser parser = CreateParser(cecil);
        List<SerializationStep> steps = parser.ParseWithConditionals(t, fields);
        ConditionalBlock? cb = Assert.Single(steps) as ConditionalBlock;
        Assert.NotNull(cb);
        Assert.Equal("flag", cb.ConditionField);
        WriteField? inner = Assert.Single(cb.Steps) as WriteField;
        Assert.NotNull(inner);
        Assert.Equal("inner", inner.FieldName);
    }

    /// <summary>Missing <c>Pack</c> on a derived type with a base implementation yields <see cref="CallBase"/>.</summary>
    [Fact]
    public void ParseWithConditionals_call_base_when_pack_missing()
    {
        const string source = @"
namespace PackAsm {
  public struct MemoryPacker {
    public void Write<T>(ref T value) where T : unmanaged { }
  }
  public class Base {
    public int x;
    public void Pack(ref MemoryPacker p) { p.Write(ref x); }
  }
  public sealed class Derived : Base {
  }
}";
        (Assembly reflection, AssemblyDefinition cecil) = TestCompilation.Compile(source);
        Type derived = reflection.GetType("PackAsm.Derived", throwOnError: true)!;
        FieldInfo[] fields = derived.GetFields(BindingFlags.Public | BindingFlags.Instance);
        PackMethodParser parser = CreateParser(cecil);
        List<SerializationStep> steps = parser.ParseWithConditionals(derived, fields);
        Assert.IsType<CallBase>(Assert.Single(steps));
    }

    /// <summary><see cref="PackMethodParser.ParseUnpackOnlySteps"/> captures <c>DateTime.UtcNow</c> assignments.</summary>
    [Fact]
    public void ParseUnpackOnlySteps_timestamp_now()
    {
        const string source = @"
namespace PackAsm {
  public struct MemoryUnpacker { }
  public sealed class U {
    public System.DateTime decodedTime;
    public void Unpack(ref MemoryUnpacker u) {
      decodedTime = System.DateTime.UtcNow;
    }
  }
}";
        (Assembly reflection, AssemblyDefinition cecil) = TestCompilation.Compile(source);
        Type u = reflection.GetType("PackAsm.U", throwOnError: true)!;
        PackMethodParser parser = CreateParser(cecil);
        List<SerializationStep> steps = parser.ParseUnpackOnlySteps(u);
        TimestampNow? ts = Assert.Single(steps) as TimestampNow;
        Assert.NotNull(ts);
        Assert.Equal("decoded_time", ts.FieldName);
    }
}
