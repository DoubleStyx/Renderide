using System.Collections;
using SharedTypeGenerator.Analysis;
using SharedTypeGenerator.IR;
using Xunit;

namespace SharedTypeGenerator.Tests.Unit;

/// <summary>Unit tests for <see cref="FieldClassifier"/>.</summary>
public sealed class FieldClassifierTests
{
    private readonly FieldClassifier _classifier;

    /// <summary>Creates a classifier using well-known types from this test assembly.</summary>
    public FieldClassifierTests()
    {
        var wellKnown = new WellKnownTypes(typeof(FieldClassifierTests).Assembly.GetTypes());
        _classifier = new FieldClassifier(wellKnown);
    }

    /// <summary><see cref="FieldClassifier.ClassifyByType"/> covers common primitives and collections.</summary>
    [Fact]
    public void ClassifyByType_covers_primitives_and_lists()
    {
        Assert.Equal(FieldKind.String, _classifier.ClassifyByType(typeof(string)));
        Assert.Equal(FieldKind.Bool, _classifier.ClassifyByType(typeof(bool)));
        Assert.Equal(FieldKind.Enum, _classifier.ClassifyByType(typeof(PlainEnum)));
        Assert.Equal(FieldKind.FlagsEnum, _classifier.ClassifyByType(typeof(FlagsEnum)));
        Assert.Equal(FieldKind.Nullable, _classifier.ClassifyByType(typeof(int?)));
        Assert.Equal(FieldKind.ValueList, _classifier.ClassifyByType(typeof(List<int>)));
        Assert.Equal(FieldKind.StringList, _classifier.ClassifyByType(typeof(List<string>)));
        Assert.Equal(FieldKind.NestedValueList, _classifier.ClassifyByType(typeof(List<List<int>>)));
        Assert.Equal(FieldKind.ObjectList, _classifier.ClassifyByType(typeof(List<TestPackable>)));
        Assert.Equal(FieldKind.PolymorphicList, _classifier.ClassifyByType(typeof(List<TestPolymorphicLeaf>)));
    }

    /// <summary>Enum element lists are distinguished from plain value lists via the pack method name.</summary>
    [Fact]
    public void Classify_write_value_list_vs_enum_value_list()
    {
        Assert.Equal(FieldKind.EnumValueList,
            _classifier.Classify(typeof(List<PlainEnum>), "WriteEnumValueList"));
        Assert.Equal(FieldKind.ValueList,
            _classifier.Classify(typeof(List<int>), "WriteEnumValueList"));
    }

    /// <summary><c>Write</c>/<c>Read</c> uses the shared atomic classification path.</summary>
    [Fact]
    public void Classify_write_read_uses_atomic_path_for_int()
    {
        Assert.Equal(FieldKind.Pod, _classifier.Classify(typeof(int), "Write"));
        Assert.Equal(FieldKind.Pod, _classifier.Classify(typeof(int), "Read"));
    }

    private enum PlainEnum
    {
        A,
        B,
    }

    [Flags]
    private enum FlagsEnum
    {
        None = 0,
        X = 1,
    }
}
