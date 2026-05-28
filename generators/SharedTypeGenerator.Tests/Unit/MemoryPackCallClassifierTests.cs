using SharedTypeGenerator.Analysis;
using Xunit;

namespace SharedTypeGenerator.Tests.Unit;

/// <summary>Unit tests for <see cref="MemoryPackCallClassifier"/>.</summary>
public sealed class MemoryPackCallClassifierTests
{
    /// <summary>Known MemoryPack calls map to semantic operations used by analysis and emission.</summary>
    [Theory]
    [InlineData("Write", new[] { "Int32" }, nameof(MemoryPackOperation.Write))]
    [InlineData("Write", new[] { "Boolean", "Boolean" }, nameof(MemoryPackOperation.PackedBools))]
    [InlineData("WriteObject", new string[] { }, nameof(MemoryPackOperation.WriteObject))]
    [InlineData("WriteObjectRequired", new string[] { }, nameof(MemoryPackOperation.WriteObjectRequired))]
    [InlineData("WriteValueList", new string[] { }, nameof(MemoryPackOperation.WriteValueList))]
    [InlineData("WriteEnumValueList", new string[] { }, nameof(MemoryPackOperation.WriteValueList))]
    [InlineData("WriteObjectList", new string[] { }, nameof(MemoryPackOperation.WriteObjectList))]
    [InlineData("WritePolymorphicList", new string[] { }, nameof(MemoryPackOperation.WritePolymorphicList))]
    [InlineData("WriteStringList", new string[] { }, nameof(MemoryPackOperation.WriteStringList))]
    [InlineData("WriteNestedValueList", new string[] { }, nameof(MemoryPackOperation.WriteNestedValueList))]
    [InlineData("Pack", new string[] { }, nameof(MemoryPackOperation.CallBase))]
    [InlineData("Unpack", new string[] { }, nameof(MemoryPackOperation.CallBase))]
    [InlineData("ReadObject", new string[] { }, nameof(MemoryPackOperation.Ignore))]
    [InlineData("NotMemoryPack", new string[] { }, nameof(MemoryPackOperation.Ignore))]
    public void Classify_maps_method_shapes(string methodName, string[] parameterTypeNames, string expectedName)
    {
        var expected = Enum.Parse<MemoryPackOperation>(expectedName);
        Assert.Equal(expected, MemoryPackCallClassifier.Classify(methodName, parameterTypeNames));
    }
}
