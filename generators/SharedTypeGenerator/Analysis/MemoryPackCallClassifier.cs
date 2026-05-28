using Mono.Cecil;

namespace SharedTypeGenerator.Analysis;

/// <summary>Classifies MemoryPack method calls into serializer operations without exposing raw method-name switches to callers.</summary>
internal static class MemoryPackCallClassifier
{
    /// <summary>Classifies a Cecil method reference from pack or unpack IL.</summary>
    public static MemoryPackOperation Classify(MethodReference callRef)
    {
        string[] parameterTypeNames = callRef.Parameters
            .Select(static p => p.ParameterType.Name)
            .ToArray();
        return Classify(callRef.Name, parameterTypeNames);
    }

    /// <summary>Classifies a method name and parameter type names; exposed for focused tests without Cecil setup.</summary>
    public static MemoryPackOperation Classify(string methodName, IReadOnlyList<string> parameterTypeNames)
    {
        return methodName switch
        {
            "Write" when parameterTypeNames.Count == 1 => MemoryPackOperation.Write,
            "Write" when AllParametersAre(parameterTypeNames, "Boolean") => MemoryPackOperation.PackedBools,
            "WriteObject" => MemoryPackOperation.WriteObject,
            "WriteObjectRequired" => MemoryPackOperation.WriteObjectRequired,
            "WriteValueList" or "WriteEnumValueList" => MemoryPackOperation.WriteValueList,
            "WriteObjectList" => MemoryPackOperation.WriteObjectList,
            "WritePolymorphicList" => MemoryPackOperation.WritePolymorphicList,
            "WriteStringList" => MemoryPackOperation.WriteStringList,
            "WriteNestedValueList" => MemoryPackOperation.WriteNestedValueList,
            "Pack" or "Unpack" => MemoryPackOperation.CallBase,
            "Read" when parameterTypeNames.Count == 1 => MemoryPackOperation.Ignore,
            "Read" when AllParametersAre(parameterTypeNames, "Boolean&") => MemoryPackOperation.Ignore,
            "ReadObject" or "ReadValueList" or "ReadEnumValueList" or "ReadObjectList" or "ReadPolymorphicList" or "ReadStringList" or "ReadNestedValueList" => MemoryPackOperation.Ignore,
            _ => MemoryPackOperation.Ignore,
        };
    }

    private static bool AllParametersAre(IReadOnlyList<string> parameterTypeNames, string expectedName) =>
        parameterTypeNames.Count > 0 && parameterTypeNames.All(name => name == expectedName);
}
