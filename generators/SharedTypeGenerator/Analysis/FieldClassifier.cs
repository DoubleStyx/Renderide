using System.Collections;
using System.Reflection;
using SharedTypeGenerator.IR;

namespace SharedTypeGenerator.Analysis;

/// <summary>
/// Central classification of CLR field types into <see cref="FieldKind"/> for IR and emission.
/// </summary>
public class FieldClassifier
{
    private readonly Type _iMemoryPackable;
    private readonly Type _polymorphicBase;

    /// <summary>Creates a classifier using well-known types from the shared assembly.</summary>
    public FieldClassifier(WellKnownTypes wellKnown)
    {
        _iMemoryPackable = wellKnown.IMemoryPackable;
        _polymorphicBase = wellKnown.PolymorphicMemoryPackableEntityDefinition;
    }

    /// <summary>Classifies a field based on its type and the C# Pack method name used to serialize it.
    /// The packMethodName disambiguates when field type alone is insufficient.</summary>
    public FieldKind Classify(Type fieldType, string packMethodName)
    {
        return packMethodName switch
        {
            "WriteObject" or "ReadObject" => FieldKind.Object,
            "WriteValueList" or "ReadValueList" or "WriteEnumValueList" or "ReadEnumValueList" => ClassifyValueListElement(fieldType),
            "WriteObjectList" or "ReadObjectList" => FieldKind.ObjectList,
            "WritePolymorphicList" or "ReadPolymorphicList" => FieldKind.PolymorphicList,
            "WriteStringList" or "ReadStringList" => FieldKind.StringList,
            "WriteNestedValueList" or "ReadNestedValueList" => FieldKind.NestedValueList,
            "Write" or "Read" => ClassifyByFieldType(fieldType, expandLists: false),
            _ => ClassifyByType(fieldType),
        };
    }

    /// <summary>Classifies a field purely by its type, without a known C# method name.
    /// Used for ExplicitLayout and GeneralStruct fields where we don't parse Pack IL.</summary>
    public FieldKind ClassifyByType(Type fieldType) =>
        ClassifyByFieldType(fieldType, expandLists: true);

    /// <summary>
    /// Core classification shared by <see cref="ClassifyByType"/> and the <c>Write</c>/<c>Read</c> overload path.
    /// When <paramref name="expandLists"/> is true, generic <see cref="IEnumerable"/> implementations are classified as lists; otherwise they fall through to POD rules.
    /// </summary>
    private FieldKind ClassifyByFieldType(Type fieldType, bool expandLists)
    {
        if (fieldType == typeof(string))
            return FieldKind.String;

        if (fieldType == typeof(bool))
            return FieldKind.Bool;

        if (fieldType.IsEnum)
            return HasFlagsAttribute(fieldType) ? FieldKind.FlagsEnum : FieldKind.Enum;

        if (fieldType.Name == "Nullable`1")
            return FieldKind.Nullable;

        if (expandLists && typeof(IEnumerable).IsAssignableFrom(fieldType) && fieldType.IsGenericType)
            return ClassifyListType(fieldType);

        if (fieldType.IsClass && fieldType.IsAssignableTo(_iMemoryPackable))
            return FieldKind.Object;

        if (fieldType.IsValueType && !fieldType.IsPrimitive && fieldType != typeof(Guid)
            && !fieldType.Name.StartsWith("SharedMemoryBufferDescriptor", StringComparison.Ordinal)
            && fieldType.IsAssignableTo(_iMemoryPackable))
            return FieldKind.ObjectRequired;

        return FieldKind.Pod;
    }

    /// <summary>When C# calls WriteValueList, it bulk-copies elements as raw values.
    /// We only distinguish enum lists (EnumValueList) from plain value lists.</summary>
    private static FieldKind ClassifyValueListElement(Type listType)
    {
        if (!listType.IsGenericType || listType.GenericTypeArguments.Length == 0)
            return FieldKind.ValueList;

        Type elemType = listType.GenericTypeArguments[0];

        if (elemType.IsEnum)
            return FieldKind.EnumValueList;

        return FieldKind.ValueList;
    }

    private FieldKind ClassifyListType(Type listType)
    {
        if (!listType.IsGenericType || listType.GenericTypeArguments.Length == 0)
            return FieldKind.ValueList;

        Type elemType = listType.GenericTypeArguments[0];

        if (elemType == typeof(string))
            return FieldKind.StringList;

        if (typeof(IEnumerable).IsAssignableFrom(elemType) && elemType.IsGenericType)
            return FieldKind.NestedValueList;

        if (elemType.IsEnum)
            return FieldKind.EnumValueList;

        if (IsPolymorphicEntity(elemType))
            return FieldKind.PolymorphicList;

        if (elemType.IsAssignableTo(_iMemoryPackable))
            return FieldKind.ObjectList;

        return FieldKind.ValueList;
    }

    private bool IsPolymorphicEntity(Type type)
    {
        Type? baseType = type.BaseType;
        while (baseType != null)
        {
            if (baseType.IsGenericType && baseType.GetGenericTypeDefinition() == _polymorphicBase)
                return true;
            baseType = baseType.BaseType;
        }

        return false;
    }

    private static bool HasFlagsAttribute(Type enumType) =>
        enumType.GetCustomAttribute<FlagsAttribute>() != null;
}
