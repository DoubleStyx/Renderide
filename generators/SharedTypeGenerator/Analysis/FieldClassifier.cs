using System.Collections;
using System.Reflection;
using SharedTypeGenerator.IR;

namespace SharedTypeGenerator.Analysis;

/// <summary>Single source of truth for classifying a field into a FieldKind.
/// Replaces the old generator's scattered boolean flag logic
/// (isString, isBool, isPod, hasFlagEnumField, useObjectRequired, etc.)
/// with one deterministic classification method.</summary>
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
            "Write" or "Read" => ClassifyWriteRead(fieldType),
            _ => ClassifyByType(fieldType),
        };
    }

    /// <summary>Classifies a field purely by its type, without a known C# method name.
    /// Used for ExplicitLayout and GeneralStruct fields where we don't parse Pack IL.</summary>
    public FieldKind ClassifyByType(Type fieldType)
    {
        if (fieldType == typeof(string))
            return FieldKind.String;

        if (fieldType == typeof(bool))
            return FieldKind.Bool;

        if (fieldType.IsEnum)
            return HasFlagsAttribute(fieldType) ? FieldKind.FlagsEnum : FieldKind.Enum;

        if (fieldType.Name == "Nullable`1")
            return FieldKind.Nullable;

        if (typeof(IEnumerable).IsAssignableFrom(fieldType) && fieldType.IsGenericType)
            return ClassifyListType(fieldType);

        if (fieldType.IsClass && fieldType.IsAssignableTo(_iMemoryPackable))
            return FieldKind.Object;

        if (fieldType.IsValueType && !fieldType.IsPrimitive && fieldType != typeof(Guid)
            && !fieldType.Name.StartsWith("SharedMemoryBufferDescriptor")
            && fieldType.IsAssignableTo(_iMemoryPackable))
            return FieldKind.ObjectRequired;

        return FieldKind.Pod;
    }

    /// <summary>Classifies a field that uses C#'s overloaded Write/Read method.
    /// When C# calls Write&lt;T&gt;, it bulk-copies the value as raw bytes.
    /// Only string, bool, enum, nullable, and reference-type objects get special treatment.</summary>
    private FieldKind ClassifyWriteRead(Type fieldType)
    {
        if (fieldType == typeof(string))
            return FieldKind.String;

        if (fieldType == typeof(bool))
            return FieldKind.Bool;

        if (fieldType.IsEnum)
            return HasFlagsAttribute(fieldType) ? FieldKind.FlagsEnum : FieldKind.Enum;

        if (fieldType.Name == "Nullable`1")
            return FieldKind.Nullable;

        if (fieldType.IsClass && fieldType.IsAssignableTo(_iMemoryPackable))
            return FieldKind.Object;

        return FieldKind.Pod;
    }

    /// <summary>When C# calls WriteValueList, it bulk-copies elements as raw values.
    /// We only distinguish enum lists (EnumValueList) from plain value lists.</summary>
    private FieldKind ClassifyValueListElement(Type listType)
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

    private bool IsIMemoryPackableValueType(Type type)
    {
        return type.IsValueType && !type.IsPrimitive && type != typeof(Guid)
            && !type.Name.StartsWith("SharedMemoryBufferDescriptor")
            && type.IsAssignableTo(_iMemoryPackable);
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

    private static bool HasFlagsAttribute(Type enumType)
    {
        return enumType.GetCustomAttribute<FlagsAttribute>() != null;
    }

    /// <summary>Checks whether a value type contains fields (bool, enum, or nested struct)
    /// that prevent it from being treated as Pod in Rust.</summary>
    private bool HasNonPodInnerFields(Type type)
    {
        if (!type.IsValueType || type.IsPrimitive || type.IsEnum || type == typeof(Guid))
            return false;
        if (type.Name.StartsWith("SharedMemoryBufferDescriptor"))
            return false;

        return type.GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)
            .Any(f => f.FieldType == typeof(bool)
                    || f.FieldType.IsEnum
                    || (f.FieldType.IsValueType && !f.FieldType.IsPrimitive
                        && f.FieldType != typeof(Guid)
                        && !f.FieldType.Name.StartsWith("SharedMemoryBufferDescriptor")
                        && HasNonPodInnerFields(f.FieldType)));
    }
}
