using System.Diagnostics.CodeAnalysis;

namespace SharedTypeGenerator.IR;

/// <summary>Each field gets exactly one classification that determines
/// how it is serialized and what Rust type it maps to.</summary>
[SuppressMessage("Microsoft.Naming", "CA1720:Identifiers should not contain type names", Justification = "Member names mirror C# / wire vocabulary (string, object).")]
public enum FieldKind
{
    /// <summary>Primitive, Guid, SharedMemoryBufferDescriptor -- write/read directly as Pod.</summary>
    Pod,

    /// <summary>bool -- write_bool/read_bool.</summary>
    Bool,

    /// <summary>string -- write_str/read_str.</summary>
    String,

    /// <summary>Non-[Flags] enum -- write_object_required/read_object_required.</summary>
    Enum,

    /// <summary>[Flags] enum -- write/read as repr(transparent) Pod.</summary>
    FlagsEnum,

    /// <summary>T? where T : unmanaged -- write_option/read_option.</summary>
    Nullable,

    /// <summary>Nullable class implementing IMemoryPackable -- write_object/read_object.</summary>
    Object,

    /// <summary>Non-nullable value type implementing IMemoryPackable -- write_object_required/read_object_required.</summary>
    ObjectRequired,

    /// <summary>List{T} where T : unmanaged (Pod) -- write_value_list/read_value_list.</summary>
    ValueList,

    /// <summary>List{T} where T is a non-[Flags] enum -- write_enum_value_list/read_enum_value_list.</summary>
    EnumValueList,

    /// <summary>List{T} where T : IMemoryPackable -- write_object_list/read_object_list.</summary>
    ObjectList,

    /// <summary>List{T} where T : PolymorphicMemoryPackableEntity -- write_polymorphic_list/read_polymorphic_list.</summary>
    PolymorphicList,

    /// <summary>List{string} -- write_string_list/read_string_list.</summary>
    StringList,

    /// <summary>List{List{T}} -- write_nested_value_list/read_nested_value_list.</summary>
    NestedValueList,
}
