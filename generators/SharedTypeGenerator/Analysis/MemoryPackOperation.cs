namespace SharedTypeGenerator.Analysis;

/// <summary>MemoryPack call categories that affect generated serialization steps.</summary>
internal enum MemoryPackOperation
{
    /// <summary>Generic scalar or unmanaged write/read operation.</summary>
    Write,

    /// <summary>Nullable object write/read operation.</summary>
    WriteObject,

    /// <summary>Required object write/read operation.</summary>
    WriteObjectRequired,

    /// <summary>Value-list write/read operation.</summary>
    WriteValueList,

    /// <summary>Object-list write/read operation.</summary>
    WriteObjectList,

    /// <summary>Polymorphic-list write/read operation.</summary>
    WritePolymorphicList,

    /// <summary>String-list write/read operation.</summary>
    WriteStringList,

    /// <summary>Nested value-list write/read operation.</summary>
    WriteNestedValueList,

    /// <summary>Multiple bool fields packed into one byte.</summary>
    PackedBools,

    /// <summary>Base type pack/unpack delegation.</summary>
    CallBase,

    /// <summary>Known read-side or unrelated call that does not produce a pack step.</summary>
    Ignore,
}
