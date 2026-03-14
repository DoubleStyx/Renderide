namespace SharedTypeGenerator.IR;

/// <summary>Classifies each C# type into exactly one structural shape
/// that determines which emission strategy to use.</summary>
public enum TypeShape
{
    /// <summary>C# enum without [Flags].</summary>
    ValueEnum,

    /// <summary>C# enum with [Flags] attribute.</summary>
    FlagsEnum,

    /// <summary>ExplicitLayout struct with all blittable fields.</summary>
    PodStruct,

    /// <summary>Class or struct implementing IMemoryPackable with Pack/Unpack methods.</summary>
    PackableStruct,

    /// <summary>Abstract class extending PolymorphicMemoryPackableEntity{T} with a type registry.</summary>
    PolymorphicBase,

    /// <summary>Value type not implementing IMemoryPackable (e.g., Guid).</summary>
    GeneralStruct,
}
