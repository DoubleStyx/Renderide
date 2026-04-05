namespace SharedTypeGenerator.IR;

/// <summary>One operation in the serialization sequence, produced by parsing
/// the Pack method IL. The same list drives both pack and unpack emission.</summary>
public abstract record SerializationStep;

/// <summary>Write/read a single field. The FieldKind determines the Rust method to call.</summary>
public record WriteField(string FieldName, FieldKind Kind) : SerializationStep;

/// <summary>Multiple bool fields packed into a single byte.</summary>
public record PackedBools(List<string> FieldNames) : SerializationStep;

/// <summary>Delegates to the base type's serialization (base.Pack / base.Unpack).</summary>
public record CallBase : SerializationStep;

/// <summary>Fields guarded by an if-check on a previously deserialized bool field.
/// Recursive: inner steps can contain nested ConditionalBlocks.</summary>
public record ConditionalBlock(string ConditionField, List<SerializationStep> Steps) : SerializationStep;

/// <summary>Writes the current UTC timestamp as an i128 nanosecond value.</summary>
public record TimestampNow(string FieldName) : SerializationStep;
