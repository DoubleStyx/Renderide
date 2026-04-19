using System.Diagnostics;
using System.Reflection;
using SharedTypeGenerator.IR;

namespace SharedTypeGenerator.Analysis;

/// <summary>Per-<see cref="TypeShape"/> analysis methods for <see cref="TypeAnalyzer"/>.</summary>
public partial class TypeAnalyzer
{
    private TypeDescriptor AnalyzePolymorphic(Type type)
    {
        List<PolymorphicVariant> variants = _polyAnalyzer.ExtractVariants(type);
        foreach (Type refType in PolymorphicAnalyzer.GetReferencedTypes(variants))
            EnqueueType(refType);

        return new TypeDescriptor
        {
            CSharpName = type.Name,
            RustName = type.Name.HumanizeType(),
            Shape = TypeShape.PolymorphicBase,
            Fields = [],
            Variants = variants,
        };
    }

    /// <summary>Shared enum member extraction for <see cref="TypeShape.ValueEnum"/> and <see cref="TypeShape.FlagsEnum"/>.</summary>
    private static TypeDescriptor AnalyzeEnumCore(Type type, TypeShape shape, string rustName)
    {
        FieldInfo valueField = type.GetField("value__")!;
        Type underlyingType = valueField.FieldType;
        string rustUnderlying = RustTypeMapper.MapPrimitiveType(underlyingType);

        Array values = Enum.GetValues(type);
        var members = new List<EnumMember>();
        var seen = new HashSet<string>();
        bool first = true;

        foreach (object value in values)
        {
            string? name = value.ToString();
            Debug.Assert(name != null);
            if (!seen.Add(name))
                continue;

            object? num = valueField.GetValue(value);
            Debug.Assert(num != null);

            members.Add(new EnumMember { Name = name, Value = num, IsDefault = first });
            first = false;
        }

        return new TypeDescriptor
        {
            CSharpName = type.Name,
            RustName = rustName,
            Shape = shape,
            Fields = [],
            UnderlyingEnumType = underlyingType,
            RustUnderlyingType = rustUnderlying,
            EnumMembers = members,
        };
    }

    private TypeDescriptor AnalyzePackableStruct(Type type)
    {
        FieldInfo[] fields = type.GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);

        var fieldDescriptors = new List<FieldDescriptor>();
        foreach (FieldInfo field in fields)
        {
            string rustType = MapRustTypeWithQueue(field.FieldType);
            FieldKind kind = _classifier.ClassifyByType(field.FieldType);

            fieldDescriptors.Add(new FieldDescriptor
            {
                CSharpName = field.Name,
                RustName = field.Name.HumanizeField(),
                RustType = rustType,
                Kind = kind,
            });
        }

        List<SerializationStep> steps = _packParser.ParseWithConditionals(type, fields);
        steps = ResolveCallBases(type, steps, fields);
        List<SerializationStep> unpackOnlySteps = _packParser.ParseUnpackOnlySteps(type);

        return new TypeDescriptor
        {
            CSharpName = type.Name,
            RustName = MapRustName(type),
            Shape = TypeShape.PackableStruct,
            Fields = fieldDescriptors,
            PackSteps = steps,
            UnpackOnlySteps = unpackOnlySteps,
        };
    }

    private TypeDescriptor AnalyzeGeneralStruct(Type type)
    {
        FieldInfo[] fields = type.GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);

        var fieldDescriptors = new List<FieldDescriptor>();
        foreach (FieldInfo field in fields)
        {
            string rustType = MapRustTypeWithQueue(field.FieldType);
            fieldDescriptors.Add(new FieldDescriptor
            {
                CSharpName = field.Name,
                RustName = field.Name.HumanizeField(),
                RustType = rustType,
                Kind = _classifier.ClassifyByType(field.FieldType),
            });
        }

        bool isPod = type == typeof(Guid);
        bool shouldPack = type == typeof(Guid);

        // Non-explicit C# structs still map to `#[repr(C)]` scalars in Rust. When every field is a
        // plain scalar and classified as `Pod`, the layout matches `bytemuck::Pod` (same rule as
        // blittable rows used for shared-memory copy helpers — e.g. `MaterialOverrideState`).
        if (!isPod
            && fieldDescriptors.Count > 0
            && fieldDescriptors.All(f => f.Kind == FieldKind.Pod && IsPlainRustScalarLayoutType(f.RustType)))
        {
            isPod = true;
        }

        List<SerializationStep> steps = [];
        if (shouldPack)
        {
            foreach (FieldInfo field in fields)
            {
                string rustName = field.Name.HumanizeField();
                steps.Add(new WriteField(rustName, FieldKind.Pod));
            }
        }

        return new TypeDescriptor
        {
            CSharpName = type.Name,
            RustName = type == typeof(Guid) ? "Guid" : MapRustName(type),
            Shape = TypeShape.GeneralStruct,
            Fields = fieldDescriptors,
            PackSteps = steps,
            IsPod = isPod,
        };
    }

    /// <summary>Returns true when <paramref name="rustType"/> is a single scalar field suitable for
    /// `#[repr(C)]` layout without glam vectors (avoids SIMD padding mismatches vs. C#).</summary>
    internal static bool IsPlainRustScalarLayoutType(string rustType) =>
        rustType is "i32" or "i64" or "u32" or "u64" or "u16" or "i16" or "u8" or "f32" or "f64";

    /// <summary>Recursively replaces CallBase steps with the inlined serialization
    /// steps from the base type's Pack method.</summary>
    private List<SerializationStep> ResolveCallBases(Type type, List<SerializationStep> steps, FieldInfo[] allFields)
    {
        var resolved = new List<SerializationStep>();
        foreach (SerializationStep step in steps)
        {
            if (step is CallBase)
            {
                Type? baseType = type.BaseType;
                if (baseType != null &&
                    !(baseType.IsGenericType && baseType.GetGenericTypeDefinition() == _polymorphicBase))
                {
                    List<SerializationStep> baseSteps = _packParser.ParseWithConditionals(baseType, allFields);
                    resolved.AddRange(ResolveCallBases(baseType, baseSteps, allFields));
                }
            }
            else if (step is ConditionalBlock cb)
            {
                resolved.Add(new ConditionalBlock(cb.ConditionField,
                    ResolveCallBases(type, cb.Steps, allFields)));
            }
            else
            {
                resolved.Add(step);
            }
        }
        return resolved;
    }
}
