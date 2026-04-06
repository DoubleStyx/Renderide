using System.Collections.Generic;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using LayoutKind = System.Runtime.InteropServices.LayoutKind;
using SharedTypeGenerator.IR;

namespace SharedTypeGenerator.Analysis;

/// <summary>Per-<see cref="TypeShape"/> analysis methods for <see cref="TypeAnalyzer"/>.</summary>
public partial class TypeAnalyzer
{
    private TypeDescriptor AnalyzePolymorphic(Type type)
    {
        List<PolymorphicVariant> variants = _polyAnalyzer.ExtractVariants(type);
        foreach (Type refType in _polyAnalyzer.GetReferencedTypes(variants))
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

    private TypeDescriptor AnalyzeValueEnum(Type type)
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
            if (!seen.Add(name)) continue;

            object? num = valueField.GetValue(value);
            Debug.Assert(num != null);

            members.Add(new EnumMember { Name = name, Value = num, IsDefault = first });
            first = false;
        }

        return new TypeDescriptor
        {
            CSharpName = type.Name,
            RustName = RustTypeMapper.MapType(type, _assembly).HumanizeType(),
            Shape = TypeShape.ValueEnum,
            Fields = [],
            UnderlyingEnumType = underlyingType,
            RustUnderlyingType = rustUnderlying,
            EnumMembers = members,
        };
    }

    private TypeDescriptor AnalyzeFlagsEnum(Type type)
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
            if (!seen.Add(name)) continue;

            object? num = valueField.GetValue(value);
            Debug.Assert(num != null);

            members.Add(new EnumMember { Name = name, Value = num, IsDefault = first });
            first = false;
        }

        return new TypeDescriptor
        {
            CSharpName = type.Name,
            RustName = MapRustName(type),
            Shape = TypeShape.FlagsEnum,
            Fields = [],
            UnderlyingEnumType = underlyingType,
            RustUnderlyingType = rustUnderlying,
            EnumMembers = members,
        };
    }

    private TypeDescriptor AnalyzePodStruct(Type type)
    {
        FieldInfo[] fields = type.GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
        var fieldDescriptors = new List<FieldDescriptor>();

        int totalSize = 0;
        bool allFieldsPod = true;

        foreach (FieldInfo field in fields)
        {
            FieldOffsetAttribute? offset = field.GetCustomAttribute<FieldOffsetAttribute>();

            Type sizeType = field.FieldType == typeof(bool) ? typeof(byte) : field.FieldType;
            if (sizeType.IsEnum) sizeType = sizeType.GetField("value__")!.FieldType;
            try { totalSize += Marshal.SizeOf(sizeType); } catch { /* skip */ }

            bool fieldRustLayoutPod = IsRustLayoutPodField(field.FieldType, new HashSet<Type>());
            if (!fieldRustLayoutPod)
                allFieldsPod = false;

            string rustType = field.FieldType == typeof(bool) ? "u8" : MapRustTypeWithQueue(field.FieldType);
            FieldKind kind = _classifier.ClassifyByType(field.FieldType);
            if (kind == FieldKind.Pod && !fieldRustLayoutPod)
                kind = FieldKind.ObjectRequired;

            fieldDescriptors.Add(new FieldDescriptor
            {
                CSharpName = field.Name,
                RustName = field.Name.HumanizeField(),
                RustType = rustType,
                Kind = kind,
                ExplicitOffset = offset?.Value,
            });
        }

        var layout = type.GetCustomAttribute<StructLayoutAttribute>();
        int declaredSize = (layout?.Value == LayoutKind.Explicit && layout.Size > 0) ? layout.Size : 0;
        if (declaredSize == 0)
            declaredSize = GetExplicitLayoutSizeViaCecil(type);

        int paddingBytes = 0;

        try
        {
            if (fields.Length > 0 && fields.Any(f => f.GetCustomAttribute<FieldOffsetAttribute>() != null) && declaredSize > 0)
            {
                var offsetSizePairs = new List<(int Offset, int Size)>();
                for (int i = 0; i < fields.Length; i++)
                {
                    FieldInfo field = fields[i];
                    int offset = field.GetCustomAttribute<FieldOffsetAttribute>()?.Value ?? 0;
                    Type st = field.FieldType == typeof(bool) ? typeof(byte) : field.FieldType;
                    if (st.IsEnum) st = st.GetField("value__")!.FieldType;
                    int size;
                    try { size = Marshal.SizeOf(st); } catch { size = 0; }
                    offsetSizePairs.Add((offset, size));
                }

                offsetSizePairs.Sort((a, b) => a.Offset.CompareTo(b.Offset));

                int paddingIndex = 0;
                for (int i = 0; i < offsetSizePairs.Count; i++)
                {
                    (int offset, int size) = offsetSizePairs[i];
                    int gapEnd = offset + size;
                    int nextStart = i + 1 < offsetSizePairs.Count
                        ? offsetSizePairs[i + 1].Offset
                        : declaredSize;
                    int gap = nextStart - gapEnd;
                    if (gap > 0)
                    {
                        string padName = paddingIndex == 0 ? "_padding" : $"_padding_{paddingIndex}";
                        fieldDescriptors.Add(new FieldDescriptor
                        {
                            CSharpName = padName,
                            RustName = padName,
                            RustType = $"[u8; {gap}]",
                            Kind = FieldKind.Pod,
                            ExplicitOffset = gapEnd,
                        });
                        paddingBytes += gap;
                        paddingIndex++;
                    }
                }
            }
            else if (declaredSize == 0)
            {
                int actualSize = Marshal.SizeOf(type);
                paddingBytes = Math.Max(0, actualSize - totalSize);
            }
        }
        catch
        {
            /* fallback: paddingBytes stays 0 */
        }

        // Top-level fields that map to Quat/Mat4/Vec4 can force Rust-only padding when combined with other fields.
        bool hasSimdCompositePaddingRisk = fields.Length > 1 && fields.Any(f =>
        {
            string rustT = f.FieldType == typeof(bool) ? "u8" : RustTypeMapper.MapType(f.FieldType, _assembly);
            return RustTypeMapper.IsGlamRustTypeRequiringCompositeNonPod(rustT);
        });
        bool isPod = allFieldsPod && !hasSimdCompositePaddingRisk;

        return new TypeDescriptor
        {
            CSharpName = type.Name,
            RustName = MapRustName(type),
            Shape = TypeShape.PodStruct,
            Fields = fieldDescriptors,
            IsPod = isPod,
            ExplicitSize = declaredSize > 0 ? declaredSize : null,
            PaddingBytes = paddingBytes,
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
