using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using Mono.Cecil;
using LayoutKind = System.Runtime.InteropServices.LayoutKind;
using NotEnoughLogs;
using SharedTypeGenerator.IR;
using SharedTypeGenerator.Logging;
using ReflectionTypeAttributes = System.Reflection.TypeAttributes;

namespace SharedTypeGenerator.Analysis;

/// <summary>Frontend orchestrator: loads a compiled C# assembly and produces
/// an ordered list of TypeDescriptors by traversing from RendererCommand.</summary>
public class TypeAnalyzer
{
    private readonly Logger _logger;
    private readonly Assembly _assembly;
    private readonly AssemblyDefinition _assemblyDef;
    private readonly Type[] _types;
    private readonly FieldClassifier _classifier;
    private readonly PackMethodParser _packParser;
    private readonly PolymorphicAnalyzer _polyAnalyzer;

    private readonly Queue<Type> _typeQueue = new();
    private readonly HashSet<Type> _generated = [];

    private readonly Type _iMemoryPackable;
    private readonly Type _polymorphicBase;

    public TypeAnalyzer(Logger logger, string assemblyPath)
    {
        _logger = logger;

        _assembly = Assembly.LoadFrom(assemblyPath);
        _assemblyDef = AssemblyDefinition.ReadAssembly(assemblyPath);
        _types = _assembly.GetTypes();

        _iMemoryPackable = _types.First(t => t.Name == "IMemoryPackable");
        _polymorphicBase = _types.First(t => t.Name == "PolymorphicMemoryPackableEntity`1").GetGenericTypeDefinition();

        _classifier = new FieldClassifier(_types);
        _packParser = new PackMethodParser(_assemblyDef, _assembly, _classifier);
        _polyAnalyzer = new PolymorphicAnalyzer(_assemblyDef, _assembly);
    }

    /// <summary>Determines the engine version string from the FrooxEngine assembly
    /// adjacent to the loaded assembly.</summary>
    public string DetectEngineVersion(string assemblyPath)
    {
        try
        {
            Assembly frooxEngine = Assembly.LoadFrom(
                Path.Combine(Path.GetDirectoryName(assemblyPath)!, "FrooxEngine.dll"));
            return frooxEngine.FullName ?? "Unknown";
        }
        catch (Exception e)
        {
            _logger.LogWarning(LogCategory.Startup, $"Couldn't detect FrooxEngine version: {e.Message}");
            return "Unknown";
        }
    }

    /// <summary>Analyzes all types reachable from RendererCommand,
    /// returning them in generation order as TypeDescriptors.</summary>
    public List<TypeDescriptor> Analyze()
    {
        var result = new List<TypeDescriptor>();

        Type rootType = _types.First(t => t.Name == "RendererCommand");
        AnalyzeAndEnqueue(rootType, result);

        while (_typeQueue.TryDequeue(out Type? type))
        {
            Debug.Assert(type != null);
            AnalyzeAndEnqueue(type, result);
        }

        return result;
    }

    private void AnalyzeAndEnqueue(Type type, List<TypeDescriptor> result)
    {
        if (!_generated.Add(type)) return;

        TypeDescriptor? descriptor = AnalyzeType(type);
        if (descriptor == null)
        {
            _logger.LogWarning(LogCategory.Analysis, $"Could not analyze type: {type.FullName}");
            return;
        }

        result.Add(descriptor);
    }

    private TypeDescriptor? AnalyzeType(Type type)
    {
        TypeShape shape = ClassifyShape(type);
        _logger.LogDebug(LogCategory.Analysis, $"Analyzing {type.FullName} as {shape}");

        return shape switch
        {
            TypeShape.PolymorphicBase => AnalyzePolymorphic(type),
            TypeShape.ValueEnum => AnalyzeValueEnum(type),
            TypeShape.FlagsEnum => AnalyzeFlagsEnum(type),
            TypeShape.PodStruct => AnalyzePodStruct(type),
            TypeShape.PackableStruct => AnalyzePackableStruct(type),
            TypeShape.GeneralStruct => AnalyzeGeneralStruct(type),
            _ => null,
        };
    }

    private TypeShape ClassifyShape(Type type)
    {
        if (type.IsEnum)
            return type.GetCustomAttribute<FlagsAttribute>() != null ? TypeShape.FlagsEnum : TypeShape.ValueEnum;

        if (IsPolymorphicBase(type))
            return TypeShape.PolymorphicBase;

        // ExplicitLayout structs are PodStruct (C# WriteValueList requires T : unmanaged, so these must be Pod)
        if (type.GetCustomAttribute<StructLayoutAttribute>()?.Value == LayoutKind.Explicit)
            return TypeShape.PodStruct;
        if ((type.Attributes & ReflectionTypeAttributes.ExplicitLayout) != 0)
            return TypeShape.PodStruct;
        if (HasExplicitLayoutViaCecil(type))
            return TypeShape.PodStruct;

        if (type != _iMemoryPackable && !type.IsAbstract && type.IsAssignableTo(_iMemoryPackable))
            return TypeShape.PackableStruct;

        if (type.IsValueType && !type.IsEnum)
            return TypeShape.GeneralStruct;

        // Fallback for abstract IMemoryPackable classes that aren't polymorphic bases
        if (type.IsAbstract && type.IsAssignableTo(_iMemoryPackable) && !IsPolymorphicBase(type))
            return TypeShape.PackableStruct;

        return TypeShape.GeneralStruct;
    }

    private bool IsPolymorphicBase(Type type)
    {
        if (type.BaseType is not { IsGenericType: true }) return false;
        return type.BaseType.GetGenericTypeDefinition() == _polymorphicBase;
    }

    /// <summary>Fallback for ExplicitLayout detection when reflection attributes are unavailable
    /// (e.g. types loaded from a different assembly context). Uses Mono.Cecil metadata.</summary>
    private bool HasExplicitLayoutViaCecil(Type type)
    {
        if (!type.IsValueType || type.IsEnum) return false;
        string? fullName = type.FullName;
        if (string.IsNullOrEmpty(fullName)) return false;
        TypeDefinition? typeDef = _assemblyDef.MainModule.GetType(fullName);
        return typeDef != null && (typeDef.Attributes & Mono.Cecil.TypeAttributes.ExplicitLayout) != 0;
    }

    /// <summary>Gets StructLayoutAttribute.Size from Cecil when reflection returns null.</summary>
    private int GetExplicitLayoutSizeViaCecil(Type type)
    {
        try
        {
            string? fullName = type.FullName;
            if (string.IsNullOrEmpty(fullName)) return 0;
            TypeDefinition? typeDef = _assemblyDef.MainModule.GetType(fullName);
            if (typeDef == null) return 0;
            // StructLayoutAttribute.Size is stored in ClassLayout table (ClassSize), not CustomAttributes
            if (typeDef.ClassSize > 0)
                return typeDef.ClassSize;
            CustomAttribute? attr = typeDef.CustomAttributes
                .FirstOrDefault(a => a.AttributeType.Name == "StructLayoutAttribute");
            if (attr == null) return 0;
            foreach (var prop in attr.Properties)
            {
                if (prop.Name == "Size" && prop.Argument.Value is int size && size > 0)
                    return size;
            }
            if (attr.ConstructorArguments.Count >= 2 && attr.ConstructorArguments[1].Value is int sizeArg && sizeArg > 0)
                return sizeArg;
        }
        catch { /* ignore */ }
        return 0;
    }

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

            if (!IsFieldTypePod(field.FieldType, []))
                allFieldsPod = false;

            string rustType = field.FieldType == typeof(bool) ? "u8" : MapRustTypeWithQueue(field.FieldType);
            FieldKind kind = _classifier.ClassifyByType(field.FieldType);

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
            // Compute gaps from ExplicitLayout and add synthetic padding fields at correct offsets
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
        catch { /* fallback: paddingBytes stays 0 */ }

        bool isPod = allFieldsPod;

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

    private string MapRustName(Type type)
    {
        if (type.DeclaringType != null)
            return (type.DeclaringType.Name + '_' + type.Name).HumanizeType();
        return type.Name.HumanizeType();
    }

    private string MapRustTypeWithQueue(Type fieldType)
    {
        var result = RustTypeMapper.Map(fieldType, _assembly);
        foreach (Type refType in result.ReferencedTypes)
            EnqueueType(refType);
        return result.RustType;
    }

    private void EnqueueType(Type type)
    {
        if (_generated.Contains(type) || _typeQueue.Contains(type)) return;
        if (type.Assembly == _assembly || type == typeof(Guid))
            _typeQueue.Enqueue(type);
    }

    private static bool IsFieldTypePod(Type ft, HashSet<Type> visited)
    {
        // All enums with explicit repr (ValueEnum and FlagsEnum) are Pod in Rust
        if (ft.IsEnum) return true;
        if (ft == typeof(bool)) return true;
        if (ft.IsPrimitive || ft == typeof(Guid) || ft.Name?.StartsWith("SharedMemoryBufferDescriptor") == true)
            return true;
        if (ft.IsValueType && !ft.IsEnum && !visited.Contains(ft))
        {
            visited.Add(ft);
            try
            {
                return ft.GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)
                    .All(f => IsFieldTypePod(f.FieldType, visited));
            }
            finally { visited.Remove(ft); }
        }
        return false;
    }
}
