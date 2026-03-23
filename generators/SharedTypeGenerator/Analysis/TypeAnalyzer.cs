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
public partial class TypeAnalyzer
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

    /// <summary>Loads <paramref name="assemblyPath"/> and prepares analyzers for <see cref="Analyze"/>.</summary>
    public TypeAnalyzer(Logger logger, string assemblyPath)
    {
        _logger = logger;

        _assembly = Assembly.LoadFrom(assemblyPath);
        _assemblyDef = AssemblyDefinition.ReadAssembly(assemblyPath);
        _types = _assembly.GetTypes();

        var wellKnown = new WellKnownTypes(_types);
        _iMemoryPackable = wellKnown.IMemoryPackable;
        _polymorphicBase = wellKnown.PolymorphicMemoryPackableEntityDefinition;

        _classifier = new FieldClassifier(wellKnown);
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
