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
        _packParser = new PackMethodParser(_assemblyDef, _classifier);
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
        catch (Exception e) when (e is FileNotFoundException or BadImageFormatException or FileLoadException or ReflectionTypeLoadException)
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
            return;

        result.Add(descriptor);
    }

    private TypeDescriptor? AnalyzeType(Type type)
    {
        TypeShape shape = ClassifyShape(type);
        _logger.LogDebug(LogCategory.Analysis, $"Analyzing {type.FullName} as {shape}");

        TypeDescriptor? descriptor = shape switch
        {
            TypeShape.PolymorphicBase => AnalyzePolymorphic(type),
            TypeShape.ValueEnum => AnalyzeEnumCore(type, TypeShape.ValueEnum, RustTypeMapper.MapType(type, _assembly).HumanizeType()),
            TypeShape.FlagsEnum => AnalyzeEnumCore(type, TypeShape.FlagsEnum, MapRustName(type)),
            TypeShape.PodStruct => AnalyzePodStruct(type),
            TypeShape.PackableStruct => AnalyzePackableStruct(type),
            TypeShape.GeneralStruct => AnalyzeGeneralStruct(type),
            _ => null,
        };

        if (descriptor == null)
        {
            _logger.LogWarning(
                LogCategory.Analysis,
                $"Could not analyze type: {type.FullName} (shape={shape}; unsupported classification, sub-analyzer returned null, or missing metadata)");
        }

        return descriptor;
    }

    /// <summary>
    /// Top-level shape dispatch: classifies <paramref name="type"/> into the <see cref="TypeShape"/>
    /// that drives sub-analyzer selection. Internal so unit tests can exercise dispatch without
    /// running a full <see cref="Analyze"/> pass.
    /// </summary>
    internal TypeShape ClassifyShape(Type type)
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
        if (CecilLayoutInspector.HasExplicitLayout(_assemblyDef, type))
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

    /// <summary>Maps a CLR type name to the Rust type identifier used in generated code (nested types use <c>Outer_Inner</c>).</summary>
    internal static string MapRustName(Type type)
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

}
