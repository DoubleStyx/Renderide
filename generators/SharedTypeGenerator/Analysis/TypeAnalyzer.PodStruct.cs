using System.Reflection;
using System.Runtime.InteropServices;
using LayoutKind = System.Runtime.InteropServices.LayoutKind;
using SharedTypeGenerator.IR;
using SharedTypeGenerator.Logging;

namespace SharedTypeGenerator.Analysis;

/// <summary>Explicit-layout <see cref="TypeShape.PodStruct"/> analysis helpers for <see cref="TypeAnalyzer"/>.</summary>
public partial class TypeAnalyzer
{
    /// <summary>Builds a single <see cref="FieldDescriptor"/> for an explicit-layout struct field.</summary>
    private FieldDescriptor BuildPodStructFieldDescriptor(FieldInfo field)
    {
        FieldOffsetAttribute? offset = field.GetCustomAttribute<FieldOffsetAttribute>();

        bool fieldRustLayoutPod = PodAnalyzer.IsRustLayoutPodField(field.FieldType, new HashSet<Type>(), _assembly);
        string rustType = field.FieldType == typeof(bool) ? "u8" : MapRustTypeWithQueue(field.FieldType);
        FieldKind kind = _classifier.ClassifyByType(field.FieldType);
        if (kind == FieldKind.Pod && !fieldRustLayoutPod)
            kind = FieldKind.ObjectRequired;

        return new FieldDescriptor
        {
            CSharpName = field.Name,
            RustName = field.Name.HumanizeField(),
            RustType = rustType,
            Kind = kind,
            ExplicitOffset = offset?.Value,
        };
    }

    /// <summary>
    /// Inserts synthetic <c>_padding</c> fields between explicit-offset regions to match declared struct size.
    /// </summary>
    /// <returns>Total padding bytes added.</returns>
    private static int ComputeExplicitLayoutGapPadding(
        FieldInfo[] fields,
        int declaredSize,
        List<FieldDescriptor> fieldDescriptors)
    {
        if (fields.Length == 0 || declaredSize <= 0 || !fields.Any(f => f.GetCustomAttribute<FieldOffsetAttribute>() != null))
            return 0;

        var offsetSizePairs = new List<(int Offset, int Size)>();
        foreach (FieldInfo field in fields)
        {
            int offset = field.GetCustomAttribute<FieldOffsetAttribute>()?.Value ?? 0;
            int size = ManagedLayoutSizing.TryGetManagedFieldSize(field, out int s) ? s : 0;
            offsetSizePairs.Add((offset, size));
        }

        offsetSizePairs.Sort((a, b) => a.Offset.CompareTo(b.Offset));

        int paddingBytes = 0;
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

        return paddingBytes;
    }

    /// <summary>Throws when explicit field extents exceed the recorded host-interop size.</summary>
    private static void VerifyHostInteropExtentAgainstFields(FieldInfo[] fields, int hostInteropSizeBytes, Type structType)
    {
        int maxEnd = ManagedLayoutSizing.MaxFieldEndBytes(fields);
        if (maxEnd > hostInteropSizeBytes)
        {
            throw new InvalidOperationException(
                $"{structType.FullName}: explicit layout field extent ({maxEnd} bytes) exceeds Marshal.SizeOf={hostInteropSizeBytes}.");
        }
    }

    /// <summary>
    /// Returns the declared explicit-layout size from <see cref="StructLayoutAttribute.Size"/>,
    /// falling back to Cecil <c>ClassLayout</c> metadata, otherwise 0.
    /// </summary>
    private int ResolveDeclaredOrCecilSize(Type type, StructLayoutAttribute? layout)
    {
        int declaredSize = (layout?.Value == LayoutKind.Explicit && layout.Size > 0) ? layout.Size : 0;
        if (declaredSize == 0)
            declaredSize = CecilLayoutInspector.GetExplicitLayoutSizeOrZero(_assemblyDef, type, _logger);
        return declaredSize;
    }

    /// <summary>
    /// Computes trailing padding bytes for a Pod struct: gap padding between explicit offsets when
    /// <paramref name="declaredSize"/> is set, otherwise the difference between
    /// <see cref="Marshal.SizeOf(Type)"/> and the summed managed field sizes.
    /// </summary>
    private int ComputePodStructPadding(Type type, FieldInfo[] fields, int declaredSize,
        List<FieldDescriptor> fieldDescriptors)
    {
        try
        {
            if (fields.Length > 0 && fields.Any(f => f.GetCustomAttribute<FieldOffsetAttribute>() != null) && declaredSize > 0)
            {
                return ComputeExplicitLayoutGapPadding(fields, declaredSize, fieldDescriptors);
            }

            if (declaredSize == 0)
            {
                try
                {
                    int actualSize = Marshal.SizeOf(type);
                    int summed = ManagedLayoutSizing.SumManagedFieldSizes(fields);
                    return Math.Max(0, actualSize - summed);
                }
                catch (Exception ex) when (ex is ArgumentException or MarshalDirectiveException)
                {
                    _logger.LogTrace(LogCategory.Analysis,
                        $"{type.FullName}: Marshal.SizeOf for padding heuristic failed: {ex.Message}");
                }
            }
        }
        catch (Exception ex) when (ex is ArgumentException or InvalidOperationException or MarshalDirectiveException)
        {
            _logger.LogTrace(LogCategory.Analysis,
                $"{type.FullName}: explicit layout padding computation failed: {ex.Message}");
        }

        return 0;
    }

    /// <summary>
    /// Whether multiple-field structs contain at least one glam SIMD composite that introduces
    /// alignment padding incompatible with whole-struct Pod emission.
    /// </summary>
    private bool HasSimdCompositePaddingRisk(FieldInfo[] fields)
    {
        if (fields.Length < 2)
            return false;
        return fields.Any(f =>
        {
            string rustT = f.FieldType == typeof(bool) ? "u8" : RustTypeMapper.MapType(f.FieldType, _assembly);
            return RustTypeMapper.IsGlamRustTypeRequiringCompositeNonPod(rustT);
        });
    }

    /// <summary>
    /// Computes the host-interop size constant emitted alongside the Rust struct.
    /// Prefers <c>max(declaredSize, fieldEnd)</c> over <see cref="Marshal.SizeOf(Type)"/> when explicit-layout
    /// information is available so the constant matches the managed on-wire record stride.
    /// </summary>
    private int? ComputeHostInteropSizeBytes(Type type, FieldInfo[] fields, int declaredSize)
    {
        try
        {
            int marshalSize = Marshal.SizeOf(type);
            int fieldEndBytes = ManagedLayoutSizing.MaxFieldEndBytes(fields);
            if (fieldEndBytes > 0 || declaredSize > 0)
            {
                int managed = Math.Max(declaredSize, fieldEndBytes);
                if (managed != marshalSize)
                {
                    _logger.LogInfo(
                        LogCategory.Analysis,
                        $"{type.FullName}: managed layout size={managed} differs from Marshal.SizeOf={marshalSize} (declaredSize={declaredSize}, fieldEnd={fieldEndBytes}); using managed layout size for HostInteropSizeBytes.");
                }
                return managed;
            }

            return marshalSize;
        }
        catch (Exception ex) when (ex is ArgumentException or MissingMethodException)
        {
            _logger.LogWarning(LogCategory.Analysis, $"{type.FullName}: Marshal.SizeOf failed: {ex.Message}");
            return null;
        }
    }

    /// <summary>Analyzes an explicit-layout Pod struct into a <see cref="TypeDescriptor"/>.</summary>
    private TypeDescriptor AnalyzePodStruct(Type type)
    {
        FieldInfo[] fields = type.GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
        var fieldDescriptors = new List<FieldDescriptor>();

        bool allFieldsPod = true;
        foreach (FieldInfo field in fields)
        {
            if (!PodAnalyzer.IsRustLayoutPodField(field.FieldType, new HashSet<Type>(), _assembly))
                allFieldsPod = false;
            fieldDescriptors.Add(BuildPodStructFieldDescriptor(field));
        }

        StructLayoutAttribute? layout = type.GetCustomAttribute<StructLayoutAttribute>();
        int declaredSize = ResolveDeclaredOrCecilSize(type, layout);
        int paddingBytes = ComputePodStructPadding(type, fields, declaredSize, fieldDescriptors);

        bool isPod = allFieldsPod && !HasSimdCompositePaddingRisk(fields);
        int? hostInteropSizeBytes = ComputeHostInteropSizeBytes(type, fields, declaredSize);
        if (hostInteropSizeBytes.HasValue)
            VerifyHostInteropExtentAgainstFields(fields, hostInteropSizeBytes.Value, type);

        return new TypeDescriptor
        {
            CSharpName = type.Name,
            RustName = MapRustName(type),
            Shape = TypeShape.PodStruct,
            Fields = fieldDescriptors,
            IsPod = isPod,
            ExplicitSize = declaredSize > 0 ? declaredSize : null,
            PaddingBytes = paddingBytes,
            HostInteropSizeBytes = hostInteropSizeBytes,
        };
    }
}
