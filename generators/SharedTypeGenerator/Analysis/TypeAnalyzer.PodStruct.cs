using System.Reflection;
using System.Runtime.InteropServices;
using LayoutKind = System.Runtime.InteropServices.LayoutKind;
using NotEnoughLogs;
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

    /// <summary>Accumulates marshal size of fields for trailing-padding heuristics when no explicit offsets exist.</summary>
    private static int SumMarshalSizesOfFields(FieldInfo[] fields)
    {
        int totalSize = 0;
        foreach (FieldInfo field in fields)
        {
            Type sizeType = field.FieldType == typeof(bool) ? typeof(byte) : field.FieldType;
            if (sizeType.IsEnum)
                sizeType = sizeType.GetField("value__")!.FieldType;
            try
            {
                totalSize += Marshal.SizeOf(sizeType);
            }
            catch (Exception ex) when (ex is ArgumentException or MarshalDirectiveException)
            {
                /* skip field for sum */
            }
        }

        return totalSize;
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
        for (int i = 0; i < fields.Length; i++)
        {
            FieldInfo field = fields[i];
            int offset = field.GetCustomAttribute<FieldOffsetAttribute>()?.Value ?? 0;
            Type st = field.FieldType == typeof(bool) ? typeof(byte) : field.FieldType;
            if (st.IsEnum)
                st = st.GetField("value__")!.FieldType;
            int size;
            try
            {
                size = Marshal.SizeOf(st);
            }
            catch (Exception ex) when (ex is ArgumentException or MarshalDirectiveException)
            {
                size = 0;
            }

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

    /// <summary>Throws when explicit field extents exceed <see cref="Marshal.SizeOf"/> for the struct.</summary>
    private static void VerifyHostInteropExtentAgainstFields(FieldInfo[] fields, int hostInteropSizeBytes, Type structType)
    {
        if (!fields.Any(f => f.GetCustomAttribute<FieldOffsetAttribute>() != null))
            return;

        int maxEnd = 0;
        foreach (FieldInfo field in fields)
        {
            FieldOffsetAttribute? fo = field.GetCustomAttribute<FieldOffsetAttribute>();
            if (fo == null)
                continue;

            Type st = field.FieldType == typeof(bool) ? typeof(byte) : field.FieldType;
            if (st.IsEnum)
                st = st.GetField("value__")!.FieldType;
            int sz;
            try
            {
                sz = Marshal.SizeOf(st);
            }
            catch (Exception ex) when (ex is ArgumentException or MarshalDirectiveException)
            {
                continue;
            }

            maxEnd = Math.Max(maxEnd, fo.Value + sz);
        }

        if (maxEnd > hostInteropSizeBytes)
        {
            throw new InvalidOperationException(
                $"{structType.FullName}: explicit layout field extent ({maxEnd} bytes) exceeds Marshal.SizeOf={hostInteropSizeBytes}.");
        }
    }

    private TypeDescriptor AnalyzePodStruct(Type type)
    {
        FieldInfo[] fields = type.GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
        var fieldDescriptors = new List<FieldDescriptor>();

        int totalSize = SumMarshalSizesOfFields(fields);
        bool allFieldsPod = true;

        foreach (FieldInfo field in fields)
        {
            bool fieldRustLayoutPod = PodAnalyzer.IsRustLayoutPodField(field.FieldType, new HashSet<Type>(), _assembly);
            if (!fieldRustLayoutPod)
                allFieldsPod = false;

            fieldDescriptors.Add(BuildPodStructFieldDescriptor(field));
        }

        StructLayoutAttribute? layout = type.GetCustomAttribute<StructLayoutAttribute>();
        int declaredSize = (layout?.Value == LayoutKind.Explicit && layout.Size > 0) ? layout.Size : 0;
        if (declaredSize == 0)
            declaredSize = CecilLayoutInspector.GetExplicitLayoutSizeOrZero(_assemblyDef, type, _logger);

        int paddingBytes = 0;

        try
        {
            if (fields.Length > 0 && fields.Any(f => f.GetCustomAttribute<FieldOffsetAttribute>() != null) && declaredSize > 0)
            {
                paddingBytes = ComputeExplicitLayoutGapPadding(fields, declaredSize, fieldDescriptors);
            }
            else if (declaredSize == 0)
            {
                try
                {
                    int actualSize = Marshal.SizeOf(type);
                    paddingBytes = Math.Max(0, actualSize - totalSize);
                }
                catch (Exception ex) when (ex is ArgumentException or MarshalDirectiveException)
                {
                    _logger.LogTrace(LogCategory.Analysis, $"{type.FullName}: Marshal.SizeOf for padding heuristic failed: {ex.Message}");
                }
            }
        }
        catch (Exception ex) when (ex is ArgumentException or InvalidOperationException or MarshalDirectiveException)
        {
            _logger.LogTrace(LogCategory.Analysis, $"{type.FullName}: explicit layout padding computation failed: {ex.Message}");
        }

        bool hasSimdCompositePaddingRisk = fields.Length > 1 && fields.Any(f =>
        {
            string rustT = f.FieldType == typeof(bool) ? "u8" : RustTypeMapper.MapType(f.FieldType, _assembly);
            return RustTypeMapper.IsGlamRustTypeRequiringCompositeNonPod(rustT);
        });
        bool isPod = allFieldsPod && !hasSimdCompositePaddingRisk;

        int? hostInteropSizeBytes = null;
        try
        {
            int marshalSize = Marshal.SizeOf(type);
            hostInteropSizeBytes = marshalSize;
            if (declaredSize > 0 && marshalSize != declaredSize)
            {
                _logger.LogWarning(
                    LogCategory.Analysis,
                    $"{type.FullName}: StructLayout.Size={declaredSize} differs from Marshal.SizeOf={marshalSize}; using Marshal.SizeOf for HostInteropSizeBytes.");
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(LogCategory.Analysis, $"{type.FullName}: Marshal.SizeOf failed: {ex.Message}");
        }

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
