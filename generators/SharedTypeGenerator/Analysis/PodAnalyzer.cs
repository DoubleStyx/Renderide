using System.Reflection;

namespace SharedTypeGenerator.Analysis;

/// <summary>Recursive classification of C# field types for Rust <c>Pod</c> / layout compatibility.</summary>
internal static class PodAnalyzer
{
    /// <summary>
    /// Whether <paramref name="ft"/> is blittable as nested raw bytes (C# layout), independent of glam SIMD rules.
    /// </summary>
    public static bool IsFieldTypePod(Type ft, HashSet<Type> visited)
    {
        if (ft.IsEnum)
            return true;
        if (ft == typeof(bool))
            return true;
        if (ft.IsPrimitive || ft == typeof(Guid) || ft.Name?.StartsWith("SharedMemoryBufferDescriptor", StringComparison.Ordinal) == true)
            return true;
        if (ft.IsValueType && !ft.IsEnum && !visited.Contains(ft))
        {
            visited.Add(ft);
            try
            {
                return ft.GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)
                    .All(f => IsFieldTypePod(f.FieldType, visited));
            }
            finally
            {
                visited.Remove(ft);
            }
        }

        return false;
    }

    /// <summary>
    /// Whether <paramref name="ft"/> can be emitted as whole-struct <c>bytemuck::Pod</c> in Rust under default SIMD glam.
    /// Differs from <see cref="IsFieldTypePod"/> when nested composites gain SIMD alignment padding vs. C# blittable layout.
    /// Restricted-variant value enums are rejected: <c>Pod</c> requires every bit pattern to be a valid value, so reading
    /// an out-of-range byte via <c>bytemuck::pod_read_unaligned</c> is UB. Such fields are routed through
    /// <c>MemoryPackable</c>, which validates the wire byte via <c>EnumRepr::from_i32</c> in the generated unpack match.
    /// </summary>
    public static bool IsRustLayoutPodField(Type ft, HashSet<Type> visited, Assembly assembly)
    {
        if (ft.IsEnum)
            return false;
        if (ft == typeof(bool))
            return true;
        if (ft.IsPrimitive || ft == typeof(Guid) || ft.Name?.StartsWith("SharedMemoryBufferDescriptor", StringComparison.Ordinal) == true)
            return true;

        if (ft.IsValueType && ft.Assembly == assembly && !ft.IsEnum)
        {
            string rustStructName = RustTypeMapper.MapType(ft, assembly);
            if (RustTypeMapper.IsGlamRustType(rustStructName))
                return true;

            if (visited.Contains(ft))
                return false;
            visited.Add(ft);
            try
            {
                FieldInfo[] fields = ft.GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                foreach (FieldInfo f in fields)
                {
                    string rustLeaf = RustTypeMapper.MapType(f.FieldType, assembly);
                    if (RustTypeMapper.IsGlamRustTypeRequiringCompositeNonPod(rustLeaf))
                    {
                        if (fields.Length == 1)
                            continue;
                        return false;
                    }

                    if (!IsRustLayoutPodField(f.FieldType, visited, assembly))
                        return false;
                }

                return true;
            }
            finally
            {
                visited.Remove(ft);
            }
        }

        return IsFieldTypePod(ft, visited);
    }
}
