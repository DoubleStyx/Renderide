using System.Collections;
using System.Reflection;

namespace SharedTypeGenerator.Analysis;

/// <summary>Pure function that maps a C# Type to its Rust type string.
/// Returns both the Rust type name and any types that need to be queued for generation.
/// No side effects -- the caller is responsible for queueing.</summary>
/// <remarks>
/// Host <c>RenderVector*</c> / <c>RenderQuaternion</c> / <c>RenderMatrix4x4</c> map to <c>glam</c>
/// (<c>Vec2</c>, <c>IVec2</c>, …, <c>Quat</c>, <c>Mat4</c>). Composite explicit-layout structs may omit
/// whole-struct <c>Pod</c> when SIMD alignment adds padding; see <see cref="IsGlamRustTypeRequiringCompositeNonPod"/>.
/// </remarks>
public static class RustTypeMapper
{
    /// <summary>Rust type string plus referenced types from the shared assembly that must be emitted.</summary>
    /// <param name="RustType">Emitted Rust type text.</param>
    /// <param name="ReferencedTypes">Types to enqueue for generation.</param>
    public record struct MappingResult(string RustType, List<Type> ReferencedTypes);

    /// <summary>Maps a C# type to its Rust equivalent, collecting any referenced
    /// types that should be generated.</summary>
    public static MappingResult Map(Type type, Assembly targetAssembly, bool inList = false)
    {
        var referenced = new List<Type>();
        string rustType = MapInternal(type, targetAssembly, inList, referenced);
        return new MappingResult(rustType, referenced);
    }

    /// <summary>Maps a C# type to its Rust equivalent without tracking references.
    /// Use when you only need the type string.</summary>
    public static string MapType(Type type, Assembly targetAssembly, bool inList = false)
    {
        return MapInternal(type, targetAssembly, inList, null);
    }

    private static string MapInternal(Type type, Assembly targetAssembly, bool inList, List<Type>? referenced)
    {
        if (type == typeof(string))
            return "Option<String>";

        if (type == typeof(byte)) return "u8";
        if (type == typeof(sbyte)) return "i8";
        if (type == typeof(short)) return "i16";
        if (type == typeof(ushort)) return "u16";
        if (type == typeof(int)) return "i32";
        if (type == typeof(uint)) return "u32";
        if (type == typeof(long)) return "i64";
        if (type == typeof(ulong)) return "u64";
        if (type == typeof(Int128)) return "i128";
        if (type == typeof(UInt128)) return "u128";
        if (type == typeof(float)) return "f32";
        if (type == typeof(double)) return "f64";
        if (type == typeof(bool)) return "bool";
        if (type == typeof(DateTime)) return "i128";

        switch (type.Name)
        {
            case "RenderVector2": return "Vec2";
            case "RenderVector2i": return "IVec2";
            case "RenderVector3": return "Vec3";
            case "RenderVector3i": return "IVec3";
            case "RenderVector4": return "Vec4";
            case "RenderVector4i": return "IVec4";
            case "RenderQuaternion": return "Quat";
            case "RenderMatrix4x4": return "Mat4";
        }

        if (typeof(IEnumerable).IsAssignableFrom(type))
            return $"Vec<{MapInternal(type.GenericTypeArguments.First(), targetAssembly, true, referenced)}>";

        if (type.Name == "Nullable`1")
            return $"Option<{MapInternal(type.GenericTypeArguments.First(), targetAssembly, inList, referenced)}>";

        if (type.IsClass && !inList)
        {
            TryAddReference(type, targetAssembly, referenced);
            return $"Option<{type.Name.HumanizeType()}>";
        }

        if (type.Name.StartsWith("SharedMemoryBufferDescriptor", StringComparison.Ordinal))
        {
            if (type.GenericTypeArguments.Length > 0)
                MapInternal(type.GenericTypeArguments.First(), targetAssembly, inList, referenced);
            return "SharedMemoryBufferDescriptor";
        }

        if (type.IsGenericType)
        {
            string baseName = type.Name.Remove(type.Name.IndexOf('`'));
            string args = string.Join(", ", type.GenericTypeArguments.Select(t => MapInternal(t, targetAssembly, inList, referenced)));
            return $"{baseName}<{args}>".HumanizeType();
        }

        if (type.DeclaringType != null)
        {
            TryAddReference(type, targetAssembly, referenced);
            TryAddReference(type.DeclaringType, targetAssembly, referenced);
            return (type.DeclaringType.Name + '_' + type.Name).HumanizeType();
        }

        TryAddReference(type, targetAssembly, referenced);
        return type.Name.HumanizeType();
    }

    private static void TryAddReference(Type type, Assembly targetAssembly, List<Type>? referenced)
    {
        if (referenced == null) return;
        if (type.Assembly == targetAssembly || type == typeof(Guid))
            referenced.Add(type);
    }

    /// <summary>Maps a bit count to the smallest standard Rust unsigned type that can hold it.</summary>
    public static string BitsToRustUintType(int bits)
    {
        if (bits <= 8) return "u8";
        if (bits <= 16) return "u16";
        if (bits <= 32) return "u32";
        if (bits <= 64) return "u64";
        return "u128";
    }

    /// <summary>Maps a C# primitive type to its Rust equivalent for enum underlying types.</summary>
    public static string MapPrimitiveType(Type type)
    {
        if (type == typeof(byte)) return "u8";
        if (type == typeof(sbyte)) return "i8";
        if (type == typeof(short)) return "i16";
        if (type == typeof(ushort)) return "u16";
        if (type == typeof(int)) return "i32";
        if (type == typeof(uint)) return "u32";
        if (type == typeof(long)) return "i64";
        if (type == typeof(ulong)) return "u64";
        return "i32";
    }

    /// <summary>Returns the Rust type name without surrounding <c>Option&lt;…&gt;</c> when present.</summary>
    public static string NormalizeRustTypeName(string rustType)
    {
        string t = rustType.Trim();
        if (t.StartsWith("Option<", StringComparison.Ordinal) && t.EndsWith('>'))
            t = t[7..^1].Trim();

        return t;
    }

    /// <summary>Strips a single outer <c>Vec&lt;T&gt;</c> wrapper when present (after <see cref="NormalizeRustTypeName"/>).</summary>
    public static string StripVecElementType(string rustType)
    {
        string t = NormalizeRustTypeName(rustType);
        if (t.StartsWith("Vec<", StringComparison.Ordinal) && t.EndsWith('>'))
            return t[4..^1].Trim();

        return t;
    }

    /// <summary>
    /// Returns true if the emitted Rust type name is a glam vector/matrix/quaternion used in shared memory.
    /// </summary>
    public static bool IsGlamRustType(string rustType)
    {
        string t = NormalizeRustTypeName(rustType);
        return t is "Vec2" or "Vec3" or "Vec4" or "Quat" or "Mat4" or "IVec2" or "IVec3" or "IVec4";
    }

    /// <summary>
    /// True for default-glam types that use <c>repr(align(16))</c> and can introduce Rust-only padding
    /// when combined with other fields in a <c>repr(C)</c> composite. Single-field wrappers remain whole-struct <c>Pod</c>.
    /// </summary>
    public static bool IsGlamRustTypeRequiringCompositeNonPod(string rustType)
    {
        string t = NormalizeRustTypeName(rustType);
        return t is "Quat" or "Mat4" or "Vec4";
    }
}
