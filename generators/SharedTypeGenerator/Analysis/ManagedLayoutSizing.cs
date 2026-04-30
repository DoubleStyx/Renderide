using System.Reflection;
using System.Runtime.InteropServices;

namespace SharedTypeGenerator.Analysis;

/// <summary>
/// Helpers for computing CLR managed-layout field sizes used when sizing explicit-layout
/// <see cref="IR.TypeShape.PodStruct"/> records and their padding.
/// </summary>
/// <remarks>
/// "Managed layout" here means: <c>bool</c> is one byte (matching the on-wire managed record stride,
/// not the four-byte P/Invoke marshaled bool), enums collapse to their underlying integer type, and
/// every other field falls back to <see cref="Marshal.SizeOf(Type)"/>. Failures from
/// <see cref="Marshal.SizeOf(Type)"/> on unmarshallable types are surfaced through the
/// <c>TryGet</c> pattern so each call site can pick its own fall-through policy.
/// </remarks>
internal static class ManagedLayoutSizing
{
    /// <summary>Resolves the size-defining type for a field under managed layout rules.</summary>
    /// <param name="field">Field whose declared type to inspect.</param>
    /// <returns><c>byte</c> for <c>bool</c>, the underlying integer type for enums, otherwise the field's declared type.</returns>
    private static Type ResolveSizeType(FieldInfo field)
    {
        Type t = field.FieldType == typeof(bool) ? typeof(byte) : field.FieldType;
        if (t.IsEnum)
            t = t.GetField("value__")!.FieldType;
        return t;
    }

    /// <summary>Tries to compute a single field's managed size in bytes.</summary>
    /// <param name="field">Field to size.</param>
    /// <param name="sizeBytes">Set to the field's managed size on success; undefined on failure.</param>
    /// <returns><c>true</c> if a size was computed; <c>false</c> if <see cref="Marshal.SizeOf(Type)"/> rejected the type.</returns>
    public static bool TryGetManagedFieldSize(FieldInfo field, out int sizeBytes)
    {
        Type sizeType = ResolveSizeType(field);
        try
        {
            sizeBytes = Marshal.SizeOf(sizeType);
            return true;
        }
        catch (Exception ex) when (ex is ArgumentException or MarshalDirectiveException)
        {
            sizeBytes = 0;
            return false;
        }
    }

    /// <summary>Sums managed sizes across <paramref name="fields"/>; unmarshallable fields contribute zero.</summary>
    /// <param name="fields">Fields to sum (typically all instance fields of a struct).</param>
    public static int SumManagedFieldSizes(FieldInfo[] fields)
    {
        int total = 0;
        foreach (FieldInfo field in fields)
        {
            if (TryGetManagedFieldSize(field, out int size))
                total += size;
        }

        return total;
    }

    /// <summary>
    /// Returns the maximum <c>FieldOffset + managed size</c> over fields with an explicit
    /// <see cref="FieldOffsetAttribute"/>. Returns 0 when no field carries an explicit offset
    /// or when every offset-bearing field is unmarshallable.
    /// </summary>
    /// <param name="fields">Fields to scan.</param>
    public static int MaxFieldEndBytes(FieldInfo[] fields)
    {
        int maxEnd = 0;
        foreach (FieldInfo field in fields)
        {
            FieldOffsetAttribute? offset = field.GetCustomAttribute<FieldOffsetAttribute>();
            if (offset == null)
                continue;
            if (!TryGetManagedFieldSize(field, out int size))
                continue;
            maxEnd = Math.Max(maxEnd, offset.Value + size);
        }

        return maxEnd;
    }
}
