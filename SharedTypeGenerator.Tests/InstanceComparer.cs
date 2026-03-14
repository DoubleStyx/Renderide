using System.Collections;
using System.Reflection;
using SharedTypeGenerator.Analysis;
using SharedTypeGenerator.IR;

namespace SharedTypeGenerator.Tests;

/// <summary>Reflection-based deep comparison for round-trip testing.
/// Excludes unpack-only fields (e.g. decoded_time) that are set during Unpack but not serialized.</summary>
public static class InstanceComparer
{
    /// <summary>Compares two instances for structural equality, excluding the given field names.</summary>
    /// <param name="expected">Original instance before round-trip.</param>
    /// <param name="actual">Instance after round-trip.</param>
    /// <param name="excludedFields">Field names to skip (e.g. from TypeDescriptor.UnpackOnlySteps).</param>
    /// <returns>Null if equal; otherwise a description of the first difference.</returns>
    public static string? Compare(object? expected, object? actual, IReadOnlySet<string>? excludedFields = null)
    {
        excludedFields ??= new HashSet<string>();
        return CompareInternal(expected, actual, excludedFields, "root");
    }

    /// <summary>Builds excluded field names from a TypeDescriptor's UnpackOnlySteps.</summary>
    public static HashSet<string> ExcludedFromDescriptor(TypeDescriptor descriptor)
    {
        var excluded = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var step in descriptor.UnpackOnlySteps)
        {
            if (step is TimestampNow ts)
                excluded.Add(ts.FieldName);
        }
        return excluded;
    }

    private static string? CompareInternal(object? expected, object? actual, IReadOnlySet<string> excludedFields, string path)
    {
        if (expected == null && actual == null)
            return null;
        if (expected == null)
            return $"{path}: expected null, got non-null";
        if (actual == null)
            return $"{path}: expected non-null, got null";

        var type = expected.GetType();
        if (actual.GetType() != type)
            return $"{path}: type mismatch {type.Name} vs {actual.GetType().Name}";

        if (type.IsPrimitive || type == typeof(string) || type == typeof(decimal) || type == typeof(Guid))
        {
            if (!expected.Equals(actual))
                return $"{path}: value mismatch '{expected}' vs '{actual}'";
            return null;
        }

        if (type.IsEnum)
        {
            if (!expected.Equals(actual))
                return $"{path}: enum mismatch {expected} vs {actual}";
            return null;
        }

        if (type.IsValueType && !type.IsPrimitive && type != typeof(Guid))
        {
            foreach (var field in type.GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance))
            {
                var rustName = field.Name.HumanizeField();
                if (excludedFields.Contains(rustName) || excludedFields.Contains(field.Name))
                    continue;

                var expVal = field.GetValue(expected);
                var actVal = field.GetValue(actual);
                var diff = CompareInternal(expVal, actVal, excludedFields, $"{path}.{field.Name}");
                if (diff != null)
                    return diff;
            }
            return null;
        }

        if (expected is IEnumerable expEnum && actual is IEnumerable actEnum && type.IsGenericType)
        {
            var expList = expEnum.Cast<object>().ToList();
            var actList = actEnum.Cast<object>().ToList();
            if (expList.Count != actList.Count)
                return $"{path}: list count mismatch {expList.Count} vs {actList.Count}";

            for (int i = 0; i < expList.Count; i++)
            {
                var diff = CompareInternal(expList[i], actList[i], excludedFields, $"{path}[{i}]");
                if (diff != null)
                    return diff;
            }
            return null;
        }

        foreach (var field in type.GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance))
        {
            var rustName = field.Name.HumanizeField();
            if (excludedFields.Contains(rustName) || excludedFields.Contains(field.Name))
                continue;

            var expVal = field.GetValue(expected);
            var actVal = field.GetValue(actual);
            var diff = CompareInternal(expVal, actVal, excludedFields, $"{path}.{field.Name}");
            if (diff != null)
                return diff;
        }

        return null;
    }
}
