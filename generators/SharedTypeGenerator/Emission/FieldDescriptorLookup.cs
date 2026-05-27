using SharedTypeGenerator.IR;

namespace SharedTypeGenerator.Emission;

/// <summary>Precomputed lookup for emitted Rust fields by Rust field name.</summary>
internal sealed class FieldDescriptorLookup
{
    private readonly Dictionary<string, FieldDescriptor> _byRustName;

    /// <summary>Creates a lookup for <paramref name="fields"/>.</summary>
    public FieldDescriptorLookup(IEnumerable<FieldDescriptor> fields)
    {
        _byRustName = new Dictionary<string, FieldDescriptor>(StringComparer.Ordinal);
        foreach (FieldDescriptor field in fields)
            _byRustName.TryAdd(field.RustName, field);
    }

    /// <summary>Returns the descriptor for <paramref name="rustName"/>, or <see langword="null"/> when no matching field exists.</summary>
    public FieldDescriptor? Find(string rustName) =>
        _byRustName.GetValueOrDefault(rustName);
}
