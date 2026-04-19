namespace SharedTypeGenerator.Tests.Unit.Support;

/// <summary>Restores an environment variable to its previous value on <see cref="Dispose"/>.</summary>
internal sealed class EnvVarScope : IDisposable
{
    /// <summary>Environment variable name passed to the constructor.</summary>
    private readonly string _name;

    /// <summary>Value captured at construction time for restore.</summary>
    private readonly string? _previous;

    /// <summary>Guards <see cref="Dispose"/> so the previous env value is restored once.</summary>
    private bool _disposed;

    /// <summary>Captures <paramref name="name"/>'s current value (may be null) for later restore.</summary>
    /// <param name="name">Environment variable name.</param>
    public EnvVarScope(string name)
    {
        _name = name;
        _previous = Environment.GetEnvironmentVariable(name);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        if (_previous is null)
            Environment.SetEnvironmentVariable(_name, null);
        else
            Environment.SetEnvironmentVariable(_name, _previous);
        GC.SuppressFinalize(this);
    }
}
