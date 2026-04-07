using NotEnoughLogs;
using NotEnoughLogs.Sinks;

namespace Renderide.Generators.Logging;

/// <summary>
/// <see cref="ILoggerSink"/> that drops <see cref="LogLevel.Warning"/> so generator logs stay focused on
/// <see cref="LogLevel.Info"/> progress, <see cref="LogLevel.Error"/> / <see cref="LogLevel.Critical"/> issues,
/// and (when enabled) <see cref="LogLevel.Trace"/>. Use <see cref="LogLevel.Error"/> or <see cref="LogLevel.Info"/>
/// for anything operators must see.
/// </summary>
public sealed class SuppressWarningsSink : ILoggerSink, IDisposable
{
    private readonly ILoggerSink _inner;
    private bool _disposed;

    /// <summary>Wraps <paramref name="inner"/> (typically <see cref="TruncatingFileSink"/>).</summary>
    public SuppressWarningsSink(ILoggerSink inner)
    {
        _inner = inner;
    }

    /// <inheritdoc />
    public void Log(LogLevel level, ReadOnlySpan<char> category, ReadOnlySpan<char> content)
    {
        if (level == LogLevel.Warning)
            return;
        _inner.Log(level, category, content);
    }

    /// <inheritdoc />
    public void Log(LogLevel level, ReadOnlySpan<char> category, ReadOnlySpan<char> format, params object[] args)
    {
        if (level == LogLevel.Warning)
            return;
        _inner.Log(level, category, format, args);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed)
            return;
        if (_inner is IDisposable d)
            d.Dispose();
        _disposed = true;
    }
}
