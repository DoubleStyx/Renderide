using NotEnoughLogs;
using NotEnoughLogs.Sinks;

namespace Renderide.Generators.Logging;

/// <summary>Forwards each log line to every child sink (console plus file, etc.).</summary>
public sealed class FanOutSink : ILoggerSink, IDisposable
{
    private readonly ILoggerSink[] _sinks;
    private bool _disposed;

    /// <summary>Creates a sink that duplicates output to all <paramref name="sinks"/>.</summary>
    public FanOutSink(params ILoggerSink[] sinks)
    {
        _sinks = sinks ?? throw new ArgumentNullException(nameof(sinks));
    }

    /// <inheritdoc />
    public void Log(LogLevel level, ReadOnlySpan<char> category, ReadOnlySpan<char> content)
    {
        foreach (ILoggerSink sink in _sinks)
            sink.Log(level, category, content);
    }

    /// <inheritdoc />
    public void Log(LogLevel level, ReadOnlySpan<char> category, ReadOnlySpan<char> format, params object[] args)
    {
        foreach (ILoggerSink sink in _sinks)
            sink.Log(level, category, format, args);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed)
            return;
        foreach (ILoggerSink sink in _sinks)
        {
            if (sink is IDisposable d)
                d.Dispose();
        }

        _disposed = true;
    }
}
