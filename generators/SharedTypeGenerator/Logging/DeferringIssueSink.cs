using NotEnoughLogs;
using NotEnoughLogs.Sinks;

namespace SharedTypeGenerator.Logging;

/// <summary>
/// <see cref="ILoggerSink"/> that forwards <see cref="LogLevel.Info"/>, <see cref="LogLevel.Debug"/>,
/// and <see cref="LogLevel.Trace"/> immediately to an inner sink, and buffers
/// <see cref="LogLevel.Warning"/>, <see cref="LogLevel.Error"/>, and <see cref="LogLevel.Critical"/>
/// until <see cref="FlushDeferred"/> runs so issues appear after normal progress logs.
/// </summary>
public sealed class DeferringIssueSink : ILoggerSink, IDisposable
{
    private readonly ILoggerSink _inner;
    private readonly List<DeferredLogLine> _buffer = [];
    private readonly object _lock = new();
    private bool _disposed;

    /// <summary>Creates a sink that defers issue-level lines to <paramref name="inner"/> on flush.</summary>
    public DeferringIssueSink(ILoggerSink inner)
    {
        _inner = inner;
    }

    /// <inheritdoc />
    public void Log(LogLevel level, ReadOnlySpan<char> category, ReadOnlySpan<char> content)
    {
        if (IsDeferredLevel(level))
        {
            lock (_lock)
                _buffer.Add(new DeferredLogLine(level, new string(category), new string(content)));
            return;
        }

        _inner.Log(level, category, content);
    }

    /// <inheritdoc />
    public void Log(LogLevel level, ReadOnlySpan<char> category, ReadOnlySpan<char> format, params object[] args)
    {
        if (IsDeferredLevel(level))
        {
            string message = string.Format(format.ToString(), args);
            lock (_lock)
                _buffer.Add(new DeferredLogLine(level, new string(category), message));
            return;
        }

        _inner.Log(level, category, format, args);
    }

    /// <summary>Writes all buffered warnings, errors, and critical lines through the inner sink, then clears the buffer.</summary>
    public void FlushDeferred()
    {
        List<DeferredLogLine> batch;
        lock (_lock)
        {
            if (_buffer.Count == 0)
                return;
            batch = _buffer.ToList();
            _buffer.Clear();
        }

        lock (Console.Out)
        {
            Console.WriteLine();
            Console.WriteLine("─── Deferred issues (warning / error / critical) ───");
        }

        foreach (DeferredLogLine line in batch)
            _inner.Log(line.Level, line.Category, line.Message);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed)
            return;
        FlushDeferred();
        if (_inner is IDisposable d)
            d.Dispose();
        _disposed = true;
    }

    private static bool IsDeferredLevel(LogLevel level) => (int)level <= (int)LogLevel.Warning;

    private readonly record struct DeferredLogLine(LogLevel Level, string Category, string Message);
}
