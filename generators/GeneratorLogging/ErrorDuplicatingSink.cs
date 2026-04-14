using System.Globalization;
using NotEnoughLogs;
using NotEnoughLogs.Sinks;

namespace Renderide.Generators.Logging;

/// <summary>
/// Forwards logs to an inner sink. When the inner is a <see cref="LogFileSink"/>, formats each line once and
/// writes it via <see cref="LogFileSink.WriteFormattedLine"/> so file output matches any stderr duplicate.
/// Additionally mirrors <see cref="LogLevel.Error"/> and <see cref="LogLevel.Critical"/> lines to <see cref="Console.Error"/>
/// (or another <see cref="TextWriter"/>) for terminal visibility.
/// </summary>
public sealed class ErrorDuplicatingSink : ILoggerSink, IDisposable
{
    private readonly ILoggerSink _inner;
    private readonly TextWriter _stderr;
    private readonly object _stderrLock = new();
    private bool _disposed;

    /// <summary>
    /// Wraps <paramref name="inner"/> (typically a <see cref="LogFileSink"/>). Optional <paramref name="stderr"/>
    /// defaults to <see cref="Console.Error"/> for duplicate error output.
    /// </summary>
    public ErrorDuplicatingSink(ILoggerSink inner, TextWriter? stderr = null)
    {
        _inner = inner ?? throw new ArgumentNullException(nameof(inner));
        _stderr = stderr ?? Console.Error;
    }

    /// <inheritdoc />
    public void Log(LogLevel level, ReadOnlySpan<char> category, ReadOnlySpan<char> content)
    {
        string categoryStr = category.ToString();
        string message = content.ToString();
        string line = LogFileSink.FormatLogLine(level, categoryStr, message);
        if (_inner is LogFileSink file)
            file.WriteFormattedLine(line);
        else
            _inner.Log(level, category, content);

        MaybeMirrorToStderr(level, line);
    }

    /// <inheritdoc />
    public void Log(LogLevel level, ReadOnlySpan<char> category, ReadOnlySpan<char> format, params object[] args)
    {
        string message = string.Format(CultureInfo.InvariantCulture, format.ToString(), args);
        string categoryStr = category.ToString();
        string line = LogFileSink.FormatLogLine(level, categoryStr, message);
        if (_inner is LogFileSink file)
            file.WriteFormattedLine(line);
        else
            _inner.Log(level, category, message);

        MaybeMirrorToStderr(level, line);
    }

    private void MaybeMirrorToStderr(LogLevel level, string line)
    {
        if (level != LogLevel.Error && level != LogLevel.Critical)
            return;
        lock (_stderrLock)
        {
            _stderr.WriteLine(line);
        }
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
