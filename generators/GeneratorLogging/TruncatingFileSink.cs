using System.Text;
using NotEnoughLogs;
using NotEnoughLogs.Sinks;

namespace Renderide.Generators.Logging;

/// <summary>
/// Appends UTF-8 log lines to a file opened with <see cref="FileMode.Create"/> so each process run starts with an empty file.
/// </summary>
public sealed class TruncatingFileSink : ILoggerSink, IDisposable
{
    private readonly StreamWriter _writer;
    private readonly object _lock = new();
    private bool _disposed;

    /// <summary>Opens <paramref name="logFilePath"/> for write, truncating any existing file.</summary>
    public TruncatingFileSink(string logFilePath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(logFilePath);
        string? dir = Path.GetDirectoryName(Path.GetFullPath(logFilePath));
        if (!string.IsNullOrEmpty(dir))
            Directory.CreateDirectory(dir);
        var stream = new FileStream(
            logFilePath,
            FileMode.Create,
            FileAccess.Write,
            FileShare.Read);
        _writer = new StreamWriter(stream, new UTF8Encoding(encoderShouldEmitUTF8Identifier: false))
        {
            AutoFlush = true,
        };
    }

    /// <inheritdoc />
    public void Log(LogLevel level, ReadOnlySpan<char> category, ReadOnlySpan<char> content)
    {
        WriteLine(level, category.ToString(), content.ToString());
    }

    /// <inheritdoc />
    public void Log(LogLevel level, ReadOnlySpan<char> category, ReadOnlySpan<char> format, params object[] args)
    {
        string message = string.Format(format.ToString(), args);
        WriteLine(level, category.ToString(), message);
    }

    private void WriteLine(LogLevel level, string category, string message)
    {
        string line = $"{DateTime.UtcNow:O} [{level}] [{category}] {message}";
        lock (_lock)
        {
            _writer.WriteLine(line);
        }
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed)
            return;
        lock (_lock)
        {
            _writer.Dispose();
        }

        _disposed = true;
    }
}
