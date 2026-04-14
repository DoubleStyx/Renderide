using System.Globalization;
using System.Text;
using NotEnoughLogs;
using NotEnoughLogs.Sinks;

namespace Renderide.Generators.Logging;

/// <summary>
/// Appends UTF-8 lines to a single log file, using a line prefix compatible with the Rust logger crate
/// (<c>[HH:MM:SS.mmm]</c> UTC + level name).
/// </summary>
public sealed class LogFileSink : ILoggerSink, IDisposable
{
    private readonly StreamWriter _writer;
    private readonly object _lock = new();
    private bool _disposed;

    /// <summary>
    /// Opens <paramref name="logFilePath"/> for write (truncating if the file already exists).
    /// Creates parent directories when missing.
    /// </summary>
    public LogFileSink(string logFilePath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(logFilePath);
        string fullPath = Path.GetFullPath(logFilePath);
        string? parent = Path.GetDirectoryName(fullPath);
        if (!string.IsNullOrEmpty(parent))
            Directory.CreateDirectory(parent);
        var stream = new FileStream(
            fullPath,
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
        string message = string.Format(CultureInfo.InvariantCulture, format.ToString(), args);
        WriteLine(level, category.ToString(), message);
    }

    /// <summary>
    /// Formats a single log line using the same rules as <see cref="LogFileSink"/> file output
    /// (<c>[HH:MM:SS.mmm]</c> UTC, level name, optional category in brackets). Used by
    /// <see cref="ErrorDuplicatingSink"/> so duplicated error lines match the log file.
    /// </summary>
    public static string FormatLogLine(LogLevel level, string category, string message)
    {
        DateTime utc = DateTime.UtcNow;
        string ts = $"{utc:HH:mm:ss}.{utc.Millisecond:000}";
        string levelName = MapLevel(level);
        return string.IsNullOrEmpty(category)
            ? FormattableString.Invariant($"[{ts}] {levelName} {message}")
            : FormattableString.Invariant($"[{ts}] {levelName} [{category}] {message}");
    }

    private void WriteLine(LogLevel level, string category, string message)
    {
        WriteFormattedLine(FormatLogLine(level, category, message));
    }

    /// <summary>
    /// Appends one full line as returned by <see cref="FormatLogLine"/> (used when a caller formats once and mirrors elsewhere).
    /// </summary>
    public void WriteFormattedLine(string line)
    {
        ArgumentNullException.ThrowIfNull(line);
        lock (_lock)
        {
            _writer.WriteLine(line);
        }
    }

    private static string MapLevel(LogLevel level) => level switch
    {
        LogLevel.Critical => "Error",
        LogLevel.Error => "Error",
        LogLevel.Warning => "Warn",
        LogLevel.Info => "Info",
        LogLevel.Debug => "Debug",
        LogLevel.Trace => "Trace",
        _ => "Info",
    };

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
