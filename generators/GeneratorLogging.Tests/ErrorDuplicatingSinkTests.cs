using System.Globalization;
using NotEnoughLogs;
using NotEnoughLogs.Sinks;
using Xunit;

namespace Renderide.Generators.Logging.Tests;

/// <summary>Unit tests for <see cref="ErrorDuplicatingSink"/> stdout/stderr mirroring.</summary>
public sealed class ErrorDuplicatingSinkTests
{
    /// <summary>Info-level logs must not be written to the error mirror stream.</summary>
    [Fact]
    public void Info_goes_to_inner_only_not_stderr()
    {
        var inner = new CollectingSink();
        var stderr = new StringWriter(CultureInfo.InvariantCulture);
        using var sink = new ErrorDuplicatingSink(inner, stderr);
        sink.Log(LogLevel.Info, "Cat", "hello");
        Assert.Single(inner.Lines);
        Assert.Equal("hello", inner.Lines[0]);
        Assert.Equal(string.Empty, stderr.ToString());
    }

    /// <summary>Error-level logs must be duplicated to the configured <see cref="TextWriter"/>.</summary>
    [Fact]
    public void Error_is_written_to_inner_and_stderr_with_same_line()
    {
        var inner = new CollectingSink();
        var stderr = new StringWriter(CultureInfo.InvariantCulture);
        using var sink = new ErrorDuplicatingSink(inner, stderr);
        sink.Log(LogLevel.Error, "Cat", "boom");
        Assert.Single(inner.Lines);
        Assert.Equal("boom", inner.Lines[0]);
        string err = stderr.ToString().TrimEnd('\r', '\n');
        Assert.EndsWith("boom", err, StringComparison.Ordinal);
        Assert.Contains(" Error ", err, StringComparison.Ordinal);
        Assert.Contains("[Cat]", err, StringComparison.Ordinal);
    }

    /// <summary>When inner is a <see cref="LogFileSink"/>, file and stderr should share one formatted line (same timestamp prefix).</summary>
    [Fact]
    public void LogFileSink_inner_matches_stderr_line_exactly()
    {
        string path = Path.Combine(Path.GetTempPath(), $"genlog-edup-{Guid.NewGuid():N}.log");
        try
        {
            var stderr = new StringWriter(CultureInfo.InvariantCulture);
            using var sink = new ErrorDuplicatingSink(new LogFileSink(path), stderr);
            sink.Log(LogLevel.Error, "Cat", "exact");
            string fileText = File.ReadAllText(path).TrimEnd('\r', '\n');
            string err = stderr.ToString().TrimEnd('\r', '\n');
            Assert.Equal(fileText, err);
        }
        finally
        {
            try
            {
                if (File.Exists(path))
                    File.Delete(path);
            }
            catch
            {
                // ignored
            }
        }
    }

    /// <summary><see cref="LogLevel.Critical"/> is mirrored like <see cref="LogLevel.Error"/>.</summary>
    [Fact]
    public void Critical_is_mirrored_to_stderr()
    {
        var inner = new CollectingSink();
        var stderr = new StringWriter(CultureInfo.InvariantCulture);
        using var sink = new ErrorDuplicatingSink(inner, stderr);
        sink.Log(LogLevel.Critical, "Cat", "fatal");
        Assert.Contains("Error", stderr.ToString(), StringComparison.Ordinal);
        Assert.Contains("fatal", stderr.ToString(), StringComparison.Ordinal);
    }

    /// <summary>Formatted log overload must mirror formatted text.</summary>
    [Fact]
    public void Format_overload_mirrors_formatted_error()
    {
        var inner = new CollectingSink();
        var stderr = new StringWriter(CultureInfo.InvariantCulture);
        using var sink = new ErrorDuplicatingSink(inner, stderr);
        sink.Log(LogLevel.Error, "Cat", "n={0}", 42);
        Assert.Single(inner.Lines);
        Assert.Equal("n=42", inner.Lines[0]);
        Assert.Contains("n=42", stderr.ToString(), StringComparison.Ordinal);
    }

    /// <summary>Captures log message text for assertions (not used with <see cref="LogFileSink"/>).</summary>
    private sealed class CollectingSink : ILoggerSink
    {
        /// <summary>Recorded message bodies (content or formatted result).</summary>
        public List<string> Lines { get; } = [];

        /// <inheritdoc />
        public void Log(LogLevel level, ReadOnlySpan<char> category, ReadOnlySpan<char> content) =>
            Lines.Add(content.ToString());

        /// <inheritdoc />
        public void Log(LogLevel level, ReadOnlySpan<char> category, ReadOnlySpan<char> format, params object[] args) =>
            Lines.Add(string.Format(CultureInfo.InvariantCulture, format.ToString(), args));
    }
}
