using NotEnoughLogs;
using NotEnoughLogs.Sinks;
using Xunit;

namespace Renderide.Generators.Logging.Tests;

/// <summary>Smoke tests for shared generator logging sinks.</summary>
public sealed class LoggingSinkTests
{
    /// <summary><see cref="FanOutSink"/> forwards each message to every child sink.</summary>
    [Fact]
    public void FanOutSink_forwards_to_all_children()
    {
        var a = new CollectingSink();
        var b = new CollectingSink();
        using var fan = new FanOutSink(a, b);
        fan.Log(LogLevel.Info, "Test", "hello");
        Assert.Single(a.Lines);
        Assert.Single(b.Lines);
        Assert.Equal("hello", a.Lines[0]);
        Assert.Equal("hello", b.Lines[0]);
    }

    /// <summary><see cref="TruncatingFileSink"/> creates a new file and appends formatted lines.</summary>
    [Fact]
    public void TruncatingFileSink_writes_utf8_lines()
    {
        string path = Path.Combine(Path.GetTempPath(), $"genlog-test-{Guid.NewGuid():N}.log");
        try
        {
            using (var sink = new TruncatingFileSink(path))
            {
                sink.Log(LogLevel.Warning, "Cat", "msg");
            }

            string text = File.ReadAllText(path);
            Assert.Contains("[Warning]", text, StringComparison.Ordinal);
            Assert.Contains("[Cat]", text, StringComparison.Ordinal);
            Assert.Contains("msg", text, StringComparison.Ordinal);
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

    /// <summary><see cref="SuppressWarningsSink"/> must not forward <see cref="LogLevel.Warning"/> to the inner sink.</summary>
    [Fact]
    public void SuppressWarningsSink_drops_warning_forwards_other_levels()
    {
        var inner = new CollectingSink();
        using var sink = new SuppressWarningsSink(inner);
        sink.Log(LogLevel.Warning, "Cat", "noise");
        sink.Log(LogLevel.Info, "Cat", "progress");
        sink.Log(LogLevel.Error, "Cat", "boom");
        Assert.Equal(2, inner.Lines.Count);
        Assert.Equal("progress", inner.Lines[0]);
        Assert.Equal("boom", inner.Lines[1]);
    }

    private sealed class CollectingSink : ILoggerSink
    {
        public List<string> Lines { get; } = [];

        public void Log(LogLevel level, ReadOnlySpan<char> category, ReadOnlySpan<char> content) =>
            Lines.Add(content.ToString());

        public void Log(LogLevel level, ReadOnlySpan<char> category, ReadOnlySpan<char> format, params object[] args) =>
            Lines.Add(string.Format(format.ToString(), args));
    }
}
