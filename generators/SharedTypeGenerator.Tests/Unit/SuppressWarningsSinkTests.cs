using NotEnoughLogs;
using NotEnoughLogs.Sinks;
using SharedTypeGenerator.Logging;
using SharedTypeGenerator.Tests.Unit.Support;
using Xunit;

namespace SharedTypeGenerator.Tests.Unit;

/// <summary>Unit tests for <see cref="SuppressWarningsSink"/>.</summary>
public sealed class SuppressWarningsSinkTests
{
    /// <summary>Warnings are dropped; other levels forward to the inner sink.</summary>
    [Theory]
    [InlineData(LogLevel.Warning, false)]
    [InlineData(LogLevel.Info, true)]
    [InlineData(LogLevel.Error, true)]
    [InlineData(LogLevel.Critical, true)]
    [InlineData(LogLevel.Trace, true)]
    public void Log_span_overload_respects_level(LogLevel level, bool expectForward)
    {
        var inner = new CollectingSink();
        using var sink = new SuppressWarningsSink(inner);
        sink.Log(level, "Cat", "hello");
        Assert.Equal(expectForward ? 1 : 0, inner.Lines.Count);
    }

    /// <summary>Formatted overload mirrors the span overload.</summary>
    [Fact]
    public void Log_format_overload_drops_warnings()
    {
        var inner = new CollectingSink();
        using var sink = new SuppressWarningsSink(inner);
        sink.Log(LogLevel.Warning, "Cat", "n={0}", 1);
        Assert.Empty(inner.Lines);
    }
}
