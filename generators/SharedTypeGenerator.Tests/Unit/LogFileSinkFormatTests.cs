using System.Text.RegularExpressions;
using NotEnoughLogs;
using SharedTypeGenerator.Logging;
using Xunit;

namespace SharedTypeGenerator.Tests.Unit;

/// <summary>Unit tests for <see cref="LogFileSink"/> formatting helpers.</summary>
public sealed class LogFileSinkFormatTests
{
    /// <summary><see cref="LogFileSink.FormatLogLine"/> uses bracketed time, level, optional category, and message.</summary>
    [Fact]
    public void FormatLogLine_shape_matches_expected_pattern()
    {
        string line = LogFileSink.FormatLogLine(LogLevel.Info, "Cat", "hello");
        Assert.Matches(new Regex(@"^\[\d{2}:\d{2}:\d{2}\.\d{3}\] Info \[Cat\] hello$"), line);
    }

    /// <summary>Formatted lines embed the mapped level name next to the timestamp.</summary>
    [Theory]
    [InlineData(LogLevel.Critical, "Error")]
    [InlineData(LogLevel.Error, "Error")]
    [InlineData(LogLevel.Warning, "Warn")]
    [InlineData(LogLevel.Info, "Info")]
    [InlineData(LogLevel.Debug, "Debug")]
    [InlineData(LogLevel.Trace, "Trace")]
    public void FormatLogLine_maps_levels_to_names(LogLevel level, string expectedToken)
    {
        string line = LogFileSink.FormatLogLine(level, "", "msg");
        Assert.Contains(" " + expectedToken + " ", line, StringComparison.Ordinal);
    }
}
