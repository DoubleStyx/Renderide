using System.Globalization;
using NotEnoughLogs;
using NotEnoughLogs.Sinks;

namespace SharedTypeGenerator.Tests.Unit.Support;

/// <summary>Captures log message bodies for assertions in sink unit tests.</summary>
internal sealed class CollectingSink : ILoggerSink
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
