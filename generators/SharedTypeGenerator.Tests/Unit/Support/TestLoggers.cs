using NotEnoughLogs;
using NotEnoughLogs.Behaviour;
using NotEnoughLogs.Sinks;

namespace SharedTypeGenerator.Tests.Unit.Support;

/// <summary>Logger factory helpers for unit tests.</summary>
internal static class TestLoggers
{
    /// <summary>Creates a test logger with a collecting sink unless a specific sink is provided.</summary>
    public static Logger Create(LogLevel maxLevel = LogLevel.Trace, ILoggerSink? sink = null) =>
        new(
            [sink ?? new CollectingSink()],
            new LoggerConfiguration
            {
                Behaviour = new DirectLoggingBehaviour(),
                MaxLevel = maxLevel,
            });
}
