using System.Runtime;
using CommandLine;
using NotEnoughLogs;
using SharedTypeGenerator;
using NotEnoughLogs.Behaviour;
using SharedTypeGenerator.Logging;
using SharedTypeGenerator.Options;

GCSettings.LatencyMode = GCLatencyMode.Batch;

Parser.Default.ParseArguments<GeneratorOptions>(args)
    .WithParsed(options =>
    {
        LogLevel maxLevel = ResolveMaxLogLevel(options.Verbose);
        using var logger = new Logger(new LoggerConfiguration
        {
            Behaviour = new DirectLoggingBehaviour(),
            MaxLevel = maxLevel,
        });
        GeneratorRunner.Run(options, logger);
    });

static LogLevel ResolveMaxLogLevel(bool verbose)
{
    if (verbose)
        return LogLevel.Trace;
#if DEBUG
    return LogLevel.Trace;
#else
    return LogLevel.Info;
#endif
}
