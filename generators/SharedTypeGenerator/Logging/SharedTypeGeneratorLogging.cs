namespace SharedTypeGenerator.Logging;

/// <summary>
/// Factory for the SharedTypeGenerator log pipeline: suppresses warnings, writes to a log file, mirrors
/// error-level lines to stderr.
/// </summary>
public static class SharedTypeGeneratorLogging
{
    /// <summary>
    /// Builds the composed sink used by SharedTypeGenerator and its tests: <see cref="SuppressWarningsSink"/> -&gt;
    /// <see cref="ErrorDuplicatingSink"/> -&gt; <see cref="LogFileSink"/>.
    /// </summary>
    /// <param name="logFilePath">Target log file path (truncated per <see cref="LogFileSink"/>).</param>
    /// <param name="stderr">Optional duplicate destination for errors; defaults to <see cref="Console.Error"/>.</param>
    public static SuppressWarningsSink CreateMainSink(string logFilePath, TextWriter? stderr = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(logFilePath);
        return new SuppressWarningsSink(new ErrorDuplicatingSink(new LogFileSink(logFilePath), stderr));
    }
}
