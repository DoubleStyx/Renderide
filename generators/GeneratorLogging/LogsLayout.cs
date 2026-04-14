using System.Globalization;

namespace Renderide.Generators.Logging;

/// <summary>
/// Resolves log paths aligned with the Rust workspace logger crate: <c>logs/&lt;component&gt;/&lt;UTC-timestamp&gt;.log</c>,
/// optionally rooted at <c>RENDERIDE_LOGS_ROOT</c>.
/// </summary>
public static class LogsLayout
{
    /// <summary>Subdirectory under the logs root for SharedTypeGenerator runs (matches user-facing layout <c>logs/SharedTypeGenerator/</c>).</summary>
    public const string SharedTypeGeneratorSubdir = "SharedTypeGenerator";

    /// <summary>
    /// Environment variable: when set, this directory is the logs root (component subdirectories are created beneath it).
    /// Same semantics as the Rust crate’s <c>RENDERIDE_LOGS_ROOT</c>.
    /// </summary>
    public const string LogsRootEnvVar = "RENDERIDE_LOGS_ROOT";

    /// <summary>
    /// Returns the logs root: <c>RENDERIDE_LOGS_ROOT</c> if set, otherwise <c>{repository root}/logs</c>
    /// where the root is <see cref="RenderidePathResolver.ResolveRenderideRoot"/>.
    /// </summary>
    public static string ResolveLogsRoot(string? gitTopLevel)
    {
        string? env = Environment.GetEnvironmentVariable(LogsRootEnvVar);
        if (!string.IsNullOrWhiteSpace(env))
            return Path.GetFullPath(env.Trim());
        string workspaceRoot = RenderidePathResolver.ResolveRenderideRoot(gitTopLevel);
        return Path.Combine(workspaceRoot, "logs");
    }

    /// <summary><c>ResolveLogsRoot</c> / <see cref="SharedTypeGeneratorSubdir"/>.</summary>
    public static string SharedTypeGeneratorLogDirectory(string? gitTopLevel) =>
        Path.Combine(ResolveLogsRoot(gitTopLevel), SharedTypeGeneratorSubdir);

    /// <summary>
    /// A new log file path for this process: <c>.../SharedTypeGenerator/YYYY-MM-DD_HH-mm-ss.log</c> in UTC,
    /// matching the Rust workspace logger crate’s <c>log_filename_timestamp</c> (second resolution).
    /// Ensures the component directory exists.
    /// </summary>
    public static string EnsureNewSharedTypeGeneratorLogFilePath(string? gitTopLevel)
    {
        string dir = SharedTypeGeneratorLogDirectory(gitTopLevel);
        Directory.CreateDirectory(dir);
        string stamp = FormatLogFilenameTimestampUtc();
        return Path.Combine(dir, stamp + ".log");
    }

    /// <summary>Filename stem <c>YYYY-MM-DD_HH-mm-ss</c> in UTC (same pattern as Rust <c>log_filename_timestamp</c>).</summary>
    public static string FormatLogFilenameTimestampUtc()
    {
        DateTime utc = DateTime.UtcNow;
        return utc.ToString("yyyy-MM-dd_HH-mm-ss", CultureInfo.InvariantCulture);
    }

    /// <summary>
    /// A unique log file under <see cref="SharedTypeGeneratorLogDirectory"/> for parallel test runs:
    /// <c>{UTC-stamp}_{guid}.log</c>.
    /// </summary>
    public static string EnsureUniqueTestSharedTypeGeneratorLogFilePath(string? gitTopLevel)
    {
        string dir = SharedTypeGeneratorLogDirectory(gitTopLevel);
        Directory.CreateDirectory(dir);
        string stamp = FormatLogFilenameTimestampUtc();
        string unique = Guid.NewGuid().ToString("N", CultureInfo.InvariantCulture);
        return Path.Combine(dir, $"{stamp}_{unique}.log");
    }
}
