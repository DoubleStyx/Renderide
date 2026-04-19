using System.Globalization;
using System.Text.RegularExpressions;
using SharedTypeGenerator.Logging;
using Xunit;

namespace SharedTypeGenerator.Tests.Unit;

/// <summary>Unit tests for <see cref="LogsLayout"/>.</summary>
public sealed class LogsLayoutTests : IDisposable
{
    private readonly string? _prevLogsRoot;

    /// <summary>Preserves <see cref="LogsLayout.LogsRootEnvVar"/> for isolation.</summary>
    public LogsLayoutTests()
    {
        _prevLogsRoot = Environment.GetEnvironmentVariable(LogsLayout.LogsRootEnvVar);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        RestoreEnv(LogsLayout.LogsRootEnvVar, _prevLogsRoot);
        GC.SuppressFinalize(this);
    }

    /// <summary><see cref="LogsLayout.ResolveLogsRoot"/> honours <c>RENDERIDE_LOGS_ROOT</c>.</summary>
    [Fact]
    public void ResolveLogsRoot_uses_env_when_set()
    {
        string temp = Path.Combine(Path.GetTempPath(), "renderide-logs-" + Guid.NewGuid().ToString("N", CultureInfo.InvariantCulture));
        Environment.SetEnvironmentVariable(LogsLayout.LogsRootEnvVar, temp);
        try
        {
            string root = LogsLayout.ResolveLogsRoot(gitTopLevel: null);
            Assert.Equal(Path.GetFullPath(temp), root);
        }
        finally
        {
            Environment.SetEnvironmentVariable(LogsLayout.LogsRootEnvVar, null);
        }
    }

    /// <summary>When unset, logs root falls back under the resolved Renderide tree.</summary>
    [Fact]
    public void ResolveLogsRoot_falls_back_without_env()
    {
        Environment.SetEnvironmentVariable(LogsLayout.LogsRootEnvVar, null);
        string root = LogsLayout.ResolveLogsRoot(gitTopLevel: null);
        Assert.EndsWith(Path.Combine("logs"), root, StringComparison.Ordinal);
    }

    /// <summary>New log paths use a UTC timestamp file name with second resolution.</summary>
    [Fact]
    public void EnsureNewSharedTypeGeneratorLogFilePath_creates_directory_and_stamp()
    {
        string temp = Path.Combine(Path.GetTempPath(), "stg-log-" + Guid.NewGuid().ToString("N", CultureInfo.InvariantCulture));
        Environment.SetEnvironmentVariable(LogsLayout.LogsRootEnvVar, temp);
        try
        {
            string path = LogsLayout.EnsureNewSharedTypeGeneratorLogFilePath(gitTopLevel: null);
            Assert.True(Directory.Exists(Path.GetDirectoryName(path)));
            string name = Path.GetFileName(path);
            Assert.Matches(new Regex(@"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.log$"), name);
        }
        finally
        {
            Environment.SetEnvironmentVariable(LogsLayout.LogsRootEnvVar, null);
            try
            {
                if (Directory.Exists(temp))
                    Directory.Delete(temp, recursive: true);
            }
            catch
            {
                // ignored
            }
        }
    }

    /// <summary>Parallel-safe log paths append a hex suffix.</summary>
    [Fact]
    public void EnsureUniqueTestSharedTypeGeneratorLogFilePath_adds_hex_suffix()
    {
        string temp = Path.Combine(Path.GetTempPath(), "stg-log-u-" + Guid.NewGuid().ToString("N", CultureInfo.InvariantCulture));
        Environment.SetEnvironmentVariable(LogsLayout.LogsRootEnvVar, temp);
        try
        {
            string path = LogsLayout.EnsureUniqueTestSharedTypeGeneratorLogFilePath(gitTopLevel: null);
            string name = Path.GetFileNameWithoutExtension(path);
            Assert.Contains("_", name, StringComparison.Ordinal);
            int lastUnderscore = name.LastIndexOf('_');
            string suffix = name[(lastUnderscore + 1)..];
            Assert.Equal(32, suffix.Length);
            Assert.Matches(new Regex("^[0-9a-f]{32}$"), suffix);
        }
        finally
        {
            Environment.SetEnvironmentVariable(LogsLayout.LogsRootEnvVar, null);
            try
            {
                if (Directory.Exists(temp))
                    Directory.Delete(temp, recursive: true);
            }
            catch
            {
                // ignored
            }
        }
    }

    /// <summary><see cref="LogsLayout.EnsureUniqueTestSharedTypeGeneratorLogFilePathUnderLogsRoot"/> places files under an explicit tree (not the repo).</summary>
    [Fact]
    public void EnsureUniqueTestSharedTypeGeneratorLogFilePathUnderLogsRoot_uses_explicit_root()
    {
        string root = Path.Combine(Path.GetTempPath(), "stg-explicit-" + Guid.NewGuid().ToString("N", CultureInfo.InvariantCulture));
        try
        {
            string path = LogsLayout.EnsureUniqueTestSharedTypeGeneratorLogFilePathUnderLogsRoot(root);
            string expectedDir = Path.Combine(Path.GetFullPath(root), LogsLayout.SharedTypeGeneratorSubdir);
            Assert.StartsWith(expectedDir + Path.DirectorySeparatorChar, path, StringComparison.Ordinal);
            Assert.True(Directory.Exists(Path.GetDirectoryName(path)));
        }
        finally
        {
            try
            {
                if (Directory.Exists(root))
                    Directory.Delete(root, recursive: true);
            }
            catch
            {
                // ignored
            }
        }
    }

    private static void RestoreEnv(string name, string? value)
    {
        if (value is null)
            Environment.SetEnvironmentVariable(name, null);
        else
            Environment.SetEnvironmentVariable(name, value);
    }
}
