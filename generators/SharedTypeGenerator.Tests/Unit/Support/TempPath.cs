namespace SharedTypeGenerator.Tests.Unit.Support;

/// <summary>Random-named file under the system temp directory; deletes on <see cref="Dispose"/>.</summary>
internal sealed class TempFile : IDisposable
{
    /// <summary>Guards <see cref="Dispose"/> so file deletion runs once.</summary>
    private bool _disposed;

    /// <summary>Creates a path that does not create a zero-byte file (unlike <see cref="Path.GetTempFileName"/>).</summary>
    public TempFile()
    {
        FilePath = System.IO.Path.Combine(System.IO.Path.GetTempPath(), System.IO.Path.GetRandomFileName());
    }

    /// <summary>Full path to the temp file.</summary>
    public string FilePath { get; }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        try
        {
            if (File.Exists(FilePath))
                File.Delete(FilePath);
        }
        catch (IOException)
        {
        }
        catch (UnauthorizedAccessException)
        {
        }

        GC.SuppressFinalize(this);
    }
}

/// <summary>Creates a unique directory under the system temp directory; deletes recursively on <see cref="Dispose"/>.</summary>
internal sealed class TempDirectory : IDisposable
{
    /// <summary>Guards <see cref="Dispose"/> so directory deletion runs once.</summary>
    private bool _disposed;

    /// <summary>Creates the directory immediately.</summary>
    public TempDirectory()
    {
        DirectoryPath = System.IO.Path.Combine(System.IO.Path.GetTempPath(), Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(DirectoryPath);
    }

    /// <summary>Full path to the temp directory.</summary>
    public string DirectoryPath { get; }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        try
        {
            if (Directory.Exists(DirectoryPath))
                Directory.Delete(DirectoryPath, recursive: true);
        }
        catch (IOException)
        {
        }
        catch (UnauthorizedAccessException)
        {
        }

        GC.SuppressFinalize(this);
    }
}
