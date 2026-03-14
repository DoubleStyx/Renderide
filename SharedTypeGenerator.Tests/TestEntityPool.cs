using System.Reflection;
using Renderite.Shared;

namespace SharedTypeGenerator.Tests;

/// <summary>IMemoryPackerEntityPool implementation for tests.
/// Creates new instances via Activator.CreateInstance (no pooling).</summary>
public sealed class TestEntityPool : IMemoryPackerEntityPool
{
    private readonly Assembly _assembly;

    public TestEntityPool(Assembly assembly)
    {
        _assembly = assembly;
    }

    public T Borrow<T>() where T : class, IMemoryPackable, new()
    {
        return Activator.CreateInstance<T>();
    }

    public void Return<T>(T value) where T : class, IMemoryPackable, new()
    {
        // No pooling; discard
    }
}
