namespace SharedTypeGenerator.Tests.Unit;

/// <summary>Minimal stand-ins for Renderite.Shared well-known types used by <see cref="SharedTypeGenerator.Analysis.WellKnownTypes"/>.</summary>
public interface IMemoryPackable
{
}

/// <summary>Open generic base used by <see cref="SharedTypeGenerator.Analysis.FieldClassifier"/> polymorphic detection.</summary>
public abstract class PolymorphicMemoryPackableEntity<T> : IMemoryPackable
{
}

/// <summary>Concrete packable reference type for list classification tests.</summary>
public sealed class TestPackable : IMemoryPackable
{
}

/// <summary>Concrete polymorphic leaf for list classification tests.</summary>
public sealed class TestPolymorphicLeaf : PolymorphicMemoryPackableEntity<int>
{
}
