# Test assembly dependency

Copy `Renderite.Shared.dll` from your Resonite installation here for roundtrip tests.

Example (Linux):
```bash
cp ~/.steam/steam/steamapps/common/Resonite/Renderite.Shared.dll generators/SharedTypeGenerator.Tests/lib/
```

Override the path via `RENDERITE_SHARED_DLL` or `RenderiteSharedDllPath` when building.
