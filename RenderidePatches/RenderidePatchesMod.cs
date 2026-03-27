using FrooxEngine;
using HarmonyLib;
using ResoniteModLoader;

namespace RenderidePatches;

/// <summary>
/// Resonite mod entry point: applies Harmony patches so <see cref="Renderite.Shared.ShaderUpload" /> carries a
/// deterministic shader label when the active renderer is Renderide.
/// </summary>
public sealed class RenderidePatchesMod : ResoniteMod
{
    /// <inheritdoc />
    public override string Name => "RenderidePatches";

    /// <inheritdoc />
    public override string Author => "DoubleStyx";

    /// <inheritdoc />
    public override string Version => "1.0.0";

    /// <inheritdoc />
    public override void OnEngineInit()
    {
        var rendererName = Engine.Current?.RenderSystem?.RendererName;
        if (!string.IsNullOrEmpty(rendererName) &&
            !rendererName.StartsWith(RenderidePatchConstants.RendererNamePrefix, StringComparison.Ordinal))
        {
            Msg($"RenderidePatches: Renderer is '{rendererName}', not Renderide; skipping Harmony patches.");
            return;
        }

        var harmony = new Harmony("com.renderide.patches");
        harmony.PatchAll(typeof(RenderidePatchesMod).Assembly);
        Msg("RenderidePatches: Harmony patches registered (shader upload labels for Renderide).");
    }
}
