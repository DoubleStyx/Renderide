namespace RenderidePatches;

/// <summary>
/// Values shared between mod init and Harmony patches; must match <c>RendererInitResult.rendererIdentifier</c> from Renderide.
/// </summary>
internal static class RenderidePatchConstants
{
    /// <summary>
    /// <see cref="FrooxEngine.RenderSystem.RendererName" /> uses this prefix for the native Renderide renderer (e.g. <c>Renderide 0.1.0 (wgpu)</c>).
    /// </summary>
    internal const string RendererNamePrefix = "Renderide";
}
