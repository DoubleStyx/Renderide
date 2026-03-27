using System.Reflection;
using FrooxEngine;
using HarmonyLib;
using Renderite.Shared;

namespace RenderidePatches;

/// <summary>
/// Patches <c>RenderSystem.SendAssetUpdate</c> (internal; resolved via reflection) so <see cref="ShaderUpload" /> commands
/// sent to Renderide use a source-derived label in <see cref="ShaderUpload.file" /> instead of a local bundle path.
/// </summary>
[HarmonyPatch]
public static class ShaderUploadLabelPatch
{
    /// <summary>
    /// Resolves the internal asset IPC dispatch method; it is not callable from this assembly by name.
    /// </summary>
    /// <returns>The method to patch.</returns>
    public static MethodBase TargetMethod()
    {
        return AccessTools.Method(typeof(RenderSystem), "SendAssetUpdate", new[] { typeof(AssetCommand) })
               ?? throw new InvalidOperationException("RenderSystem.SendAssetUpdate(AssetCommand) not found.");
    }

    /// <summary>
    /// Runs before the renderer IPC send; mutates <paramref name="assetCommand" /> when it is a <see cref="ShaderUpload" />.
    /// </summary>
    /// <param name="__instance">The render system dispatching the command.</param>
    /// <param name="assetCommand">The asset command about to be sent.</param>
    [HarmonyPrefix]
    public static void PrefixApplyShaderLabel(RenderSystem __instance, AssetCommand assetCommand)
    {
        if (assetCommand is not ShaderUpload upload)
        {
            return;
        }

        var rendererName = __instance.RendererName;
        if (string.IsNullOrEmpty(rendererName) ||
            !rendererName.StartsWith(RenderidePatchConstants.RendererNamePrefix, StringComparison.Ordinal))
        {
            return;
        }

        var shader = __instance.Shaders.TryGet(upload.assetId);
        if (shader == null)
        {
            return;
        }

        var label = ShaderLabelBuilder.TryBuildShaderLabel(shader);
        if (label != null)
        {
            upload.file = label;
        }
    }
}
