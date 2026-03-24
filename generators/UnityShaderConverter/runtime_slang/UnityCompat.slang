// Unity / ShaderLab compatibility shims for Slang → WGSL conversion.
// Grows incrementally as more Resonite shaders are brought online.
//
// Include this file first in every generated translation unit (see SlangEmitter). It must appear
// before any #include of Unity .cginc headers so HLSLSupport.cginc selects the DX11-style texture
// path (Texture2D + SamplerState) instead of the legacy CG path (sampler2D / tex2D intrinsics).

#ifndef UNITY_SHADER_CONVERTER_COMPAT_INCLUDED
#define UNITY_SHADER_CONVERTER_COMPAT_INCLUDED

// -----------------------------------------------------------------------------
// Unity platform macros (match ShaderLab / D3D11-style compilation)
// SHADER_API_D3D11 forces UNITY_COMPILER_HLSL and the Texture2D branch in HLSLSupport.cginc.
// SHADER_TARGET satisfies Unity #if (SHADER_TARGET < 30) guards with a desktop SM4+ value.
// -----------------------------------------------------------------------------

#ifndef SHADER_API_D3D11
#define SHADER_API_D3D11 1
#endif

#ifndef SHADER_TARGET
#define SHADER_TARGET 50
#endif

/// Converter-only PBS selection: <c>UnityPBSLighting.cginc</c> requires a BRDF choice when <c>SHADER_TARGET</c> is at least 30.
#ifndef UNITY_PBS_USE_BRDF1
#define UNITY_PBS_USE_BRDF1 1
#endif

// -----------------------------------------------------------------------------
// Types and semantics shared with Unity headers
// -----------------------------------------------------------------------------

/// HLSL <c>clip</c> is not available in WGSL; map to early fragment discard for Slang targets.
#ifndef clip
#define clip(x) if ((x) < 0.0) discard
#endif

typedef float4 fixed4;
typedef float3 fixed3;
typedef float2 fixed2;
typedef float fixed;

#define SV_Target SV_Target0

// -----------------------------------------------------------------------------
// Legacy sampler type names (Cg-era signatures in UnityCG.cginc / Resonite Common.cginc).
// Unity’s DX11 path uses Texture2D + SamplerState; these aliases keep old parameter types valid.
// -----------------------------------------------------------------------------

#ifndef sampler2D
#define sampler2D Texture2D
#endif
#ifndef sampler3D
#define sampler3D Texture3D
#endif
#ifndef samplerCUBE
#define samplerCUBE TextureCube
#endif

// -----------------------------------------------------------------------------
// CG-style sampling for split texture + sampler declarations (UNITY_DECLARE_TEX2D etc.).
// Texture2D _MainTex pairs with SamplerState sampler_MainTex (sampler##tex token pasting).
// Do not enable UNITY_COMPILER_DXC here; Unity’s min16float / sampler2D_f block targets DXC, not WGSL.
// -----------------------------------------------------------------------------

#ifndef UNITY_SHADER_CONVERTER_LEGACY_TEX_MACROS
#define UNITY_SHADER_CONVERTER_LEGACY_TEX_MACROS

// Two-level join improves macro argument expansion before ## in some preprocessors.
#define UNITY_SC_JOIN(a, b) UNITY_SC_JOIN2(a, b)
#define UNITY_SC_JOIN2(a, b) a##b
#define UNITY_SC_SAMPLER_FOR(tex) UNITY_SC_JOIN(sampler, tex)

// Slang pastes `sampler`+`mainTex` into `samplermainTex`. UnityCG’s VertexLight uses split samplers inconsistently; bind a harmless fallback so the TU type-checks (VertexLight is effectively legacy).
SamplerState unity_usc_vertexlight_sampler_compat;
#define samplermainTex unity_usc_vertexlight_sampler_compat

// Borrowed sampler for UnityUI.cginc helpers that take sampler2D parameters; split Texture2D + SamplerState
// has no per-parameter sampler name. Matches Unity’s reuse-sampler pattern until explicit sampler parameters exist.
SamplerState unity_usc_ui_shared_sampler;

// Resonite Common.cginc EVR_* macros pass texture names through nested expansions; Slang can paste `sampler##formal`
// before substituting the actual texture. Sampling with one shared state matches “borrow sampler” behavior.
SamplerState unity_usc_resonite_macro_sampler;

// Use `usc_t` (not `tex`) as the macro parameter name: `tex` can interact badly with Slang/Unity token pasting.
#define tex2D(usc_t, uv) ((usc_t).Sample(UNITY_SC_SAMPLER_FOR(usc_t), (uv)))
#define tex2Dbias(usc_t, t) ((usc_t).SampleBias(UNITY_SC_SAMPLER_FOR(usc_t), (t).xy, (t).w))
#define tex2Dlod(usc_t, t) ((usc_t).SampleLevel(UNITY_SC_SAMPLER_FOR(usc_t), (t).xy, (t).w))
#define tex2Dgrad(usc_t, t, dx, dy) ((usc_t).SampleGrad(UNITY_SC_SAMPLER_FOR(usc_t), (t), (dx), (dy)))

// Projected sample (float4: coord.xy / coord.w). For float3 divisors use tex2Dproj3.
#define tex2Dproj(usc_t, coord) ((usc_t).Sample(UNITY_SC_SAMPLER_FOR(usc_t), (coord).xy / (coord).w))
#define tex2Dproj3(usc_t, coord) ((usc_t).Sample(UNITY_SC_SAMPLER_FOR(usc_t), (coord).xy / (coord).z))

#define tex3D(usc_t, uvw) ((usc_t).Sample(UNITY_SC_SAMPLER_FOR(usc_t), (uvw)))
#define tex3Dbias(usc_t, t) ((usc_t).SampleBias(UNITY_SC_SAMPLER_FOR(usc_t), (t).xyz, (t).w))
#define tex3Dlod(usc_t, t) ((usc_t).SampleLevel(UNITY_SC_SAMPLER_FOR(usc_t), (t).xyz, (t).w))
#define tex3Dgrad(usc_t, t, dx, dy) ((usc_t).SampleGrad(UNITY_SC_SAMPLER_FOR(usc_t), (t), (dx), (dy)))
#define tex3Dproj(usc_t, t) ((usc_t).Sample(UNITY_SC_SAMPLER_FOR(usc_t), (t).xyz / (t).w))

#define texCUBE(usc_t, dir) ((usc_t).Sample(UNITY_SC_SAMPLER_FOR(usc_t), float3((dir).xyz)))
#define texCUBEbias(usc_t, t) ((usc_t).SampleBias(UNITY_SC_SAMPLER_FOR(usc_t), (t).xyz, (t).w))
#define texCUBElod(usc_t, t) ((usc_t).SampleLevel(UNITY_SC_SAMPLER_FOR(usc_t), (t).xyz, (t).w))
#define texCUBEgrad(usc_t, t, dx, dy) ((usc_t).SampleGrad(UNITY_SC_SAMPLER_FOR(usc_t), (t).xyz, (dx), (dy)))
#define texCUBEproj(usc_t, t) ((usc_t).Sample(UNITY_SC_SAMPLER_FOR(usc_t), (t).xyz / (t).w))

#endif // UNITY_SHADER_CONVERTER_LEGACY_TEX_MACROS

#endif // UNITY_SHADER_CONVERTER_COMPAT_INCLUDED
