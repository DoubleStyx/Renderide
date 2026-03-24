// Injected by UnityShaderConverter after the shader’s leading #include block (see SlangEmitter).
// Slang’s preprocessor mishandles Unity’s `sampler##tex` when `tex` is `_MainTex` (becomes `samplermainTex`).
// Redeclare texture/sampler macros using a two-step `UNITY_SC_JOIN(sampler, tex)` so names match Unity (e.g. `sampler_MainTex`).
// Each `#undef` is wrapped in `#ifdef` so Slang does not warn when Unity headers never defined the macro.

#ifndef UNITY_SHADER_CONVERTER_POST_UNITY_INCLUDED
#define UNITY_SHADER_CONVERTER_POST_UNITY_INCLUDED

#ifndef UNITY_SC_JOIN
#define UNITY_SC_JOIN(a, b) UNITY_SC_JOIN2(a, b)
#define UNITY_SC_JOIN2(a, b) a##b
#endif
#ifndef UNITY_SC_SAMPLER_FOR
#define UNITY_SC_SAMPLER_FOR(tex) UNITY_SC_JOIN(sampler, tex)
#endif

#ifdef UNITY_DECLARE_TEX2D
#undef UNITY_DECLARE_TEX2D
#endif
#define UNITY_DECLARE_TEX2D(tex) Texture2D tex; SamplerState UNITY_SC_SAMPLER_FOR(tex)
#ifdef UNITY_DECLARE_TEX2D_NOSAMPLER
#undef UNITY_DECLARE_TEX2D_NOSAMPLER
#endif
#define UNITY_DECLARE_TEX2D_NOSAMPLER(tex) Texture2D tex
#ifdef UNITY_DECLARE_TEX2D_NOSAMPLER_INT
#undef UNITY_DECLARE_TEX2D_NOSAMPLER_INT
#endif
#define UNITY_DECLARE_TEX2D_NOSAMPLER_INT(tex) Texture2D<int4> tex
#ifdef UNITY_DECLARE_TEX2D_NOSAMPLER_UINT
#undef UNITY_DECLARE_TEX2D_NOSAMPLER_UINT
#endif
#define UNITY_DECLARE_TEX2D_NOSAMPLER_UINT(tex) Texture2D<uint4> tex
#ifdef UNITY_SAMPLE_TEX2D
#undef UNITY_SAMPLE_TEX2D
#endif
#define UNITY_SAMPLE_TEX2D(tex, coord) (tex).Sample(UNITY_SC_SAMPLER_FOR(tex), coord)
#ifdef UNITY_SAMPLE_TEX2D_LOD
#undef UNITY_SAMPLE_TEX2D_LOD
#endif
#define UNITY_SAMPLE_TEX2D_LOD(tex, coord, lod) (tex).SampleLevel(UNITY_SC_SAMPLER_FOR(tex), coord, lod)
#ifdef UNITY_SAMPLE_TEX2D_SAMPLER
#undef UNITY_SAMPLE_TEX2D_SAMPLER
#endif
#define UNITY_SAMPLE_TEX2D_SAMPLER(tex, samplertex, coord) (tex).Sample(UNITY_SC_SAMPLER_FOR(samplertex), coord)
#ifdef UNITY_SAMPLE_TEX2D_SAMPLER_LOD
#undef UNITY_SAMPLE_TEX2D_SAMPLER_LOD
#endif
#define UNITY_SAMPLE_TEX2D_SAMPLER_LOD(tex, samplertex, coord, lod) (tex).SampleLevel(UNITY_SC_SAMPLER_FOR(samplertex), coord, lod)

#ifdef UNITY_DECLARE_TEX2D_HALF
#undef UNITY_DECLARE_TEX2D_HALF
#endif
#define UNITY_DECLARE_TEX2D_HALF(tex) Texture2D tex; SamplerState UNITY_SC_SAMPLER_FOR(tex)
#ifdef UNITY_DECLARE_TEX2D_FLOAT
#undef UNITY_DECLARE_TEX2D_FLOAT
#endif
#define UNITY_DECLARE_TEX2D_FLOAT(tex) Texture2D tex; SamplerState UNITY_SC_SAMPLER_FOR(tex)
#ifdef UNITY_DECLARE_TEX2D_NOSAMPLER_HALF
#undef UNITY_DECLARE_TEX2D_NOSAMPLER_HALF
#endif
#define UNITY_DECLARE_TEX2D_NOSAMPLER_HALF(tex) Texture2D tex
#ifdef UNITY_DECLARE_TEX2D_NOSAMPLER_FLOAT
#undef UNITY_DECLARE_TEX2D_NOSAMPLER_FLOAT
#endif
#define UNITY_DECLARE_TEX2D_NOSAMPLER_FLOAT(tex) Texture2D tex

#ifdef UNITY_DECLARE_TEXCUBE
#undef UNITY_DECLARE_TEXCUBE
#endif
#define UNITY_DECLARE_TEXCUBE(tex) TextureCube tex; SamplerState UNITY_SC_SAMPLER_FOR(tex)
#ifdef UNITY_DECLARE_TEXCUBE_NOSAMPLER
#undef UNITY_DECLARE_TEXCUBE_NOSAMPLER
#endif
#define UNITY_DECLARE_TEXCUBE_NOSAMPLER(tex) TextureCube tex
#ifdef UNITY_ARGS_TEXCUBE
#undef UNITY_ARGS_TEXCUBE
#endif
#define UNITY_ARGS_TEXCUBE(tex) TextureCube tex, SamplerState UNITY_SC_SAMPLER_FOR(tex)
#ifdef UNITY_PASS_TEXCUBE
#undef UNITY_PASS_TEXCUBE
#endif
#define UNITY_PASS_TEXCUBE(tex) tex, UNITY_SC_SAMPLER_FOR(tex)
#ifdef UNITY_PASS_TEXCUBE_SAMPLER
#undef UNITY_PASS_TEXCUBE_SAMPLER
#endif
#define UNITY_PASS_TEXCUBE_SAMPLER(tex, samplertex) tex, UNITY_SC_SAMPLER_FOR(samplertex)
#ifdef UNITY_PASS_TEXCUBE_SAMPLER_LOD
#undef UNITY_PASS_TEXCUBE_SAMPLER_LOD
#endif
#define UNITY_PASS_TEXCUBE_SAMPLER_LOD(tex, samplertex, lod) tex, UNITY_SC_SAMPLER_FOR(samplertex), lod
#ifdef UNITY_ARGS_TEXCUBE_NOSAMPLER
#undef UNITY_ARGS_TEXCUBE_NOSAMPLER
#endif
#define UNITY_ARGS_TEXCUBE_NOSAMPLER(tex) TextureCube tex
#ifdef UNITY_PASS_TEXCUBE_NOSAMPLER
#undef UNITY_PASS_TEXCUBE_NOSAMPLER
#endif
#define UNITY_PASS_TEXCUBE_NOSAMPLER(tex) tex
#ifdef UNITY_SAMPLE_TEXCUBE
#undef UNITY_SAMPLE_TEXCUBE
#endif
#define UNITY_SAMPLE_TEXCUBE(tex, coord) (tex).Sample(UNITY_SC_SAMPLER_FOR(tex), coord)
#ifdef UNITY_SAMPLE_TEXCUBE_LOD
#undef UNITY_SAMPLE_TEXCUBE_LOD
#endif
#define UNITY_SAMPLE_TEXCUBE_LOD(tex, coord, lod) (tex).SampleLevel(UNITY_SC_SAMPLER_FOR(tex), coord, lod)
#ifdef UNITY_SAMPLE_TEXCUBE_SAMPLER
#undef UNITY_SAMPLE_TEXCUBE_SAMPLER
#endif
#define UNITY_SAMPLE_TEXCUBE_SAMPLER(tex, samplertex, coord) (tex).Sample(UNITY_SC_SAMPLER_FOR(samplertex), coord)
#ifdef UNITY_SAMPLE_TEXCUBE_SAMPLER_LOD
#undef UNITY_SAMPLE_TEXCUBE_SAMPLER_LOD
#endif
#define UNITY_SAMPLE_TEXCUBE_SAMPLER_LOD(tex, samplertex, coord, lod) (tex).SampleLevel(UNITY_SC_SAMPLER_FOR(samplertex), coord, lod)

#ifdef UNITY_DECLARE_TEX3D
#undef UNITY_DECLARE_TEX3D
#endif
#define UNITY_DECLARE_TEX3D(tex) Texture3D tex; SamplerState UNITY_SC_SAMPLER_FOR(tex)
#ifdef UNITY_DECLARE_TEX3D_NOSAMPLER
#undef UNITY_DECLARE_TEX3D_NOSAMPLER
#endif
#define UNITY_DECLARE_TEX3D_NOSAMPLER(tex) Texture3D tex
#ifdef UNITY_SAMPLE_TEX3D
#undef UNITY_SAMPLE_TEX3D
#endif
#define UNITY_SAMPLE_TEX3D(tex, coord) (tex).Sample(UNITY_SC_SAMPLER_FOR(tex), coord)
#ifdef UNITY_SAMPLE_TEX3D_LOD
#undef UNITY_SAMPLE_TEX3D_LOD
#endif
#define UNITY_SAMPLE_TEX3D_LOD(tex, coord, lod) (tex).SampleLevel(UNITY_SC_SAMPLER_FOR(tex), coord, lod)
#ifdef UNITY_SAMPLE_TEX3D_SAMPLER
#undef UNITY_SAMPLE_TEX3D_SAMPLER
#endif
#define UNITY_SAMPLE_TEX3D_SAMPLER(tex, samplertex, coord) (tex).Sample(UNITY_SC_SAMPLER_FOR(samplertex), coord)
#ifdef UNITY_SAMPLE_TEX3D_SAMPLER_LOD
#undef UNITY_SAMPLE_TEX3D_SAMPLER_LOD
#endif
#define UNITY_SAMPLE_TEX3D_SAMPLER_LOD(tex, samplertex, coord, lod) (tex).SampleLevel(UNITY_SC_SAMPLER_FOR(samplertex), coord, lod)
#ifdef UNITY_DECLARE_TEX3D_FLOAT
#undef UNITY_DECLARE_TEX3D_FLOAT
#endif
#define UNITY_DECLARE_TEX3D_FLOAT(tex) Texture3D tex; SamplerState UNITY_SC_SAMPLER_FOR(tex)
#ifdef UNITY_DECLARE_TEX3D_HALF
#undef UNITY_DECLARE_TEX3D_HALF
#endif
#define UNITY_DECLARE_TEX3D_HALF(tex) Texture3D tex; SamplerState UNITY_SC_SAMPLER_FOR(tex)

#ifdef UNITY_DECLARE_TEX2DARRAY_MS
#undef UNITY_DECLARE_TEX2DARRAY_MS
#endif
#define UNITY_DECLARE_TEX2DARRAY_MS(tex) Texture2DMSArray<float> tex; SamplerState UNITY_SC_SAMPLER_FOR(tex)
#ifdef UNITY_DECLARE_TEX2DARRAY_MS_NOSAMPLER
#undef UNITY_DECLARE_TEX2DARRAY_MS_NOSAMPLER
#endif
#define UNITY_DECLARE_TEX2DARRAY_MS_NOSAMPLER(tex) Texture2DArray<float> tex
#ifdef UNITY_DECLARE_TEX2DARRAY
#undef UNITY_DECLARE_TEX2DARRAY
#endif
#define UNITY_DECLARE_TEX2DARRAY(tex) Texture2DArray tex; SamplerState UNITY_SC_SAMPLER_FOR(tex)
#ifdef UNITY_DECLARE_TEX2DARRAY_NOSAMPLER
#undef UNITY_DECLARE_TEX2DARRAY_NOSAMPLER
#endif
#define UNITY_DECLARE_TEX2DARRAY_NOSAMPLER(tex) Texture2DArray tex
#ifdef UNITY_ARGS_TEX2DARRAY
#undef UNITY_ARGS_TEX2DARRAY
#endif
#define UNITY_ARGS_TEX2DARRAY(tex) Texture2DArray tex, SamplerState UNITY_SC_SAMPLER_FOR(tex)
#ifdef UNITY_PASS_TEX2DARRAY
#undef UNITY_PASS_TEX2DARRAY
#endif
#define UNITY_PASS_TEX2DARRAY(tex) tex, UNITY_SC_SAMPLER_FOR(tex)
#ifdef UNITY_ARGS_TEX2DARRAY_NOSAMPLER
#undef UNITY_ARGS_TEX2DARRAY_NOSAMPLER
#endif
#define UNITY_ARGS_TEX2DARRAY_NOSAMPLER(tex) Texture2DArray tex
#ifdef UNITY_PASS_TEX2DARRAY_NOSAMPLER
#undef UNITY_PASS_TEX2DARRAY_NOSAMPLER
#endif
#define UNITY_PASS_TEX2DARRAY_NOSAMPLER(tex) tex
#ifdef UNITY_SAMPLE_TEX2DARRAY
#undef UNITY_SAMPLE_TEX2DARRAY
#endif
#define UNITY_SAMPLE_TEX2DARRAY(tex, coord) (tex).Sample(UNITY_SC_SAMPLER_FOR(tex), coord)
#ifdef UNITY_SAMPLE_TEX2DARRAY_LOD
#undef UNITY_SAMPLE_TEX2DARRAY_LOD
#endif
#define UNITY_SAMPLE_TEX2DARRAY_LOD(tex, coord, lod) (tex).SampleLevel(UNITY_SC_SAMPLER_FOR(tex), coord, lod)
#ifdef UNITY_SAMPLE_TEX2DARRAY_SAMPLER
#undef UNITY_SAMPLE_TEX2DARRAY_SAMPLER
#endif
#define UNITY_SAMPLE_TEX2DARRAY_SAMPLER(tex, samplertex, coord) (tex).Sample(UNITY_SC_SAMPLER_FOR(samplertex), coord)
#ifdef UNITY_SAMPLE_TEX2DARRAY_SAMPLER_LOD
#undef UNITY_SAMPLE_TEX2DARRAY_SAMPLER_LOD
#endif
#define UNITY_SAMPLE_TEX2DARRAY_SAMPLER_LOD(tex, samplertex, coord, lod) (tex).SampleLevel(UNITY_SC_SAMPLER_FOR(samplertex), coord, lod)

#ifdef UNITY_DECLARE_TEXCUBEARRAY
#undef UNITY_DECLARE_TEXCUBEARRAY
#endif
#define UNITY_DECLARE_TEXCUBEARRAY(tex) TextureCubeArray tex; SamplerState UNITY_SC_SAMPLER_FOR(tex)
#ifdef UNITY_DECLARE_TEXCUBEARRAY_NOSAMPLER
#undef UNITY_DECLARE_TEXCUBEARRAY_NOSAMPLER
#endif
#define UNITY_DECLARE_TEXCUBEARRAY_NOSAMPLER(tex) TextureCubeArray tex
#ifdef UNITY_ARGS_TEXCUBEARRAY
#undef UNITY_ARGS_TEXCUBEARRAY
#endif
#define UNITY_ARGS_TEXCUBEARRAY(tex) TextureCubeArray tex, SamplerState UNITY_SC_SAMPLER_FOR(tex)
#ifdef UNITY_PASS_TEXCUBEARRAY
#undef UNITY_PASS_TEXCUBEARRAY
#endif
#define UNITY_PASS_TEXCUBEARRAY(tex) tex, UNITY_SC_SAMPLER_FOR(tex)
#ifdef UNITY_ARGS_TEXCUBEARRAY_NOSAMPLER
#undef UNITY_ARGS_TEXCUBEARRAY_NOSAMPLER
#endif
#define UNITY_ARGS_TEXCUBEARRAY_NOSAMPLER(tex) TextureCubeArray tex
#ifdef UNITY_PASS_TEXCUBEARRAY_NOSAMPLER
#undef UNITY_PASS_TEXCUBEARRAY_NOSAMPLER
#endif
#define UNITY_PASS_TEXCUBEARRAY_NOSAMPLER(tex) tex

#if defined(SHADER_API_PSSL)
#ifdef UNITY_SAMPLE_TEXCUBEARRAY
#undef UNITY_SAMPLE_TEXCUBEARRAY
#endif
#define UNITY_SAMPLE_TEXCUBEARRAY(tex, coord) (tex).Sample(UNITY_SC_SAMPLER_FOR(tex), float4((coord).xyz, round((coord).w)))
#ifdef UNITY_SAMPLE_TEXCUBEARRAY_LOD
#undef UNITY_SAMPLE_TEXCUBEARRAY_LOD
#endif
#define UNITY_SAMPLE_TEXCUBEARRAY_LOD(tex, coord, lod) (tex).SampleLevel(UNITY_SC_SAMPLER_FOR(tex), float4((coord).xyz, round((coord).w)), lod)
#ifdef UNITY_SAMPLE_TEXCUBEARRAY_SAMPLER
#undef UNITY_SAMPLE_TEXCUBEARRAY_SAMPLER
#endif
#define UNITY_SAMPLE_TEXCUBEARRAY_SAMPLER(tex, samplertex, coord) (tex).Sample(UNITY_SC_SAMPLER_FOR(samplertex), float4((coord).xyz, round((coord).w)))
#ifdef UNITY_SAMPLE_TEXCUBEARRAY_SAMPLER_LOD
#undef UNITY_SAMPLE_TEXCUBEARRAY_SAMPLER_LOD
#endif
#define UNITY_SAMPLE_TEXCUBEARRAY_SAMPLER_LOD(tex, samplertex, coord, lod) (tex).SampleLevel(UNITY_SC_SAMPLER_FOR(samplertex), float4((coord).xyz, round((coord).w)), lod)
#else
#ifdef UNITY_SAMPLE_TEXCUBEARRAY
#undef UNITY_SAMPLE_TEXCUBEARRAY
#endif
#define UNITY_SAMPLE_TEXCUBEARRAY(tex, coord) (tex).Sample(UNITY_SC_SAMPLER_FOR(tex), coord)
#ifdef UNITY_SAMPLE_TEXCUBEARRAY_LOD
#undef UNITY_SAMPLE_TEXCUBEARRAY_LOD
#endif
#define UNITY_SAMPLE_TEXCUBEARRAY_LOD(tex, coord, lod) (tex).SampleLevel(UNITY_SC_SAMPLER_FOR(tex), coord, lod)
#ifdef UNITY_SAMPLE_TEXCUBEARRAY_SAMPLER
#undef UNITY_SAMPLE_TEXCUBEARRAY_SAMPLER
#endif
#define UNITY_SAMPLE_TEXCUBEARRAY_SAMPLER(tex, samplertex, coord) (tex).Sample(UNITY_SC_SAMPLER_FOR(samplertex), coord)
#ifdef UNITY_SAMPLE_TEXCUBEARRAY_SAMPLER_LOD
#undef UNITY_SAMPLE_TEXCUBEARRAY_SAMPLER_LOD
#endif
#define UNITY_SAMPLE_TEXCUBEARRAY_SAMPLER_LOD(tex, samplertex, coord, lod) (tex).SampleLevel(UNITY_SC_SAMPLER_FOR(samplertex), coord, lod)
#endif

// -----------------------------------------------------------------------------
// Depth / screenspace: HLSLSupport’s non-stereo UNITY_DECLARE_DEPTH_TEXTURE is
// `sampler2D_float tex`, which becomes Texture2D-only after our sampler2D typedef.
// tex2D then references SamplerState UNITY_SC_SAMPLER_FOR(tex), which was never declared.
// Mirror Unity’s split-sampler pattern for the non-stereo branch only (stereo uses arrays).
// -----------------------------------------------------------------------------
#if !defined(UNITY_STEREO_INSTANCING_ENABLED) && !defined(UNITY_STEREO_MULTIVIEW_ENABLED)
#ifdef UNITY_DECLARE_DEPTH_TEXTURE
#undef UNITY_DECLARE_DEPTH_TEXTURE
#endif
#define UNITY_DECLARE_DEPTH_TEXTURE(tex) Texture2D_float tex; SamplerState UNITY_SC_SAMPLER_FOR(tex);
#ifdef UNITY_DECLARE_SCREENSPACE_TEXTURE
#undef UNITY_DECLARE_SCREENSPACE_TEXTURE
#endif
#define UNITY_DECLARE_SCREENSPACE_TEXTURE(tex) Texture2D_float tex; SamplerState UNITY_SC_SAMPLER_FOR(tex);
#endif

#endif // UNITY_SHADER_CONVERTER_POST_UNITY_INCLUDED
