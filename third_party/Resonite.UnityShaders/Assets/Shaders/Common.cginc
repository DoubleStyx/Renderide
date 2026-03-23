#ifndef EVRCOMMON
#define EVRCOMMON
#include "UnityCG.cginc"
#include "UnityStandardUtils.cginc"
#include "EVRPoissonDisc.cginc"
UNITY_INSTANCING_BUFFER_START(EVR_INST_BUFFER)
// Do our regular uniforms first.
UNITY_DEFINE_INSTANCED_PROP(fixed4, _Color)
UNITY_DEFINE_INSTANCED_PROP(fixed4, _EmissionColor)
UNITY_DEFINE_INSTANCED_PROP(float4, _SpecularColor)
UNITY_DEFINE_INSTANCED_PROP(float, _NormalScale)
UNITY_DEFINE_INSTANCED_PROP(float, _Lerp)
UNITY_DEFINE_INSTANCED_PROP(float, _Glossiness)
UNITY_DEFINE_INSTANCED_PROP(float, _Metallic)

// Now do our variant specific uniforms.

#if defined(_COLORMASK) || defined(_COLORMAP) || defined(_LERP)
UNITY_DEFINE_INSTANCED_PROP(fixed4, _Color1)

UNITY_DEFINE_INSTANCED_PROP(float4, _EmissionColor1)

UNITY_DEFINE_INSTANCED_PROP(float, _Glossiness1)

UNITY_DEFINE_INSTANCED_PROP(float, _Metallic1)

UNITY_DEFINE_INSTANCED_PROP(float4, _SpecularColor1)

UNITY_DEFINE_INSTANCED_PROP(float, _NormalScale1)
#if defined(_COLORMAP) || defined(_COLORMASK)
UNITY_DEFINE_INSTANCED_PROP(fixed4, _Color2)
UNITY_DEFINE_INSTANCED_PROP(fixed4, _Color3)

UNITY_DEFINE_INSTANCED_PROP(float4, _EmissionColor2)
UNITY_DEFINE_INSTANCED_PROP(float4, _EmissionColor3)

UNITY_DEFINE_INSTANCED_PROP(float, _Glossiness2)
UNITY_DEFINE_INSTANCED_PROP(float, _Glossiness3)

UNITY_DEFINE_INSTANCED_PROP(float, _Metallic2)
UNITY_DEFINE_INSTANCED_PROP(float, _Metallic3)

UNITY_DEFINE_INSTANCED_PROP(float4, _SpecularColor2)
UNITY_DEFINE_INSTANCED_PROP(float4, _SpecularColor3)
#endif
#endif

#ifdef _PACKED_NORMALMAP
UNITY_DEFINE_INSTANCED_PROP(float, _NormalScale2)
UNITY_DEFINE_INSTANCED_PROP(float, _NormalScale3)
#endif

#ifdef _DETAIL_NORMALMAP
UNITY_DEFINE_INSTANCED_PROP(float, _DetailNormalMapScale)
#endif

UNITY_INSTANCING_BUFFER_END(EVR_INST_BUFFER)

#if defined(_ALBEDOTEX) || defined(_EMISSIONTEX) || defined(_METALLICMAP) || defined(_SPECULARMAP) || defined(_OCCLUSION)
// Basically, force the albedos on just so we have a sampler we can attach our textures to.
#define SAMPLER_REQUIRED 1
#endif

// Albedo texture definitions
#if defined(SAMPLER_REQUIRED) || defined(_ALBEDOTEX)
	UNITY_DECLARE_TEX2D(_MainTex);
	#if defined(_LERP) || defined(_COLORMAP)
		UNITY_DECLARE_TEX2D(_MainTex1);
		#if defined(_COLORMAP)
			UNITY_DECLARE_TEX2D(_MainTex2);
			UNITY_DECLARE_TEX2D(_MainTex3);
		#endif
	#endif
#endif

// Emissive texture definitions
#if defined(_EMISSIONTEX)
	UNITY_DECLARE_TEX2D_NOSAMPLER(_EmissionMap);
	#if defined(_LERP) || defined(_COLORMAP)
		UNITY_DECLARE_TEX2D_NOSAMPLER(_EmissionMap1);
		#if defined(_COLORMAP)
			UNITY_DECLARE_TEX2D_NOSAMPLER(_EmissionMap2);
			UNITY_DECLARE_TEX2D_NOSAMPLER(_EmissionMap3);
		#endif
	#endif
#endif

#if defined(_PACKED_EMISSIONTEX)
	UNITY_DECLARE_TEX2D(_PackedEmissionMap);
#endif

// Occlusion texture definitions
#if defined(_OCCLUSION)
	UNITY_DECLARE_TEX2D_NOSAMPLER(_OcclusionMap);
	#if defined(_LERP) || defined(_COLORMAP)
		UNITY_DECLARE_TEX2D_NOSAMPLER(_OcclusionMap1);
		#if defined(_COLORMAP)
			UNITY_DECLARE_TEX2D_NOSAMPLER(_OcclusionMap2);
			UNITY_DECLARE_TEX2D_NOSAMPLER(_OcclusionMap3);
		#endif
	#endif
#endif

// Specular texture definitions
#if defined(_SPECULARMAP)
	UNITY_DECLARE_TEX2D_NOSAMPLER(_SpecularMap);
	#if defined(_LERP) || defined(_COLORMAP)
		UNITY_DECLARE_TEX2D_NOSAMPLER(_SpecularMap1);
		#if defined(_COLORMAP)
			UNITY_DECLARE_TEX2D_NOSAMPLER(_SpecularMap2);
			UNITY_DECLARE_TEX2D_NOSAMPLER(_SpecularMap3);
		#endif
	#endif
#endif

#if defined(_METALLICMAP)
	UNITY_DECLARE_TEX2D_NOSAMPLER(_MetallicMap);
	#if defined(_LERP) || defined(_COLORMAP)
		UNITY_DECLARE_TEX2D_NOSAMPLER(_MetallicMap1);
		#if defined(_COLORMAP)
			UNITY_DECLARE_TEX2D_NOSAMPLER(_MetallicMap2);
			UNITY_DECLARE_TEX2D_NOSAMPLER(_MetallicMap3);
		#endif
	#endif
#endif

// Normal map definitions
#if defined(_NORMALMAP)
	UNITY_DECLARE_TEX2D(_NormalMap);
	float4 _NormalMap_ST; 
	#if defined(_LERP)
		UNITY_DECLARE_TEX2D(_NormalMap1);
	#endif
#endif

// Packed normal map definitions
#if defined(_PACKED_NORMALMAP)
	UNITY_DECLARE_TEX2D(_PackedNormalMap01);
	UNITY_DECLARE_TEX2D(_PackedNormalMap23);
#endif

#ifdef _DETAIL_NORMALMAP
	UNITY_DECLARE_TEX2D(_DetailNormalMap);
#endif

#if defined(_LERPTEX) || defined(_LERPTEX_POLARUV)
	sampler2D _LerpTex;

	float4 _LerpTex_ST;
#endif

#ifdef _COLORMASK
	UNITY_DECLARE_TEX2D(_ColorMask);
#endif

#ifdef _COLORMAP
	UNITY_DECLARE_TEX2D(_ColorMap);
#endif

#if defined(_COLORMAP) || defined(_LERP)
	#if defined(_COLORMAP)
		#ifdef _METALLICMAP
			UNITY_DECLARE_TEX2D(_MetallicGloss01);
			UNITY_DECLARE_TEX2D(_MetallicGloss23);
		#endif

		#ifdef _HEIGHTMAP
			float _HeightTransitionRange;
			UNITY_DECLARE_TEX2D(_PackedHeightMap);
		#endif
	#endif
#endif

#if defined(OBJECT_POS_OFFSET) || defined(VERTEX_POS_OFFSET)
sampler2D _PositionOffsetMap;
float4 _PositionOffsetMap_ST;
float4 _PositionOffsetMagnitude;
#endif

#ifdef _DETAIL_ALBEDOTEX
sampler2D _DetailAlbedoMap;
#endif

#if defined(_NORMALMAP) || defined(_DETAIL_NORMALMAP)
float _Distort;
#endif

#ifdef _ALPHACLIP
fixed _AlphaClip;
#endif

#if defined(_ALPHATEST) || defined(_MASK_TEXTURE_CLIP)
fixed _Cutoff;
#endif

#ifdef VERTEX_OFFSET
sampler2D _VertexOffsetMap;
float4 _VertexOffsetMap_ST;
float _VertexOffsetMagnitude;
float _VertexOffsetBias;
#endif

#ifdef UV_OFFSET
sampler2D _UVOffsetMap;
float _UVOffsetMagnitude;
float _UVOffsetBias;
#endif

#ifdef _RIM
float4 _RimColor;
float _RimPower;
#endif

#ifdef _INTERSECT
float _BeginTransitionStart;
float _BeginTransitionEnd;
float _EndTransitionStart;
float _EndTransitionEnd;

float4 _IntersectColor;
float4 _IntersectEmissionColor;
#endif

UNITY_DECLARE_DEPTH_TEXTURE(_CameraDepthTexture);

float _GammaCurve;

#ifdef _SLICE
half _EdgeTransitionStart;
half _EdgeTransitionEnd;

half4 _EdgeColor;
float4 _EdgeEmissionColor;

uniform float _SlicerCount;
uniform half4 _Slicers[8];
#endif

#ifdef _TRIPLANAR
float _TriBlendPower;
float4 _MainTex_ST;
#endif

#ifdef _FADE_DEPTH
float _DepthDivisor;

float evrDepthFade(float2 screenUv, float vertexDepth) {
	float depth = DECODE_EYEDEPTH(tex2D(_CameraDepthTexture, screenUv).r);
	return (depth - vertexDepth) / _DepthDivisor;
}
#endif

#if defined(_BLUR)
float2 _Spread;
float _Iterations;
#endif

#if defined(_GRAB_PASS)
sampler2D _BackgroundTexture;
#endif

#if defined(_GRAB_PASS_PER_OBJ)
sampler2D _GrabTexture;
#endif

#ifdef SPREAD_TEX
sampler2D _SpreadTex;
float4 _SpreadTex_ST;
#endif

#ifdef _REFRACT
float _RefractionStrength;
float _DepthBias;
#endif

#if defined(_NORMALMAP)
// Normal maps always require bitangents.
#define _BITANGENT 1
#endif

struct evr_appdata_t 
{
	float4 vertex : POSITION;
	float3 normal : NORMAL;

	#if defined(_VERTEXCOLORS)
	float4 vcolor : COLOR;
	#endif

	#if defined(_BITANGENT)
	float4 tangent : TANGENT;
	#endif
	
	#if defined(_TEXTURE) || defined(_NORMALMAP) || defined(_TEXTURE_NORMALMAP) || defined(_MASK_TEXTURE_MUL) || defined(_MASK_TEXTURE_CLIP) || defined(_LERPTEX) || defined(_LERPTEX_POLARUV)
	float2 texcoord : TEXCOORD0;
	#endif

	#if defined(_SECONDARY_TEXCOORD)
	float2 texcoord1 : TEXCOORD1;
	#endif

	#if INSTANCING_ON
	UNITY_VERTEX_INPUT_INSTANCE_ID
	#endif
};

struct evr_v2f 
{
	float4 vertex : SV_POSITION;
	float3 normal : NORMAL;

	#if defined(_TEXTURE) || defined(_NORMALMAP) || defined(_TEXTURE_NORMALMAP) || defined(_MASK_TEXTURE_MUL) || defined(_MASK_TEXTURE_CLIP) || defined(_LERPTEX) || defined(_LERPTEX_POLARUV)
	half2 texcoord : TEXCOORD0;
	#endif

	#if defined(_SECONDARY_TEXCOORD)
	half2 texcoord1 : TEXCOORD1;
	#endif

	#if defined(_BITANGENT)
	float3 tangent : TANGENT;
	float3 bitangent : TEXCOORD2;
	#endif
	
	float4 position : TEXCOORD3;
	
	#if defined(_VERTEXCOLORS)
	float4 vcolor : COLOR;
	#endif
	
	UNITY_VERTEX_OUTPUT_STEREO

	#if INSTANCING_ON
	UNITY_VERTEX_INPUT_INSTANCE_ID
	#endif
};

#if INSTANCING_ON
#define EVR_SETUP_INSTANCING_VERTEX(v, o) UNITY_SETUP_INSTANCE_ID(v); \
											UNITY_TRANSFER_INSTANCE_ID(v, o);

#define EVR_SETUP_INSTANCING_FRAGMENT(i) UNITY_SETUP_INSTANCE_ID(i);
#else
#define EVR_SETUP_INSTANCING_VERTEX(v, o) // no-op
#define EVR_SETUP_INSTANCING_FRAGMENT(i) // no-op
#endif

float2 PolarUV(float2 rawUv, float radiusPow)
{
	const float ANGLE_LENGTH = (3.141592653589 * 2);
	const float DIV_ANGLE = 1.0 / ANGLE_LENGTH;

	// compute polar coordinates
	float radius = pow(length(rawUv), radiusPow);
	float angle = atan2(rawUv.x, rawUv.y) + ANGLE_LENGTH * 0.5;

	return float2(angle * DIV_ANGLE, radius);
}

float2 TransformPolarMapping(float2 polarUv, float4 tex_ST, out float2 uvddx, out float2 uvddy)
{
	// compute modified angle for computing partial derivatives to avoid discontinuity
	polarUv = polarUv * tex_ST.xy + tex_ST.zw;

	float x0 = polarUv.x;
	float x1 = fmod(abs(polarUv.x) + 0.5, 1);

	float2 coord0 = float2(x0, polarUv.y);
	float2 coord1 = float2(x1, polarUv.y);

	float2 uvddx0 = ddx(coord0);
	float2 uvddy0 = ddy(coord0);
	float2 uvddx1 = ddx(coord1);
	float2 uvddy1 = ddy(coord1);

	uvddx = uvddx0;
	uvddy = uvddy0;

	float l0 = length(uvddx0) + length(uvddy0);
	float l1 = length(uvddx1) + length(uvddy1);

	if (l1 < l0)
	{
		uvddx = uvddx1;
		uvddy = uvddy1;
	}

	return polarUv;
}

float4 SampleTex2Dpolar(sampler2D tex, float2 polarUv, float4 tex_ST)
{
	float2 uvddx, uvddy;
	float2 uv = TransformPolarMapping(polarUv, tex_ST, uvddx, uvddy);
	return tex2Dgrad(tex, uv, uvddx, uvddy);
}

// Polar mapping
float2 PolarMapping(float2 rawUv, float4 tex_ST, float radiusPow, out float2 uvddx, out float2 uvddy)
{
	rawUv = PolarUV(rawUv, radiusPow);
	return TransformPolarMapping(rawUv, tex_ST, uvddx, uvddy);
}

bool IsNaN(float x)
{
	return (asuint(x) & 0x7fffffff) > 0x7f800000;
}

#define CONVERT_SRGB_TO_LINEAR(val) (val <= 0.04045f) ? val / 12.92f : pow((val + 0.055f) / 1.055f, 2.4f)

/**
 * \brief Converts a floating point value with a given profile to linear space.
 * \param inVal The value to be converted
 * \param inProfile The value's profile.  0 = linear, 1 = sRGB, 2 = HDRsRGB
 * \return A linearized value.
 */
float ConvertFloatProfileToLinear(in float inVal, int inProfile)
{
	switch (inProfile)
	{
	case 0:
		return inVal;
	case 1:
		if ((inVal < 1.0f && inVal > -1.0f))
			return CONVERT_SRGB_TO_LINEAR(inVal);
		else
			return inVal;
	case 2:
		if ((inVal < 1.0f && inVal > -1.0f))
			return CONVERT_SRGB_TO_LINEAR(inVal);
		else
			return pow(inVal, 1.0f / 1.5f);
	}

	return inVal;
}

/**
 * \brief 
 * \param inColor 
 * \param profile 0 = linear, 1 = sRGB, 2 = HDRsRGB
 * \return 
 */
float4 ConvertProfileToLinear(in float4 inColor,int inProfile)
{
	#if defined(_VERTEX_HDRSRGBALPHA_COLOR)
	return float4(ConvertFloatProfileToLinear(inColor.r, inProfile), 
				ConvertFloatProfileToLinear(inColor.g, inProfile),
				ConvertFloatProfileToLinear(inColor.b, inProfile),
				pow(inColor.a, 1.0/2.2));
	#endif
	
	return float4(ConvertFloatProfileToLinear(inColor.r, inProfile), 
					ConvertFloatProfileToLinear(inColor.g, inProfile),
					ConvertFloatProfileToLinear(inColor.b, inProfile),
					inColor.a);
}

// Vertex color conversions.
float4 ApplyVertexColor(float4 inColor)
{
	#if defined(_VERTEX_LINEAR_COLOR)
	return ConvertProfileToLinear(inColor, 0);
	#endif

	#if defined(_VERTEX_SRGB_COLOR)
	return ConvertProfileToLinear(inColor, 1);
	#endif

	#if defined(_VERTEX_HDRSRGB_COLOR)
	return ConvertProfileToLinear(inColor, 2);
	#endif

	#if defined(_VERTEX_HDRSRGBALPHA_COLOR)
	return ConvertProfileToLinear(inColor, 2);
	#endif
	
	return inColor;
}

#ifdef _VERTEXCOLORS
#define EVR_APPLY_VERTEX_COLORS(v, o) o.vcolor = v.vcolor;
#define EVR_APPLY_VERTEX_COLORS_FRAG(c, i) c *= ApplyVertexColor(i.vcolor);
#else
#define EVR_APPLY_VERTEX_COLORS(v, o) // no-op
#define EVR_APPLY_VERTEX_COLORS_FRAG(c, i) // no-op
#endif

#define EVR_APPLY_NORMAL_INFO(v, o)	o.normal = UnityObjectToWorldNormal(v.normal);

#ifdef _NORMALMAP
#define EVR_APPLY_NORMALMAP_INFO_FRAG(i, t, tex)	float3x3 tangentTransform = float3x3(i.tangent, i.bitangent, i.normal); \
													float3 bumpNormal = UnpackNormal(tex2D(tex, i.t)); \
													i.normal = normalize(mul(bumpNormal, tangentTransform));

#define EVR_APPLY_NORMALMAP_INFO_FRAG_TRANSFORM(i, t, tex, scale)	float3x3 tangentTransform = float3x3(i.tangent, i.bitangent, i.normal); \
															float3 bumpNormal = UnpackScaleNormal(UNITY_SAMPLE_TEX2D(tex, TRANSFORM_TEX(i.t, tex)), scale); \
															i.normal = normalize(mul(bumpNormal, tangentTransform));

#define EVR_APPLY_NORMALMAP_INFO_FRAG_LERP_TRANSFORM(tc1, tc2, tex1, tex2, i, l)	float3x3 tangentTransform = float3x3(i.tangent, i.bitangent, i.normal); \
																				float3 bumpNormal = lerp(UnpackNormal(tex2D(tex1, TRANFORM_TEX(i.tc1, tex1))), UnpackNormal(tex2D(tex2, TRANSFORM_TEX(i.tc2, tex2))), l); \
																				i.normal = normalize(mul(bumpNormal, tangentTransform));

#else
#define EVR_APPLY_NORMALMAP_INFO_FRAG(i, t, tex) // no-op
#define EVR_APPLY_NORMALMAP_INFO_FRAG_TRANSFORM(i, t, tex, scale) // no-op 
#define EVR_APPLY_NORMALMAP_INFO_FRAG_LERP(tc1, tc2, tex1, tex2, i, l) // no-op
#endif

#ifdef _BITANGENT
#define EVR_APPLY_BITANGENT(v, o)	o.tangent = normalize(mul(unity_ObjectToWorld, float4(v.tangent.xyz, 0)).xyz); \
									o.bitangent = normalize(cross(o.normal, o.tangent) * v.tangent.w);
#else
#define EVR_APPLY_BITANGENT(v, o) // no-op
#endif

#if defined(_TEXTURE) || defined(_NORMALMAP) || defined(_TEXTURE_NORMALMAP) || defined(_MASK_TEXTURE_MUL) || defined(_MASK_TEXTURE_CLIP) || defined(_LERPTEX) || defined(_LERPTEX_POLARUV)
#define EVR_APPLY_TEXCOORD_INFO(n, m, v, o) o.n = v.m;
#define EVR_APPLY_TEXCOORD_INFO_TRANSFORM(n, m, v, tex, o) o.n = TRANSFORM_TEX(v.m, tex);
#else
#define EVR_APPLY_TEXCOORD_INFO(n, m, v, o)
#define EVR_APPLY_TEXCOORD_INFO_TRANSFORM(n, m, v, tex, o) // no-op
#endif

#if defined(WORLD_SPACE)
#define EVR_APPLY_VERTEX_POSITION(v, o) o.vertex = UnityObjectToClipPos(v.vertex); \
										o.position = mul(unity_ObjectToWorld, v.vertex);
#else
#define EVR_APPLY_VERTEX_POSITION(v, o) o.vertex = UnityObjectToClipPos(v.vertex); \
										o.position = v.vertex;
#endif

#define EVR_APPLY_ALPHA_OUTPUT(alpha, output) output.a = pow(alpha, _GammaCurve);

fixed4 evrCalculateBlur(float2 screenUv, float2 uv, float depth, float iterations) {
	half4 c = half4(0, 0, 0, 0);
#ifdef _BLUR
#if !defined(POISSON_DISC)
	half angleStep = TAU / iterations;
#endif

#ifdef SPREAD_TEX
	float2 spread = _Spread * tex2D(_SpreadTex, TRANSFORM_TEX(uv, _SpreadTex)).rg;
#else
	float2 spread = _Spread;
#endif
	spread *= saturate(evrDepthFade(screenUv, depth));
#if !defined(SHADER_API_GLES)
	[loop]
#endif
	for (int n = 0; n < iterations; n++)
	{
#if !defined(POISSON_DISC)
		float angle = angleStep * n;
#if defined(_GRAB_PASS)
		c += tex2D(_BackgroundTexture, screenUv + float2(-cos(angle), sin(angle)) * spread);
#elif defined(_GRAB_PASS_PER_OBJ)
		c += tex2D(_GrabTexture, screenUv + float2(-cos(angle), sin(angle)) * spread);
#endif
#else
#if defined(_GRAB_PASS)
		c += tex2D(_BackgroundTexture, screenUv + (POISSON2D_SAMPLES[n] * 2 - 1) * spread);
#elif defined(_GRAB_PASS_PER_OBJ)
		c += tex2D(_GrabTexture, screenUv + (POISSON2D_SAMPLES[n] * 2 - 1) * spread);
#endif
#endif
	}
#endif
	return c / iterations;
}

fixed2 evrCalculateRefractionCoords(float2 screenUv, float2 uv, float depth, float3 tangent, float3 bitangent, float3 normal, float4 grabPos) {

	
	float2 grabUv = screenUv;
#ifdef _REFRACT
	normal = normalize(normal);

#ifdef _NORMALMAP
	float3x3 tangentTransform = float3x3(tangent, bitangent, normal);
	float3 bumpNormal = UnpackNormal(UNITY_SAMPLE_TEX2D(_NormalMap, TRANSFORM_TEX(uv, _NormalMap)));

	// Compute pertrubed normal, replacing the old one
	normal = normalize(mul(bumpNormal, tangentTransform));
#endif

	normal = mul((float3x3)UNITY_MATRIX_V, normal);

	float2 vignettePos = saturate((1 - abs(screenUv * 2 - 1)) * 32);
	float refractStr =  lerp(0, saturate(evrDepthFade(screenUv, depth)), vignettePos.x * vignettePos.y);

	grabUv -= (normal.xy / grabPos.w) * (_RefractionStrength * refractStr);
	float samplDepth = SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, grabUv);
	if (samplDepth > LinearEyeDepth(depth) + _DepthBias)
		grabUv = screenUv;
#endif
	return grabUv;
}

evr_v2f evr_vert(evr_appdata_t v)
{
	evr_v2f o;

	EVR_SETUP_INSTANCING_VERTEX(v, o);

	UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);

	EVR_APPLY_VERTEX_POSITION(v, o);

	EVR_APPLY_TEXCOORD_INFO(texcoord, texcoord, v, o);

	EVR_APPLY_NORMAL_INFO(v, o);
	
	EVR_APPLY_BITANGENT(v, o);
	
	EVR_APPLY_VERTEX_COLORS(v, o);

	return o;
}

#endif
