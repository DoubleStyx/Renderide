#include "UnityPBSLighting.cginc"
#include "AutoLight.cginc"
#include "UnityCG.cginc"

#ifdef OCCLUSION_METALLIC
#define OCCLUSION_MAP
#define METALLICGLOSS_MAP
#endif

#ifdef RAMPMASK_OUTLINEMASK_THICKNESS
#define RAMP_MASK
#define OUTLINE_MASK
#define THICKNESS_MAP
#endif

struct VertexInput
{
    float4 vertex : POSITION;
    float2 uv : TEXCOORD0;
    float2 uv1 : TEXCOORD1;
    float3 normal : NORMAL;
    float4 tangent : TANGENT;
    float4 color : COLOR;

	UNITY_VERTEX_INPUT_INSTANCE_ID
};

struct VertexOutput
{	
    #if defined(Geometry)
        float4 pos : CLIP_POS;
        float4 vertex : SV_POSITION; // We need both of these in order to shadow Outlines correctly
    #else
        float4 pos : SV_POSITION;
    #endif

    float2 uv : TEXCOORD0;
    float2 uv1 : TEXCOORD1;
    float3 ntb[3] : TEXCOORD2; //texcoord 3, 4 || Holds World Normal, Tangent, and Bitangent
    float4 worldPos : TEXCOORD5;
    float4 color : TEXCOORD6;
    float3 normal : TEXCOORD8;
    float4 screenPos : TEXCOORD9;
    float3 objPos : TEXCOORD11;

    //float distanceToOrigin : TEXCOORD10;
    SHADOW_COORDS(7)
    UNITY_FOG_COORDS(10)

	UNITY_VERTEX_OUTPUT_STEREO
};

#if defined(Geometry)
    struct v2g
    {
        float4 pos : CLIP_POS;
        float4 vertex : SV_POSITION;
        float2 uv : TEXCOORD0;
        float2 uv1 : TEXCOORD1;
        float3 ntb[3] : TEXCOORD2; //texcoord 3, 4 || Holds World Normal, Tangent, and Bitangent
        float4 worldPos : TEXCOORD5;
        float4 color : TEXCOORD6;
        float3 normal : TEXCOORD8;
        float4 screenPos : TEXCOORD9;
        float3 objPos : TEXCOORD11;

        //float distanceToOrigin : TEXCOORD10;
        SHADOW_COORDS(7)
        UNITY_FOG_COORDS(10)

		UNITY_VERTEX_OUTPUT_STEREO
    };

    struct g2f
    {
        float4 pos : SV_POSITION;
        float2 uv : TEXCOORD0;
        float2 uv1 : TEXCOORD1;
        float3 ntb[3] : TEXCOORD2; //texcoord 3, 4 || Holds World Normal, Tangent, and Bitangent
        float4 worldPos : TEXCOORD5;
        float4 color : TEXCOORD6;
        float4 screenPos : TEXCOORD8;
        float3 objPos : TEXCOORD10;

        //float distanceToOrigin : TEXCOORD9;
        SHADOW_COORDS(7)
        UNITY_FOG_COORDS(9)

		UNITY_VERTEX_OUTPUT_STEREO
    };
#endif

struct XSLighting
{
    half4 albedo;

#ifdef NORMAL_MAP
    half4 normalMap;
#endif
    //half4 detailNormal;
    //half4 detailMask;

#ifdef METALLICGLOSS_MAP
    half4 metallicGlossMap;
#endif

    //half4 reflectivityMask;
    //half4 specularMap;

#ifdef THICKNESS_MAP
    half4 thickness;
#endif

#ifdef OCCLUSION_MAP
    half4 occlusion;
#endif

#ifdef EMISSION_MAP
    half4 emissionMap;
#endif

#ifdef RAMP_MASK
    half4 rampMask;
#endif

    #if defined(AlphaToMask) && defined(Masked) || defined(Dithered)
        half4 cutoutMask;
    #endif

    half3 diffuseColor;
    half attenuation;
    half3 normal;
    half3 tangent;
    half3 bitangent;
    // World-space position (xyz only; Slang requires explicit half3 vs half4 separation).
    half3 worldPos;
    half3 color;
    half alpha;
    float isOutline;
    float2 screenUV;
    float3 objPos;
};

struct TextureUV
{	
    half2 uv0;
    half2 uv1;

    half2 albedoUV;

    //half2 specularMapUV;

#ifdef METALLICGLOSS_MAP
    half2 metallicGlossMapUV;
#endif

    //half2 detailMaskUV;
#ifdef NORMAL_MAP
    half2 normalMapUV;
#endif
    //half2 detailNormalUV;

#ifdef THICKNESS_MAP
    half2 thicknessMapUV;
#endif

#ifdef OCCLUSION_MAP
    half2 occlusionUV;
#endif

    //half2 reflectivityMaskUV;

#ifdef EMISSION_MAP
    half2 emissionMapUV;
#endif

#ifdef OUTLINE_MASK
    half2 outlineMaskUV;
#endif
};

struct DotProducts
{
    half ndl;
    half vdn;
    half vdh;
    half tdh;
    half bdh;
    half ndh;
    half rdv;
    half ldh;
    half svdn;
};

UNITY_DECLARE_TEX2D(_MainTex); half4 _MainTex_ST;

#ifdef NORMAL_MAP
UNITY_DECLARE_TEX2D_NOSAMPLER(_BumpMap); half4 _BumpMap_ST;
#endif

//UNITY_DECLARE_TEX2D_NOSAMPLER(_DetailNormalMap); half4 _DetailNormalMap_ST;
//UNITY_DECLARE_TEX2D_NOSAMPLER(_DetailMask); half4 _DetailMask_ST;
//UNITY_DECLARE_TEX2D_NOSAMPLER(_SpecularMap); half4 _SpecularMap_ST;

#ifdef METALLICGLOSS_MAP
UNITY_DECLARE_TEX2D_NOSAMPLER(_MetallicGlossMap); half4 _MetallicGlossMap_ST;
#endif

//UNITY_DECLARE_TEX2D_NOSAMPLER(_ReflectivityMask); half4 _ReflectivityMask_ST;

#ifdef THICKNESS_MAP
UNITY_DECLARE_TEX2D_NOSAMPLER(_ThicknessMap); half4 _ThicknessMap_ST;
#endif

#ifdef EMISSION_MAP
UNITY_DECLARE_TEX2D_NOSAMPLER(_EmissionMap); half4 _EmissionMap_ST;
#endif

#ifdef RAMP_MASK
UNITY_DECLARE_TEX2D_NOSAMPLER(_RampSelectionMask);
#endif

#if defined(AlphaToMask) && defined(Masked) || defined(Dithered)
UNITY_DECLARE_TEX2D_NOSAMPLER(_CutoutMask); half4 _CutoutMask_ST;
#endif

#ifdef OCCLUSION_MAP
sampler2D _OcclusionMap; half4 _OcclusionMap_ST;
#endif

#ifdef OUTLINE_MASK
sampler2D _OutlineMask;
#endif

#ifdef MATCAP
sampler2D _Matcap;
uniform half4 _MatcapTint;
#endif

sampler2D _Ramp;
//samplerCUBE _BakedCubemap;

uniform half4 _Color, _ShadowRim, 
    _OutlineColor, _SSColor, _OcclusionColor,
    _EmissionColor, _RimColor;

uniform half _Cutoff;
uniform half _FadeDitherDistance;
//half _EmissionToDiffuse/*, _ScaleWithLightSensitivity*/;
uniform half _Saturation;
uniform half _Metallic, _Glossiness, _Reflectivity/*, _ClearcoatStrength, _ClearcoatSmoothness*/;
uniform half _BumpScale/*, _DetailNormalMapScale*/;
uniform half _SpecularIntensity, _SpecularArea/*, _AnisotropicAX, _AnisotropicAY*/, _SpecularAlbedoTint;

uniform half _RimRange, _RimThreshold, _RimIntensity, _RimSharpness, _RimAlbedoTint, _RimCubemapTint, _RimAttenEffect;
uniform half _ShadowRimRange, _ShadowRimThreshold, _ShadowRimSharpness, _ShadowSharpness, _ShadowRimAlbedoTint;

uniform half _SSDistortion, _SSPower, _SSScale;
uniform half _OutlineWidth;

uniform int _FadeDither;
//int _SpecMode, _SpecularStyle, _ReflectionMode, _ReflectionBlendMode, _ClearCoat;
//int _ReflectionMode;
//int _ScaleWithLight;
uniform int _OutlineAlbedoTint, _OutlineEmissive;
uniform int _UVSetAlbedo, _UVSetNormal, _UVSetDetNormal, 
    _UVSetDetMask, _UVSetMetallic, _UVSetSpecular,
    _UVSetThickness, _UVSetOcclusion, _UVSetReflectivity,
    _UVSetEmission;

// half _HalftoneDotSize, _HalftoneDotAmount, _HalftoneLineAmount;

//Defines for helper functions
#define grayscaleVec float3(0.2125, 0.7154, 0.0721)