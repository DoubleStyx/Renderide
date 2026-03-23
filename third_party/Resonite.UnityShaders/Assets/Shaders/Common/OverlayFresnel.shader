Shader "OverlayFresnel" {
	Properties{

		_Exp("Exponent", Float) = 1.0
		_GammaCurve("Power", Float) = 2.2

		_BehindFarTex("Behind Far Texture", 2D) = "white" {}
		_BehindNearTex("Behind Near Texture", 2D) = "white" {}

		_FrontFarTex("Front Far Texture", 2D) = "white" {}
		_FrontNearTex("Front Near Texture", 2D) = "white" {}

		_BehindFarColor("Behind FarColor", Color) = (0,0,0,1)
		_BehindNearColor("Behind NearColor", Color) = (1,1,1,1)

		_FrontFarColor("Front FarColor", Color) = (0,0,0,1)
		_FrontNearColor("Front NearColor", Color) = (1,1,1,1)

		_NormalMap("Normal Map", 2D) = "bump" {}

		_SrcBlend("SrcBlend", Float) = 1.0
		_DstBlend("DstBlend", Float) = 0.0
		_ZWrite("ZWrite", Float) = 1.0
		_Cull("Cull", Float) = 2.0

		_PolarPow("Polar Mapping Power", Float) = 1.0

		_OffsetFactor("Offset Factor", Float) = 0
		_OffsetUnits("Offset Units", Float) = 0
	}

		

		SubShader{
		Tags{ "RenderType" = "Opaque" }
		LOD 100

		Pass
		{

		Blend[_SrcBlend][_DstBlend], One One
		BlendOp Add, Max
		ZWrite[_ZWrite]
		Cull[_Cull]
		ZTest Greater
		Offset[_OffsetFactor],[_OffsetUnits]

		CGPROGRAM


#pragma vertex evr_vert
#pragma fragment frag
#pragma multi_compile_fog

#pragma multi_compile _ _TEXTURE
#pragma multi_compile _ _NORMALMAP
#pragma multi_compile _ _MUL_ALPHA_INTENSITY
#pragma multi_compile _ _POLARUV
#pragma target 3.0
		#define WORLD_SPACE 1
#include "UnityCG.cginc"
#include "../Common.cginc"

		fixed4 _BehindFarColor;
	fixed4 _BehindNearColor;

	float _Exp;

#ifdef _TEXTURE
	sampler2D _BehindFarTex;
	float4 _BehindFarTex_ST;

	sampler2D _BehindNearTex;
	float4 _BehindNearTex_ST;
#endif

#ifdef _POLARUV
	float _PolarPow;
#endif

	fixed4 frag(evr_v2f i) : SV_Target
	{
		UNITY_SETUP_INSTANCE_ID(i);
		UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);

		i.normal = normalize(i.normal);

		EVR_APPLY_NORMALMAP_INFO_FRAG_TRANSFORM(i, texcoord, _NormalMap, 1);

	float3 viewDir = normalize(_WorldSpaceCameraPos.xyz - i.position.xyz);

	float fresnel = pow(1 - abs(dot(i.normal, viewDir)), _Exp);

	fixed4 farColor = _BehindFarColor;
	fixed4 nearColor = _BehindNearColor;

#ifdef _TEXTURE

#ifdef _POLARUV
	float2 polarUv = PolarUV(i.texcoord * 2 - 1, _PolarPow);
	farColor *= SampleTex2Dpolar(_BehindFarTex, polarUv, _BehindFarTex_ST);
	nearColor *= SampleTex2Dpolar(_BehindNearTex, polarUv, _BehindNearTex_ST);
#else
	farColor *= tex2D(_BehindFarTex, TRANSFORM_TEX(i.texcoord, _BehindFarTex));
	nearColor *= tex2D(_BehindNearTex, TRANSFORM_TEX(i.texcoord, _BehindNearTex));
#endif

#endif

	// compute final color by blending between the two based on the fresnel

	fixed4 col = lerp(nearColor, farColor, fresnel);

#ifdef _MUL_ALPHA_INTENSITY
	float mulfactor = (col.r + col.g + col.b) * 0.3333333;
	col.a *= mulfactor * mulfactor;
#endif

	UNITY_APPLY_FOG(i.fogCoord, col);
	//UNITY_OPAQUE_ALPHA(col.a);
	return col;
	}
		ENDCG
	} // pass


	Pass
	{

		Blend[_SrcBlend][_DstBlend], One One
		BlendOp Add, Max
		ZWrite[_ZWrite]
		Cull[_Cull]
		ZTest LEqual
		Offset[_OffsetFactor],[_OffsetUnits]

		CGPROGRAM


#pragma vertex evr_vert
#pragma fragment frag
#pragma multi_compile_fog

#pragma multi_compile _ _TEXTURE
#pragma multi_compile _ _NORMALMAP
#pragma multi_compile _ _MUL_ALPHA_INTENSITY
#pragma multi_compile _ _POLARUV
		
#define WORLD_SPACE 1
#include "UnityCG.cginc"
#include "../Common.cginc"

		fixed4 _FrontFarColor;
	fixed4 _FrontNearColor;

	float _Exp;

#ifdef _TEXTURE
	sampler2D _FrontFarTex;
	float4 _FrontFarTex_ST;

	sampler2D _FrontNearTex;
	float4 _FrontNearTex_ST;
#endif

#ifdef _POLARUV
	float _PolarPow;
#endif

	fixed4 frag(evr_v2f i) : SV_Target
	{
		UNITY_SETUP_INSTANCE_ID(i);
		UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);

		i.normal = normalize(i.normal);
		EVR_APPLY_NORMALMAP_INFO_FRAG_TRANSFORM(i, texcoord, _NormalMap, 1);

	float3 viewDir = normalize(_WorldSpaceCameraPos.xyz - i.position.xyz);

	float fresnel = pow(pow(1 - abs(dot(i.normal, viewDir)), _Exp), _GammaCurve);

	fixed4 farColor = _FrontFarColor;
	fixed4 nearColor = _FrontNearColor;

#ifdef _TEXTURE

#ifdef _POLARUV
	float2 polarUv = PolarUV(i.texcoord * 2 - 1, _PolarPow);
	farColor *= SampleTex2Dpolar(_FrontFarTex, polarUv, _FrontFarTex_ST);
	nearColor *= SampleTex2Dpolar(_FrontNearTex, polarUv, _FrontNearTex_ST);
#else
	farColor *= tex2D(_FrontFarTex, TRANSFORM_TEX(i.texcoord, _FrontFarTex));
	nearColor *= tex2D(_FrontNearTex, TRANSFORM_TEX(i.texcoord, _FrontNearTex));
#endif

#endif

	// compute final color by blending between the two based on the fresnel

	fixed4 col = lerp(nearColor, farColor, fresnel);

#ifdef _MUL_ALPHA_INTENSITY
	float mulfactor = (col.r + col.g + col.b) * 0.3333333;
	col.a *= mulfactor * mulfactor;
#endif

	UNITY_APPLY_FOG(i.fogCoord, col);
	//UNITY_OPAQUE_ALPHA(col.a);
	return col;
	}
		ENDCG
	} // pass


	}

}
