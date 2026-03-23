// Upgrade NOTE: upgraded instancing buffer 'Props' to new syntax.

Shader "PBSDisplaceSpecular"
{
	Properties{
		_Color("Color", Color) = (1,1,1,1)
		_MainTex("Albedo (RGB)", 2D) = "white" {}

		_NormalMap("NormalMap", 2D) = "bump" {}
		_NormalScale("Normal Scale", Float) = 1

		_EmissionColor("EmissionColor", Color) = (0,0,0,0)
		_EmissionMap("EmissionMap", 2D) = "black" {}

		_OcclusionMap("Occlusion", 2D) = "white" {}

		_SpecularColor("SpecularColor", Color) = (1,1,1,0.5)
		_SpecularMap("SpecularMap", 2D) = "white" {}

		_AlphaClip("AlphaClip", Range(0,1)) = 0.5
 
		_OffsetFactor("Offset Factor", Float) = 0.0
		_OffsetUnits("Offset Units", Float) = 0.0

		_VertexOffsetMap("Vertex Offset Map", 2D) = "black" {}
		_VertexOffsetMagnitude("Vertex Offset Magnitude", Float) = 0.1
		_VertexOffsetBias("Vertex Offset Bias", Float) = 0

		_UVOffsetMap("UV Offset Map", 2D) = "black" {}
		_UVOffsetMagnitude("UV Offset Magnitude", Float) = 0.1
		_UVOffsetBias("UV Offset Bias", Float) = 0

		_PositionOffsetMap("Position Offset Map", 2D) = "black"
		_PositionOffsetMagnitude("Position Offset Magnitude", Vector) = (1, 1, 0, 0)
	}

		

		SubShader{
		Tags{ "RenderType" = "Opaque" }
		//Tags{ "RenderType" = "Transparent" "Queue" = "Transparent" }
		Cull[_Cull]
		Offset[_OffsetFactor],[_OffsetUnits]
		LOD 200

		CGPROGRAM
		// Physically based Standard lighting model, and enable shadows on all light types
#pragma surface surf_specular StandardSpecular fullforwardshadows vertex:vertDisplace addshadow
#define _GLOSSYENV 1

		// Use shader model 3.0 target, to get nicer looking lighting
#pragma target 3.0

#pragma multi_compile _ _ALBEDOTEX
#pragma multi_compile _ _EMISSIONTEX
#pragma multi_compile _ _NORMALMAP
#pragma multi_compile _ _SPECULARMAP
#pragma multi_compile _ _OCCLUSION

#pragma multi_compile _ _ALPHACLIP

#pragma multi_compile _ VERTEX_OFFSET
#pragma multi_compile _ UV_OFFSET

#pragma multi_compile _ OBJECT_POS_OFFSET VERTEX_POS_OFFSET
				
			#include "../SurfaceShaders.cginc"
/*
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

#ifdef _ALBEDOTEX
		sampler2D _MainTex;
#endif

#ifdef _ALPHACLIP
		fixed _AlphaClip;
#endif

#ifdef _NORMALMAP
	sampler2D _NormalMap;
	float _NormalScale;
#endif

#ifdef _EMISSIONTEX
	sampler2D _EmissionMap;
#endif

#ifdef _SPECULARMAP
	sampler2D _SpecularMap;
#else
	half4 _SpecularColor;
#endif

#ifdef _OCCLUSION
	sampler2D _OcclusionMap;
#endif

#if defined(OBJECT_POS_OFFSET) || defined(VERTEX_POS_OFFSET)
		sampler2D _PositionOffsetMap;
		float4 _PositionOffsetMap_ST;
		float4 _PositionOffsetMagnitude;
#endif

	fixed4 _Color;
	float4 _EmissionColor;

	struct Input
	{
		float2 uv_MainTex;
		float facing : FACE;
#ifdef UV_OFFSET
		float2 uv_UVOffsetMap;
#endif
	};

	void vert(inout appdata_full v)
	{
#ifdef VERTEX_OFFSET
		float2 uv = v.texcoord;

#ifdef OBJECT_POS_OFFSET
		float2 position = unity_ObjectToWorld._m03_m23;
#elif VERTEX_POS_OFFSET
		float2 position = mul(unity_ObjectToWorld, float4(v.vertex.xyz, 1)).xz;
#endif

#if defined(OBJECT_POS_OFFSET) || defined(VERTEX_POS_OFFSET)
		float2 uvOffset = tex2Dlod(_PositionOffsetMap, float4(TRANSFORM_TEX(position, _PositionOffsetMap), 0, 0)).xy;
		uvOffset *= _PositionOffsetMagnitude.xy;
		uv += uvOffset;
#endif

		float vertOffset = tex2Dlod(_VertexOffsetMap, float4(TRANSFORM_TEX(uv, _VertexOffsetMap), 0, 0)).x;
		vertOffset = vertOffset * _VertexOffsetMagnitude + _VertexOffsetBias;

		v.vertex.xyz += v.normal.xyz * vertOffset;
#endif
	}

	void surf(Input IN, inout SurfaceOutputStandardSpecular o)
	{
		float2 mainUv = IN.uv_MainTex;

#ifdef UV_OFFSET
		fixed2 uvOffset = tex2D(_UVOffsetMap, IN.uv_UVOffsetMap).xy;
		uvOffset = uvOffset * _UVOffsetMagnitude + _UVOffsetBias;

		mainUv += uvOffset;
#endif

#ifdef _ALBEDOTEX
		fixed4 c = tex2D(_MainTex, mainUv) * _Color;
#else
		fixed4 c = _Color;
#endif

#ifdef _ALPHACLIP
		clip(c.a - _AlphaClip);
#endif

		o.Albedo = c.rgb;

#ifdef _NORMALMAP
		o.Normal = UnpackScaleNormal(tex2D(_NormalMap, mainUv), _NormalScale);
#endif
		if (IN.facing < 0.5)
			o.Normal.z *= -1;

#ifdef _OCCLUSION
		o.Occlusion = tex2D(_OcclusionMap, mainUv).r;
#endif

#ifdef _SPECULARMAP
		half4 s = tex2D(_SpecularMap, mainUv);
#else
		half4 s = _SpecularColor;
#endif
		o.Specular = s.rgb;
		o.Smoothness = s.a;

		o.Emission = _EmissionColor;
#ifdef _EMISSIONTEX
		o.Emission *= tex2D(_EmissionMap, mainUv).rgb;
#endif

		o.Alpha = c.a;
	}*/
	ENDCG
	}
		FallBack "Diffuse"
}
