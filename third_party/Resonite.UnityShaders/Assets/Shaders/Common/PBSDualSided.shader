// Upgrade NOTE: upgraded instancing buffer 'Props' to new syntax.

Shader "PBSDualSided"
{
	Properties{
		_Color("Color", Color) = (1,1,1,1)
		_MainTex("Albedo (RGB)", 2D) = "white" {}

		_NormalMap("NormalMap", 2D) = "bump" {}
		_NormalScale("Normal Scale", Float) = 1

		_EmissionColor("EmissionColor", Color) = (0,0,0,0)
		_EmissionMap("EmissionMap", 2D) = "black" {}

		_OcclusionMap("Occlusion", 2D) = "white" {}

		_Glossiness ("Smoothness", Range(0,1)) = 0.5
		_Metallic ("Metallic", Range(0,1)) = 0.0

		_MetallicMap("MetallicMap", 2D) = "black" {}

		_AlphaClip ("AlphaClip", Range(0,1)) = 0.5

		_Cull ("Culling", Float) = 0

		_OffsetFactor("Offset Factor", Float) = 0.0
		_OffsetUnits("Offset Units", Float) = 0.0
	}

		

	SubShader {
		Tags{ "RenderType" = "Opaque" }
		//Tags { "RenderType"="Transparent" "Queue"="Transparent" }
		Cull [_Cull]
		Offset[_OffsetFactor],[_OffsetUnits]
		LOD 200
		
		CGPROGRAM
		// Physically based Standard lighting model, and enable shadows on all light types
		#pragma surface surf Standard fullforwardshadows addshadow
		#define _GLOSSYENV 1

		// Use shader model 3.0 target, to get nicer looking lighting
		#pragma target 3.0
		

		#pragma multi_compile _ _ALPHACLIP

		#pragma multi_compile _ _ALBEDOTEX
		#pragma multi_compile _ _EMISSIONTEX
		#pragma multi_compile _ _NORMALMAP
		#pragma multi_compile _ _METALLICMAP
		#pragma multi_compile _ _OCCLUSION

		#pragma multi_compile _ VCOLOR_ALBEDO VCOLOR_EMIT VCOLOR_METALLIC

#if defined(VCOLOR_ALBEDO) || defined(VCOLOR_EMIT) || defined(VCOLOR_METALLIC)
#define VCOLOR
#endif

#ifdef _ALBEDOTEX
		sampler2D _MainTex;
#endif

#ifdef _NORMALMAP
		sampler2D _NormalMap;
		float _NormalScale;
#endif

#ifdef _EMISSIONTEX
		sampler2D _EmissionMap;
#endif

#ifdef _METALLICMAP
		sampler2D _MetallicMap;
#else
		half _Glossiness;
		half _Metallic;
#endif

#ifdef _OCCLUSION
		sampler2D _OcclusionMap;
#endif

#ifdef _ALPHACLIP
		fixed _AlphaClip;
#endif

		half4 _Color;
		float4 _EmissionColor;

		struct Input
		{
			float2 uv_MainTex;
			float facing : FACE;
#ifdef VCOLOR
			half4 color : COLOR;
#endif
		};

		void surf (Input IN, inout SurfaceOutputStandard o)
		{
#ifdef _ALBEDOTEX
			half4 c = tex2D (_MainTex, IN.uv_MainTex) * _Color;
#else
			half4 c = _Color;
#endif

#ifdef VCOLOR_ALBEDO
			c *= IN.color;
#endif

#ifdef _ALPHACLIP
			clip(c.a - _AlphaClip);
#endif

			o.Albedo = c.rgb;

#ifdef _NORMALMAP
			o.Normal = UnpackScaleNormal(tex2D(_NormalMap, IN.uv_MainTex), _NormalScale);
#endif
			if (IN.facing < 0.5)
				o.Normal.z *= -1;

#ifdef _OCCLUSION
			o.Occlusion = tex2D(_OcclusionMap, IN.uv_MainTex).r;
#endif

#ifdef _METALLICMAP
			half4 m = tex2D(_MetallicMap, IN.uv_MainTex);
			
			o.Metallic = m.r;
			o.Smoothness = m.a;
#else
			o.Metallic = _Metallic;
			o.Smoothness = _Glossiness;
#endif

#ifdef VCOLOR_METALLIC
			o.Metallic *= (IN.color.r + IN.color.g + IN.color.b) * 0.333333;
			o.Smoothness *= IN.color.a;
#endif

			float4 emission = _EmissionColor;
#ifdef _EMISSIONTEX
			emission *= tex2D(_EmissionMap, IN.uv_MainTex);
#endif

#ifdef VCOLOR_EMIT
			emission *= IN.color;
#endif

			o.Emission = emission.rgb;

			o.Alpha = c.a;
		}
		ENDCG
	}
	FallBack "Diffuse"
}
