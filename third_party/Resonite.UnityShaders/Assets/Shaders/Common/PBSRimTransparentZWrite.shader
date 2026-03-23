// Upgrade NOTE: upgraded instancing buffer 'Props' to new syntax.

Shader "PBSRimTransparentZWrite"
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

		_RimColor ("Rim Color", Color) = (1,0,0,1)
		_RimPower("Rim Power", Range(0,10)) = 3.0
	}

		

	SubShader {
		//Tags{ "RenderType" = "Opaque" }
		Tags { "RenderType"="Transparent" "Queue"="Transparent" }
		Offset[_OffsetFactor],[_OffsetUnits]
		LOD 200

		Pass{
			ColorMask 0
		}
		
		CGPROGRAM
		// Physically based Standard lighting model, and enable shadows on all light types
		#pragma surface surf Standard alpha fullforwardshadows
		#define _GLOSSYENV 1

		// Use shader model 3.0 target, to get nicer looking lighting
		#pragma target 3.0
		

		#pragma multi_compile _ _ZWRITE
		#pragma multi_compile _ _ALBEDOTEX
		#pragma multi_compile _ _EMISSIONTEX
		#pragma multi_compile _ _NORMALMAP
		#pragma multi_compile _ _METALLICMAP
		#pragma multi_compile _ _OCCLUSION

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

		fixed4 _Color;
		float4 _EmissionColor;
		float4 _RimColor;
		float _RimPower;

		struct Input
		{
			float2 uv_MainTex;
			float3 viewDir;
		};

		void surf (Input IN, inout SurfaceOutputStandard o)
		{
#ifdef _ALBEDOTEX
			fixed4 c = tex2D (_MainTex, IN.uv_MainTex) * _Color;
#else
			fixed4 c = _Color;
#endif
			o.Albedo = c.rgb;

#ifdef _NORMALMAP
			o.Normal = UnpackScaleNormal(tex2D(_NormalMap, IN.uv_MainTex), _NormalScale);
#endif

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

			float4 emission = _EmissionColor;
#ifdef _EMISSIONTEX
			emission *= tex2D(_EmissionMap, IN.uv_MainTex);
#endif

			half rim = 1.0 - saturate(dot(normalize(IN.viewDir), o.Normal));
			o.Emission = _RimColor.rgb * pow(rim, _RimPower) + emission.rgb;

			o.Alpha = c.a;
		}
		ENDCG
	}
	FallBack "Diffuse"
}
