// Upgrade NOTE: upgraded instancing buffer 'Props' to new syntax.

Shader "PBSRimTransparentSpecular"
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

		_RimColor ("Rim Color", Color) = (1,0,0,1)
		_RimPower("Rim Power", Range(0,10)) = 3.0
	}

		

	SubShader {
		//Tags{ "RenderType" = "Opaque" }
		Tags { "RenderType"="Transparent" "Queue"="Transparent" }
		Offset[_OffsetFactor],[_OffsetUnits]
		LOD 200
		
		CGPROGRAM
		// Physically based Standard lighting model, and enable shadows on all light types
		#pragma surface surf StandardSpecular alpha fullforwardshadows
		#define _GLOSSYENV 1

		// Use shader model 3.0 target, to get nicer looking lighting
		#pragma target 3.0
		

		#pragma multi_compile _ _EMISSIONTEX
		#pragma multi_compile _ _NORMALMAP
		#pragma multi_compile _ _SPECULARMAP
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

#ifdef _SPECULARMAP
		sampler2D _SpecularMap;
#else
		half4 _SpecularColor;
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

		void surf (Input IN, inout SurfaceOutputStandardSpecular o)
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

#ifdef _SPECULARMAP
			half4 s = tex2D(_SpecularMap, IN.uv_MainTex);
#else
			half4 s = _SpecularColor;
#endif
			o.Specular = s.rgb;
			o.Smoothness = s.a;

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
