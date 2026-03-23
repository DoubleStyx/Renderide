// Upgrade NOTE: upgraded instancing buffer 'Props' to new syntax.

Shader "ColorMask"
{
	Properties{
		_Color("Color0", Color) = (1,0,0,1)
		_Color1("Color1", Color) = (0,1,0,1)
		_Color2("Color2", Color) = (0,0,1,1)
		_Color3("Color3", Color) = (1,1,1,1)

		_MainTex("Albedo (RGB)", 2D) = "white" {}
		_ColorMask("ColorMask", 2D) = "black" {}

		_NormalMap("NormalMap", 2D) = "bump" {}
		_NormalScale("Normal Scale", Float) = 1

		_EmissionColor("EmissionColor0", Color) = (0,0,0,0)
		_EmissionColor1("EmissionColor1", Color) = (0,0,0,0)
		_EmissionColor2("EmissionColor2", Color) = (0,0,0,0)
		_EmissionColor3("EmissionColor3", Color) = (0,0,0,0)

		_EmissionMap("EmissionMap", 2D) = "black" {}

		_OcclusionMap("Occlusion", 2D) = "white" {}

		_Glossiness("Smoothness", Range(0,1)) = 0.5
		_Metallic("Metallic", Range(0,1)) = 0.0

		_MetallicMap("MetallicMap", 2D) = "black" {}
	}



		SubShader{
			Tags{ "RenderType" = "Opaque" }
			//Tags { "RenderType"="Transparent" "Queue"="Transparent" }
			Offset[_OffsetFactor],[_OffsetUnits]
			LOD 200

			CGPROGRAM
		// Physically based Standard lighting model, and enable shadows on all light types
		#pragma surface surf Standard fullforwardshadows
		#define _GLOSSYENV 1

		// Use shader model 3.0 target, to get nicer looking lighting
		#pragma target 3.0
		

		#pragma multi_compile _ _ALBEDOTEX
		#pragma multi_compile _ _EMISSIONTEX
		#pragma multi_compile _ _NORMALMAP
		#pragma multi_compile _ _METALLICMAP
		#pragma multi_compile _ _OCCLUSION
#pragma mutli_compile _ _MULTI_VALUES

		sampler2D _ColorMask;

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
		fixed4 _Color1;
		fixed4 _Color2;
		fixed4 _Color3;

		float4 _EmissionColor;
		float4 _EmissionColor1;
		float4 _EmissionColor2;
		float4 _EmissionColor3;

		struct Input
		{
			float2 uv_MainTex;
			float2 uv_ColorMask;
			float3 viewDir;
		};

		void surf(Input IN, inout SurfaceOutputStandard o)
		{
			fixed4 mask = tex2D(_ColorMask, IN.uv_ColorMask);

			half4 c = _Color * mask.r + _Color1 * mask.g + _Color2 * mask.b + _Color3 * mask.a;

			float weight = saturate(1 / (mask.r + mask.g + mask.b + mask.a));

			c *= weight;

#ifdef _ALBEDOTEX
			c *= tex2D(_MainTex, IN.uv_MainTex);
#endif
			o.Albedo = c.rgb;

#ifdef _NORMALMAP
			o.Normal = UnpackScaleNormal(tex2D(_NormalMap, IN.uv_MainTex), _NormalScale);
#else
			o.Normal = half3(0, 0, 1);
#endif

#ifdef _OCCLUSION
			o.Occlusion = tex2D(_OcclusionMap, IN.uv_MainTex).r;
#endif

#ifdef _METALLICMAP
			half4 m = tex2D(_MetallicMap, IN.uv_MainTex);

#ifdef _MULTI_VALUES
			m.r *= _Metallic;
			m.a *= _Glossiness;
#endif
			o.Metallic = m.r;
			o.Smoothness = m.a;
#else
			o.Metallic = _Metallic;
			o.Smoothness = _Glossiness;
#endif

			float4 emission = _EmissionColor * mask.r + _EmissionColor1 * mask.g + _EmissionColor2 * mask.b + _EmissionColor3 * mask.a;
			emission *= weight;

#ifdef _EMISSIONTEX
			emission *= tex2D(_EmissionMap, IN.uv_MainTex);
#endif

			half rim = 1.0 - saturate(dot(normalize(IN.viewDir), o.Normal));
			o.Emission = emission.rgb;

			o.Alpha = c.a;
		}
		ENDCG
	}
		FallBack "Diffuse"
}
