Shader "PBSMultiUVSpecular"
{
    Properties
    {
		_Color("Color", Color) = (1,1,1,1)
		_MainTex("Albedo (RGB)", 2D) = "white" {}
		_SecondaryAlbedo ("Secondary Albedo", 2D) = "white" {}

		_NormalMap("NormalMap", 2D) = "bump" {}
		_NormalScale("Normal Scale", Float) = 1

		_EmissionColor("EmissionColor", Color) = (0,0,0,0)
		_EmissionMap("EmissionMap", 2D) = "black" {}

		_SecondaryEmissionColor("Secondary EmissionColor", Color) = (0,0,0,0)
		_SecondaryEmissionMap("Secondary EmissionMap", 2D) = "black" {}

		_OcclusionMap("Occlusion", 2D) = "white" {}

		_SpecularColor("SpecularColor", Color) = (1,1,1,0.5)
		_SpecularMap("SpecularMap", 2D) = "white" {}

		_AlphaClip("AlphaClip", Range(0,1)) = 0.5

		_OffsetFactor("Offset Factor", Float) = 0.0
		_OffsetUnits("Offset Units", Float) = 0.0

		_AlbedoUV ("Albedo UV", Float) = 0
		_SecondaryAlbedoUV ("Secondary Albedo UV", Float) = 0
		_EmissionUV ("Secondary Emission UV", Float) = 0
		_SecondaryEmissionUV ("Emission UV", Float) = 0
		_NormalUV ("Normal Map UV", Float) = 0
		_OcclusionUV ("Occlusion UV", Float) = 0
		_SpecularUV ("SpecularMap UV", Float) = 0
    }




    SubShader
    {
		Tags{ "RenderType" = "Opaque" }
		//Tags { "RenderType"="Transparent" "Queue"="Transparent" }
		Offset[_OffsetFactor],[_OffsetUnits]
		LOD 200

        CGPROGRAM
        // Physically based Standard lighting model, and enable shadows on all light types
		#pragma surface surf StandardSpecular fullforwardshadows vertex:vert
		#define _GLOSSYENV 1

		// Use shader model 3.0 target, to get nicer looking lighting
		#pragma target 4.0
		

		#pragma multi_compile _ _DUAL_ALBEDO
		#pragma multi_compile _ _EMISSIONTEX _DUAL_EMISSIONTEX
		#pragma multi_compile _ _NORMALMAP
		#pragma multi_compile _ _SPECULARMAP
		#pragma multi_compile _ _OCCLUSION

		#pragma multi_compile _ _ALPHACLIP

#ifdef _DUAL_EMISSIONTEX
#define _EMISSIONTEX
#endif

		sampler2D _MainTex;
		float4 _MainTex_ST;
		half _AlbedoUV;

#ifdef _DUAL_ALBEDO
		sampler2D _SecondaryAlbedo;
		half _SecondaryAlbedoUV;
		float4 _SecondaryAlbedo_ST;
#endif

#ifdef _ALPHACLIP
		fixed _AlphaClip;
#endif

#ifdef _NORMALMAP
		sampler2D _NormalMap;
		half _NormalScale;
		half _NormalUV;
		float4 _NormalMap_ST;
#endif

#ifdef _EMISSIONTEX
		sampler2D _EmissionMap;
		half _EmissionUV;
		float4 _EmissionMap_ST;
#endif

#ifdef _DUAL_EMISSIONTEX
		float4 _SecondaryEmissionColor;
		sampler2D _SecondaryEmissionMap;
		half _SecondaryEmissionUV;
		float4 _SecondaryEmissionMap_ST;
#endif


#ifdef _SPECULARMAP
		sampler2D _SpecularMap;
		half _SpecularUV;
		float4 _SpecularMap_ST;
#else
		half4 _SpecularColor;
#endif

#ifdef _OCCLUSION
		sampler2D _OcclusionMap;
		float4 _OcclusionMap_ST;
		half _OcclusionUV;
#endif

		fixed4 _Color;
		float4 _EmissionColor;

		struct Input
		{
			float4 albedo_coord;
#ifdef _EMISSIONTEX
			float4 emission_coord;
#endif
#if defined(_NORMALMAP) || defined(_OCCLUSION)
			float4 normal_occlusion_coord;
#endif
#ifdef _SPECULARMAP
			float2 specular_coord;
#endif
		};

        // Add instancing support for this shader. You need to check 'Enable Instancing' on materials that use the shader.
        // See https://docs.unity3d.com/Manual/GPUInstancing.html for more information about instancing.
        // #pragma instancing_options assumeuniformscaling
        UNITY_INSTANCING_BUFFER_START(Props)
        // put more per-instance properties here
        UNITY_INSTANCING_BUFFER_END(Props)

		float2 GetUV(in appdata_full v, half index)
		{
			if (index < 1)
				return v.texcoord;
			if (index < 2)
				return v.texcoord1;
			if (index < 3)
				return v.texcoord2;

			return v.texcoord3;
		}

		void vert(inout appdata_full v, out Input o)
		{
			UNITY_INITIALIZE_OUTPUT(Input, o);

			o.albedo_coord.xy = TRANSFORM_TEX(GetUV(v, _AlbedoUV), _MainTex);

#ifdef _EMISSIONTEX
			o.emission_coord.xy = TRANSFORM_TEX(GetUV(v, _EmissionUV), _EmissionMap);
#endif
#ifdef _NORMALMAP
			o.normal_occlusion_coord.xy = TRANSFORM_TEX(GetUV(v, _NormalUV), _NormalMap);
#endif
#ifdef _SPECULARMAP
			o.specular_coord = TRANSFORM_TEX(GetUV(v, _SpecularUV), _SpecularMap);
#endif
#ifdef _OCCLUSION
			o.normal_occlusion_coord.zw = TRANSFORM_TEX(GetUV(v, _OcclusionUV), _OcclusionMap);
#endif

#ifdef _DUAL_EMISSIONTEX
			o.emission_coord.zw = TRANSFORM_TEX(GetUV(v, _SecondaryEmissionUV), _SecondaryEmissionMap);
#endif
#ifdef _DUAL_ALBEDO
			o.albedo_coord.zw = TRANSFORM_TEX(GetUV(v, _SecondaryAlbedoUV), _SecondaryAlbedo);
#endif
		}

        void surf (Input IN, inout SurfaceOutputStandardSpecular o)
        {
			fixed4 c = tex2D(_MainTex, IN.albedo_coord.xy) * _Color;

#ifdef _DUAL_ALBEDO
			c *= tex2D(_SecondaryAlbedo, IN.albedo_coord.zw);
#endif

#ifdef _ALPHACLIP
			clip(c.a - _AlphaClip);
#endif

			o.Albedo = c.rgb;

#ifdef _NORMALMAP
			o.Normal = UnpackScaleNormal(tex2D(_NormalMap, IN.normal_occlusion_coord.xy), _NormalScale);
#endif

#ifdef _OCCLUSION
			o.Occlusion = tex2D(_OcclusionMap, IN.normal_occlusion_coord.zw).r;
#endif

#ifdef _SPECULARMAP
			half4 s = tex2D(_SpecularMap, IN.specular_coord);
#else
			half4 s = _SpecularColor;
#endif

			o.Emission = _EmissionColor;
#ifdef _EMISSIONTEX
			o.Emission *= tex2D(_EmissionMap, IN.emission_coord.xy).rgb;
#endif

#ifdef _DUAL_EMISSIONTEX
			o.Emission += tex2D(_SecondaryEmissionMap, IN.emission_coord.zw).rgb * _SecondaryEmissionColor.rgb;
#endif

			o.Alpha = c.a;
        }
        ENDCG
    }
    FallBack "Diffuse"
}
