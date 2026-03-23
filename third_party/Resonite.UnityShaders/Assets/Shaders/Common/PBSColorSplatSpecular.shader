Shader "PBSColorSplatSpecular"
{
	Properties
	{
		_ColorMap("Color Map", 2D) = "white" {}
		_PackedHeightMap("Packed Height Map", 2D) = "white" {}

		_Color("Color0", Color) = (1,1,1,1)
		_Color1("Color1", Color) = (1,1,1,1)
		_Color2("Color2", Color) = (1,1,1,1)
		_Color3("Color3", Color) = (1,1,1,1)

		_HeightTransitionRange("Height Transition", Range(0, 1)) = 0.1

		_Albedo("Albedo 0", 2D) = "white" {}
		_Albedo1("Albedo 1", 2D) = "white" {}
		_Albedo2("Albedo 2", 2D) = "white" {}
		_Albedo3("Albedo 3", 2D) = "white" {}

		_PackedNormalMap01("Packed Normal 01", 2D) = "black" {}
		_PackedNormalMap23("Packed Normal 23", 2D) = "black" {}

		_NormalScale0("Normal Scale 0", Float) = 1
		_NormalScale1("Normal Scale 1", Float) = 1
		_NormalScale2("Normal Scale 2", Float) = 1
		_NormalScale3("Normal Scale 3", Float) = 1

		_EmissionColor("Emission Color 0", Color) = (0, 0, 0, 0)
		_EmissionColor1("Emission Color 1", Color) = (0, 0, 0, 0)
		_EmissionColor2("Emission Color 2", Color) = (0, 0, 0, 0)
		_EmissionColor3("Emission Color 3", Color) = (0, 0, 0, 0)

		_EmissionMap("Emission Map 0", 2D) = "white" {}
		_EmissionMap1("Emission Map 1", 2D) = "white" {}
		_EmissionMap2("Emission Map 2", 2D) = "white" {}
		_EmissionMap3("Emission Map 3", 2D) = "white" {}

		_PackedEmissionMap("Packed Emission Map", 2D) = "white" {}

		_SpecularColor("Specular Color 0", Color) = (0.5, 0.5, 0.5, 0.5)
		_SpecularColor1("Specular Color 1", Color) = (0.5, 0.5, 0.5, 0.5)
		_SpecularColor2("Specular Color 2", Color) = (0.5, 0.5, 0.5, 0.5)
		_SpecularColor3("Specular Color 3", Color) = (0.5, 0.5, 0.5, 0.5)

		_SpecularMap("Specular Map 0", 2D) = "white" {}
		_SpecularMap1("Specular Map 1", 2D) = "white" {}
		_SpecularMap2("Specular Map 2", 2D) = "white" {}
		_SpecularMap3("Specular Map 3", 2D) = "white" {}

		_OffsetFactor("Offset Factor", Float) = 0.0
		_OffsetUnits("Offset Units", Float) = 0.0
	}
		SubShader
		{
			Tags { "RenderType" = "Opaque" }
			Offset[_OffsetFactor],[_OffsetUnits]
			LOD 200

			CGPROGRAM
			// Physically based Standard lighting model, and enable shadows on all light types
			#pragma surface surf StandardSpecular fullforwardshadows
			#define _GLOSSYENV 1 

			// Use shader model 3.0 target, to get nicer looking lighting
			#pragma target 3.0
			

			#pragma multi_compile _ _HEIGHTMAP
			#pragma multi_compile _ _PACKED_NORMALMAP
			#pragma multi_compile _ _EMISSIONTEX _PACKED_EMISSIONTEX
			#pragma multi_compile _ _SPECULARMAP

			sampler2D _ColorMap;

			float4 _Color;
			float4 _Color1;
			float4 _Color2;
			float4 _Color3;

			UNITY_DECLARE_TEX2D(_Albedo);
			UNITY_DECLARE_TEX2D(_Albedo1);
			UNITY_DECLARE_TEX2D(_Albedo2);
			UNITY_DECLARE_TEX2D(_Albedo3);

			float4 _EmissionColor;
			float4 _EmissionColor1;
			float4 _EmissionColor2;
			float4 _EmissionColor3;

	#ifdef _EMISSIONTEX
			UNITY_DECLARE_TEX2D_NOSAMPLER(_EmissionMap);
			UNITY_DECLARE_TEX2D_NOSAMPLER(_EmissionMap1);
			UNITY_DECLARE_TEX2D_NOSAMPLER(_EmissionMap2);
			UNITY_DECLARE_TEX2D_NOSAMPLER(_EmissionMap3);
	#elif defined(_PACKED_EMISSIONTEX)
			sampler2D _PackedEmissionMap;
	#endif

	#ifdef _SPECULARMAP
			UNITY_DECLARE_TEX2D_NOSAMPLER(_SpecularMap);
			UNITY_DECLARE_TEX2D_NOSAMPLER(_SpecularMap1);
			UNITY_DECLARE_TEX2D_NOSAMPLER(_SpecularMap2);
			UNITY_DECLARE_TEX2D_NOSAMPLER(_SpecularMap3);
#endif


	#ifdef _PACKED_NORMALMAP
			float _NormalScale0;
			float _NormalScale1;
			float _NormalScale2;
			float _NormalScale3;

			sampler2D _PackedNormalMap01;
			sampler2D _PackedNormalMap23;
	#endif

	#ifdef _HEIGHTMAP
			float _HeightTransitionRange;
			sampler2D _PackedHeightMap;
	#endif

			float4 _SpecularColor;
			float4 _SpecularColor1;
			float4 _SpecularColor2;
			float4 _SpecularColor3;

			struct Input
			{
				float2 uv_Albedo;
				float2 uv_ColorMap;
			};

			// Add instancing support for this shader. You need to check 'Enable Instancing' on materials that use the shader.
			// See https://docs.unity3d.com/Manual/GPUInstancing.html for more information about instancing.
			// #pragma instancing_options assumeuniformscaling
			UNITY_INSTANCING_BUFFER_START(Props)
				// put more per-instance properties here
			UNITY_INSTANCING_BUFFER_END(Props)

			void surf(Input IN, inout SurfaceOutputStandardSpecular o)
			{
				fixed4 map = tex2D(_ColorMap, IN.uv_ColorMap);

	#ifdef _HEIGHTMAP
				fixed4 heights = tex2D(_PackedHeightMap, IN.uv_Albedo);

				heights *= map; // multiply heights by the map contribution

				fixed maxHeight = max(max(heights.x, heights.y), max(heights.z, heights.w));

				heights -= maxHeight;
				heights += _HeightTransitionRange;
				heights /= _HeightTransitionRange;
				map = saturate(heights);
	#endif

				// normalize the map, so all components add up to 1
				map /= map.x + map.y + map.z + map.w;

				fixed4 c0 = UNITY_SAMPLE_TEX2D(_Albedo, IN.uv_Albedo) * _Color;
				fixed4 c1 = UNITY_SAMPLE_TEX2D(_Albedo1, IN.uv_Albedo) * _Color1;
				fixed4 c2 = UNITY_SAMPLE_TEX2D(_Albedo2, IN.uv_Albedo) * _Color2;
				fixed4 c3 = UNITY_SAMPLE_TEX2D(_Albedo3, IN.uv_Albedo) * _Color3;

				fixed4 c = c0 * map.x + c1 * map.y + c2 * map.z + c3 * map.w;

				o.Albedo = c.rgb;
				o.Alpha = c.a;

				half4 e0 = _EmissionColor;
				half4 e1 = _EmissionColor1;
				half4 e2 = _EmissionColor2;
				half4 e3 = _EmissionColor3;

	#ifdef _EMISSIONTEX
				e0 *= UNITY_SAMPLE_TEX2D_SAMPLER(_EmissionMap, _Albedo, IN.uv_Albedo);
				e1 *= UNITY_SAMPLE_TEX2D_SAMPLER(_EmissionMap1, _Albedo1, IN.uv_Albedo);
				e2 *= UNITY_SAMPLE_TEX2D_SAMPLER(_EmissionMap2, _Albedo2, IN.uv_Albedo);
				e3 *= UNITY_SAMPLE_TEX2D_SAMPLER(_EmissionMap3, _Albedo3, IN.uv_Albedo);
	#elif defined(_PACKED_EMISSIONTEX)
				float4 packedEmission = tex2D(_PackedEmissionMap, IN.uv_Albedo);

				e0 *= packedEmission.x;
				e1 *= packedEmission.y;
				e2 *= packedEmission.z;
				e3 *= packedEmission.w;
	#endif

				half4 e = e0 * map.x + e1 * map.y + e2 * map.z + e3 * map.w;
				o.Emission = e.rgb;

	#ifdef _PACKED_NORMALMAP
				fixed4 n01 = tex2D(_PackedNormalMap01, IN.uv_Albedo);
				fixed4 n23 = tex2D(_PackedNormalMap23, IN.uv_Albedo);

				fixed3 n0;
				fixed3 n1;
				fixed3 n2;
				fixed3 n3;

				n0.xy = (n01.xy * 2 - 1) * _NormalScale0;
				n1.xy = (n01.zw * 2 - 1) * _NormalScale1;
				n2.xy = (n23.xy * 2 - 1) * _NormalScale2;
				n3.xy = (n23.zw * 2 - 1) * _NormalScale3;

				n0.z = sqrt(1 - saturate(dot(n0.xy, n0.xy)));
				n1.z = sqrt(1 - saturate(dot(n1.xy, n1.xy)));
				n2.z = sqrt(1 - saturate(dot(n2.xy, n2.xy)));
				n3.z = sqrt(1 - saturate(dot(n3.xy, n3.xy)));

				fixed3 n = n0 * map.x + n1 * map.y + n2 * map.z + n3 * map.w;

				o.Normal = normalize(n);
	#else
				o.Normal = half3(0, 0, 1);
	#endif

	#ifdef _SPECULARMAP
				fixed4 s0 = UNITY_SAMPLE_TEX2D_SAMPLER(_SpecularMap, _Albedo, IN.uv_Albedo);
				fixed4 s1 = UNITY_SAMPLE_TEX2D_SAMPLER(_SpecularMap1, _Albedo1, IN.uv_Albedo);
				fixed4 s2 = UNITY_SAMPLE_TEX2D_SAMPLER(_SpecularMap2, _Albedo2, IN.uv_Albedo);
				fixed4 s3 = UNITY_SAMPLE_TEX2D_SAMPLER(_SpecularMap3, _Albedo3, IN.uv_Albedo);

				s0 *= _SpecularColor;
				s1 *= _SpecularColor1;
				s2 *= _SpecularColor2;
				s3 *= _SpecularColor3;
	#else
				fixed4 s0 = _SpecularColor;
				fixed4 s1 = _SpecularColor1;
				fixed4 s2 = _SpecularColor2;
				fixed4 s3 = _SpecularColor3;
	#endif

				fixed4 s = s0 * map.x + s1 * map.y + s2 * map.z + s3 * map.w;

				o.Specular = s.rgb;
				o.Smoothness = s.a;
			}
			ENDCG
		}
			FallBack "Diffuse"
}
