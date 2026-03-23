Shader "PBSSliceTransparentSpecular"
{
    Properties
    {
		_Color("Color", Color) = (1,1,1,1)
		_EdgeColor ("Edge Color", Color) = (1, 1, 1, 1)

		_MainTex("Albedo (RGB)", 2D) = "white" {}

		_NormalMap("NormalMap", 2D) = "bump" {}
		_NormalScale("Normal Scale", Float) = 1

		_EmissionColor("EmissionColor", Color) = (0,0,0,0)
		_EdgeEmissionColor ("Edge Emission Color", Color) = (1, 1, 1, 1)
		_EmissionMap("EmissionMap", 2D) = "black" {}

		_OcclusionMap("Occlusion", 2D) = "white" {}

		_SpecularColor("SpecularColor", Color) = (1,1,1,0.5)
		_SpecularMap("SpecularMap", 2D) = "white" {}

		_AlphaClip("AlphaClip", Range(0,1)) = 0.5

		_EdgeTransitionStart ("Slice Edge Transition Start", Float) = 0
		_EdgeTransitionEnd ("Slice Edge Transition End", Float) = 0.1

		// Detail maps
		_DetailAlbedoMap("Detail Albedo", 2D) = "grey" {}
		_DetailNormalMapScale("Detail Normal Scale", Float) = 1.0
		[Normal] _DetailNormalMap("Detail Normal", 2D) = "bump" {}

		_Cull("Culling", Float) = 0

		_OffsetFactor("Offset Factor", Float) = 0.0
		_OffsetUnits("Offset Units", Float) = 0.0
    }

	

    SubShader
    {
		//Tags { "RenderType" = "Opaque" }
		Tags { "RenderType"="Transparent" "Queue"="Transparent" }
		Cull[_Cull]
		Offset[_OffsetFactor],[_OffsetUnits]
		LOD 200

        CGPROGRAM
        // Physically based Standard lighting model, and enable shadows on all light types
        #pragma surface surf StandardSpecular alpha fullforwardshadows vertex:vert addshadow
		#define _GLOSSYENV 1

        // Use shader model 3.0 target, to get nicer looking lighting
        #pragma target 3.0
		

		#pragma multi_compile _ _ALBEDOTEX _DETAIL_ALBEDOTEX
		#pragma multi_compile _ _EMISSIONTEX
		#pragma multi_compile _ _NORMALMAP _DETAIL_NORMALMAP
		#pragma multi_compile _ _METALLICMAP
		#pragma multi_compile _ _OCCLUSION

		#pragma multi_compile WORLD_SPACE OBJECT_SPACE

#if defined(_ALBEDOTEX) || defined(_DETAIL_ALBEDOTEX)
		sampler2D _MainTex;
#endif

#ifdef _DETAIL_ALBEDOTEX
		sampler2D _DetailAlbedoMap;
#endif

#if defined(_NORMALMAP) || defined(_DETAIL_NORMALMAP)
		sampler2D _NormalMap;
		float _NormalScale;
#endif

#ifdef _DETAIL_NORMALMAP
		sampler2D _DetailNormalMap;
		float _DetailNormalMapScale;
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

		half _EdgeTransitionStart;
		half _EdgeTransitionEnd;

		half4 _Color;
		half4 _EdgeColor;
		float4 _EmissionColor;
		float4 _EdgeEmissionColor;

		uniform float _SlicerCount;
		uniform half4 _Slicers[8];

		struct Input
		{
			float2 uv_MainTex;

#ifdef _DETAIL_ALBEDOTEX
			float2 uv_DetailAlbedoMap;
#endif
#ifdef _DETAIL_NORMALMAP
			float2 uv_DetailNormalMap;
#endif

			float3 pos;
			float facing : FACE;
		};

        // Add instancing support for this shader. You need to check 'Enable Instancing' on materials that use the shader.
        // See https://docs.unity3d.com/Manual/GPUInstancing.html for more information about instancing.
        // #pragma instancing_options assumeuniformscaling
        UNITY_INSTANCING_BUFFER_START(Props)
            // put more per-instance properties here
        UNITY_INSTANCING_BUFFER_END(Props)
			
		half plane_distance(half3 p, half3 normal, half offset)
		{
			return dot(p, normal) + offset;
		}

		void vert(inout appdata_full v, out Input o) 
		{
			UNITY_INITIALIZE_OUTPUT(Input, o);

#ifdef OBJECT_SPACE
			o.pos = v.vertex.xyz;
#elif defined(WORLD_SPACE)
			o.pos = mul(unity_ObjectToWorld, v.vertex).xyz;
#endif
		}

		void surf(Input IN, inout SurfaceOutputStandardSpecular o)
		{
			half minDistance = 60000;

			for (float i = 0; i < 8; i++)
			{
				half4 slicer = _Slicers[i];

				if (all(slicer.xyz == 0))
					break;

				half dist = plane_distance(IN.pos, slicer.xyz, slicer.w);

				minDistance = min(minDistance, dist);
			}

			if (minDistance < 0)
				discard;

			half edgeLerp = 1 - saturate((minDistance - _EdgeTransitionStart) / (_EdgeTransitionEnd - _EdgeTransitionStart));

#if defined(_ALBEDOTEX) || defined(_DETAIL_ALBEDOTEX)
			half4 c = tex2D(_MainTex, IN.uv_MainTex) * lerp(_Color, _EdgeColor, edgeLerp);
#ifdef _DETAIL_ALBEDOTEX
			half4 detail = tex2D(_DetailAlbedoMap, IN.uv_DetailAlbedoMap);
			c.rgb *= detail * unity_ColorSpaceDouble.rgb;
#endif
#else
			half4 c = lerp(_Color, _EdgeColor, edgeLerp);
#endif


			o.Albedo = c.rgb;

#if defined(_NORMALMAP) || defined(_DETAIL_NORMALMAP)
			o.Normal = UnpackScaleNormal(tex2D(_NormalMap, IN.uv_MainTex), _NormalScale);

#ifdef _DETAIL_NORMALMAP
			half3 detailNormal = UnpackScaleNormal(tex2D(_DetailNormalMap, IN.uv_DetailNormalMap), _DetailNormalMapScale);
			o.Normal = BlendNormals(o.Normal, detailNormal);
#endif

#endif
			if (IN.facing < 0.5)
				o.Normal.z *= -1;

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

			o.Emission = lerp(emission.rgb, _EdgeEmissionColor, edgeLerp);

			o.Alpha = c.a;
		}
        ENDCG
    }
    FallBack "Diffuse"
}
