Shader "PBSDistanceLerpSpecularTransparent"
{
	Properties{
		_Color("Color", Color) = (1,1,1,1)
		_MainTex("Albedo (RGB)", 2D) = "white" {}

		_NormalMap("NormalMap", 2D) = "bump" {}
		_NormalScale("Normal Scale", Float) = 1

		_EmissionColor("EmissionColor", Color) = (0,0,0,0)
		_EmissionMap("EmissionMap", 2D) = "black" {}

		_OcclusionMap("Occlusion", 2D) = "white" {}

		_Glossiness("Smoothness", Range(0,1)) = 0.5
		_Metallic("Metallic", Range(0,1)) = 0.0

		_MetallicMap("MetallicMap", 2D) = "black" {}

		_DistanceGridSize("Distance Grid Size", Vector) = (0, 0, 0, 0)
		_DistanceGridOffset("Distance Grid Offset", Vector) = (0, 0, 0, 0)

		_DisplaceDistanceFrom("Displace Distance From", Float) = 1
		_DisplaceDistanceTo("Displace Distance To", Float) = 0

		_DisplaceMagnitudeFrom("Displace Magnitude From", Float) = 0
		_DisplaceMagnitudeTo("Displace Magnitude To", Float) = 0.1

		_EmissionDistanceFrom("Emission Distance From", Float) = 1
		_EmissionDistanceTo("Emission Distance To", Float) = 0

		_EmissionColorFrom("Emission Color From", Color) = (0, 0, 0, 0)
		_EmissionColorTo("Emission Color To", Color) = (1.5, 1.5, 1.5, 0)

		_DisplacementDirection("Displacement Direction", Vector) = (0, 1, 0, 0)

		//_Cull("Culling", Float) = 0
	}


		SubShader{
			Tags{ "RenderType" = "Opaque" }
			//Tags { "RenderType"="Transparent" "Queue"="Transparent" }
			Cull[_Cull]
			Offset[_OffsetFactor],[_OffsetUnits]
			LOD 200

			CGPROGRAM
			// Physically based Standard lighting model, and enable shadows on all light types
			#pragma surface surf StandardSpecular alpha fullforwardshadows vertex:vert addshadow
			#define _GLOSSYENV 1

			// Use shader model 3.0 target, to get nicer looking lighting
			#pragma target 3.0
			

			//#pragma multi_compile _ _ALBEDOTEX
			#pragma multi_compile _ _NORMALMAP
			//#pragma multi_compile _ _EMISSIONTEX
			#pragma multi_compile _ _SPECULARMAP
			//#pragma multi_compile _ _OCCLUSION
			#pragma multi_compile WORLD_SPACE LOCAL_SPACE
			#pragma multi_compile _ OVERRIDE_DISPLACE_DIRECTION

			#define _ALBEDOTEX
			#define _EMISSIONTEX
			#define _OCCLUSION

			half4 _DistanceGridSize;
			half4 _DistanceGridOffset;

			half _DisplaceDistanceFrom;
			half _DisplaceDistanceTo;

			half _DisplaceMagnitudeFrom;
			half _DisplaceMagnitudeTo;

			half _EmissionDistanceFrom;
			half _EmissionDistanceTo;

			half4 _EmissionColorFrom;
			half4 _EmissionColorTo;

#ifdef OVERRIDE_DISPLACE_DIRECTION
			half3 _DisplacementDirection;
#endif

			int _PointCount;

			half3 _Points[16];
			half4 _TintColors[16];

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

			struct Input
			{
				float2 uv_MainTex;
				float3 emission;

				float facing : FACE;
			};

			void vert(inout appdata_full v, out Input o)
			{
#ifdef WORLD_SPACE
				float3 referencePoint = mul(unity_ObjectToWorld, v.vertex).xyz;
#else
				float3 referencePoint = v.vertex.xyz;
#endif

				bool3 mask = _DistanceGridSize == 0;

				float3 origReferencePoint = referencePoint;

				referencePoint += _DistanceGridOffset;
				referencePoint /= _DistanceGridSize;
				referencePoint = round(referencePoint);
				referencePoint *= _DistanceGridSize;
				referencePoint = mask ? origReferencePoint : referencePoint;

				// determine the displacement distance
				half displace = 0;
				half3 emission = 0;

				float distanceInverseScale = 1 / (_DisplaceDistanceTo - _DisplaceDistanceFrom);
				float emissionInverseScale = 1 / (_EmissionDistanceTo - _EmissionDistanceFrom);

				for (int i = 0; i < _PointCount; i++)
				{
					float d = distance(referencePoint, _Points[i]);

					float displaceLerp = saturate((d - _DisplaceDistanceFrom) * distanceInverseScale);
					float emissionLerp = saturate((d - _EmissionDistanceFrom) * emissionInverseScale);

					displace += lerp(_DisplaceMagnitudeFrom, _DisplaceMagnitudeTo, displaceLerp);
					emission += _TintColors[i] * lerp(_EmissionColorFrom, _EmissionColorTo, emissionLerp);
				}

#ifdef OVERRIDE_DISPLACE_DIRECTION
				v.vertex.xyz += _DisplacementDirection * displace;
#else
				v.vertex.xyz += v.normal.xyz * displace;
#endif

				o.emission = emission;

				o.facing = 0;
				o.uv_MainTex = float2(0, 0);
			}

			void surf(Input IN, inout SurfaceOutputStandardSpecular o)
			{
	#ifdef _ALBEDOTEX
				fixed4 c = tex2D(_MainTex, IN.uv_MainTex) * _Color;
	#else
				fixed4 c = _Color;
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

				o.Emission = emission.rgb + IN.emission;
				o.Alpha = c.a;
			}
			ENDCG
		}
			FallBack "Diffuse"
}
