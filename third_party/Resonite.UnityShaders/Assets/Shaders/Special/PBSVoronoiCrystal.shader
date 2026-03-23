Shader "PBSVoronoiCrystal" {
	Properties{
		_Scale("Scale", Vector) = (1, 1, 0, 0)

		_ColorTint ("Color Tint", Color) = (1, 1, 1, 1)
		_ColorGradient("Color Gradient", 2D) = "white" {}
		_EmissionColor("Emission Color", Color) = (0, 0, 0, 0)
		_EmissionGradient("Emission Gradient", 2D) = "white" {}
		_GlossGradient("Smoothness Gradient", 2D) = "white" {}
		_NormalMap("NormalMap", 2D) = "bump" {}

		_AnimationOffset("Animation Offset", Float) = 0

		_NormalStrength("Normal Strength", Float) = 1

		_EdgeThickness("Edge Thickness", Range(0, 1)) = 0.1
		_EdgeColor("Edge Color", Color) = (0, 0, 0, 1)
		_EdgeEmission("Edge Emission", Color) = (0, 0, 0, 0)
		_EdgeGloss("Edge Smoothness", Range(0, 1)) = 0.8
		_EdgeMetallic("Edge Metallic", Range(0, 1)) = 0.1
		_EdgeNormalStrength("Edge Normal Strength", Float) = 0.5

		_Glossiness("Smoothness", Range(0, 1)) = 0.5
		_Metallic("Metallic", Range(0, 1)) = 0.0

		[HideInInspector]
		_Global ("Dummy", 2D) = "black" {}
	}

		

		SubShader{
			Tags { "RenderType" = "Opaque" }
			Offset[_OffsetFactor],[_OffsetUnits]
			LOD 200

			CGPROGRAM
		// Physically based Standard lighting model, and enable shadows on all light types
		#pragma surface surf Standard fullforwardshadows
		#define _GLOSSYENV 1

		#include "UnityPBSLighting.cginc"

		// Use shader model 3.0 target, to get nicer looking lighting
		#pragma target 3.0
		

			uniform float4 _EmissionColor;

			uniform float4 _ColorTint;

			sampler2D _ColorGradient;
			sampler2D _GlossGradient;
			sampler2D _EmissionGradient;
			sampler2D _NormalMap;

			uniform float2 _Scale;

			uniform float _NormalStrength;

			uniform float _EdgeThickness;
			uniform float4 _EdgeColor;
			uniform float4 _EdgeEmission;
			uniform float _EdgeGloss;
			uniform float _EdgeMetallic;
			uniform float _EdgeNormalStrength;

			uniform float _AnimationOffset;

			struct Input {
				float2 uv_Global;
				float2 uv_NormalMap;
				float3 viewDir;
				float3 worldPos;
			};

			uniform half _Glossiness;
			uniform half _Metallic;

			// Add instancing support for this shader. You need to check 'Enable Instancing' on materials that use the shader.
			// See https://docs.unity3d.com/Manual/GPUInstancing.html for more information about instancing.
			// #pragma instancing_options assumeuniformscaling
			UNITY_INSTANCING_BUFFER_START(Props)
				// put more per-instance properties here
			UNITY_INSTANCING_BUFFER_END(Props)

			//void vert(inout appdata_full v)
			//{
			//	v.vertex.xyz += v.normal * _Amount;
			//}

			float2 random2(float2 p)
			{
				return frac(sin(float2(dot(p, float2(127.1, 311.7)), dot(p, float2(269.5, 183.3))))*43758.5453);
			}

			void surf(Input IN, inout SurfaceOutputStandard o)
			{
				float2 uv = IN.uv_Global * _Scale;

				float2 i_uv = floor(uv);
				float2 f_uv = frac(uv);

				// this actually needs to come from external value, because otherwise frac at the end doesn't work and returns constant value
				float minDist = 2;
				float secondMinDist = 2;

				float2 minPoint = float2(0, 0);

				for (int y = -1; y <= 1; y++)
					for (int x = -1; x <= 1; x++)
					{
						float2 neighbor = float2(x, y);
						float2 tile = i_uv + neighbor;

						tile %= _Scale;

						if (tile.x < 0)
							tile.x += _Scale.x;
						if (tile.y < 0)
							tile.y += _Scale.y;

						float2 p = random2(tile);
						float2 origPoint = p;

						p = 0.5 + 0.5*sin(_AnimationOffset + 6.2831*p);

						float2 diff = neighbor + p - f_uv;
						float dist = length(diff);

						if (dist < minDist)
						{
							secondMinDist = minDist;
							minDist = dist;
							minPoint = origPoint;
						}
						else if (dist < secondMinDist)
							secondMinDist = dist;
					}

				float2 cellOffset = 0.5 + 0.5*sin(_AnimationOffset + 6.2831*minPoint);

				float borderDist = secondMinDist - minDist;

				float aaf = fwidth(borderDist);
				float borderLerp = smoothstep(_EdgeThickness - aaf, _EdgeThickness, borderDist);

				float2 edgeDir = normalize(float2(ddx(borderDist), ddy(borderDist))) * _EdgeNormalStrength;

				float3 edgeNormal = normalize(float3(edgeDir, 1));
				float3 cellNormal = UnpackScaleNormal(tex2D(_NormalMap, IN.uv_NormalMap + minPoint), _NormalStrength);

				float3 edgeColor = _EdgeColor.xyz;
				float3 cellColor = tex2D(_ColorGradient, cellOffset).xyz * _ColorTint.xyz;

				float3 cellEmission = tex2D(_EmissionGradient, cellOffset) * _EmissionColor;

				//o.Normal = UnpackNormal(tex2D(_NormalMap, IN.uv_NormalMap + minPoint));
				o.Normal = lerp(edgeNormal, cellNormal, borderLerp);

				o.Albedo = lerp(edgeColor, cellColor, borderLerp); //tex2D(_Gradient, minPoint); //float3(1, 1, 1);
				// Metallic and smoothness come from slider variables
				o.Metallic = lerp(_EdgeMetallic, _Metallic, borderLerp);
				o.Smoothness = lerp(_EdgeGloss, _Glossiness * tex2D(_GlossGradient, minPoint).x, borderLerp);
				o.Emission = lerp(_EdgeEmission.xyz, cellEmission, borderLerp);
			}
			ENDCG
		}
		FallBack "Diffuse"
}
