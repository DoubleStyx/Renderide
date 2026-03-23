// Upgrade NOTE: replaced '_Object2World' with 'unity_ObjectToWorld'

// Normal Mapping for a Triplanar Shader - Ben Golus 2017
// Unity Surface Shader example shader

// Implements correct triplanar normals in a Surface Shader with out computing or passing additional information from the
// vertex shader. Instead works around some oddities with how Surface Shaders handle the tangent space vectors. Attempting
// to directly access the tangent matrix data results in a shader generation error. This works around the issue by tricking
// the surface shader into not using those vectors until actually in the generated shader code.

Shader "PBSTriplanarTransparentSpecular"
{
	Properties
	{
		_Color("Color", Color) = (1,1,1,1)
		_MainTex("Albedo (RGB)", 2D) = "white" {}

		_NormalMap("NormalMap", 2D) = "bump" {}
		_NormalScale("Normal Scale", Float) = 1

		_EmissionColor("EmissionColor", Color) = (0,0,0,0)
		_EmissionMap("EmissionMap", 2D) = "black" {}

		_OcclusionMap("Occlusion", 2D) = "white" {}

		_SpecularColor("SpecularColor", Color) = (1,1,1,0.5)
		_SpecularMap("SpecularMap", 2D) = "white" {}

		_TriBlendPower("Triplanar Blend Power", Float) = 4.0

		_Cull("Culling", Float) = 0
	}



		SubShader{
			//Tags{ "RenderType" = "Opaque" }
			Tags { "RenderType" = "Transparent" "Queue" = "Transparent" }
			Cull[_Cull]
			Offset[_OffsetFactor],[_OffsetUnits]
			LOD 200

			CGPROGRAM
			// Physically based Standard lighting model, and enable shadows on all light types
	#pragma surface surf StandardSpecular alpha fullforwardshadows vertex:vert
	#define _GLOSSYENV 1

			// Use shader model 3.0 target, to get nicer looking lighting
	#pragma target 3.0
	

	#pragma multi_compile _WORLDSPACE _OBJECTSPACE
	#pragma multi_compile _ _ALBEDOTEX
	#pragma multi_compile _ _EMISSIONTEX
	#pragma multi_compile _ _NORMALMAP
	#pragma multi_compile _ _SPECULARMAP
	#pragma multi_compile _ _OCCLUSION


	#include "UnityStandardUtils.cginc"

			// flip UVs horizontally to correct for back side projection
	#define TRIPLANAR_CORRECT_PROJECTED_U

			// offset UVs to prevent obvious mirroring
			// #define TRIPLANAR_UV_OFFSET

			// hack to work around the way Unity passes the tangent to world matrix to surface shaders to prevent compiler errors
	#if defined(INTERNAL_DATA) && (defined(UNITY_PASS_FORWARDBASE) || defined(UNITY_PASS_FORWARDADD) || defined(UNITY_PASS_DEFERRED) || defined(UNITY_PASS_META))
	#define WorldToTangentNormalVector(data,normal) mul(normal, half3x3(data.internalSurfaceTtoW0, data.internalSurfaceTtoW1, data.internalSurfaceTtoW2))
	#else
	#define WorldToTangentNormalVector(data,normal) normal
	#endif

			// Reoriented Normal Mapping
			// http://blog.selfshadow.com/publications/blending-in-detail/
			// Altered to take normals (-1 to 1 ranges) rather than unsigned normal maps (0 to 1 ranges)
			half3 blend_rnm(half3 n1, half3 n2)
		{
			n1.z += 1;
			n2.xy = -n2.xy;

			return n1 * dot(n1, n2) / n1.z - n2;
		}

	#ifdef _ALBEDOTEX
			sampler2D _MainTex;
	#endif

			float4 _MainTex_ST;

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

		float _TriBlendPower;

		struct Input {
			float facing : FACE;
	#ifdef _WORLDSPACE
			float3 worldPos;
	#else
			float3 objPos;
	#endif
			float3 worldNormal;
			INTERNAL_DATA
		};

		void vert(inout appdata_full v, out Input o)
		{
			UNITY_INITIALIZE_OUTPUT(Input, o);
	#ifdef _OBJECTSPACE
			o.objPos = v.vertex;
	#endif
		}

		void surf(Input IN, inout SurfaceOutputStandardSpecular o)
		{
			// work around bug where IN.worldNormal is always (0,0,0)!
			IN.worldNormal = WorldNormalVector(IN, float3(0, 0, 1));

	#ifdef _OBJECTSPACE
			// TODO!!! For some reason in appdata_full the normal is wrong and doesn't get passed, so we need to transform it here
			IN.worldNormal = mul((float3x3)unity_WorldToObject, IN.worldNormal);
	#endif

			// calculate triplanar blend
			half3 triblend = saturate(pow(abs(IN.worldNormal), _TriBlendPower));
			triblend /= max(dot(triblend, half3(1,1,1)), 0.0001);

			// calculate triplanar uvs
			// applying texture scale and offset values ala TRANSFORM_TEX macro
	#ifdef _WORLDSPACE
			float3 pos = IN.worldPos;
	#else
			float3 pos = IN.objPos;
	#endif

			float2 uvX = pos.zy * _MainTex_ST.xy + _MainTex_ST.zy;
			float2 uvY = pos.xz * _MainTex_ST.xy + _MainTex_ST.zy;
			float2 uvZ = pos.xy * _MainTex_ST.xy + _MainTex_ST.zy;

			// offset UVs to prevent obvious mirroring
	#if defined(TRIPLANAR_UV_OFFSET)
			uvY += 0.33;
			uvZ += 0.67;
	#endif

			// minor optimization of sign(). prevents return value of 0
			half3 axisSign = IN.worldNormal < 0 ? -1 : 1;

			// flip UVs horizontally to correct for back side projection
	#if defined(TRIPLANAR_CORRECT_PROJECTED_U)
			uvX.x *= axisSign.x;
			uvY.x *= axisSign.y;
			uvZ.x *= -axisSign.z;
	#endif

			// albedo textures
	#ifdef _ALBEDOTEX
			fixed4 colX = tex2D(_MainTex, uvX);
			fixed4 colY = tex2D(_MainTex, uvY);
			fixed4 colZ = tex2D(_MainTex, uvZ);
			fixed4 col = (colX * triblend.x + colY * triblend.y + colZ * triblend.z) * _Color;
	#else
			fixed4 col = _Color;
	#endif

	#ifdef _SPECULARMAP
			half4 specX = tex2D(_SpecularMap, uvX);
			half4 specY = tex2D(_SpecularMap, uvY);
			half4 specZ = tex2D(_SpecularMap, uvZ);
			half4 spec = (specX * triblend.x + specY * triblend.y + specZ * triblend.z);
	#else
			half4 spec = _SpecularColor;
	#endif

			half4 emit = _EmissionColor;
	#ifdef _EMISSIONTEX
			half4 emitX = tex2D(_EmissionMap, uvX);
			half4 emitY = tex2D(_EmissionMap, uvY);
			half4 emitZ = tex2D(_EmissionMap, uvZ);
			emit *= (emitX * triblend.x + emitY * triblend.y + emitZ * triblend.z);
	#endif

	#ifdef _OCCLUSION
			// occlusion textures
			half occX = tex2D(_OcclusionMap, uvX).g;
			half occY = tex2D(_OcclusionMap, uvY).g;
			half occZ = tex2D(_OcclusionMap, uvZ).g;
			//half occ = LerpOneTo(colX * triblend.x + colY * triblend.y + colZ * triblend.z, _OcclusionStrength);
			half occ = occX * triblend.x + occY * triblend.y + occZ * triblend.z;
	#else
			half occ = 1;
	#endif

	#ifdef _NORMALMAP
			// tangent space normal maps
			half3 tnormalX = UnpackScaleNormal(tex2D(_NormalMap, uvX), _NormalScale);
			half3 tnormalY = UnpackScaleNormal(tex2D(_NormalMap, uvY), _NormalScale);
			half3 tnormalZ = UnpackScaleNormal(tex2D(_NormalMap, uvZ), _NormalScale);

			// flip normal maps' x axis to account for flipped UVs
	#if defined(TRIPLANAR_CORRECT_PROJECTED_U)
			tnormalX.x *= axisSign.x;
			tnormalY.x *= axisSign.y;
			tnormalZ.x *= -axisSign.z;
	#endif

			half3 absVertNormal = abs(IN.worldNormal);

			// swizzle world normals to match tangent space and apply reoriented normal mapping blend
			tnormalX = blend_rnm(half3(IN.worldNormal.zy, absVertNormal.x), tnormalX);
			tnormalY = blend_rnm(half3(IN.worldNormal.xz, absVertNormal.y), tnormalY);
			tnormalZ = blend_rnm(half3(IN.worldNormal.xy, absVertNormal.z), tnormalZ);

			// apply world space sign to tangent space Z
			tnormalX.z *= axisSign.x;
			tnormalY.z *= axisSign.y;
			tnormalZ.z *= axisSign.z;

			// sizzle tangent normals to match world normal and blend together
			half3 worldNormal = normalize(
				tnormalX.zyx * triblend.x +
				tnormalY.xzy * triblend.y +
				tnormalZ.xyz * triblend.z
			);

	#ifdef _OBJECTSPACE
			worldNormal = mul((float3x3)unity_ObjectToWorld, worldNormal);
	#endif

			half3 normal = WorldToTangentNormalVector(IN, worldNormal);
	#else
			half3 normal = half3(0, 0, 1);
	#endif

			// set surface ouput properties
			o.Albedo = col.rgb;
			o.Alpha = col.a;
			o.Occlusion = occ;
			o.Emission = emit.rgb;
			o.Specular = spec.rgb;
			o.Smoothness = spec.a;

			// convert world space normals into tangent normals
			o.Normal = normal;

			if (IN.facing < 0.5)
				o.Normal.z *= -1;
		}
		ENDCG
		}
			FallBack "Diffuse"
}
