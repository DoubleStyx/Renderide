Shader "PBSLerp" {
	Properties{
		_Color("Color0", Color) = (1,1,1,1)
		_Color1("Color0", Color) = (1,1,1,1)

		_Lerp("Lerp", Float) = 0
		_LerpTex("LerpTexture", 2D) = "white" {}

		_MainTex("Albedo (RGB) 0", 2D) = "white" {}
		_MainTex1("Albedo (RGB) 1", 2D) = "white" {}

		_NormalMap("NormalMap 0", 2D) = "bump" {}
		_NormalMap1("NormalMap 1", 2D) = "bump" {}
		_NormalScale("Normal Scale", Float) = 1
		_NormalScale1("Normal Scale", Float) = 1

		_EmissionColor("EmissionColor0", Color) = (0,0,0,0)
		_EmissionColor1("EmissionColor1", Color) = (0,0,0,0)

		_EmissionMap("EmissionMap0", 2D) = "black" {}
		_EmissionMap1("EmissionMap1", 2D) = "black" {}

		_Occlusion("Occlusion0", 2D) = "white" {}
		_Occlusion1("Occlusion1", 2D) = "white" {}

		// Metalic

		_Glossiness("Smoothness 0", Range(0,1)) = 0.5
		_Glossiness1("Smoothness 1", Range(0,1)) = 0.5

		_Metallic("Metallic 0", Range(0,1)) = 0.0
		_Metallic1("Metallic 1", Range(0,1)) = 0.0

		_MetallicMap("MetallicMap0", 2D) = "black" {}
		_MetallicMap1("MetallicMap1", 2D) = "black" {}

		_AlphaClip("Alpha clip", Float) = 0.5
			_Cull("_Cull", Float) = 2.0
	}



		SubShader{
			Tags { "RenderType" = "Opaque" }
			Offset[_OffsetFactor],[_OffsetUnits]
			Cull[_Cull]
			LOD 200

			CGPROGRAM
			#pragma surface surf Standard fullforwardshadows addshadow
			#define _GLOSSYENV 1

			#pragma multi_compile _ _LERPTEX

			#pragma multi_compile _ _ALBEDOTEX
			#pragma multi_compile _ _EMISSIONTEX
			#pragma multi_compile _ _NORMALMAP
			#pragma multi_compile _ _METALLICMAP
			#pragma multi_compile _ _OCCLUSION
#pragma multi_compile _ _MULTI_VALUES
		#pragma multi_compile _ _DUALSIDED
		#pragma multi_compile _ _ALPHACLIP

		// Use shader model 3.0 target, to get nicer looking lighting
		#pragma target 3.0
		

#ifdef _LERPTEX
		sampler2D _LerpTex;
#endif
		half _Lerp;

#ifdef _ALBEDOTEX
		sampler2D _MainTex;
		sampler2D _MainTex1;
#endif

#ifdef _NORMALMAP
		sampler2D _NormalMap;
		sampler2D _NormalMap1;
		float _NormalScale;
		float _NormalScale1;
#endif

#ifdef _OCCLUSION
		sampler2D _Occlusion;
		sampler2D _Occlusion1;
#endif

#ifdef _EMISSIONTEX
		sampler2D _EmissionMap;
		sampler2D _EmissionMap1;
#endif

#ifdef _METALLICMAP
		sampler2D _MetallicMap;
		sampler2D _MetallicMap1;
#endif		
		half _Glossiness;
		half _Glossiness1;

		half _Metallic;
		half _Metallic1;

		fixed3 _EmissionColor;
		fixed3 _EmissionColor1;

		fixed4 _Color;
		fixed4 _Color1;

#ifdef _ALPHACLIP
		float _AlphaClip;
#endif

		struct Input {
			float2 uv_LerpTex;
			float2 uv_MainTex;
			float2 uv_MainTex1;

#ifdef _DUALSIDED
			float facing : FACE;
#endif 
		};

		void surf(Input IN, inout SurfaceOutputStandard o)
		{
			// Compute lerp factor first
#ifdef _LERPTEX
			half l = tex2D(_LerpTex, IN.uv_LerpTex);
#ifdef _MULTI_VALUES
			l *= _Lerp;
#endif
#else
			half l = _Lerp;
#endif

			// Albedo

			fixed4 c0 = _Color;
			fixed4 c1 = _Color1;

#ifdef _ALBEDOTEX
			c0 *= tex2D(_MainTex, IN.uv_MainTex);
			c1 *= tex2D(_MainTex1, IN.uv_MainTex1);
#endif

			fixed4 c = lerp(c0, c1, l);

#ifdef _ALPHACLIP
			clip(c.a - _AlphaClip);
#endif

			o.Albedo = c.rgb;
			o.Alpha = c.a;

			// Normal

#ifdef _NORMALMAP
			fixed3 n0 = UnpackScaleNormal(tex2D(_NormalMap, IN.uv_MainTex), _NormalScale);
			fixed3 n1 = UnpackScaleNormal(tex2D(_NormalMap1, IN.uv_MainTex1), _NormalScale1);

			o.Normal = lerp(n0, n1, l);
#endif

#ifdef _DUALSIDED
			if (IN.facing < 0.5)
				o.Normal.z *= -1;
#endif

			// Occlusion (duh)
#ifdef _OCCLUSION
			o.Occlusion = lerp(tex2D(_Occlusion, IN.uv_MainTex).r, tex2D(_Occlusion1, IN.uv_MainTex1).r, l);
#endif

			// Emission

			half3 e0 = _EmissionColor;
			half3 e1 = _EmissionColor1;

#ifdef _EMISSIONTEX
			e0 *= tex2D(_EmissionMap, IN.uv_MainTex).rgb;
			e1 *= tex2D(_EmissionMap1, IN.uv_MainTex1).rgb;
#endif

			o.Emission = lerp(e0, e1, l);

			// Metallic and smoothness
#ifdef _METALLICMAP
			half4 m0 = tex2D(_MetallicMap, IN.uv_MainTex);
			half4 m1 = tex2D(_MetallicMap1, IN.uv_MainTex1);

#ifdef _MULTI_VALUES
			m0.r *= _Metallic;
			m0.a *= _Glossiness;

			m1.r *= _Metallic1;
			m1.a *= _Glossiness1;
#endif


			half4 m = lerp(m0, m1, l);

			o.Metallic = m.r;
			o.Smoothness = m.a;
#else
			o.Metallic = lerp(_Metallic, _Metallic1, l);
			o.Smoothness = lerp(_Glossiness, _Glossiness1, l);
#endif
		}
		ENDCG
	}
	FallBack "Transparent/Cutout/VertexLit"
}
