Shader "Art/PaintPBS" {
	Properties {
		_Color ("Color", Color) = (1,1,1,1)
		_MainTex ("Albedo (RGB)", 2D) = "white" {}
		_PaintTex ("Paint Pattern (RGBA)", 2D) = "white" {}
		_PaintTexOffsets("Paint Tex Offsets", Vector) = ( 0, 0.333, 0.5, 0.777 )
		_PaintTexShifts("Paint Tex Shifts", Vector) = (-0.7, 0.2, -0.4, 1)
		_PaintTexScales("Paint Tex Scales", Vector) = ( 1, 0.95, 0.89, 1.13 )
		_SideFadeSize ("Side Fade Size", Range(0, 0.5)) = 0.1
		_Glossiness ("Smoothness", Range(0,1)) = 0.5
		_Metallic ("Metallic", Range(0,1)) = 0.0
		_Pow("Power", Float) = 1
		_PaintBias ("Paint Bias", Float) = 0
		_PaintGain ("Paint Gain", Float) = 1
		_OutputScale ("Output Scale", Float) = 10
	}
	SubShader {
		Tags{ "RenderType" = "Transparent" "Queue" = "Transparent" }
		LOD 200

		CGPROGRAM
		// Physically based Standard lighting model, and enable shadows on all light types
		#pragma surface surf Standard alpha:fade fullforwardshadows

		// Use shader model 3.0 target, to get nicer looking lighting
		#pragma target 3.0
		

		sampler2D _MainTex;
		sampler2D _PaintTex;

		struct Input {
			float2 uv_MainTex;
			float2 uv_PaintTex;
		};

		float4 _PaintTexOffsets;
		float4 _PaintTexShifts;
		float4 _PaintTexScales;

		half _SideFadeSize;
		half _Glossiness;
		half _Metallic;
		fixed4 _Color;

		float _Pow;
		float _PaintBias;
		float _PaintGain;
		float _OutputScale;

		// Add instancing support for this shader. You need to check 'Enable Instancing' on materials that use the shader.
		// See https://docs.unity3d.com/Manual/GPUInstancing.html for more information about instancing.
		// #pragma instancing_options assumeuniformscaling
		UNITY_INSTANCING_BUFFER_START(Props)
			// put more per-instance properties here
		UNITY_INSTANCING_BUFFER_END(Props)

		void surf (Input IN, inout SurfaceOutputStandard o)
		{
			// Albedo comes from a texture tinted by color
			fixed4 c = tex2D (_MainTex, IN.uv_MainTex) * _Color;
			o.Albedo = c.rgb;
			// Metallic and smoothness come from slider variables
			o.Metallic = _Metallic;
			o.Smoothness = _Glossiness;

			c.a *= saturate(min(IN.uv_MainTex.x / _SideFadeSize, (1-IN.uv_MainTex.x) / _SideFadeSize));

			float4 offsets = IN.uv_PaintTex.y * _PaintTexScales + _PaintTexOffsets
				 + IN.uv_PaintTex.x * _PaintTexShifts;

			float4 p = float4(
				tex2D(_PaintTex, float2(IN.uv_PaintTex.x, offsets.x)).r,
				tex2D(_PaintTex, float2(IN.uv_PaintTex.x, offsets.y)).g,
				tex2D(_PaintTex, float2(IN.uv_PaintTex.x, offsets.z)).b,
				tex2D(_PaintTex, float2(IN.uv_PaintTex.x, offsets.w)).a);

			float paint = (p.x + p.y + p.z + p.w) * 0.25 * _PaintGain + _PaintBias;

			float strength = saturate((c.a + pow(paint, _Pow) - 1) * _OutputScale);

			o.Alpha = strength;
		}
		ENDCG
	}
	FallBack "Diffuse"
}
