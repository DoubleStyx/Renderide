Shader "Filters/Blur"
{
	Properties
	{
		_Iterations("Iterations", Float) = 4
		_Spread("Spread", Vector) = (0.1, 0.1, 0, 0)
		_SpreadTex("SpreadTexture", 2D) = "white" {}
		_RefractionStrength ("Refraction", Float) = 0.01
			_DepthDivisor("Depth Divisor", Float) = 1

		_NormalMap("Normal Map", 2D) = "bump" {}

		_SrcBlend("SrcBlend", Float) = 1.0
		_DstBlend("DstBlend", Float) = 0.0
		_ZWrite("ZWrite", Float) = 1.0
		_Cull("Cull", Float) = 2.0

		_StencilComp("Stencil Comparison", Float) = 8
		_Stencil("Stencil ID", Float) = 0
		_StencilOp("Stencil Operation", Float) = 0
		_StencilWriteMask("Stencil Write Mask", Float) = 255
		_StencilReadMask("Stencil Read Mask", Float) = 255

		_ColorMask("Color Mask", Float) = 15
	}



		SubShader
	{
		// Draw ourselves after all opaque geometry
		Tags
		{
			"Queue" = "Transparent+500"
			"IgnoreProjector" = "True"
			"RenderType" = "Transparent"
			"PreviewType" = "Plane"
			"CanUseSpriteAtlas" = "True"
		}

		Stencil
		{
			Ref[_Stencil]
			Comp[_StencilComp]
			Pass[_StencilOp]
			ReadMask[_StencilReadMask]
			WriteMask[_StencilWriteMask]
		}

		Blend[_SrcBlend][_DstBlend], One One
		BlendOp Add, Max
		ZWrite[_ZWrite]
		Cull[_Cull]
		ZTest[_ZTest]

		Offset[_OffsetFactor],[_OffsetUnits]

		ColorMask[_ColorMask]

		// Grab the screen behind the object into _BackgroundTexture
		GrabPass
		{
			"_BackgroundTexture"
		}

		// Render the object with the texture generated above, and invert the colors
		Pass
	{
		CGPROGRAM

#pragma multi_compile_instancing
#define _FADE_DEPTH
#define _BLUR
#define _GRAB_PASS
#pragma vertex vert
#pragma fragment frag
#include "../Common.cginc"
#include "UnityUI.cginc"

#pragma multi_compile _ SPREAD_TEX
#pragma multi_compile _ REFRACT REFRACT_NORMALMAP
#pragma multi_compile _ RECTCLIP
#pragma multi_compile _ POISSON_DISC

#pragma exclude_renderers d3d11_9x

float4 _Rect;

		float _RefractionStrength;

		struct v2f
		{
			float2 uv : TEXCOORD0;
			float4 grabPos : TEXCOORD1;
			float depth : TEXCOORD2;
			float4 pos : SV_POSITION;
			float3 normal : NORMAL;
			float3 tangent : TANGENT;
			float3 bitangent : TEXCOORD3;

			float2 position : TEXCOORD4;
		};

		v2f vert(appdata_full v)
		{
			v2f o;

			UNITY_SETUP_INSTANCE_ID(v);
			// use UnityObjectToClipPos from UnityCG.cginc to calculate 
			// the clip-space of the vertex
			o.pos = UnityObjectToClipPos(v.vertex);
			o.uv = v.texcoord.xy;

			o.normal = UnityObjectToWorldNormal(v.normal);
			o.tangent = normalize(mul(unity_ObjectToWorld, float4(v.tangent.xyz, 0)).xyz);
			o.bitangent = normalize(cross(o.normal, o.tangent) * v.tangent.w);

			// use ComputeGrabScreenPos function from UnityCG.cginc
			// to get the correct texture coordinate
			o.grabPos = ComputeGrabScreenPos(o.pos);

			o.position = v.vertex.xy;

			COMPUTE_EYEDEPTH(o.depth);
			return o;
		}

		half4 frag(v2f i) : SV_Target
		{
#ifdef RECTCLIP
			clip(UnityGet2DClipping(i.position, _Rect) - 0.1);
#endif
			float2 grabUv = i.grabPos.xy / i.grabPos.w;

#ifdef REFRACT
			i.normal = normalize(i.normal);
#elif REFRACT_NORMALMAP
			i.normal = normalize(i.normal);

			float3x3 tangentTransform = float3x3(i.tangent, i.bitangent, i.normal);
			float3 bumpNormal = UnpackNormal(tex2D(_NormalMap, TRANSFORM_TEX(i.uv, _NormalMap)));

			// Compute pertrubed normal, replacing the old one
			i.normal = normalize(mul(bumpNormal, tangentTransform));
#endif
#if defined(REFRACT) || defined(REFRACT_NORMALMAP)
			i.normal = mul((float3x3)UNITY_MATRIX_V, i.normal);
			grabUv -= (i.normal.xy / i.grabPos.w) * _RefractionStrength;
#endif 
			return evrCalculateBlur(grabUv, i.uv, i.depth, _Iterations);
		}
		ENDCG
	}

	}
}