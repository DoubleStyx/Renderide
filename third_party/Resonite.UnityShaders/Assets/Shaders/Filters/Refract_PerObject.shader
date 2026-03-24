Shader "Filters/Refract_PerObject"
{
	Properties
	{
		_RefractionStrength ("Refraction", Float) = 0.01

		_NormalMap("Normal Map", 2D) = "bump" {}

		_DepthBias("Depth Bias", Float) = 0.01
			_DepthDivisor("DepthStart", Float) = 0

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

		Offset[_OffsetFactor],[_OffsetUnits]
		Blend[_SrcBlend][_DstBlend], One One
		BlendOp Add, Max
		ZWrite[_ZWrite]
		Cull[_Cull]
		ZTest[_ZTest]
		ColorMask[_ColorMask]

		// Grab the screen behind the object into _GrabTexture
		GrabPass
		{
			
		}

		// Render the object with the texture generated above, and invert the colors
		Pass
	{
		CGPROGRAM

#pragma multi_compile_instancing

#pragma vertex vert
#pragma fragment frag

#define _FADE_DEPTH
#define _REFRACT
#include "UnityCG.cginc"
#include "UnityUI.cginc"
#include "UnityStandardUtils.cginc"
#include "../Common.cginc"

#pragma multi_compile _ _NORMALMAP
#pragma multi_compile _ RECTCLIP

#ifdef RECTCLIP
		float4 _Rect;
#endif

		struct v2f
		{
			float2 uv : TEXCOORD0;
			float4 grabPos : TEXCOORD1;
			float depth : TEXCOORD2;
			float camDist : TEXCOORD3;
			float4 pos : SV_POSITION;
			float3 normal : NORMAL;
#ifdef _NORMALMAP
			float3 tangent : TANGENT;
			float3 bitangent : TEXCOORD4;
#endif

#ifdef RECTCLIP
			float2 position : TEXCOORD5;
#endif
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
#ifdef _NORMALMAP
			o.tangent = normalize(mul(unity_ObjectToWorld, float4(v.tangent.xyz, 0)).xyz);
			o.bitangent = normalize(cross(o.normal, o.tangent) * v.tangent.w);
#endif

			// use ComputeGrabScreenPos function from UnityCG.cginc
			// to get the correct texture coordinate
			o.grabPos = ComputeGrabScreenPos(o.pos);

#ifdef RECTCLIP
			o.position = v.vertex.xy;
#endif
			COMPUTE_EYEDEPTH(o.depth);
			o.camDist = length(ObjSpaceViewDir(v.vertex));
			return o;
		}

		sampler2D _GrabTexture;

		half4 frag(v2f i) : SV_Target
		{
#ifdef RECTCLIP
			clip(UnityGet2DClipping(i.position, _Rect) - 0.1);
#endif

			half4 c = half4(0, 0, 0, 0);
			float2 grabUv = i.grabPos.xy / i.grabPos.w;

			float3 tangent, normal, binormal;

#ifdef _NORMALMAP
			tangent = i.tangent;
			binormal = i.bitangent;
#endif
			normal = i.normal;

			grabUv = evrCalculateRefractionCoords(grabUv, i.uv, i.depth, tangent, binormal, normal, i.grabPos);

			return tex2D(_GrabTexture, grabUv);
		}
		ENDCG
	}

	}
}