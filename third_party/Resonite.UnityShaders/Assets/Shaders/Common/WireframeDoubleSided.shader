Shader "WireframeDoubleSided"
{
	Properties
	{
		_LineColor("Line Color", Color) = (1,1,1,1)
		_FillColor("Fill Color", Color) = (1,1,1,0)

		_InnerLineColor("Inner Line Color", Color) = (1,1,1,1)
		_InnerFillColor("Inner Fill Color", Color) = (1,1,1,0)

		_LineFarColor("Line Far Color", Color) = (1,1,1,0)
		_FillFarColor("Fill Far Color", Color) = (0,0,0,0)

		_InnerLineFarColor("Line Far Color", Color) = (1,1,1,0)
		_InnerFillFarColor("Fill Far Color", Color) = (0,0,0,0)

		_Exp("Exp", Float) = 1

		_MainTex("Main Texture", 2D) = "white" {}

		_Thickness("Thickness", Float) = 1

		_ZWrite("ZWrite", Float) = 0
	}
		SubShader
	{
		Tags{ "RenderType" = "Transparent" "Queue" = "Transparent" }

		Pass
	{
			Blend SrcAlpha OneMinusSrcAlpha
			ZWrite[_ZWrite]
			Cull Front
			Offset[_OffsetFactor],[_OffsetUnits]
			LOD 200

		CGPROGRAM


#pragma target 5.0

#include "UnityCG.cginc"
#include "WireframeShared.cginc"
#pragma vertex vert
#pragma fragment frag
#pragma geometry geom

#pragma multi_compile_instancing

#pragma multi_compile _ _SCREENSPACE
#pragma multi_compile _ _FRESNEL

	sampler2D _MainTex;
	float4 _MainTex_ST;

	float4 _InnerLineColor;
	float4 _InnerFillColor;

#ifdef _FRESNEL
	float4 _InnerLineFarColor;
	float4 _InnerFillFarColor;
	float _Exp;
#endif

	struct appdata
	{
		float4 vertex : POSITION;
		float3 normal : NORMAL;
		float4 texcoord : TEXCOORD0;

		UNITY_VERTEX_INPUT_INSTANCE_ID
	};

	struct v2g
	{
		float4	pos		: POSITION;
		float2  uv		: TEXCOORD0;
		float3  normal	: NORMAL;
		float4  posWorld : TEXCOORD1;

		UNITY_VERTEX_OUTPUT_STEREO
	};

	struct g2f
	{
		float4	pos		: POSITION;
		float2	uv		: TEXCOORD0;
		float3  normal	: NORMAL;

		float3 dist		: TEXCOORD1;

		float4 posWorld : TEXCOORD2;

		UNITY_VERTEX_OUTPUT_STEREO
	};

	/// Vertex Shader
	v2g vert(appdata v)
	{
		v2g output;

		UNITY_SETUP_INSTANCE_ID(v);
		UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);

		output.pos = UnityObjectToClipPos(v.vertex);
		output.uv = TRANSFORM_TEX(v.texcoord, _MainTex);
		output.normal = UnityObjectToWorldNormal(v.normal);
		output.posWorld = mul(unity_ObjectToWorld, v.vertex);

		return output;
	}

	// Geometry Shader
	[maxvertexcount(3)]
	void geom(triangle v2g p[3], inout TriangleStream<g2f> triStream)
	{
		float3 dist;
#ifdef _SCREENSPACE
		dist = geom_screenspace(p[0].pos, p[1].pos, p[2].pos);
#else
		dist = geom_objectspace(p[0].posWorld, p[1].posWorld, p[2].posWorld);
#endif

		g2f pIn;

		//add the first point
		pIn.pos = p[0].pos;
		pIn.uv = p[0].uv;
		pIn.dist = float3(dist.x, 0, 0);
		pIn.posWorld = p[0].posWorld;
		pIn.normal = p[0].normal;
		UNITY_TRANSFER_VERTEX_OUTPUT_STEREO(p[0], pIn);
		triStream.Append(pIn);

		//add the second point
		pIn.pos = p[1].pos;
		pIn.uv = p[1].uv;
		pIn.dist = float3(0, dist.y, 0);
		pIn.posWorld = p[1].posWorld;
		pIn.normal = p[1].normal;
		UNITY_TRANSFER_VERTEX_OUTPUT_STEREO(p[1], pIn);
		triStream.Append(pIn);

		//add the third point
		pIn.pos = p[2].pos;
		pIn.uv = p[2].uv;
		pIn.dist = float3(0, 0, dist.z);
		pIn.posWorld = p[2].posWorld;
		pIn.normal = p[2].normal;
		UNITY_TRANSFER_VERTEX_OUTPUT_STEREO(p[2], pIn);
		triStream.Append(pIn);
	}

	// Fragment Shader
	float4 frag(g2f i) : COLOR
	{
		//UNITY_SETUP_INSTANCE_ID(i);
		UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);

		float l = line_lerp(i.dist);

#ifdef _FRESNEL
	float3 viewDir = normalize(_WorldSpaceCameraPos.xyz - i.posWorld.xyz);
	float fresnel = pow(1 - abs(dot(i.normal, viewDir)), _Exp);

	float4 fillColor = lerp(_InnerFillColor, _InnerFillFarColor, fresnel);
	float4 lineColor = lerp(_InnerLineColor, _InnerLineFarColor, fresnel);
#else
	float4 fillColor = _InnerFillColor;
	float4 lineColor = _InnerLineColor;
#endif

	float4 c = lerp(fillColor, lineColor, l) * tex2D(_MainTex, i.uv);

	return c;
	}

		ENDCG
	}

			Pass
	{
		Blend SrcAlpha OneMinusSrcAlpha
		ZWrite[_ZWrite]
		Cull Back
		Offset[_OffsetFactor],[_OffsetUnits]
		LOD 200

		CGPROGRAM


#pragma target 5.0
#include "UnityCG.cginc"
#include "WireframeShared.cginc"
#pragma vertex vert
#pragma fragment frag
#pragma geometry geom

#pragma multi_compile_instancing

#pragma multi_compile _ _SCREENSPACE
#pragma multi_compile _ _FRESNEL

	sampler2D _MainTex;
	float4 _MainTex_ST;

	float4 _LineColor;
	float4 _FillColor;

#ifdef _FRESNEL
	float4 _LineFarColor;
	float4 _FillFarColor;
	float _Exp;
#endif

	struct appdata
	{
		float4 vertex : POSITION;
		float3 normal : NORMAL;
		float4 texcoord : TEXCOORD0;

		UNITY_VERTEX_INPUT_INSTANCE_ID
	};

	struct v2g
	{
		float4	pos		: POSITION;
		float2  uv		: TEXCOORD0;
		float3  normal	: NORMAL;
		float4  posWorld : TEXCOORD1;

		UNITY_VERTEX_OUTPUT_STEREO
	};

	struct g2f
	{
		float4	pos		: POSITION;
		float2	uv		: TEXCOORD0;
		float3  normal	: NORMAL;

		float3 dist		: TEXCOORD1;

		float4 posWorld : TEXCOORD2;

		UNITY_VERTEX_OUTPUT_STEREO
	};


	// Vertex Shader
	v2g vert(appdata v)
	{
		v2g output;

		UNITY_SETUP_INSTANCE_ID(v);
		UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);

		output.pos = UnityObjectToClipPos(v.vertex);
		output.uv = TRANSFORM_TEX(v.texcoord, _MainTex);
		output.normal = UnityObjectToWorldNormal(v.normal);
		output.posWorld = mul(unity_ObjectToWorld, v.vertex);

		return output;
	}

	// Geometry Shader
	[maxvertexcount(3)]
	void geom(triangle v2g p[3], inout TriangleStream<g2f> triStream)
	{
		float3 dist;
#ifdef _SCREENSPACE
		dist = geom_screenspace(p[0].pos, p[1].pos, p[2].pos);
#else
		dist = geom_objectspace(p[0].posWorld, p[1].posWorld, p[2].posWorld);
#endif

		g2f pIn;

		//add the first point
		pIn.pos = p[0].pos;
		pIn.uv = p[0].uv;
		pIn.dist = float3(dist.x, 0, 0);
		pIn.posWorld = p[0].posWorld;
		pIn.normal = p[0].normal;
		UNITY_TRANSFER_VERTEX_OUTPUT_STEREO(p[0], pIn);
		triStream.Append(pIn);

		//add the second point
		pIn.pos = p[1].pos;
		pIn.uv = p[1].uv;
		pIn.dist = float3(0, dist.y, 0);
		pIn.posWorld = p[1].posWorld;
		pIn.normal = p[1].normal;
		UNITY_TRANSFER_VERTEX_OUTPUT_STEREO(p[1], pIn);
		triStream.Append(pIn);

		//add the third point
		pIn.pos = p[2].pos;
		pIn.uv = p[2].uv;
		pIn.dist = float3(0, 0, dist.z);
		pIn.posWorld = p[2].posWorld;
		pIn.normal = p[2].normal;
		UNITY_TRANSFER_VERTEX_OUTPUT_STEREO(p[2], pIn);
		triStream.Append(pIn);
	}

	// Fragment Shader
	float4 frag(g2f i) : COLOR
	{
		//UNITY_SETUP_INSTANCE_ID(i);
		UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);

		float l = line_lerp(i.dist);

#ifdef _FRESNEL
	float3 viewDir = normalize(_WorldSpaceCameraPos.xyz - i.posWorld.xyz);
	float fresnel = pow(1 - abs(dot(i.normal, viewDir)), _Exp);

	float4 fillColor = lerp(_FillColor, _FillFarColor, fresnel);
	float4 lineColor = lerp(_LineColor, _LineFarColor, fresnel);
#else
	float4 fillColor = _FillColor;
	float4 lineColor = _LineColor;
#endif

	float4 c = lerp(fillColor, lineColor, l) * tex2D(_MainTex, i.uv);

	return c;
	}

		ENDCG
	}


	}
}
