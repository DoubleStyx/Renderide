// Upgrade NOTE: replaced 'mul(UNITY_MATRIX_MVP,*)' with 'UnityObjectToClipPos(*)'

//Algorithms and shaders based on code from this journal
//http://cgg-journal.com/2008-2/06/index.html

#ifndef UCLA_GAMELAB_WIREFRAME
#define UCLA_GAMELAB_WIREFRAME

#include "UnityCG.cginc"

#define WIREFRAME_V2G(i) float4 posWorld : TEXCOORD##i;
#define WIREFRAME_G2F(i, j) float3 dist : TEXCOORD##i; \
	float4 posWorld : TEXCOORD##j;

// DATA STRUCTURES //
// Vertex to Geometry
struct UCLAGL_v2g
{
	float4	pos		: POSITION;		// vertex position
	float2  uv		: TEXCOORD0;	// vertex uv coordinate
	float3  normal	: NORMAL;
	float3 posObject : TEXCOORD3;
	float4 posWorld : TEXCOORD4;
};

// Geometry to  UCLAGL_fragment
struct UCLAGL_g2f
{
	float4	pos		: POSITION;		// fragment position
	float4 posWorld : TEXCOORD3;
	float2	uv		: TEXCOORD0;	// fragment uv coordinate
	float3  normal	: NORMAL;
	float3  dist	: TEXCOORD1;	// distance to each edge of the triangle
};

float _Thickness;

// Geometry Shader
float3 geom_screenspace(float4 a, float4 b, float4 c)
{
	//points in screen space
	float2 p0 = _ScreenParams.xy * a.xy / a.w;
	float2 p1 = _ScreenParams.xy * b.xy / b.w;
	float2 p2 = _ScreenParams.xy * c.xy / c.w;
	
	//edge vectors
	float2 v0 = p2 - p1;
	float2 v1 = p2 - p0;
	float2 v2 = p1 - p0;

	//area of the triangle
 	float area = abs(v1.x*v2.y - v1.y * v2.x);

		//values based on distance to the edges
	float dist0 = area / length(v0);
	float dist1 = area / length(v1);
	float dist2 = area / length(v2);

	return float3(dist0, dist1, dist2);
}

float3 geom_objectspace(float4 a, float4 b, float4 c)
{
	float3 p0 = a;
	float3 p1 = b;
	float3 p2 = c;

	float3 v0 = p2 - p1;
	float3 v1 = p2 - p0;
	float3 v2 = p1 - p0;

	float angle = acos(dot(v0, v1) / length(v0));
	float area = 0.5f * length(v0) * length(v1) * sin(angle);

	//values based on distance to the edges
	float dist0 = area / length(v0);
	float dist1 = area / length(v1);
	float dist2 = area / length(v2);

	return float3(dist0, dist1, dist2);
}

float line_lerp(float3 dist)
{			
	//find the smallest distance
	float d = min( dist.x, min( dist.y, dist.z));

	float aaf = fwidth(d);
	d = 1 - smoothstep(_Thickness - aaf, _Thickness, d);

	return d;
}


#endif