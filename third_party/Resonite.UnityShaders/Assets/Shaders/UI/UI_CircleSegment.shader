Shader "UI/CircleSegment" {

	Properties
	{
		_FillTint("Fill Tint", Color) = (1,1,1,1)
		_OutlineTint("Outline Tint", Color) = (1,1,1,1)

		_OverlayTint("OverlayTint", Color) = (1,1,1,0.5)

		_SrcBlend("SrcBlend", Float) = 1.0
		_DstBlend("DstBlend", Float) = 0.0
		_ZWrite("ZWrite", Float) = 1.0
		_Cull("Cull", Float) = 2.0
		_ZTest("ZTest", Float) = 2

		_StencilComp("Stencil Comparison", Float) = 8
		_Stencil("Stencil ID", Float) = 0
		_StencilOp("Stencil Operation", Float) = 0
		_StencilWriteMask("Stencil Write Mask", Float) = 255
		_StencilReadMask("Stencil Read Mask", Float) = 255

		_ColorMask("Color Mask", Float) = 15

		_Rect("Rect", Vector) = (0,0,1,1)
	}

		SubShader
	{
			Tags
			{
				"Queue" = "Transparent"
				"IgnoreProjector" = "True"
				"RenderType" = "Transparent"
				"PreviewType" = "Plane"
				"CanUseSpriteAtlas" = "True"
				"DisableBatching" = "True"
			}

		Stencil
		{
			Ref[_Stencil]
			Comp[_StencilComp]
			Pass[_StencilOp]
			ReadMask[_StencilReadMask]
			WriteMask[_StencilWriteMask]
		}

		Lighting Off
		Blend[_SrcBlend][_DstBlend], One One
		BlendOp Add, Max
		ZWrite[_ZWrite]
		Cull[_Cull]
		ZTest[_ZTest]
		Offset[_OffsetFactor],[_OffsetUnits]
		ColorMask[_ColorMask]

		Pass
		{

		CGPROGRAM
		// Physically based Standard lighting model, and enable shadows on all light types
#pragma vertex vert
#pragma fragment frag

#pragma multi_compile_instancing

#pragma multi_compile _ RECTCLIP
#pragma multi_compile _ OVERLAY



#include "UnityCG.cginc"
#include "UnityUI.cginc"

#define PI 3.14159265358979323846264338327

#define DATA \
				float4 fill_color : COLOR;	\
				float4 border_color : TANGENT;	\
				float2 angle_data : TEXCOORD1;	\
				float2 radius_data : TEXCOORD2;	\
				float2 extra_data : TEXCOORD3;

#define COPY_DATA \
				o.fill_color = v.fill_color; \
				o.border_color = v.border_color; \
				o.angle_data = v.angle_data; \
				o.radius_data = v.radius_data; \
				o.extra_data = v.extra_data;

#define ANGLE_OFFSET(n) n.angle_data.x
#define ANGLE_LENGTH(n) n.angle_data.y
#define RADIUS_START(n) n.radius_data.x
#define RADIUS_END(n) n.radius_data.y

#define FILL_COLOR(n) n.fill_color
#define BORDER_COLOR(n) n.border_color

#define BORDER_SIZE(n) n.extra_data.x
#define CORNER_RADIUS(n) n.extra_data.y

	uniform float4 _FillTint;
	uniform float4 _OutlineTint;

	struct a2v
	{
		float4 vertex : POSITION;
		float2 uv : TEXCOORD0;

		UNITY_VERTEX_INPUT_INSTANCE_ID

		DATA
	};

	struct v2f
	{
		float4 pos : POSITION;
		float2 uv : TEXCOORD0;

		UNITY_VERTEX_OUTPUT_STEREO

			float2 position : TEXCOORD4;

		DATA

			float4 projPos : TEXCOORD5;
	};

float4 _Rect;

	float4 _OverlayTint;
	UNITY_DECLARE_DEPTH_TEXTURE(_CameraDepthTexture);

	////////////////////////////////

	float positive(float val)
	{
		return (val < 0) ? 0 : val;
	}

	float negative(float val)
	{
		if (val > 0)
			return 0;

		return val;
	}

	float angle_compensation(float angleOffset, float angleLength)
	{
		return PI + angleLength * -0.5;
	}

	v2f vert(a2v v)
	{
		v2f o;

		UNITY_SETUP_INSTANCE_ID(v);
		UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);

		float angle_dif = ANGLE_OFFSET(v) - angle_compensation(ANGLE_OFFSET(v), ANGLE_LENGTH(v));

		float2x2 rot = float2x2(
			cos(angle_dif), -sin(angle_dif),
			sin(angle_dif), cos(angle_dif)
			);

		o.pos = UnityObjectToClipPos(v.vertex);
		o.uv = mul(rot, v.uv);	// apply the offset

		o.position = v.vertex.xy;

		COPY_DATA

		o.projPos = ComputeScreenPos(o.pos);
		COMPUTE_EYEDEPTH(o.projPos.z);

		return o;
	}

	float compute_strength(float angle_dist, float radius_dist, float corner_radius)
	{
		float dist;
	
		if(angle_dist < corner_radius && radius_dist < corner_radius)
		{
			float2 xy = float2(corner_radius - radius_dist, corner_radius - angle_dist);
			dist = corner_radius - length(xy);
		}
		else
			dist = min(angle_dist, radius_dist);
	
		//if(dist <= 0) return 0; else return 1;
	
		float width = fwidth(dist);
	
		return saturate(dist / width);
	}

	fixed4 frag(v2f i) : COLOR
	{
		//UNITY_SETUP_INSTANCE_ID(i);
		UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);

#ifdef RECTCLIP
			clip(UnityGet2DClipping(i.position, _Rect) - 0.1);
#endif

		// compute the angle from the center
		float angle = atan2(-i.uv.y, i.uv.x) + PI;
		float radius = length(i.uv);
	
		angle -= angle_compensation(ANGLE_OFFSET(i), ANGLE_LENGTH(i));

		float angle_end = ANGLE_LENGTH(i) - angle;

		float angle_dist = min(angle, angle_end) * radius;
	
		float radius_from_dist = radius - RADIUS_START(i);
		float radius_to_dist = RADIUS_END(i) - radius;

		float radius_dist = min(radius_from_dist, radius_to_dist);
	
		float corner_radius = CORNER_RADIUS(i);
	
		float remainingAngleLength = (PI * 2 - ANGLE_LENGTH(i)) * RADIUS_START(i);
	
		corner_radius = min(corner_radius, remainingAngleLength);
	
		float border_size = min(BORDER_SIZE(i), remainingAngleLength);
	
		angle_dist += max(0, BORDER_SIZE(i) - border_size);
	
		float borderLerp = compute_strength(angle_dist, radius_dist, corner_radius);
		float fillLerp = compute_strength(angle_dist - border_size, radius_dist - BORDER_SIZE(i), corner_radius);
	
		if (borderLerp <= 0)
			discard;
	
		fixed4 c;
	
		fixed4 border_c = BORDER_COLOR(i) * _OutlineTint;
		border_c.a *= borderLerp;
	
		c = lerp(border_c, FILL_COLOR(i) * _FillTint, fillLerp);
	
#ifdef OVERLAY
			float sceneZ = LinearEyeDepth(SAMPLE_DEPTH_TEXTURE_PROJ(_CameraDepthTexture, UNITY_PROJ_COORD(i.projPos)));
			float partZ = i.projPos.z;

			if (partZ > sceneZ)
				c *= _OverlayTint;
#endif

		return c;
	}

	ENDCG
}
	}
}
