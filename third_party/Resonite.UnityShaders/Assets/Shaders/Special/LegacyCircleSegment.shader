// Upgrade NOTE: replaced 'mul(UNITY_MATRIX_MVP,*)' with 'UnityObjectToClipPos(*)'

Shader "Legacy/CircleSegment" {

	SubShader{
		Tags{ "RenderType" = "Transparent" "Queue" = "Transparent" }
		Blend SrcAlpha OneMinusSrcAlpha
		ZWrite Off
		Cull Off
		LOD 200

		Pass
		{

		CGPROGRAM
		// Physically based Standard lighting model, and enable shadows on all light types
#pragma vertex vert
#pragma fragment frag

#pragma multi_compile_instancing

		// Use shader model 3.0 target, to get nicer looking lighting
#pragma target 3.0


#include "UnityCG.cginc"

#define PI 3.14159265358979323846264338327

#define DATA \
				float4 fill_color : COLOR;	\
				float4 border_color : TANGENT;	\
				float2 angle_data : TEXCOORD1;	\
				float2 radius_data : TEXCOORD2;	\
				float2 extra_data : TEXCOORD3;

#define COPY_DATA \
				o.fill_color = i.fill_color; \
				o.border_color = i.border_color; \
				o.angle_data = i.angle_data; \
				o.radius_data = i.radius_data; \
				o.extra_data = i.extra_data;

#define ANGLE_OFFSET i.angle_data.x
#define ANGLE_LENGTH i.angle_data.y
#define RADIUS_START i.radius_data.x
#define RADIUS_END i.radius_data.y

#define FILL_COLOR i.fill_color
#define BORDER_COLOR i.border_color

#define BORDER_SIZE i.extra_data.x

		struct a2v
	{
		float4 pos : POSITION;
		float2 uv : TEXCOORD0;

		UNITY_VERTEX_INPUT_INSTANCE_ID

		DATA
	};

	struct v2f
	{
		float4 pos : POSITION;
		float2 uv : TEXCOORD0;

		UNITY_VERTEX_OUTPUT_STEREO

		DATA
	};

	////////////////////////////////

	float positive(float val)
	{
		if (val < 0)
			return 0;
		return val;
	}

	float negative(float val)
	{
		if (val > 0)
			return 0;
		return val;
	}

	v2f vert(a2v i)
	{
		v2f o;

		UNITY_SETUP_INSTANCE_ID(i);
		UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);

		float angle_dif = ANGLE_OFFSET;

		float2x2 rot = float2x2(
			cos(angle_dif), -sin(angle_dif),
			sin(angle_dif), cos(angle_dif)
			);

		o.pos = UnityObjectToClipPos(i.pos);
		o.uv = mul(rot, i.uv);	// apply the offset

		COPY_DATA

			return o;
	}

	fixed4 frag(v2f i) : COLOR
	{
		//UNITY_SETUP_INSTANCE_ID(i);
		UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);

	// compute the angle from the center
	float angle = atan2(-i.uv.y, i.uv.x) + PI;
float radius = length(i.uv);

float angularBorderSize = BORDER_SIZE / radius;

float angle_end = ANGLE_LENGTH - angle;
float overflow = positive(((ANGLE_LENGTH + angularBorderSize) - PI * 2));

float angle_dist = positive(min(angle + overflow, angle_end + overflow));

float radius_from_dist = radius - RADIUS_START;
float radius_to_dist = RADIUS_END - radius;

float radius_dist = positive(min(radius_from_dist, radius_to_dist));

float dist = min(radius_dist, angle_dist);

if (dist <= 0)
	return 0;
else
{
	if (radius_dist < BORDER_SIZE || angle_dist < angularBorderSize)
		return BORDER_COLOR;
	else
		return FILL_COLOR;
}
}

	ENDCG
}
	}
}
