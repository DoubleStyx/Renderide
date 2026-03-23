#define BILLBOARD_GEOMETRY_SHADER [maxvertexcount(4)] \
	void geom(point v2g p[1], inout TriangleStream<g2f> triStream) { geom_inst(p, triStream); }

float2 billboard_size(v2g p);
void rotate_billboard(v2g p, inout float3 m0, inout float3 m1, inout float3 m2);
void setup_vertex(v2g p, inout g2f v);

float3 _nonJitteredWorldSpaceCameraPos;

void geom_inst(point v2g p[1], inout TriangleStream<g2f> triStream)
{
	UNITY_SETUP_INSTANCE_ID(p[0]);

	float4 worldPos = mul(unity_ObjectToWorld, p[0].pos);
	float3 forward = -normalize(_nonJitteredWorldSpaceCameraPos - worldPos.xyz);

	float3 m2 = forward;
	float3 m0 = normalize(cross(float3(0, 1, 0), m2));
	float3 m1 = normalize(cross(m2, m0));

	rotate_billboard(p[0], m0, m1, m2);

	// Deffer this so it's only calculated once? How to do it neatly?
	float4x4 m = unity_ObjectToWorld;
	float scale = sqrt(m[0][0] * m[0][0] + m[0][1] * m[0][1] + m[0][2] * m[0][2]);

	float2 size = billboard_size(p[0]) * scale;
	m0 *= size.x;
	m1 *= size.y;

	float4 pos0 = worldPos + float4((-m0 - m1), 0);
	float4 pos1 = worldPos + float4((+m0 - m1), 0);
	float4 pos2 = worldPos + float4((+m0 + m1), 0);
	float4 pos3 = worldPos + float4((-m0 + m1), 0);

	// TODO!!! can possibly transform the vectors instead and compute in projection space?
	// Would save two matrix transformations
	pos0 = mul(UNITY_MATRIX_VP, pos0);
	pos1 = mul(UNITY_MATRIX_VP, pos1);
	pos2 = mul(UNITY_MATRIX_VP, pos2);
	pos3 = mul(UNITY_MATRIX_VP, pos3);

	g2f o;

	UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);

	o.pos = pos1;
	o.uv = float2(1, 0);
	setup_vertex(p, o);
	triStream.Append(o);

	o.pos = pos0;
	o.uv = float2(0, 0);
	setup_vertex(p, o);
	triStream.Append(o);

	o.pos = pos2;
	o.uv = float2(1, 1);
	setup_vertex(p, o);
	triStream.Append(o);

	o.pos = pos3;
	o.uv = float2(0, 1);
	setup_vertex(p, o);
	triStream.Append(o);
}

void rotate_by_angle(float angle, inout float3 m0, inout float3 m1, inout float3 m2)
{
	float3x3 rot = float3x3(m0, m1, m2);

	float3x3 r = float3x3(
		float3(cos(angle), -sin(angle), 0),
		float3(sin(angle), cos(angle), 0),
		float3(0, 0, 1));

	rot = mul(r, rot);

	m0 = rot._m00_m01_m02;
	m1 = rot._m10_m11_m12;
}