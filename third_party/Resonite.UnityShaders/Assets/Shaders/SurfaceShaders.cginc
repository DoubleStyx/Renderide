#ifndef EVRSURFACESHADERS
#define EVRSURFACESHADERS
#include "Common.cginc"
#include "UnityPBSLighting.cginc"
#include "UnityStandardUtils.cginc"

struct Input
{
	float2 uv_MainTex;

#if defined(_LERP)
	float2 uv_MainTex1;
	float2 uv_LerpTex;
#endif
	
	#ifdef _COLORMASK
	float2 uv_ColorMask;
	#endif

	#ifdef _COLORMAP
	float2 uv_ColorMap;
	#endif

	#ifdef UV_OFFSET
	float2 uv_UVOffsetMap;
	#endif
	
	#ifdef _RIM
	float3 viewDir;
	#endif
	
	#ifdef _DUALSIDED
	float facing : FACE;
	#endif 
	
	#ifdef VCOLOR
	half4 vcolor : COLOR;
	#endif

	#ifdef _DISTANCE_LERP
	float3 emission;
	#endif

	#ifdef _INTERSECT
	float4 screenPos;
	float eyeDepth : TEXCOORD1;
	#endif

	#ifdef _SLICE
	float3 pos;
	#endif

	#ifdef _DETAIL_ALBEDOTEX
	float2 uv_DetailAlbedoMap;
	#endif
	#ifdef _DETAIL_NORMALMAP
	float2 uv_DetailNormalMap;
	#endif

	#ifdef _TRIPLANAR
	
#ifdef _WORLDSPACE
		float3 worldPos;
#else
		float3 objPos;
#endif
		float3 worldNormal;
		INTERNAL_DATA
	#endif
};

struct TriplanarBlend
{
	float2 uvX;
	float2 uvY;
	float2 uvZ;
	half3 triblend;
	half3 axisSign;
};

half3 blend_rnm(half3 n1, half3 n2)
{
	n1.z += 1;
	n2.xy = -n2.xy;

	return n1 * dot(n1, n2) / n1.z - n2;
}

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

void vertDisplace(inout appdata_full v)
{
	#ifdef VERTEX_OFFSET
	float2 uv = v.texcoord;

	#ifdef OBJECT_POS_OFFSET
	float2 position = unity_ObjectToWorld._m03_m23;
	#elif VERTEX_POS_OFFSET
	float2 position = mul(unity_ObjectToWorld, float4(v.vertex.xyz, 1)).xz;
	#endif

	#if defined(OBJECT_POS_OFFSET) || defined(VERTEX_POS_OFFSET)
	float2 uvOffset = tex2Dlod(_PositionOffsetMap, float4(TRANSFORM_TEX(position, _PositionOffsetMap), 0, 0)).xy;
	uvOffset *= _PositionOffsetMagnitude.xy;
	uv += uvOffset;
	#endif

	float vertOffset = tex2Dlod(_VertexOffsetMap, float4(TRANSFORM_TEX(uv, _VertexOffsetMap), 0, 0)).x;
	vertOffset = vertOffset * _VertexOffsetMagnitude + _VertexOffsetBias;

	v.vertex.xyz += v.normal.xyz * vertOffset;
	#endif
}

half GenerateLerp(Input IN)
{
	half l = 0;
	
	#ifdef _LERP
	l = UNITY_ACCESS_INSTANCED_PROP(EVR_INST_BUFFER, _Lerp);
	
	#ifdef _LERPTEX
#ifdef _MULTI_VALUES
	l *= tex2D(_LerpTex, IN.uv_LerpTex);
#else
	l =  tex2D(_LerpTex, IN.uv_LerpTex);
#endif
	#endif
	#endif
	return l;
}

fixed4 GenerateAlbedo(Input IN, half l, fixed4 mask)
{
	fixed4 c = UNITY_ACCESS_INSTANCED_PROP(EVR_INST_BUFFER, _Color);

	#ifdef _COLORMAP
	c *= UNITY_SAMPLE_TEX2D(_MainTex, IN.uv_MainTex) * mask.x;
	c += UNITY_SAMPLE_TEX2D(_MainTex1, IN.uv_MainTex) * UNITY_ACCESS_INSTANCED_PROP(EVR_INST_BUFFER, _Color1) * mask.y;
	c += UNITY_SAMPLE_TEX2D(_MainTex2, IN.uv_MainTex) * UNITY_ACCESS_INSTANCED_PROP(EVR_INST_BUFFER, _Color2) * mask.z;
	c += UNITY_SAMPLE_TEX2D(_MainTex3, IN.uv_MainTex) * UNITY_ACCESS_INSTANCED_PROP(EVR_INST_BUFFER, _Color3) * mask.w;
	#endif
	
	#ifdef _COLORMASK
	c = c * mask.r +
		UNITY_ACCESS_INSTANCED_PROP(EVR_INST_BUFFER, _Color1) * mask.g +
		UNITY_ACCESS_INSTANCED_PROP(EVR_INST_BUFFER, _Color2) * mask.b +
		UNITY_ACCESS_INSTANCED_PROP(EVR_INST_BUFFER, _Color3) * mask.a;

	#endif
	
	#if defined(_ALBEDOTEX) || defined(SAMPLER_REQUIRED) && !defined(_COLORMAP)
	c *= UNITY_SAMPLE_TEX2D(_MainTex, IN.uv_MainTex);
	#endif
	
	#ifdef _LERP
	fixed4 c1 = UNITY_ACCESS_INSTANCED_PROP(EVR_INST_BUFFER, _Color1);
	
	#if defined(_ALBEDOTEX) || defined(SAMPLER_REQUIRED) 
	c1 *= UNITY_SAMPLE_TEX2D(_MainTex1, IN.uv_MainTex1);
	#endif

	c = lerp(c, c1, l);
	#endif
	
	#ifdef _ALPHACLIP
	clip(c.a - _AlphaClip);
#endif
	
	return c;
}

half3 GenerateNormal(Input IN, half l, fixed4 map)
{
	half3 n = half3(0, 0, 1);

	#ifdef _NORMALMAP
	n = UnpackScaleNormal(UNITY_SAMPLE_TEX2D(_NormalMap, IN.uv_MainTex), UNITY_ACCESS_INSTANCED_PROP(EVR_INST_BUFFER, _NormalScale));

	#ifdef _LERP
	n = lerp(n, UnpackScaleNormal(UNITY_SAMPLE_TEX2D(_NormalMap1, IN.uv_MainTex1), UNITY_ACCESS_INSTANCED_PROP(EVR_INST_BUFFER, _NormalScale1)), l);
	#endif
	#endif

	#ifdef _PACKED_NORMALMAP
	fixed4 n01 = UNITY_SAMPLE_TEX2D(_PackedNormalMap01, IN.uv_MainTex);
	fixed4 n23 = UNITY_SAMPLE_TEX2D(_PackedNormalMap23, IN.uv_MainTex);

	fixed3 n0;
	fixed3 n1;
	fixed3 n2;
	fixed3 n3;

	n0.xy = (n01.xy * 2 - 1) * UNITY_ACCESS_INSTANCED_PROP(EVR_INST_BUFFER, _NormalScale);
	n1.xy = (n01.zw * 2 - 1) * UNITY_ACCESS_INSTANCED_PROP(EVR_INST_BUFFER, _NormalScale1);
	n2.xy = (n23.xy * 2 - 1) * UNITY_ACCESS_INSTANCED_PROP(EVR_INST_BUFFER, _NormalScale2);
	n3.xy = (n23.zw * 2 - 1) * UNITY_ACCESS_INSTANCED_PROP(EVR_INST_BUFFER, _NormalScale3);

	n0.z = sqrt(1 - saturate(dot(n0.xy, n0.xy)));
	n1.z = sqrt(1 - saturate(dot(n1.xy, n1.xy)));
	n2.z = sqrt(1 - saturate(dot(n2.xy, n2.xy)));
	n3.z = sqrt(1 - saturate(dot(n3.xy, n3.xy)));

	n = n0 * map.x + n1 * map.y + n2 * map.z + n3 * map.w;
	n = normalize(n);
	#endif

	#ifdef _DUALSIDED
	if (IN.facing < 0.5)
		n.z *= -1;
	#endif
	
	return n;
}

fixed GenerateOcclusion(Input IN, half l)
{
	fixed o = 1;

	#ifdef _OCCLUSION
	o = UNITY_SAMPLE_TEX2D_SAMPLER(_OcclusionMap, _MainTex, IN.uv_MainTex);

	#ifdef _LERP
	o = lerp(o, UNITY_SAMPLE_TEX2D_SAMPLER(_OcclusionMap1, _MainTex, IN.uv_MainTex1), l);
	#endif
	#endif
	
	return o;
}

half3 GenerateEmission(Input IN, half l, fixed4 mask)
{
	half3 e = UNITY_ACCESS_INSTANCED_PROP(EVR_INST_BUFFER, _EmissionColor);

	#ifdef _COLORMASK
	e = e * mask.r +
		UNITY_ACCESS_INSTANCED_PROP(EVR_INST_BUFFER, _EmissionColor1) * mask.g +
		UNITY_ACCESS_INSTANCED_PROP(EVR_INST_BUFFER, _EmissionColor2) * mask.b +
		UNITY_ACCESS_INSTANCED_PROP(EVR_INST_BUFFER, _EmissionColor3) * mask.a;

	#endif
	
	#ifdef _EMISSIONTEX
	e *= UNITY_SAMPLE_TEX2D_SAMPLER(_EmissionMap, _MainTex, IN.uv_MainTex);
	#endif

	#ifdef _COLORMAP
	e *= mask.r;
	half3 e1 = UNITY_ACCESS_INSTANCED_PROP(EVR_INST_BUFFER, _EmissionColor1) * mask.g;
	half3 e2 = UNITY_ACCESS_INSTANCED_PROP(EVR_INST_BUFFER, _EmissionColor2) * mask.b;
	half3 e3 = UNITY_ACCESS_INSTANCED_PROP(EVR_INST_BUFFER, _EmissionColor3) * mask.a;

	#ifdef _EMISSIONTEX
	e1 *= UNITY_SAMPLE_TEX2D_SAMPLER(_EmissionMap1, _MainTex1, IN.uv_MainTex);
	e2 *= UNITY_SAMPLE_TEX2D_SAMPLER(_EmissionMap2, _MainTex2, IN.uv_MainTex);
	e3 *= UNITY_SAMPLE_TEX2D_SAMPLER(_EmissionMap3, _MainTex3, IN.uv_MainTex);
	#elif defined(_PACKED_EMISSIONTEX)
	float4 packedEmission = UNITY_SAMPLE_TEX2D(_PackedEmissionMap, IN.uv_MainTex);
	e *= packedEmission.x;
	e1 *= packedEmission.y;
	e2 *= packedEmission.z;
	e3 *= packedEmission.w;
	
	#endif
	e = e + e1 + e2 + e3;
	#endif

	#ifdef _LERP
	half3 e1 = UNITY_ACCESS_INSTANCED_PROP(EVR_INST_BUFFER, _EmissionColor1);
	#ifdef _EMISSIONTEX
	e1 *= UNITY_SAMPLE_TEX2D_SAMPLER(_EmissionMap1, _MainTex, IN.uv_MainTex1);
	#endif
	e = lerp(e, e1, l);
	#endif
	
	return e;
}

half2 GenerateMetallicSmoothness(Input IN, half l, fixed4 map)
{
	half2 ms = half2(UNITY_ACCESS_INSTANCED_PROP(EVR_INST_BUFFER, _Metallic),
					UNITY_ACCESS_INSTANCED_PROP(EVR_INST_BUFFER, _Glossiness));

	#ifdef _COLORMAP
	half2 m1 = half2(UNITY_ACCESS_INSTANCED_PROP(EVR_INST_BUFFER, _Metallic1),
					UNITY_ACCESS_INSTANCED_PROP(EVR_INST_BUFFER, _Glossiness1));
	
	half2 m2 = half2(UNITY_ACCESS_INSTANCED_PROP(EVR_INST_BUFFER, _Metallic2),
					UNITY_ACCESS_INSTANCED_PROP(EVR_INST_BUFFER, _Glossiness2));
	
	half2 m3 = half2(UNITY_ACCESS_INSTANCED_PROP(EVR_INST_BUFFER, _Metallic3),
					UNITY_ACCESS_INSTANCED_PROP(EVR_INST_BUFFER, _Glossiness3));
	#ifdef _METALLICMAP
	fixed4 m01 = UNITY_SAMPLE_TEX2D(_MetallicGloss01, IN.uv_MainTex);
	fixed4 m23 = UNITY_SAMPLE_TEX2D(_MetallicGloss23, IN.uv_MainTex);
#ifdef _MULTI_VALUES
	ms *= m01.xy;
	m1 *= m01.zw;
	m2 *= m23.xy;
	m3 *= m23.zw;
#else
	ms = m01.xy;
	m1 = m01.zw;
	m2 = m23.xy;
	m3 = m23.zw;
#endif
	#endif

	ms = ms * map.x + m1 * map.y + m2 * map.z + m3 * map.w;

	#else
	
	#ifdef _METALLICMAP
	half4 mrs = UNITY_SAMPLE_TEX2D_SAMPLER(_MetallicMap, _MainTex, IN.uv_MainTex);
#ifdef _MULTI_VALUES
	ms.x *= mrs.x;
	ms.y *= mrs.a;
#else
	ms.x = mrs.x;
	ms.y = mrs.a;
#endif
	#endif

	#ifdef _LERP
	half2 ms1 = half2(UNITY_ACCESS_INSTANCED_PROP(EVR_INST_BUFFER, _Metallic1),
						UNITY_ACCESS_INSTANCED_PROP(EVR_INST_BUFFER, _Glossiness1));
	#ifdef _METALLICMAP
	half4 mrs1 = UNITY_SAMPLE_TEX2D_SAMPLER(_MetallicMap1, _MainTex, IN.uv_MainTex1);
#ifdef _MULTI_VALUES
	ms1.x *= mrs1.x;
	ms1.y *= mrs1.a;
#else
	ms1.x = mrs1.x;
	ms1.y = mrs1.a;
#endif
	#endif
	ms = lerp(ms, ms1, l);
	#endif
	#endif
	
	return ms;
}

half4 GenerateSpecular(Input IN, half l, fixed4 map)
{
	half4 s = UNITY_ACCESS_INSTANCED_PROP(EVR_INST_BUFFER, _SpecularColor);
	// Colormap
	#ifdef _COLORMAP
	fixed4 s1 = UNITY_ACCESS_INSTANCED_PROP(EVR_INST_BUFFER, _SpecularColor1);
	fixed4 s2 = UNITY_ACCESS_INSTANCED_PROP(EVR_INST_BUFFER, _SpecularColor2);
	fixed4 s3 = UNITY_ACCESS_INSTANCED_PROP(EVR_INST_BUFFER, _SpecularColor3);
	// Colormap + specular map
	#ifdef _SPECULARMAP
	// Colormap + specular map + multivalue
#ifdef _MULTI_VALUES
	s *= UNITY_SAMPLE_TEX2D_SAMPLER(_SpecularMap, _MainTex, IN.uv_MainTex);
	s1 *= UNITY_SAMPLE_TEX2D_SAMPLER(_SpecularMap1, _MainTex1, IN.uv_MainTex);
	s2 *= UNITY_SAMPLE_TEX2D_SAMPLER(_SpecularMap2, _MainTex2, IN.uv_MainTex);
	s3 *= UNITY_SAMPLE_TEX2D_SAMPLER(_SpecularMap3, _MainTex3, IN.uv_MainTex);
#else
	s = UNITY_SAMPLE_TEX2D_SAMPLER(_SpecularMap, _MainTex, IN.uv_MainTex);
	s1 = UNITY_SAMPLE_TEX2D_SAMPLER(_SpecularMap1, _MainTex1, IN.uv_MainTex);
	s2 = UNITY_SAMPLE_TEX2D_SAMPLER(_SpecularMap2, _MainTex2, IN.uv_MainTex);
	s3 = UNITY_SAMPLE_TEX2D_SAMPLER(_SpecularMap3, _MainTex3, IN.uv_MainTex);
#endif
	#endif
	s = s * map.x + s1 * map.y + s2 * map.z + s3 * map.w;
	#else

	// Specular map
	#ifdef _SPECULARMAP
	// Specular map + multivalue
#ifdef _MULTI_VALUES
	s = UNITY_SAMPLE_TEX2D_SAMPLER(_SpecularMap, _MainTex, IN.uv_MainTex) * s;
#else
	s = UNITY_SAMPLE_TEX2D_SAMPLER(_SpecularMap, _MainTex, IN.uv_MainTex);
#endif
	#endif

	#ifdef _LERP
	half4 s1 = UNITY_ACCESS_INSTANCED_PROP(EVR_INST_BUFFER, _SpecularColor1);
	#ifdef _SPECULARMAP
#ifdef _MULTI_VALUES
	s1 *= UNITY_SAMPLE_TEX2D_SAMPLER(_SpecularMap1, _MainTex1, IN.uv_MainTex1);
#else
	s1 = UNITY_SAMPLE_TEX2D_SAMPLER(_SpecularMap1, _MainTex1, IN.uv_MainTex1);
#endif
	#endif
	s = lerp(s, s1, l);
	#endif
	#endif
	return s;
}

void surf_specular (Input IN, inout SurfaceOutputStandardSpecular o)
{
	UNITY_INITIALIZE_OUTPUT(SurfaceOutputStandardSpecular, o);
	fixed4 mask = 0;
	
	#ifdef UV_OFFSET
	fixed2 uvOffset = tex2D(_UVOffsetMap, IN.uv_UVOffsetMap).xy;
	uvOffset = uvOffset * _UVOffsetMagnitude + _UVOffsetBias;

	IN.uv_MainTex += uvOffset;
	#endif
	
	#ifdef _COLORMASK
	mask = UNITY_SAMPLE_TEX2D(_ColorMask, IN.uv_MainTex);
	#endif

	#ifdef _COLORMAP
	mask = UNITY_SAMPLE_TEX2D(_ColorMap, IN.uv_MainTex);
	
	#ifdef _HEIGHTMAP
	fixed4 heights = UNITY_SAMPLE_TEX2D(_PackedHeightMap, IN.uv_MainTex);

	heights *= mask; // multiply heights by the map contribution

	fixed maxHeight = max(max(heights.x, heights.y), max(heights.z, heights.w));

	heights -= maxHeight;
	heights += _HeightTransitionRange;
	heights /= _HeightTransitionRange;
	mask = saturate(heights);
	#endif
	
	#endif

	#if defined(_COLORMASK) || defined(_COLORMAP)
	mask *= 1 / (mask.r + mask.g + mask.b + mask.a);
	#endif
	
	half l = GenerateLerp(IN);
	fixed4 c = GenerateAlbedo(IN, l, mask);
	o.Normal = GenerateNormal(IN, l, mask);
	
	o.Albedo = c.rgb;
	o.Alpha = c.a;
	o.Emission = GenerateEmission(IN, l, mask);
	o.Occlusion = GenerateOcclusion(IN, l);
	half4 spec = GenerateSpecular(IN, l, mask);
	o.Specular = spec.rgb;
	o.Smoothness = spec.a;
}

void surf_metallic (Input IN, inout SurfaceOutputStandard o)
{
	UNITY_INITIALIZE_OUTPUT(SurfaceOutputStandard, o);
	fixed4 mask = 0;
	
	#ifdef UV_OFFSET
	fixed2 uvOffset = tex2D(_UVOffsetMap, IN.uv_UVOffsetMap).xy;
	uvOffset = uvOffset * _UVOffsetMagnitude + _UVOffsetBias;

	IN.uv_MainTex += uvOffset;
	#endif
	
	#ifdef _COLORMASK
	mask = UNITY_SAMPLE_TEX2D(_ColorMask, IN.uv_ColorMask);
	#endif

	#ifdef _COLORMAP
	mask = UNITY_SAMPLE_TEX2D(_ColorMap, IN.uv_ColorMap);
	
	#ifdef _HEIGHTMAP
	fixed4 heights = UNITY_SAMPLE_TEX2D(_PackedHeightMap, IN.uv_MainTex);

	heights *= mask; // multiply heights by the map contribution

	fixed maxHeight = max(max(heights.x, heights.y), max(heights.z, heights.w));

	heights -= maxHeight;
	heights += _HeightTransitionRange;
	heights /= _HeightTransitionRange;
	mask = saturate(heights);
	#endif
	
	#endif

	#if defined(_COLORMASK) || defined(_COLORMAP)
	mask /= saturate(mask.r + mask.g + mask.b + mask.a);
	#endif
	
	half l = GenerateLerp(IN);
	fixed4 c = GenerateAlbedo(IN, l, mask);
	o.Normal = GenerateNormal(IN, l, mask);
	
	o.Albedo = c.rgb;
	o.Alpha = c.a;
	o.Emission = GenerateEmission(IN, l, mask);
	o.Occlusion = GenerateOcclusion(IN, l);
	half2 ms = GenerateMetallicSmoothness(IN, l, mask);
	o.Metallic = ms.r;
	o.Smoothness = ms.g;
}

#endif
