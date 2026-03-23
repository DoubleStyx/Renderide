// Unity / ShaderLab compatibility shims for Slang → WGSL conversion.
// Grows incrementally as more Resonite shaders are brought online.

#ifndef UNITY_SHADER_CONVERTER_COMPAT_INCLUDED
#define UNITY_SHADER_CONVERTER_COMPAT_INCLUDED

typedef float4 fixed4;
typedef float3 fixed3;
typedef float2 fixed2;
typedef float fixed;

#define SV_Target SV_Target0

#endif
