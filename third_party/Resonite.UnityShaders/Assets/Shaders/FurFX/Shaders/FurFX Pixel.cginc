// Upgrade NOTE: replaced '_Object2World' with 'unity_ObjectToWorld'
// Upgrade NOTE: replaced '_World2Object' with 'unity_WorldToObject'
// Upgrade NOTE: replaced 'mul(UNITY_MATRIX_MVP,*)' with 'UnityObjectToClipPos(*)'

#include "AutoLight.cginc"
#include "UnityCG.cginc"

struct appdata {
	float4 vertex : POSITION;
	fixed3 normal : NORMAL;
	half4 texcoord : TEXCOORD0;
};

struct v2f {
	float4 pos : SV_POSITION;
	half4 uv : TEXCOORD0;
	half4 uv2 : TEXCOORD1;
	fixed4 color : COLOR0;	
	fixed4 rim : TEXCOORD2;
	fixed4 ref : TEXCOORD3;	
	LIGHTING_COORDS(4,5)
	//SHADOW_COORDS(6)
};

uniform fixed _FurLength;
uniform fixed _EdgeFade;
uniform float4 _ForceGlobal;
uniform float4 _ForceLocal;
uniform fixed _HairHardness;
uniform fixed _HairThinness;
uniform fixed _SkinAlpha;
uniform fixed _HairShading;
uniform fixed _HairColoring;
uniform fixed4 _Color;
uniform fixed4 _SpecColor;
uniform fixed4 _RimColor;
uniform half _RimPower;
uniform half _Shininess;
uniform fixed _Reflection;
uniform fixed _Cutoff;
uniform fixed _ShadowStrength;

uniform fixed4 _LightColor0; 

sampler2D _MainTex;
sampler2D _NoiseTex;
samplerCUBE _Cube;

uniform half4 _MainTex_ST;
uniform half4 _NoiseTex_ST;

v2f VertexProgramGrab (appdata v) 
{
	v2f o;
	
	o.uv.xy = TRANSFORM_TEX(v.texcoord, _MainTex);
	o.uv.zw = TRANSFORM_TEX(v.texcoord, _NoiseTex);	
	
	fixed3 P = v.vertex.xyz + (v.normal * _FurLength * FUR_MULTIPLIER * _HairHardness);
	P += (mul(unity_WorldToObject,clamp(_ForceGlobal,-1,1)).xyz + mul(UNITY_MATRIX_P,clamp(_ForceLocal,-1,1)).xyz) * pow(FUR_MULTIPLIER,2) * _FurLength;
	o.pos = UnityObjectToClipPos(fixed4(P,1.0));
	//TRANSFER_VERTEX_TO_FRAGMENT(o);
	return o;
	
}


v2f VertexProgram (appdata v) 
{
	v2f o;
	//UNITY_INITIALIZE_OUTPUT(vertexOutput,o);

	o.pos = UnityObjectToClipPos(v.vertex);

	//TRANSFER_SHADOW(o);
	TRANSFER_VERTEX_TO_FRAGMENT(o);

	o.uv.xy = TRANSFORM_TEX(v.texcoord, _MainTex);
	o.uv.zw = TRANSFORM_TEX(v.texcoord, _NoiseTex);

	//normalizacja
	//fixed mag = length(_ForewceGlobal.xyz + _ForceLocal.xyz);
	//_ForceGlobal /= mag;
	//_ForceLocal /= mag;
	

	fixed3 P = v.vertex.xyz + (v.normal * _FurLength * FUR_MULTIPLIER * _HairHardness);

	//P += mul(_World2Object,clamp(_ForceGlobal,-1,1)).xyz * pow(FUR_MULTIPLIER,2) * _FurLength;
	
	P += (mul(unity_WorldToObject,clamp(_ForceGlobal,-1,1)).xyz + mul(UNITY_MATRIX_P,clamp(_ForceLocal,-1,1)).xyz) * pow(FUR_MULTIPLIER,2) * _FurLength;

	o.pos = UnityObjectToClipPos(fixed4(P,1.0));
	//o.pos.xyz += mul(_Object2World,_ForceLocal).xyz * pow(FUR_MULTIPLIER,2) * _FurLength;
	
	fixed4 znormal = 1 - dot(v.normal, fixed4(0,0,1,0));
	o.uv2.xy = mul( UNITY_MATRIX_TEXTURE1, v.texcoord + znormal * 0.0011 * FUR_MULTIPLIER );	

	fixed3 posWorld = mul(unity_ObjectToWorld, v.vertex).xyz;
    fixed3 normalDirection = normalize(fixed3(mul(fixed4(v.normal, 0.0), unity_WorldToObject).xyz));
    fixed3 lightDirection = normalize(_WorldSpaceLightPos0.xyz);
    fixed3 viewDirection = normalize(_WorldSpaceCameraPos.xyz - posWorld);
    
    fixed3 diffuseReflection = _LightColor0.xyz * _Color.xyz * max(0.0, dot(normalDirection, lightDirection));
    fixed3 ambientLighting = UNITY_LIGHTMODEL_AMBIENT.xyz * _Color.xyz;	
    
    fixed3 specularReflection;
    
    if (dot(normalDirection, lightDirection) < 0.0) 
    	{
        	specularReflection = fixed3(0.0, 0.0, 0.0); 
        }
        else
        {
            specularReflection = _LightColor0.xyz * _SpecColor.xyz * pow(max(0.0, dot(reflect(-lightDirection, normalDirection).xyz, viewDirection)), _Shininess);
        }
	
	 fixed3 vertexLighting = fixed3(0.0, 0.0, 0.0);
	 #ifdef VERTEXLIGHT_ON
            for (int index = 0; index < 1; index++)
            {    
               fixed4 lightPosition = fixed4(unity_4LightPosX0[index], 
                  unity_4LightPosY0[index], 
                  unity_4LightPosZ0[index], 1.0);
                  
               fixed3 vertexToLightSource = 
                  fixed3(lightPosition.xyz - posWorld);        
               fixed3 lightDirection = normalize(vertexToLightSource);
               
               fixed squaredDistance = 
                  dot(vertexToLightSource, vertexToLightSource);
               fixed attenuation = 1.0 / (1.0 + 
                  unity_4LightAtten0[index] * squaredDistance);
                  
               fixed3 diffuseReflection2 =  
                  attenuation * unity_LightColor[index].xyz 
                  * _Color.xyz * max(0.0, 
                  dot(normalDirection, lightDirection));         
 
               vertexLighting = 
                  vertexLighting + diffuseReflection2;
            }	
            #endif	
	
    half rim = 1.0 - saturate(dot (viewDirection, normalDirection));
    o.rim = fixed4(_RimColor.rgb * pow (rim, _RimPower),1.0);	
    
    //float4 nD = normalize(float3(mul(float4(v.normal, 0.0), _World2Object)));    
    o.ref = reflect( -fixed4(viewDirection,0.0), fixed4(normalDirection,0.0) );
							
    o.color = fixed4((ambientLighting + diffuseReflection + specularReflection + vertexLighting), 1.0);
	return o;
	
}

fixed4 frag (v2f i) : COLOR
{

	fixed4 o = tex2D(_MainTex, i.uv.xy);
	fixed4 shadowcol = tex2D(_NoiseTex, i.uv2.xy * _HairThinness);
	fixed3 noise = tex2D(_NoiseTex, i.uv.zw * _HairThinness).rgb;
	//half4 g = tex2Dproj( _GrabTexture, IN.GrabUV);

	clip ( FUR_MULTIPLIER > max(o.a,_SkinAlpha) ? -1 : 1 );	
	
	o.rgb -= shadowcol.rgb * _HairColoring;
	o.rgb -= (pow(1-FUR_MULTIPLIER,4)) * _HairShading;
	
	o.rgb = clamp(o.rgb,0,1);
	
	//o.a = min(noise * (1-FUR_MULTIPLIER*FUR_MULTIPLIER),1) * max(o.a,_SkinAlpha);
	o.a = clamp(noise * max(o.a,_SkinAlpha) - (FUR_MULTIPLIER*FUR_MULTIPLIER)*_EdgeFade,0,1);

	clip (o.a - _Cutoff);
	
	float4 reflTex = texCUBE(_Cube, i.ref);	
	
	fixed att = max(LIGHT_ATTENUATION(i),1-_ShadowStrength);
		
	o.rgb*=i.color * 2;
	
	o.rgb = lerp(UNITY_LIGHTMODEL_AMBIENT.xyz * _Color.xyz,o.rgb,att);
	
	o.rgb+=i.rim * 2;
	
	o.rgb += reflTex.rgb * _Reflection * o.a * att;

	return o;
}

fixed4 fragShadow (v2f i) : COLOR
{

	fixed4 o = tex2D(_MainTex, i.uv.xy);
	fixed4 shadowcol = tex2D(_NoiseTex, i.uv2.xy * _HairThinness);
	fixed3 noise = tex2D(_NoiseTex, i.uv.zw * _HairThinness).rgb;
	//half4 g = tex2Dproj( _GrabTexture, IN.GrabUV);

	clip ( FUR_MULTIPLIER > max(o.a,_SkinAlpha) ? -1 : 1 );

	o.rgb -= shadowcol.rgb * _HairColoring;
	o.rgb -= (pow(1-FUR_MULTIPLIER,4)) * _HairShading;
	
	o.rgb = clamp(o.rgb,0,1);
	
	//o.a = min(noise * (1-FUR_MULTIPLIER*FUR_MULTIPLIER),1) * max(o.a,_SkinAlpha);
	o.a = clamp(noise - (FUR_MULTIPLIER*FUR_MULTIPLIER)*_EdgeFade,0,1);
		
    //o.a =  noise;			
		
	o.rgb*=i.color * 2 * LIGHT_ATTENUATION(i);
	
	o.rgb = 0;
	o.a = (1-LIGHT_ATTENUATION(i) * 0.5); 
	

	return o;
}

fixed4 fragBase (v2f i) : COLOR
{
	fixed4 o = tex2D(_MainTex, i.uv.xy);
    
	o.rgb*=i.color * 2;
	
	fixed att = max(LIGHT_ATTENUATION(i),1-_ShadowStrength);
	
	o.rgb = lerp(UNITY_LIGHTMODEL_AMBIENT.xyz * _Color.xyz,o.rgb,att);	
	
	o.a = 1.0;
	return o;
}
