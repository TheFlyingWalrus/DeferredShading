#version 450 core
out vec4 FragColor;
  
in vec2 TexCoords;

uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D gAlbedoSpec;
uniform vec3 viewPos;

struct PointLight {
    vec3 position;
	float linear;
	vec3 diffuse;
	float quadratic;
	vec3 specular;
	float intensity;
};

const int NR_LIGHTS = 32;
uniform int pointLightCount;

layout (std140, binding = 1) uniform Lights {
	PointLight[NR_LIGHTS] pointLights;
};

void main()
{             
    // retrieve data from G-buffer
    vec3 FragPos = texture(gPosition, TexCoords).rgb;
    vec3 Normal = texture(gNormal, TexCoords).rgb;
    vec3 Albedo = texture(gAlbedoSpec, TexCoords).rgb;
    float Specular = texture(gAlbedoSpec, TexCoords).a;
    
    // then calculate lighting as usual
    vec3 lighting = Albedo * 0.1; // hard-coded ambient component
    vec3 viewDir = normalize(viewPos - FragPos);
    for(int i = 0; i < pointLightCount; ++i)
    {
        // diffuse
		vec3 lightVec = pointLights[i].position - FragPos;
        vec3 lightDir = normalize(lightVec);
		float dirFactor = max(dot(Normal, lightDir), 0.0);
        vec3 diffuse = dirFactor * Albedo * pointLights[i].diffuse;
        lighting += diffuse;

		// specular
		float specLight = pow(max(dot(viewDir, reflect(-lightDir, Normal)), 0.0), 128) * Specular;
		//lighting += specLight * pointLights[i].specular;

		//Attenuation and intensity
		float distance = length(lightVec); 
		lighting *= 1.0 / (1.0 + pointLights[i].linear * distance + pointLights[i].quadratic * (distance * distance)); 
		lighting *= pointLights[i].intensity;
    }
    
    FragColor = vec4(lighting, 1.0);
} 