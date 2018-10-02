#include <glad\glad.h>
#include "glfw3.h"
#include "math.h"
#include "models.h"
#include <iostream>
#include <sstream>
#include <vector>
#include <random>
#include <assert.h>

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void APIENTRY glDebugOutput(GLenum source, GLenum type, GLuint id, GLenum severity,
	GLsizei length, const GLchar *message, const void *userParam);
void processInput(GLFWwindow *window);

template <typename T>
constexpr auto ARRAY_SIZE(const T& iarray)
{
	return (sizeof(iarray) / sizeof(iarray[0]));
}

/* 
	Input structs
	Using GLFW input conventions
*/
struct KeyEvent
{
	int key;
	int action;
	int scancode;
	int mods;
	time_t time;
};

struct MouseEvent
{

};

struct Camera
{
	vec3f position;
	vec3f up;
	vec3f forward;
};

struct MouseData
{
	float x, y, last_x, last_y;
};

struct Material {
	unsigned int diffuse;
	unsigned int specular;
	unsigned int emissive;
	float shininess;
};

struct SpotLight {
	vec3f position;
	float cutoff;
	vec3f direction;
	float outerCutoff;
	vec3f ambient;
	float constant;
	vec3f diffuse;
	float linear;
	vec3f specular;
	float quadratic;
};

struct DirectionalLight {
	vec3f direction;
	float intensity;
	vec4f ambient;
	vec4f diffuse;
	vec4f specular;
};

struct PointLight {
	vec3f position;
	float linear;
	vec3f diffuse;
	float quadratic;
	vec3f specular;
	float intensity;
};

struct Entity
{
	Model model;
	vec3f position;
	vec3f velocity;
	vec4f rotation;
	vec4f orientation;
	vec3f scale;
};

struct Scene
{
	static const int MAX_ENTITIES = 1024;
	static const int MAX_LIGHTS = 32;
	Entity entities[MAX_ENTITIES];
	unsigned int entity_count;
	PointLight lights[MAX_LIGHTS];
	unsigned int light_count;
};

struct GBuffer
{
	GLuint id;
	GLuint albedoSpec;
	GLuint position;
	GLuint normal;
	GLuint depthBuffer;
};

struct GeometryShader
{
	//Identifier
	GLint id;

	//Uniforms
		//Matrices
	GLint viewLoc;
	GLint modelLoc;
	GLint projectionLoc;
	GLint normalLoc;

		//Material
	GLint diffuseLoc;
	GLint specularLoc;
};

struct LightingShader
{
	static const size_t MAX_POINT_LIGHTS = 32;

	//Identifier
	GLint id;

	//Uniforms
	GLint positionLoc;
	GLint normalLoc;
	GLint albedoSpecLoc;

	GLint viewPosLoc;
	GLint pointLightCountLoc;

	//Uniform blocks
	GLuint pointLightsBlock;
};

//settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

//Colors
static const vec3f red = { 1.f, 0.f, 0.f };
static const vec3f green = { 0.f, 1.f, 0.f };
static const vec3f blue = { 0.f, 0.f, 1.f };
static const vec3f white = { 1.f, 1.f, 1.f };
static const vec3f black = { 0.f, 0.f, 0.f };

mat4f lookAt(Camera camera)
{
	return lookAt(camera.position, camera.forward + camera.position, camera.up);
}

void APIENTRY glDebugOutput(GLenum source,
	GLenum type,
	GLuint id,
	GLenum severity,
	GLsizei length,
	const GLchar *message,
	const void *userParam)
{
	// ignore non-significant error/warning codes
	if (id == 131169 || id == 131185 || id == 131218 || id == 131204) return;

	std::cout << "---------------" << std::endl;
	std::cout << "Debug message (" << id << "): " << message << std::endl;

	switch (source)
	{
	case GL_DEBUG_SOURCE_API:             std::cout << "Source: API"; break;
	case GL_DEBUG_SOURCE_WINDOW_SYSTEM:   std::cout << "Source: Window System"; break;
	case GL_DEBUG_SOURCE_SHADER_COMPILER: std::cout << "Source: Shader Compiler"; break;
	case GL_DEBUG_SOURCE_THIRD_PARTY:     std::cout << "Source: Third Party"; break;
	case GL_DEBUG_SOURCE_APPLICATION:     std::cout << "Source: Application"; break;
	case GL_DEBUG_SOURCE_OTHER:           std::cout << "Source: Other"; break;
	} std::cout << std::endl;

	switch (type)
	{
	case GL_DEBUG_TYPE_ERROR:               std::cout << "Type: Error"; break;
	case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: std::cout << "Type: Deprecated Behaviour"; break;
	case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:  std::cout << "Type: Undefined Behaviour"; break;
	case GL_DEBUG_TYPE_PORTABILITY:         std::cout << "Type: Portability"; break;
	case GL_DEBUG_TYPE_PERFORMANCE:         std::cout << "Type: Performance"; break;
	case GL_DEBUG_TYPE_MARKER:              std::cout << "Type: Marker"; break;
	case GL_DEBUG_TYPE_PUSH_GROUP:          std::cout << "Type: Push Group"; break;
	case GL_DEBUG_TYPE_POP_GROUP:           std::cout << "Type: Pop Group"; break;
	case GL_DEBUG_TYPE_OTHER:               std::cout << "Type: Other"; break;
	} std::cout << std::endl;

	switch (severity)
	{
	case GL_DEBUG_SEVERITY_HIGH:         std::cout << "Severity: high"; break;
	case GL_DEBUG_SEVERITY_MEDIUM:       std::cout << "Severity: medium"; break;
	case GL_DEBUG_SEVERITY_LOW:          std::cout << "Severity: low"; break;
	case GL_DEBUG_SEVERITY_NOTIFICATION: std::cout << "Severity: notification"; break;
	} std::cout << std::endl;
	std::cout << std::endl;
}

bool LoadShaderFromFile(const std::string path, const GLint shaderID)
{
	FILE * stream;
	errno_t error;
	const char* cstr = path.c_str();
	if ((error = fopen_s(&stream, path.c_str(), "r")) != 0)
		return false;
	fseek(stream, 0, SEEK_END);
	long size = ftell(stream);
	fseek(stream, 0, SEEK_SET);
	char * StringBuffer = (char*)malloc(size);
	StringBuffer[0] = '\0';
	char line[256];
	bool success = false;
	if (StringBuffer)
	{
		while (fgets(line, 256, stream))
		{
			if (ferror(stream))
				break;
			strcat_s(StringBuffer, size, line);
		}
		success = true;
	}
	else {
		fclose(stream);
		return false;
	}
	fclose(stream);
	glShaderSource(shaderID, 1, &StringBuffer, NULL);
	free(StringBuffer);
	return success;
}

bool BuildShaderProgram(GLint * program, std::vector<std::string> vertex_shaders,
						std::vector<std::string> fragment_shaders, std::ostream* log = nullptr)
{
	std::vector<GLint> ids;
	bool success = true;
	GLint types[] = { GL_VERTEX_SHADER, GL_FRAGMENT_SHADER };
	std::vector<std::string>* paths[] = { &vertex_shaders, &fragment_shaders };
	for (int i = 0; i < 2; i++)
	{
		for (std::string path : *(paths[i]))
		{
			GLint id = glCreateShader(types[i]);
			//Load source-code
			if (!LoadShaderFromFile(path, id))
			{
				if (log)
				{
					std::string error_msg = "Error loading shader path: " + path;
					error_msg += "\n";
					*log << error_msg;
				}
				success = false;
				continue;
			}

			//Compile shader
			glCompileShader(id);
			GLint compiled;
			glGetShaderiv(id, GL_COMPILE_STATUS, &compiled);
			if (!compiled)
			{
				GLsizei write_size;
				glGetShaderiv(id, GL_INFO_LOG_LENGTH, &write_size);
				if (log)
				{
					std::string info;
					info.resize(write_size);
					glGetShaderInfoLog(id, write_size, 0, &*info.begin());
					*log << info;
				}
				success = false;
				continue;
			}
			ids.push_back(id);
		}
	}

	//Do linking
	unsigned int p = glCreateProgram();
	for (GLint id : ids)
		glAttachShader(p, id);

	glLinkProgram(p);
	GLint linked;
	glGetProgramiv(p, GL_LINK_STATUS, &linked);
	if (!linked)
	{
		if (log)
		{
			GLsizei write_size;
			glGetProgramiv(p, GL_INFO_LOG_LENGTH, &write_size);
			std::string info;
			info.resize(write_size);
			glGetProgramInfoLog(p, write_size, 0, &*info.begin());
			*log << info;
		}
		success = false;
	}

	//Clear out shaders
	for (GLint id : ids)
		glDeleteShader(id);

	*program = p;
	return success;
}

GBuffer createGBuffer()
{
	GBuffer gb;

	glGenFramebuffers(1, &gb.id);
	glBindFramebuffer(GL_FRAMEBUFFER, gb.id);

	// - position color buffer
	glGenTextures(1, &gb.position);
	glBindTexture(GL_TEXTURE_2D, gb.position);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGB, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gb.position, 0);

	// - normal color buffer
	glGenTextures(1, &gb.normal);
	glBindTexture(GL_TEXTURE_2D, gb.normal);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGB, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, gb.normal, 0);

	// - diffuse + specular color buffer
	glGenTextures(1, &gb.albedoSpec);
	glBindTexture(GL_TEXTURE_2D, gb.albedoSpec);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, gb.albedoSpec, 0);

	// - tell OpenGL which color attachments we'll use (of this framebuffer) for rendering 
	unsigned int attachments[3] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2 };
	glDrawBuffers(3, attachments);

	// - depth buffer
	glGenRenderbuffers(1, &gb.depthBuffer);
	glBindRenderbuffer(GL_RENDERBUFFER, gb.depthBuffer);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, SCR_WIDTH, SCR_HEIGHT);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, gb.depthBuffer);

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		std::cout << "Framebuffer not complete!" << std::endl;

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	return gb;
}

GeometryShader createGeometryShader()
{
	static bool RUN_ONCE = true;
	assert(RUN_ONCE);
	RUN_ONCE = false;

	GeometryShader shader;
	BuildShaderProgram(&shader.id, { "g_buffer_vs.glsl" }, { "g_buffer_fs.glsl" }, &std::cout);
	shader.projectionLoc = glGetUniformLocation(shader.id, "projection");
	shader.viewLoc = glGetUniformLocation(shader.id, "view");
	shader.modelLoc = glGetUniformLocation(shader.id, "model");
	shader.normalLoc = glGetUniformLocation(shader.id, "normal");
	shader.diffuseLoc = glGetUniformLocation(shader.id, "texture_diffuse");
	shader.specularLoc = glGetUniformLocation(shader.id, "texture_specular");

	glUseProgram(shader.id);

	glUniform1i(shader.diffuseLoc, 0);
	glUniform1i(shader.specularLoc, 1);

	return shader;
}

LightingShader createLightingShader(PointLight * pointLightsBuffer, size_t pointLightsBufferSize)
{
	static bool RUN_ONCE = true;
	assert(RUN_ONCE);
	RUN_ONCE = false;

	LightingShader shader = {};
	if (!BuildShaderProgram(&shader.id, { "light_vs.glsl" }, { "light_fs.glsl" }, &std::cout))
	{
		std::cout << "Error creating lighting shader!" << std::endl;
		return shader;
	}
	glUseProgram(shader.id);
	shader.viewPosLoc = glGetUniformLocation(shader.id, "viewPos");
	shader.pointLightCountLoc = glGetUniformLocation(shader.id, "pointLightCount");

	glUniform1i(glGetUniformLocation(shader.id, "gPosition"), 0);
	glUniform1i(glGetUniformLocation(shader.id, "gNormal"), 1);
	glUniform1i(glGetUniformLocation(shader.id, "gAlbedoSpec"), 2);

	// Allocating lights buffer object and binding to block
	glGenBuffers(1, &shader.pointLightsBlock);
	glBindBuffer(GL_UNIFORM_BUFFER, shader.pointLightsBlock);
	glBindBufferBase(GL_UNIFORM_BUFFER, 1, shader.pointLightsBlock);
	glBufferData(GL_UNIFORM_BUFFER, pointLightsBufferSize, pointLightsBuffer, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	return shader;
}

// renderQuad() renders a 1x1 XY quad in NDC
// -----------------------------------------
void renderQuad()
{
	static unsigned int quadVAO = 0;
	static unsigned int quadVBO;

	if (quadVAO == 0)
	{
		float quadVertices[] = {
			// positions        // texture Coords
			-1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
			-1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
			1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
			1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
		};
		// setup plane VAO
		glGenVertexArrays(1, &quadVAO);
		glGenBuffers(1, &quadVBO);
		glBindVertexArray(quadVAO);
		glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
	}
	glBindVertexArray(quadVAO);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	glBindVertexArray(0);
}

void GeometryPass(GeometryShader shader, GBuffer gBuffer, Scene * scene, mat4f view, mat4f projection)
{
	glUseProgram(shader.id);

	//Uniforms
	glUniformMatrix4fv(shader.viewLoc, 1, GL_FALSE, value_ptr(view));
	glUniformMatrix4fv(shader.projectionLoc, 1, GL_FALSE, value_ptr(projection));

	glBindFramebuffer(GL_FRAMEBUFFER, gBuffer.id);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);

	for (unsigned int i = 0; i < scene->entity_count; i++)
	{
		mat4f model = mat4f_identity();
		mat4f_rotate_quat(&model, scene->entities[i].orientation);
		mat4f_translate(&model, scene->entities[i].position);
		mat4f_scale(&model, scene->entities[i].scale);
		glUniformMatrix4fv(shader.modelLoc, 1, GL_FALSE, value_ptr(model));
		glActiveTexture(GL_TEXTURE0);
		//glBindTexture(GL_TEXTURE_2D, scene->entities[i].model.diffuse);
		glActiveTexture(GL_TEXTURE1);
		//glBindTexture(GL_TEXTURE_2D, scene->entities[i].model.specular);
		//glBindVertexArray(scene->entities[i].model.vao);
		glDrawArrays(GL_TRIANGLES, 0, 36);
	}

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void LightingPass(LightingShader shader, GBuffer gBuffer, Scene * scene, Camera camera)
{
	glUseProgram(shader.id);

	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
	glBindFramebuffer(GL_READ_FRAMEBUFFER, gBuffer.id);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, gBuffer.position);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, gBuffer.normal);
	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, gBuffer.albedoSpec);

	//Update pointlights
	glBindBuffer(GL_UNIFORM_BUFFER, shader.pointLightsBlock);
	glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(PointLight) * scene->light_count, scene->lights);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	//Update uniforms
	glUniform3fv(shader.viewPosLoc, 1, value_ptr(camera.position));
	glUniform1i(shader.pointLightCountLoc, scene->light_count);

	glDisable(GL_DEPTH_TEST);
	renderQuad();
}

void renderLights(GLuint shaderProgram, GBuffer gBuffer, PointLight* lights, size_t lightsCount, Model lightModel, mat4f projection, mat4f view)
{
	glUseProgram(shaderProgram);

	//glBindVertexArray(lightModel.vao);

	glBindFramebuffer(GL_READ_FRAMEBUFFER, gBuffer.id);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0); // write to default framebuffer
	glBlitFramebuffer(
		0, 0, SCR_WIDTH, SCR_HEIGHT, 0, 0, SCR_WIDTH, SCR_HEIGHT, GL_DEPTH_BUFFER_BIT, GL_NEAREST
	);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, value_ptr(projection));
	glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, value_ptr(view));

	glEnable(GL_DEPTH_TEST);

	for (int i = 0; i < lightsCount; i++)
	{
		mat4f model = mat4f_identity();
		mat4f_translate(&model, lights[i].position);
		glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, value_ptr(model));
		glUniform3fv(glGetUniformLocation(shaderProgram, "lightColor"), 1, value_ptr(lights[i].diffuse));
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 36);
	}

	glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
}

void renderTextureToScreen(GLint screenShaderProgram, GLint textureID)
{
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
	glUseProgram(screenShaderProgram);
	
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, textureID);

	glDisable(GL_DEPTH_TEST);
	renderQuad();
}

void pollMouse(GLFWwindow * window, MouseData * mouse)
{
	double x, y;
	glfwGetCursorPos(window, &x, &y);
	mouse->last_x = mouse->x;
	mouse->last_y = mouse->y;
	mouse->x = (float)x;
	mouse->y = (float)y;
}

void pushKeyEvent(KeyEvent key)
{
	std::vector<KeyEvent> events;

}

KeyEvent pullKeyEvent()
{
	return {};
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	KeyEvent event;
	event.key = key;
	event.scancode = scancode;
	event.action = action;
	event.mods = mods;
	event.time = glfwGetTime();
	pushKeyEvent(event);
}

int main()
{
	// glfw: initialize and configure 
	// ------------------------------
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);
	glfwWindowHint(GLFW_SAMPLES, 4);

	// Setup window
	GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	// glad: load all OpenGL function pointers
	// ---------------------------------------
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}

	// Setup debugging
	GLint flags; 
	glGetIntegerv(GL_CONTEXT_FLAGS, &flags);
	if (flags & GL_CONTEXT_FLAG_DEBUG_BIT)
	{
		glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);
		glEnable(GL_DEBUG_OUTPUT);
		glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
		glDebugMessageCallback(glDebugOutput, nullptr);
		glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_TRUE);
	}

	glEnable(GL_DEPTH_TEST);

	//Load textures
	Texture wooden_container_texture, steelframed_container_texture, steelframed_container_specular, awesomeface_texture, matrix_texture;
	loadTexture(&wooden_container_texture, "container.jpg");
	loadTexture(&steelframed_container_texture, "container2.png");
	loadTexture(&steelframed_container_specular, "container2_specular.png");
	loadTexture(&awesomeface_texture, "awesomeface.png");
	loadTexture(&matrix_texture, "matrix.jpg");

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0); //Usually not needed.
	
	float t = 0.f;
	float timeStep = 1.f / 144.f;
	float dt = 0.f;

	//Mouse
	MouseData mouse = {};
	float mouse_sensitivity = 0.0005f;

	//Camera
	Camera camera;
	camera.forward = { 0.f, 0.f, -1.f };
	camera.up = { 0.f, 1.f, 0.f };
	camera.position = { 0.f, 0.f, -1.f };
	float cameraSpeed = 5.f;
	float cameraSpeedFast = 20.f;

	//Models
	Model airboatModel;
	loadModel(&airboatModel, "airboat.obj.txt", &std::cout);

	//Scene
	Scene * scene = (Scene*)calloc(1, sizeof(Scene));
	{
		int dim = powf(Scene::MAX_ENTITIES, 1.f / 3);
		for (int i = 0; i < dim; i++)
		{
			float d = 5.f;
			float m = dim * d / 2;
				for (int j = 0; j < dim; j++)
				{
					for (int k = 0; k < dim; k++)
					{
						Entity cubeEntity = {};
						cubeEntity.model = airboatModel;
						cubeEntity.position = { (float)i * d - m, (float)j * d - m, (float)k * d - m};
						cubeEntity.scale = { 1.f, 1.f, 1.f };
						cubeEntity.velocity = { randf(-1.f, 1.f), randf(-1.f, 1.f), randf(-1.f, 1.f) };
						//cubeEntity.orientation = EulerToQuat(randf(0.f, 1.f), randf(0.f, 1.f), randf(0.f, 1.f));
						scene->entities[scene->entity_count++] = cubeEntity;
						if (scene->entity_count > Scene::MAX_ENTITIES)
							break;
					}
				}
		}
	}

	for (int i = 0; i < Scene::MAX_LIGHTS; i++)
	{
		float variance = 25.f;
		//scene->lights[i].position = { 0.f, 0.f, 0.f };
		scene->lights[i].position = { randf(-variance, variance), randf(-variance, variance), randf(-variance, variance) };
		//scene->lights[i].diffuse = { 1.f, 1.f, 1.f };
		scene->lights[i].diffuse = { randf(0.5f, 1.f), randf(0.5f, 1.f), randf(0.5f, 1.f) };
		scene->lights[i].intensity = 1.f;
		scene->lights[i].linear = randf(0.001f, 0.025f);
		scene->lights[i].quadratic = randf(0.001f, .00005f);
		scene->lights[i].specular = { 1.f, 1.f, 1.f };
		scene->light_count++;
	}

	//Shaders
	GeometryShader geometryShader = createGeometryShader();
	LightingShader lightingShader = createLightingShader(scene->lights, sizeof(PointLight) * scene->MAX_LIGHTS);
	GLint screenShaderProgram;
	BuildShaderProgram(&screenShaderProgram, { "screen_shader_vs.glsl" }, { "screen_shader_fs.glsl" }, &std::cout);
	GLint lightboxShaderProgram;
	BuildShaderProgram(&lightboxShaderProgram, { "lightbox_vs.glsl" }, { "lightbox_fs.glsl" }, &std::cout);

	//GBuffer
	GBuffer gBuffer = createGBuffer();

	float oldCutoff = 0.f;
	bool gamma_correct = false;
	float gamma = 2.2f;

	//Callbacks
	glfwSetKeyCallback(window, key_callback);

	//Matrices
	mat4f projection = frustumProjectionMat4f(85.f, (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 1000.f);

	while (!glfwWindowShouldClose(window))
	{
		processInput(window);
		pollMouse(window, &mouse);

		vec3f movement = {};
		vec3f side = camera.up.cross(camera.forward).normalized();
		float speed = cameraSpeed;
		if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
			speed = cameraSpeedFast;
		if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
			movement += side.scale(-speed * dt);
		if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
			movement += side.scale(speed * dt);
		if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
			movement += camera.forward.scale(speed * dt);
		if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
			movement += camera.forward.scale(-speed * dt);
		if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
			camera.up = rotate_vec3f_around_axis(camera.up, camera.up.cross(side), 0.0001f * dt);
		if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
			camera.up = rotate_vec3f_around_axis(camera.up, camera.up.cross(side), -0.0001f * dt);
		camera.position += movement;
		float pitch = (mouse.x - mouse.last_x) * mouse_sensitivity;
		float roll = (mouse.last_y - mouse.y) * mouse_sensitivity;

		camera.forward = rotate_vec3f_around_axis(camera.forward, side, roll * 5.f);
		camera.forward = rotate_vec3f_around_axis(camera.forward, camera.up, pitch * 5.f);

		// Avoid axis-flipping camera around forward.y +/- 1
		if (camera.forward * side.cross(camera.up) < 0.f)
		{
			camera.forward = (camera.up * (camera.forward * camera.up) * 3.f - camera.forward);
			camera.forward.normalize();
		}

		// Update entities
		for (int i = 0; i<scene->entity_count; i++)
		{
			scene->entities[i].position += scene->entities[i].velocity * dt;
		}

		GeometryPass(geometryShader, gBuffer, scene, lookAt(camera), projection);
		
		LightingPass(lightingShader, gBuffer, scene, camera);
		
		renderLights(lightboxShaderProgram, gBuffer, scene->lights, scene->light_count, airboatModel, projection, lookAt(camera));

		glfwSwapBuffers(window);
		glfwPollEvents();

		//Timestep
		float time = glfwGetTime();
		dt = time - t;
		t = time;

		if (dt < timeStep)
			Sleep((timeStep - dt) * 10);
	}

	// Free mallocs
	free(scene);

	// glfw: terminate, clearing all previously allocated GLFW resources.
	glfwTerminate();
	return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	// make sure the viewport matches the new window dimensions; note that width and 
	// height will be significantly larger than specified on retina displays.
	glViewport(0, 0, width, height);
}