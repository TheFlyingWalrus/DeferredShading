#pragma once
#include "math.h"
#include <GLAD\glad.h>
#include "glfw3.h"
#include <vector>

struct Vertex
{
	Vertex() : position(), normal(), texCoord() {}
	Vertex(vec3f position, vec3f normal, vec2f texCoord) : position(position), normal(normal), texCoord(texCoord) {}
	vec3f position;
	vec3f normal;
	vec2f texCoord;
};

struct Texture
{
	enum Type
	{
		DIFFUSE,
		SPECULAR,
		NORMAL
	};

	unsigned int id;
	Type type;
};

struct Mesh
{
	std::vector<Vertex> vertices;
	std::vector<unsigned int> indices;
	std::vector<Texture> textures;

	GLuint VAO;
	GLuint VBO;
	GLuint EBO;
};

struct Model
{
	std::vector<Mesh> meshes;
	std::string path;
};

/*
	Texture, Model and Mesh functions
*/
bool loadTexture(Texture * texture, const char * path, std::ostream* log = nullptr, 
	Texture::Type type = Texture::Type::DIFFUSE, bool flip = true,
	unsigned int wrap_s = GL_REPEAT, unsigned int wrap_t = GL_REPEAT,
	unsigned int min_filter = GL_LINEAR, unsigned int mag_filter = GL_LINEAR);
bool loadMesh(Mesh * mesh);
bool unloadMesh(Mesh * mesh);
bool loadModel(Model * model, const char * path, std::ostream * log = nullptr);
bool unloadModel(Model * model);

void drawMesh(GLFWwindow * window, Mesh * mesh);
void drawModel(GLFWwindow * window, Model * model);

/* 
	File loading
*/
bool loadObjModel(Model * model, std::ostream * log = nullptr);
void loadObjMtllib(const char * path, std::vector<Texture> * materials);

