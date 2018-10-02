#include "models.h"
#include "math.h"
#include <vector>
#include <fstream>
#include <sstream>
#include <glad\glad.h>
#include <stdio.h>
#include <iostream>
#include <regex>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

struct ModelData
{
	
};

bool loadTexture(Texture * texture, const char * path, std::ostream * log, Texture::Type type, bool flip,
	unsigned int wrap_s, unsigned int wrap_t, unsigned int min_filter, unsigned int mag_filter)
{
	stbi_set_flip_vertically_on_load(flip);
	glGenTextures(1, &texture->id);
	glBindTexture(GL_TEXTURE_2D, texture->id);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap_s);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap_t);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, min_filter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, mag_filter);
	int width, height, nrChannels;
	unsigned char *data = stbi_load(path, &width, &height, &nrChannels, 0);
	if (data)
	{
		unsigned int COLOR_MODE = nrChannels <= 3 ? GL_RGB : GL_RGBA;
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, COLOR_MODE, GL_UNSIGNED_BYTE, data);
		glGenerateMipmap(GL_TEXTURE_2D);
	}
	else if(log)
	{
		*log << "Error loading texture: " << path << std::endl;
		return false;
	}
	stbi_image_free(data);
	return true;
}

bool loadMesh(Mesh* mesh)
{
	//Clear previous errors
	glGetError();

	glGenVertexArrays(1, &mesh->VAO);
	glGenBuffers(1, &mesh->VBO);
	glGenBuffers(1, &mesh->EBO);

	glBindVertexArray(mesh->VAO);
	glBindBuffer(GL_ARRAY_BUFFER, mesh->VBO);

	glBufferData(GL_ARRAY_BUFFER, mesh->vertices.size() * sizeof(Vertex), &mesh->vertices[0], GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh->EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh->indices.size() * sizeof(unsigned int), &mesh->indices[0], GL_STATIC_DRAW);

	// vertex positions
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
	// vertex normals
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));
	// vertex texture coords
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, texCoord));

	glBindVertexArray(0);

	return glGetError() == GL_NO_ERROR ? true : false;
}

bool unloadMesh(Mesh* mesh)
{
	glGetError();
	glDeleteVertexArrays(1, &mesh->VAO);
	glDeleteBuffers(1, &mesh->VBO);
	glDeleteBuffers(1, &mesh->EBO);
	mesh->VAO = 0;
	mesh->VBO = 0;
	mesh->EBO = 0;
	return glGetError() == GL_NO_ERROR ? true : false;
}

bool loadModel(Model * model, const char * path, std::ostream * log)
{
	model->path = std::string(path);
	//Only supporting .OBJ for now
	if (!loadObjModel(model, log))
		return false;
	return loadMesh(&model->meshes[0]);
}	

bool unloadModel(Model * model)
{
	bool result = true;
	for (Mesh mesh : model->meshes)
		result &= unloadMesh(&mesh);
	return result;
}

bool loadObjModel(Model * model, std::ostream* log)
{
	std::ifstream stream;
	try {
		stream.open(model->path, std::ifstream::in);
	}catch (std::ifstream::failure) {
		if (log)
			*log << "Failed to open .OBJ file path: " << model->path.c_str() << std::endl;
		return false;
	}

	std::vector<vec3f> temp_vertices;
	std::vector<vec3f> temp_normals;
	std::vector<vec2f> temp_texCoords;

	std::vector<unsigned int> vertexIndices;
	std::vector<unsigned int> normalIndices;
	std::vector<unsigned int> texCoordIndices;

	std::vector<unsigned int> materials;

	while (true)
	{
		std::string line;
		std::getline(stream, line);
		if ( stream.eof() )
			break;
		if (std::regex_match(line, std::regex("^#.*")) || line.length() == 0)
			continue;
		if (std::regex_match(line, std::regex("^mtllib .*")))
		{
			std::string mtllib_path;
			scanf_s(line.c_str(), "mtllib %s\n", &mtllib_path);
			//Load material into textures here
		}
		else if ( std::regex_match(line, std::regex("^v -?[0-9]*\.[0-9]* -?[0-9]*\.[0-9]* -?[0-9]*\.[0-9]*$")) )
		{
			vec3f vertex;
			scanf_s(line.c_str(), "v %f %f %f\n", &vertex.x, &vertex.y, &vertex.z);
			temp_vertices.push_back(vertex);
		}
		else if (std::regex_match(line, std::regex("^vt -?[0-9]*\.[0-9]* -?[0-9]*\.[0-9]*$")))
		{
			vec2f texCoord;
			scanf_s(line.c_str(), "vt %f %f\n", &texCoord.x, &texCoord.y);
			temp_texCoords.push_back(texCoord);
		}
		else if (std::regex_match(line, std::regex("^vn -?[0-9]*\.[0-9]* -?[0-9]*\.[0-9]* -?[0-9]*\.[0-9]*$")))
		{
			vec3f normal;
			scanf_s(line.c_str(), "vn %f %f %f\n", &normal.x, &normal.y, &normal.z);
			temp_normals.push_back(normal);
		}
		else if (std::regex_match(line, std::regex("^f /[0-9]/[0-9]/[0-9] /[0-9]/[0-9]/[0-9] /[0-9]/[0-9]/[0-9]$")))
		{
			unsigned int vertexIndex[3], normalIndex[3], texCoordIndex[3];
			if (scanf_s(line.c_str(), "f %d/%d/%d %d/%d/%d %d/%d/%d\n",
				&vertexIndex[0], &texCoordIndex[0], &normalIndex[0],
				&vertexIndex[1], &texCoordIndex[1], &normalIndex[1],
				&vertexIndex[2], &texCoordIndex[2], &normalIndex[2]) != 9) {
				return false;
			}

			vertexIndices.push_back(vertexIndex[0]);
			vertexIndices.push_back(vertexIndex[1]);
			vertexIndices.push_back(vertexIndex[2]);
			texCoordIndices.push_back(texCoordIndex[0]);
			texCoordIndices.push_back(texCoordIndex[1]);
			texCoordIndices.push_back(texCoordIndex[2]);
			normalIndices.push_back(normalIndex[0]);
			normalIndices.push_back(normalIndex[1]);
			normalIndices.push_back(normalIndex[2]);
		}
		else if (log)
			*log << "Can't parse line: '" << line << "' in .OBJ file: " << model->path << std::endl;
	}

	if (vertexIndices.size() != normalIndices.size() || vertexIndices.size() != texCoordIndices.size())
	{
		if (log)
			*log << "Malformed .OBJ file: " << model->path << ", unequal number of positions, normals or texture coordinates\n";
		return false;
	}

	Mesh mesh;
	model->meshes.push_back(mesh);

	for (int i = 0; i < vertexIndices.size(); i++)
	{
		Vertex vertex;
		vertex.position = temp_vertices[vertexIndices[i] - 1];
		vertex.normal = temp_normals[normalIndices[i] - 1];
		vertex.texCoord = temp_texCoords[texCoordIndices[i] - 1];
		model->meshes[0].vertices.push_back(vertex);
	}
}

void loadObjMtllib(const char * path, std::vector<Texture> * materials)
{
	while (true)
	{
		
	}
}