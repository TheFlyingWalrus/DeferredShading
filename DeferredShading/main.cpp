#include <glad\glad.h>
#include "glfw3.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <iostream>
#include <sstream>
#include <vector>
#include <random>

typedef float time_t;

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);

template <typename T>
constexpr auto ARRAY_SIZE(const T& iarray)
{
	return (sizeof(iarray) / sizeof(iarray[0]));
}

struct vec3f
{
	float x, y, z;
};

struct vec4f
{
	float x, y, z, w;
};

struct mat3f
{
	vec3f a, b, c;
};

struct mat4f
{
	vec4f a, b, c, d;
};

/* 
	Input structs
	Using GLFW input convetions
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
	float constant;
	vec3f ambient;
	float linear;
	vec3f diffuse;
	float quadratic;
	vec3f specular;
	float intensity;
};

struct Model
{
	//Vertex Array Object
	GLint vao;

	//Texture data
	GLint diffuse;
	GLint specular;
};

struct Entity
{
	Model model;
	vec3f position;
	vec4f orientation;
	vec3f scale;
};

struct Scene
{
	static const int MAX_ENTITIES = 1024;
	Entity* entities;
	unsigned int entity_count;
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
	//Identifier
	GLint id;

	//Uniforms
	GLint positionLoc;
	GLint normalLoc;
	GLint albedoSpecLoc;
};

//settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

//Constants
static const float PI = acosf(-1);

//Colors
static const vec3f red = { 1.f, 0.f, 0.f };
static const vec3f green = { 0.f, 1.f, 0.f };
static const vec3f blue = { 0.f, 0.f, 1.f };
static const vec3f white = { 1.f, 1.f, 1.f };
static const vec3f black = { 0.f, 0.f, 0.f };

inline float* value_ptr(vec3f v)
{
	return &v.x;
}

inline float* value_ptr(vec4f v)
{
	return &v.x;
}

inline float* value_ptr(mat3f m)
{
	return &m.a.x;
}

inline float* value_ptr(mat4f m)
{
	return &m.a.x;
}

inline float randf(float min, float max)
{
	float r = (float)rand() / RAND_MAX;
	r *= max - min;
	r += min;
	return r;
}

inline float length(const vec3f v)
{
	return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

inline vec3f scale_vec3f(const vec3f v, const float s)
{
	return { v.x * s, v.y * s, v.z * s };
}

inline vec3f add_vec3f(const vec3f a, const vec3f b)
{
	return { a.x + b.x, a.y + b.y, a.z + b.z };
}

inline vec3f sub_vec3f(const vec3f a, const vec3f b)
{
	return { a.x - b.x, a.y - b.y, a.z - b.z };
}

inline float dot_vec3f(const vec3f a, const vec3f b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline vec3f cross_vec3f(const vec3f a, const vec3f b)
{
	return { a.y * b.z - b.y * a.z, a.z * b.x - b.z * a.x, a.x * b.y - b.x * a.y };
}

inline vec3f normalize_vec3f(const vec3f v)
{
	float il = 1.f / sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
	return { v.x * il, v.y * il, v.z * il };
}

/* Second-degree polynomial approximating inverse square root of length(v) */
inline vec3f normalize_fast(const vec3f v)
{
	float d = v.x * v.x + v.y * v.y + v.z * v.z;
	d = 15.f / 8.f - 5.f / 4.f * d + 3.f / 8.f * d * d;
	return { v.x * d, v.y * d, v.z * d };
}

inline vec3f rotate_vec3f_by_quat(const vec3f v, const vec4f q)
{
	vec3f u = { q.x, q.y, q.z };
	vec3f uxv = cross_vec3f(u, v);
	float s = q.w;
	float uv = dot_vec3f(u, v);
	float uu = dot_vec3f(u, u);
	u = scale_vec3f(u, 2 * uv);
	uxv = scale_vec3f(uxv, 2 * s);
	return add_vec3f(add_vec3f(u, uxv), scale_vec3f(v, s * s - uu));
}

inline vec3f rotate_vec3f_around_axis(const vec3f v, const vec3f axis, float radians)
{
	/* Constructing Quaternion */
	float A, B, C, D;
	A = cosf(radians / 2.f);
	B = sinf(radians / 2.f) * axis.x;
	C = sinf(radians / 2.f) * axis.y;
	D = sinf(radians / 2.f) * axis.z;
	/* Calculating coefficients */
	float AA, BB, CC, DD, AB, AC, AD, BC, BD, CD;
	AA = A * A;
	BB = B * B;
	CC = C * C;
	DD = D * D;
	AB = A * B;
	AC = A * C;
	AD = A * D;
	BC = B * C;
	BD = B * D;
	CD = C * D;
	/* Calculating v rotated by quaternion */
	vec3f u;
	u.x = v.x * (AA + BB - CC - DD) + 2 * v.y * (BC - AD) + 2 * v.z * (BD + AC);
	u.y = 2 * v.x * (BC + AD) + v.y * (AA - BB + CC - DD) + 2 * v.z * (CD - AB);
	u.z = 2 * v.x * (BD - AC) + 2 * v.y * (CD + AB) + v.z * (AA - BB - CC + DD);
	return u;
}

/*
Mat4f functions for applying transforms directly to a mat4f
Useful for concatenating many transforms;
*/
void mat4f_scalar(const float s, mat4f * m)
{
	m->a.x *= s;
	m->a.y *= s;
	m->a.z *= s;
	m->a.w *= s;

	m->b.x *= s;
	m->b.y *= s;
	m->b.z *= s;
	m->b.w *= s;

	m->c.x *= s;
	m->c.y *= s;
	m->c.z *= s;
	m->c.w *= s;

	m->d.x *= s;
	m->d.y *= s;
	m->d.z *= s;
	m->d.w *= s;
}

void mat4f_scale(mat4f * m, const vec3f scale)
{
	m->a.x *= scale.x;
	m->a.y *= scale.y;
	m->a.z *= scale.z;

	m->b.x *= scale.x;
	m->b.y *= scale.y;
	m->b.z *= scale.z;

	m->c.x *= scale.x;
	m->c.y *= scale.y;
	m->c.z *= scale.z;
}

void mat4f_translate(mat4f * m, const vec3f t)
{
	vec4f tmp = m->d;
	tmp.x += m->a.x * t.x + m->b.x * t.y + m->c.x * t.z;
	tmp.y += m->a.y * t.x + m->b.y * t.y + m->c.y * t.z;
	tmp.z += m->a.z * t.x + m->b.z * t.y + m->c.z * t.z;
	tmp.w += m->a.w * t.x + m->b.w * t.y + m->c.w * t.z;
	m->d = tmp;
}

void mat4f_transpose(mat4f * m)
{
	vec4f temp = m->a;
	m->a.y = m->b.x;
	m->a.z = m->c.x;
	m->a.w = m->d.x;
	m->b.x = temp.y;
	m->c.x = temp.z;
	m->d.x = temp.w;

	temp = m->b;
	m->b.z = m->c.y;
	m->b.w = m->d.y;
	m->c.y = temp.z;
	m->d.y = temp.w;

	temp = m->c;
	m->c.w = m->d.z;
	m->d.z = temp.w;
}

void mat4f_rotate_quat(mat4f * m, const vec4f q)
{
	float xx, xy, xz, xw, yy, yz, yw, zz, zw, ww;
	xx = 2 * q.x * q.x;
	yy = 2 * q.y * q.y;
	zz = 2 * q.z * q.z;
	ww = 2 * q.w * q.w;
	xy = 2 * q.x * q.y;
	xz = 2 * q.x * q.z;
	xw = 2 * q.x * q.w;
	yz = 2 * q.y * q.z;
	yw = 2 * q.y * q.w;
	zw = 2 * q.z * q.w;

	vec4f tmp;
	tmp = m->a;
	m->a.x = tmp.x * (1 - yy - zz) + tmp.y * (xy + zw) + tmp.z * (xz - yw);
	m->a.y = tmp.x * (xy - zw) + tmp.y * (1 - xx - zz) + tmp.z * (xw + yz);
	m->a.z = tmp.x * (yw + xz) + tmp.y * (yz - xw) + tmp.z * (1 - xx - yy);

	tmp = m->b;
	m->b.x = tmp.x * (1 - yy - zz) + tmp.y * (xy + zw) + tmp.z * (xz - yw);
	m->b.y = tmp.x * (xy - zw) + tmp.y * (1 - xx - zz) + tmp.z * (xw + yz);
	m->b.z = tmp.x * (yw + xz) + tmp.y * (yz - xw) + tmp.z * (1 - xx - yy);

	tmp = m->c;
	m->c.x = tmp.x * (1 - yy - zz) + tmp.y * (xy + zw) + tmp.z * (xz - yw);
	m->c.y = tmp.x * (xy - zw) + tmp.y * (1 - xx - zz) + tmp.z * (xw + yz);
	m->c.z = tmp.x * (yw + xz) + tmp.y * (yz - xw) + tmp.z * (1 - xx - yy);

	tmp = m->d;
	m->d.x = tmp.x * (1 - yy - zz) + tmp.y * (xy + zw) + tmp.z * (xz - yw);
	m->d.y = tmp.x * (xy - zw) + tmp.y * (1 - xx - zz) + tmp.z * (xw + yz);
	m->d.z = tmp.x * (yw + xz) + tmp.y * (yz - xw) + tmp.z * (1 - xx - yy);
}

void mat4f_multiply(mat4f * m, mat4f m2)
{
	mat4f_transpose(&m2);
	vec4f tmp;
	tmp.x = m->a.x * m2.a.x + m->a.y * m2.a.y + m->a.z * m2.a.z + m->a.w * m2.a.w;
	tmp.y = m->a.x * m2.b.x + m->a.y * m2.b.y + m->a.z * m2.b.z + m->a.w * m2.b.w;
	tmp.z = m->a.x * m2.c.x + m->a.y * m2.c.y + m->a.z * m2.c.z + m->a.w * m2.d.w;
	tmp.w = m->a.x * m2.d.x + m->a.y * m2.d.y + m->a.z * m2.d.z + m->a.w * m2.d.w;
	m->a = tmp;

	tmp.x = m->b.x * m2.a.x + m->b.y * m2.a.y + m->b.z * m2.a.z + m->b.w * m2.a.w;
	tmp.y = m->b.x * m2.b.x + m->b.y * m2.b.y + m->b.z * m2.b.z + m->b.w * m2.b.w;
	tmp.z = m->b.x * m2.c.x + m->b.y * m2.c.y + m->b.z * m2.c.z + m->b.w * m2.d.w;
	tmp.w = m->b.x * m2.d.x + m->b.y * m2.d.y + m->b.z * m2.d.z + m->b.w * m2.d.w;
	m->b = tmp;

	tmp.x = m->c.x * m2.a.x + m->c.y * m2.a.y + m->c.z * m2.a.z + m->c.w * m2.a.w;
	tmp.y = m->c.x * m2.b.x + m->c.y * m2.b.y + m->c.z * m2.b.z + m->c.w * m2.b.w;
	tmp.z = m->c.x * m2.c.x + m->c.y * m2.c.y + m->c.z * m2.c.z + m->c.w * m2.d.w;
	tmp.w = m->c.x * m2.d.x + m->c.y * m2.d.y + m->c.z * m2.d.z + m->c.w * m2.d.w;
	m->c = tmp;

	tmp.x = m->d.x * m2.a.x + m->d.y * m2.a.y + m->d.z * m2.a.z + m->d.w * m2.a.w;
	tmp.y = m->d.x * m2.b.x + m->d.y * m2.b.y + m->d.z * m2.b.z + m->d.w * m2.b.w;
	tmp.z = m->d.x * m2.c.x + m->d.y * m2.c.y + m->d.z * m2.c.z + m->d.w * m2.c.w;
	tmp.w = m->d.x * m2.d.x + m->d.y * m2.d.y + m->d.z * m2.d.z + m->d.w * m2.d.w;
	m->d = tmp;
}

mat4f lookAt(const vec3f p, const vec3f to, const vec3f up)
{
	mat4f m;
	vec3f A, B, C;
	C = normalize_vec3f(sub_vec3f(to, p));
	B = normalize_vec3f(up);
	A = normalize_vec3f(cross_vec3f(C, B));
	B = cross_vec3f(A, C);
	m.a = { A.x, B.x, -C.x, 0.f };
	m.b = { A.y, B.y, -C.y, 0.f };
	m.c = { A.z, B.z, -C.z, 0.f };
	m.d = { -(p.x * A.x + p.y * A.y + p.z * A.z),
		-(p.x * B.x + p.y * B.y + p.z * B.z),
		p.x * C.x + p.y * C.y + p.z * C.z, 1.f };
	return m;
}

mat4f lookAt(const Camera camera)
{
	return lookAt(camera.position, add_vec3f(camera.forward, camera.position), camera.up);
}

struct Mesh
{
	unsigned int verticies_count;
	vec3f * vertices;
};

inline float rad(float degree)
{
	return 0.0174532925f * degree;
}

inline float lerp(float from, float to, float t)
{
	return from + (to - from) * t;
}

inline vec4f EulerToQuat(float x, float y, float z)
{
	float cosx = cosf(PI * x);
	float cosy = cosf(PI * y);
	float cosz = cosf(PI * z);
	float sinx = sinf(PI * x);
	float siny = sinf(PI * y);
	float sinz = sinf(PI * z);
	float cosycosz = cosy * cosz;
	float sinysinz = siny * sinz;
	vec4f q;
	q.w = cosx * cosycosz + sinx * sinysinz;
	q.x = sinx * cosycosz - cosx * sinysinz;
	q.y = cosx * siny * cosz + sinx * cosy * sinz;
	q.z = cosx * cosy * sinz - sinx * siny * cosz;
	return q;
}

inline vec4f EulerToQuat(const vec3f rotation)
{
	return EulerToQuat(rotation.x, rotation.y, rotation.z);
}

inline vec3f QuatToEuler(const float x, const float y, const float z, const float w)
{
	vec3f e;
	e.x = atan2f(2 * (w * x + y * z), 1 - 2 * (x * x + y * y));
	e.y = asinf(2 * (w * y - z * x));
	e.z = atan2f(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
	return scale_vec3f(e, 1.f / (2 * PI));
}

inline vec3f QuatToEuler(const vec4f q)
{
	return QuatToEuler(q.x, q.y, q.z, q.w);
}

inline vec4f quat_normalize(const vec4f q)
{
	float d_inv = 1.f / (q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
	return { d_inv * q.x, d_inv * q.y, d_inv * q.z, d_inv * q.w };
}

inline vec4f quat_conjugate(const vec4f q)
{
	return { -q.x, -q.y, -q.z, q.w };
}

inline vec4f quat_multiply(const vec4f q1, const vec4f q2)
{
	vec4f q;
	float A, B, C, D, E, F, G, H;
	A = (q1.w + q1.x) * (q2.w + q2.x);
	B = (q1.z - q1.y) * (q2.y - q2.z);
	C = (q1.w - q1.x) * (q2.y + q2.z);
	D = (q1.y + q1.z) * (q2.w - q2.x);
	E = (q1.x + q1.z) * (q2.x + q2.y);
	F = (q1.x - q1.z) * (q2.x - q2.y);
	G = (q1.w + q1.y) * (q2.w - q2.z);
	H = (q1.w - q1.y) * (q2.w + q2.z);
	q.w = B + (-E - F + G + H) / 2;
	q.x = A - (E + F + G + H) / 2;
	q.y = C + (E - F + G - H) / 2;
	q.z = D + (E - F - G + H) / 2;
	return q;
}

inline mat4f QuatToMat4f(vec4f q)
{
	mat4f m;
	float wx, wy, wz, xx, yy, yz, xy, xz, zz, x2, y2, z2;
	x2 = q.x + q.x; y2 = q.y + q.y; z2 = q.z + q.z;
	xx = q.x * x2; xy = q.x * y2; xz = q.x * z2;
	yy = q.y * y2; yz = q.y * z2; zz = q.z * z2;
	wx = q.w * x2; wy = q.w * y2; wz = q.w * z2;

	m.a.x = 1.0f - (yy + zz); m.a.y = xy - wz;
	m.a.z = xz + wy; m.a.w = 0.0f;

	m.b.x = xy + wz; m.b.y = 1.0f - (xx + zz);
	m.b.z = yz - wx; m.b.w = 0.0f;

	m.c.x = xz - wy; m.c.y = yz + wx;
	m.c.z = 1.0f - (xx + yy); m.c.w = 0.0f;

	m.d.x = 0.f; m.d.y = 0.f;
	m.d.z = 0.f; m.d.w = 1.f;
	return m;
}

inline vec4f QuatSlerp(const vec4f q1, const vec4f q2, const float t = 1.0f)
{
	float omega, cosom, sinom, scale0, scale1;
	cosom = q1.x * q2.x + q1.y * q2.y + q1.z * q2.z + q1.w * q2.w;

	if ((1.0f - cosom) > 0.001f) {
		omega = acosf(cosom);
		sinom = sinf(omega);
		scale0 = sinf((1.0f - t) * omega) / sinom;
		scale1 = sinf(t * omega) / sinom;
	}
	else {
		scale0 = 1.0f - t;
		scale1 = t;
	}

	vec4f res;
	res.x = scale0 * q1.x + scale1 * q2.x;
	res.y = scale0 * q1.y + scale1 * q2.y;
	res.z = scale0 * q1.z + scale1 * q2.z;
	res.w = scale0 * q1.w + scale1 * q2.w;
	return res;
}

inline vec4f add_vec4f(vec4f a, vec4f b)
{
	vec4f v;
	v.x = a.x + b.x;
	v.y = a.y + b.y;
	v.z = a.z + b.z;
	v.w = a.w + b.w;
	return v;
}

inline vec4f scale(vec4f v, vec4f s)
{
	return { v.x * s.x, v.y * s.y, v.z * s.z, v.w };
}

//WRONG?
inline float det(const mat3f m)
{
	return  m.a.x * m.b.y * m.c.z + m.a.y * m.b.z * m.c.x + m.a.z * m.b.x * m.c.y - \
		m.a.z * m.b.y * m.c.x - m.a.x * m.b.z * m.c.y - m.a.y * m.b.x * m.c.z;
}

//WRONG
mat3f gl_normal_matrix(const mat4f * const m)
{
	mat3f n;
	n.a = { m->a.x, m->a.y, m->a.z };
	n.b = { m->b.x, m->b.y, m->b.z };
	n.c = { m->c.x, m->c.y, m->c.z };
	float inv_det = 1.f / det(n);
	n.a.x = inv_det * (m->b.y * m->c.z - m->c.y * m->b.z);
	n.a.y = inv_det * (m->c.x * m->b.z - m->b.x * m->c.z);
	n.a.z = inv_det * (m->b.x * m->c.y - m->b.y * m->c.z);

	n.b.x = inv_det * (m->c.y * m->a.z - m->a.y * m->c.z);
	n.b.y = inv_det * (m->a.x * m->c.z - m->c.x * m->a.z);
	n.b.z = inv_det * (m->c.x * m->a.y - m->a.x * m->c.y);

	n.c.x = inv_det * (m->a.y * m->b.z - m->b.y * m->a.z);
	n.c.y = inv_det * (m->b.x * m->a.z - m->a.x * m->b.z);
	n.c.z = inv_det * (m->a.x * m->b.y - m->b.x * m->a.y);
	return n;
}

inline mat4f mat4f_identity()
{
	mat4f m;
	m.a = { 1.f, 0.f, 0.f, 0.f };
	m.b = { 0.f, 1.f, 0.f, 0.f };
	m.c = { 0.f, 0.f, 1.f, 0.f };
	m.d = { 0.f, 0.f, 0.f, 1.f };
	return m;
}

inline mat4f add_mat4f(mat4f m1, mat4f m2)
{
	mat4f res;
	res.a = add_vec4f(m1.a, m2.a);
	res.b = add_vec4f(m1.b, m2.b);
	res.c = add_vec4f(m1.c, m2.c);
	res.d = add_vec4f(m1.d, m2.d);
	return res;
}

inline mat4f scaleMat4f(vec3f s, float t = 1.f)
{
	mat4f r;
	r.a = { s.x * t, 0.f, 0.f, 0.f };
	r.b = { 0.f, s.y * t, 0.f, 0.f };
	r.c = { 0.f, 0.f, s.z * t, 0.f };
	r.d = { 0.f, 0.f, 0.f, 1.f };
	return r;
}

inline mat4f translateMat4f(vec3f T, float t = 1.f)
{
	mat4f r;
	r.a = { 1.f, 0.f, 0.f, 0.f };
	r.b = { 0.f, 1.f, 0.f, 0.f };
	r.c = { 0.f, 0.f, 1.f, 0.f };
	r.d = { T.x * t, T.y * t, T.z * t, 1.f * t };
	return r;
}

mat4f rotateMat4f(vec3f r, float t = 1.f)
{
	return QuatToMat4f(EulerToQuat(r));
}

mat4f frustumProjectionMat4f(float fov, float ratio, float n, float f)
{
	mat4f m;
	float l, r, b, t;
	l = -n * sinf(fov);
	r = -l;
	t = (r - l) / (2.f * ratio);
	b = -t;
	m.a = { 2.f * n / (r - l), 0.f, 0.f, 0.f };
	m.b = { 0.f, 2 * n / (t - b), 0.f, 0.f };
	m.c = { (r + l) / (r - l), (t + b) / (t - b), -(f + n) / (f - n), -1.f };
	m.d = { 0.f, 0.f, -2 * f * n / (f - n), 0.f };
	return m;
}

mat4f orthographicProjectionMat4f(float l, float r, float b, float t, float n, float f)
{
	mat4f m;
	m.a = { 2.f / (r - l), 0.f, 0.f, -(r + l) / (r - l) };
	m.b = { 0.f, 2.f / (t - b), 0.f, -(t + b) / (t - b) };
	m.c = { 0.f, 0.f, -2.f / (f - n), -(f + n) / (f - n) };
	m.d = { 0.f, 0.f, 0.f, 1.f };
	return m;
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

bool BuildShaderProgram(GLint * program,
						std::vector<std::string> vertex_shaders, std::vector<std::string> fragment_shaders,
						std::ostream* log = nullptr)
{
	std::vector<GLint> ids;
	unsigned int programId;
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

bool LoadTextureFromFile(const char * path, unsigned int * id, bool flip = true,
	unsigned int wrap_s = GL_REPEAT, unsigned int wrap_t = GL_REPEAT,
	unsigned int min_filter = GL_LINEAR, unsigned int mag_filter = GL_LINEAR)
{
	stbi_set_flip_vertically_on_load(flip);
	glGenTextures(1, id);
	glBindTexture(GL_TEXTURE_2D, *id);
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
	else
		return false;
	stbi_image_free(data);
	return true;
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
	BuildShaderProgram(&shader.id, { "g_buffer_vs.txt" }, { "g_buffer_fs.txt" }, &std::cout);
	shader.projectionLoc = glGetUniformLocation(shader.id, "projection");
	shader.viewLoc = glGetUniformLocation(shader.id, "view");
	shader.modelLoc = glGetUniformLocation(shader.id, "model");
	shader.normalLoc = glGetUniformLocation(shader.id, "normal");
	shader.diffuseLoc = glGetUniformLocation(shader.id, "texture_diffuse");
	shader.specularLoc = glGetUniformLocation(shader.id, "texture_specular");

	return shader;
}

LightingShader createLightingShader()
{
	static bool RUN_ONCE = true;
	assert(RUN_ONCE);
	RUN_ONCE = false;

	LightingShader shader = {};
	shader.id = BuildShaderProgram(&shader.id, { "light_vs.txt" }, { "light_fs.txt" }, &std::cout);
	shader.albedoSpecLoc = glGetUniformLocation(shader.id, "albedo");
	shader.normalLoc = glGetUniformLocation(shader.id, "normal");
	shader.positionLoc = glGetUniformLocation(shader.id, "position");

	return shader;
}

// renderQuad() renders a 1x1 XY quad in NDC
// -----------------------------------------
unsigned int quadVAO = 0;
unsigned int quadVBO;
void renderQuad()
{
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

void GeometryPass(GeometryShader shader, GBuffer gBuffer, Scene scene, mat4f view, mat4f projection)
{
	glUseProgram(shader.id);

	//Matrices
	glUniformMatrix4fv(shader.viewLoc, 1, GL_FALSE, value_ptr(view));
	glUniformMatrix4fv(shader.projectionLoc, 1, GL_FALSE, value_ptr(projection));

	glBindFramebuffer(GL_FRAMEBUFFER, gBuffer.id);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	mat4f model;
	for (int i = 0; i < scene.entity_count; i++)
	{
		model = mat4f_identity();
		mat4f_rotate_quat(&model, scene.entities[i].orientation);
		mat4f_translate(&model, scene.entities[i].position);
		mat4f_scale(&model, scene.entities[i].scale);
		glUniformMatrix4fv(shader.modelLoc, 1, GL_FALSE, value_ptr(model));
		glUniform1i(shader.diffuseLoc, scene.entities[i].model.diffuse);
		glUniform1i(shader.specularLoc, scene.entities[i].model.specular);
		glBindVertexArray(scene.entities[i].model.vao);
		glDrawArrays(GL_TRIANGLES, 0, 36);
	}

	//glBindFramebuffer(GL_FRAMEBUFFER, 0);
	//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	renderQuad();
}

void LightingPass(LightingShader shader, GBuffer gBuffer, Scene scene)
{
	glUseProgram(shader.id);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, gBuffer.position);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, gBuffer.normal);
	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, gBuffer.albedoSpec);
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
	glfwWindowHint(GLFW_SAMPLES, 4);

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
	
	// set up vertex data (and buffer(s)) and configure vertex attributes
	// ------------------------------------------------------------------
	float vertices[] = {
		// positions          // normals           // texture coords
		-0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f, 0.0f,
		0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  1.0f, 0.0f,
		0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  1.0f, 1.0f,
		0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  1.0f, 1.0f,
		-0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f, 1.0f,
		-0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f, 0.0f,

		-0.5f, -0.5f,  0.5f,  0.0f,  0.0f, 1.0f,   0.0f, 0.0f,
		0.5f, -0.5f,  0.5f,  0.0f,  0.0f, 1.0f,   1.0f, 0.0f,
		0.5f,  0.5f,  0.5f,  0.0f,  0.0f, 1.0f,   1.0f, 1.0f,
		0.5f,  0.5f,  0.5f,  0.0f,  0.0f, 1.0f,   1.0f, 1.0f,
		-0.5f,  0.5f,  0.5f,  0.0f,  0.0f, 1.0f,   0.0f, 1.0f,
		-0.5f, -0.5f,  0.5f,  0.0f,  0.0f, 1.0f,   0.0f, 0.0f,

		-0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  1.0f, 0.0f,
		-0.5f,  0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  1.0f, 1.0f,
		-0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  0.0f, 1.0f,
		-0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  0.0f, 1.0f,
		-0.5f, -0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  0.0f, 0.0f,
		-0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  1.0f, 0.0f,

		0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f,
		0.5f,  0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f,
		0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  0.0f, 1.0f,
		0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  0.0f, 1.0f,
		0.5f, -0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  0.0f, 0.0f,
		0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f,

		-0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  0.0f, 1.0f,
		0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  1.0f, 1.0f,
		0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  1.0f, 0.0f,
		0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  1.0f, 0.0f,
		-0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  0.0f, 0.0f,
		-0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  0.0f, 1.0f,

		-0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  0.0f, 1.0f,
		0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  1.0f, 1.0f,
		0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  1.0f, 0.0f,
		0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  1.0f, 0.0f,
		-0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  0.0f, 0.0f,
		-0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  0.0f, 1.0f
	};

	unsigned int VBO, VAO;
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);

	// bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);

	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
	glEnableVertexAttribArray(2);

	//Load textures
	unsigned int wooden_container_texture, steelframed_container_texture, steelframed_container_specular, awesomeface_texture, matrix_texture;
	LoadTextureFromFile("container.jpg", &wooden_container_texture, false);
	LoadTextureFromFile("container2.png", &steelframed_container_texture, false);
	LoadTextureFromFile("container2_specular.png", &steelframed_container_specular, false);
	LoadTextureFromFile("awesomeface.png", &awesomeface_texture, false);
	LoadTextureFromFile("matrix.jpg", &matrix_texture, false);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, wooden_container_texture);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, awesomeface_texture);
	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, steelframed_container_texture);
	glActiveTexture(GL_TEXTURE3);
	glBindTexture(GL_TEXTURE_2D, steelframed_container_specular);
	glActiveTexture(GL_TEXTURE4);
	glBindTexture(GL_TEXTURE_2D, matrix_texture);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0); //Usually not needed.

	glPolygonMode(GL_FRONT, GL_TRIANGLES);

	float t = 0.f;
	float timeStep = 0.01f;
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
	float cameraSpeedFast = 10.f;

	//Scene
	Scene scene = {};
	scene.entities = (Entity*)malloc(sizeof(Entity) * Scene::MAX_ENTITIES);
	for (int i = 0; i < 100; i++)
	{
		Entity cubeEntity = {};
		cubeEntity.model.diffuse = steelframed_container_texture;
		cubeEntity.model.specular = steelframed_container_specular;
		cubeEntity.model.vao = VAO;
		cubeEntity.position = { randf(-10.f, 10.f), randf(-10.f, 10.f), randf(-10.f, 10.f) };
		cubeEntity.scale = { 1.f, 1.f, 1.f };
		cubeEntity.orientation = EulerToQuat(0.f, 0.f, 0.f);
		scene.entities[scene.entity_count++] = cubeEntity;
	}

	//Shaders
	GeometryShader geometryShader = createGeometryShader();
	LightingShader lightingShader = createLightingShader();

	//GBuffer
	GBuffer gBuffer = createGBuffer();

	float oldCutoff = 0.f;
	bool gamma_correct = false;
	float gamma = 2.2f;

	//Callbacks
	glfwSetKeyCallback(window, key_callback);

	while (!glfwWindowShouldClose(window))
	{
		processInput(window);
		pollMouse(window, &mouse);

		vec3f movement = {};
		vec3f side = normalize_vec3f(cross_vec3f(camera.up, camera.forward));
		float speed = cameraSpeed;
		if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
			speed = cameraSpeedFast;
		if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
			movement = add_vec3f(movement, scale_vec3f(side, speed * dt));
		if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
			movement = add_vec3f(movement, scale_vec3f(side, -speed * dt));
		if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
			movement = add_vec3f(movement, scale_vec3f(camera.forward, speed * dt));
		if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
			movement = add_vec3f(movement, scale_vec3f(camera.forward, -speed * dt));
		if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
			camera.up = rotate_vec3f_around_axis(camera.up, cross_vec3f(camera.up, side), 0.0001f * dt);
		if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
			camera.up = rotate_vec3f_around_axis(camera.up, cross_vec3f(camera.up, side), -0.0001f * dt);
		if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS)
		{
			if (!gamma_correct)
				gamma = 2.2f;
			else
				gamma = 1.f;
			gamma_correct = !gamma_correct;
		}

		camera.position = add_vec3f(camera.position, movement);
		float pitch = (mouse.last_x - mouse.x) * mouse_sensitivity;
		float roll = (mouse.y - mouse.last_y) * mouse_sensitivity;

		camera.forward = rotate_vec3f_around_axis(camera.forward, side, roll * 5.f);
		camera.forward = rotate_vec3f_around_axis(camera.forward, camera.up, pitch * 5.f);

		// Avoid axis-flipping camera around forward.y +/- 1
		if (dot_vec3f(camera.forward, cross_vec3f(side, camera.up)) < 0.f)
			camera.forward = normalize_vec3f(sub_vec3f(scale_vec3f(camera.up, 3.f * dot_vec3f(camera.forward, camera.up)), camera.forward));

		GeometryPass(geometryShader, gBuffer, scene, lookAt(camera), frustumProjectionMat4f(85.f, (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.f));
		glBindFramebuffer(GL_READ_FRAMEBUFFER, gBuffer.id);
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0); // write to default framebuffer
												   // blit to default framebuffer. Note that this may or may not work as the internal formats of both the FBO and default framebuffer have to match.
												   // the internal formats are implementation defined. This works on all of my systems, but if it doesn't on yours you'll likely have to write to the 		
												   // depth buffer in another shader stage (or somehow see to match the default framebuffer's internal format with the FBO's internal format).
		glBlitFramebuffer(0, 0, SCR_WIDTH, SCR_HEIGHT, 0, 0, SCR_WIDTH, SCR_HEIGHT, GL_DEPTH_BUFFER_BIT, GL_NEAREST);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glDrawBuffer(gBuffer.id);


		glfwSwapBuffers(window);
		glfwPollEvents();

		//Timestep
		float time = glfwGetTime();
		dt = time - t;
		t = time;

		if (dt < timeStep)
			Sleep((timeStep - dt) * 10);
	}

	//0 optional: de-allocate all resources once they've outlived their purpose:
	glDeleteVertexArrays(1, &VAO);
	glDeleteBuffers(1, &VBO);

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