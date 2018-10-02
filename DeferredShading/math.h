#pragma once
#include <math.h>

//Constants
static const float PI = acosf(-1);

struct vec2f
{
	float x, y;
};

struct vec3f
{
	float x, y, z;
	inline vec3f() : x(0.f), y(0.f), z(0.f) {}
	inline vec3f(float x, float y, float z) : x(x), y(y), z(z) {}
	inline vec3f(const vec3f& other) : x(other.x), y(other.y), z(other.z) {}
	inline vec3f operator+(const vec3f& rhs) const { return vec3f(x + rhs.x, y + rhs.y, z + rhs.z); }
	inline vec3f operator-(const vec3f& rhs) const { return vec3f(x - rhs.x, y - rhs.y, z - rhs.z); }
	inline vec3f operator*(const float scale) const { return { x * scale, y * scale, z * scale }; }
	inline float operator*(const vec3f& rhs) const { return x * rhs.x + y * rhs.y + z * rhs.z; }
	inline void operator+=(const vec3f& rhs) { x += rhs.x; y += rhs.y; z += rhs.z; }
	inline void operator-=(const vec3f& rhs) { x -= rhs.x; y = rhs.y; z -= rhs.z; }
	inline void operator*=(const float scale) { x *= scale; y *= scale; z *= scale; }

	inline vec3f cross(const vec3f& rhs) { return vec3f(y * rhs.z - rhs.y * z, z * rhs.x - rhs.z * x, x * rhs.y - rhs.x * y); };
	inline vec3f scale(const float s) const { return { x * s, y * s, z * s }; }

	inline void normalize()
	{
		float d = x * x + y * y + z * z;
		d = 15.f / 8.f - 5.f / 4.f * d + 3.f / 8.f * d * d;
		x *= d; y *= d; z *= d;
	}
	inline vec3f normalized() const { vec3f v = { x, y, z }; v.normalize(); return v; }
};

struct vec4f
{
	float x, y, z, w;
	inline vec4f() : x(0.f), y(0.f), z(0.f), w(0.f) {}
	inline vec4f(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
	inline vec4f(const vec4f& other) : x(other.x), y(other.y), z(other.z), w(other.w) {}
	inline vec4f operator+(const vec4f& rhs) { return vec4f(x + rhs.x, y + rhs.y, z + rhs.z, w + rhs.w); } const
	inline vec4f operator-(const vec4f& rhs) { return vec4f(x - rhs.x, y - rhs.y, z - rhs.z, w - rhs.w); } const
	inline float operator*(const vec4f& rhs) { return x * rhs.x + y * rhs.y + z * rhs.z + w * rhs.w; } const
	inline void operator+=(const vec4f& rhs) { x += rhs.x; y += rhs.y; z += rhs.z; w += rhs.w; }
	inline void operator-=(const vec4f& rhs) { x -= rhs.x; y = rhs.y; z -= rhs.z; w -= rhs.w; }
	inline void operator*=(const float scale) { x *= scale; y *= scale; z *= scale; w *= scale; }
};

struct mat4f
{
	vec4f a, b, c, d;
	mat4f() : a(1.f, 0.f, 0.f, 0.f), b(0.f, 1.f, 0.f, 0.f), c(0.f, 0.f, 1.f, 0.f), d(0.f, 0.f, 0.f, 1.f) {}
	mat4f(vec4f a, vec4f b, vec4f c, vec4f d) : a(a), b(b), c(c), d(d) {}
	mat4f(const mat4f& other) : a(other.a), b(other.b), c(other.c), d(other.d) {}
};

void mat4f_scalar(const float s, mat4f * m);
void mat4f_scale(mat4f * m, const vec3f scale);
void mat4f_translate(mat4f * m, const vec3f t);
void mat4f_transpose(mat4f * m);
void mat4f_rotate_quat(mat4f * m, const vec4f q);
void mat4f_multiply(mat4f * m, mat4f m2);
mat4f mat4f_identity();

mat4f lookAt(const vec3f p, const vec3f to, const vec3f up);
mat4f frustumProjectionMat4f(float fov, float ratio, float n, float f);
mat4f orthographicProjectionMat4f(float l, float r, float b, float t, float n, float f);

vec3f rotate_vec3f_by_quat(const vec3f v, const vec4f q);
vec3f rotate_vec3f_around_axis(const vec3f v, const vec3f axis, float radians);
vec4f EulerToQuat(float x, float y, float z);
vec4f EulerToQuat(const vec3f rotation);
vec3f QuatToEuler(const float x, const float y, const float z, const float w);
vec3f QuatToEuler(const vec4f q);
mat4f QuatToMat4f(vec4f q);
vec4f quat_normalize(const vec4f q);
vec4f quat_conjugate(const vec4f q);
vec4f quat_multiply(const vec4f q1, const vec4f q2);
vec4f QuatSlerp(const vec4f q1, const vec4f q2, const float t = 1.0f);

float randf(float min, float max);
float rad(float degree);
float lerp(float from, float to, float t);

float * value_ptr(vec3f v);
float * value_ptr(vec4f v);
float * value_ptr(mat4f m);