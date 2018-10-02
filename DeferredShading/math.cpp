#include "math.h"
#include <random>

float* value_ptr(vec3f v)
{
	return &v.x;
}

float* value_ptr(vec4f v)
{
	return &v.x;
}

float* value_ptr(mat4f m)
{
	return &m.a.x;
}

float randf(float min, float max)
{
	float r = (float)rand() / RAND_MAX;
	r *= max - min;
	r += min;
	return r;
}

float length(const vec3f v)
{
	return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

vec3f scale_vec3f(const vec3f v, const float s)
{
	return { v.x * s, v.y * s, v.z * s };
}

vec3f add_vec3f(const vec3f a, const vec3f b)
{
	return { a.x + b.x, a.y + b.y, a.z + b.z };
}

vec3f sub_vec3f(const vec3f a, const vec3f b)
{
	return { a.x - b.x, a.y - b.y, a.z - b.z };
}

float dot_vec3f(const vec3f a, const vec3f b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

vec3f cross_vec3f(const vec3f a, const vec3f b)
{
	return { a.y * b.z - b.y * a.z, a.z * b.x - b.z * a.x, a.x * b.y - b.x * a.y };
}

vec3f normalize_vec3f(const vec3f v)
{
	float il = 1.f / sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
	return { v.x * il, v.y * il, v.z * il };
}

/* Second-degree polynomial approximating inverse square root of length(v) */
vec3f normalize_fast(const vec3f v)
{
	float d = v.x * v.x + v.y * v.y + v.z * v.z;
	d = 15.f / 8.f - 5.f / 4.f * d + 3.f / 8.f * d * d;
	return { v.x * d, v.y * d, v.z * d };
}

vec3f rotate_vec3f_by_quat(const vec3f v, const vec4f q)
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

vec3f rotate_vec3f_around_axis(const vec3f v, const vec3f axis, float radians)
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

struct Mesh
{
	unsigned int verticies_count;
	vec3f * vertices;
};

float rad(float degree)
{
	return 0.0174532925f * degree;
}

float lerp(float from, float to, float t)
{
	return from + (to - from) * t;
}

vec4f EulerToQuat(float x, float y, float z)
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

vec4f EulerToQuat(const vec3f rotation)
{
	return EulerToQuat(rotation.x, rotation.y, rotation.z);
}

vec3f QuatToEuler(const float x, const float y, const float z, const float w)
{
	vec3f e;
	e.x = atan2f(2 * (w * x + y * z), 1 - 2 * (x * x + y * y));
	e.y = asinf(2 * (w * y - z * x));
	e.z = atan2f(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
	return scale_vec3f(e, 1.f / (2 * PI));
}

vec3f QuatToEuler(const vec4f q)
{
	return QuatToEuler(q.x, q.y, q.z, q.w);
}

vec4f quat_normalize(const vec4f q)
{
	float d_inv = 1.f / (q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
	return { d_inv * q.x, d_inv * q.y, d_inv * q.z, d_inv * q.w };
}

vec4f quat_conjugate(const vec4f q)
{
	return { -q.x, -q.y, -q.z, q.w };
}

vec4f quat_multiply(const vec4f q1, const vec4f q2)
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

mat4f QuatToMat4f(vec4f q)
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

/**
Should look into lerp-interpolation instead
*/
vec4f QuatSlerp(const vec4f q1, const vec4f q2, const float t)
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

vec4f add_vec4f(vec4f a, vec4f b)
{
	vec4f v;
	v.x = a.x + b.x;
	v.y = a.y + b.y;
	v.z = a.z + b.z;
	v.w = a.w + b.w;
	return v;
}

vec4f scale(vec4f v, vec4f s)
{
	return { v.x * s.x, v.y * s.y, v.z * s.z, v.w };
}

mat4f mat4f_identity()
{
	mat4f m;
	m.a = { 1.f, 0.f, 0.f, 0.f };
	m.b = { 0.f, 1.f, 0.f, 0.f };
	m.c = { 0.f, 0.f, 1.f, 0.f };
	m.d = { 0.f, 0.f, 0.f, 1.f };
	return m;
}

mat4f add_mat4f(mat4f m1, mat4f m2)
{
	mat4f res;
	res.a = add_vec4f(m1.a, m2.a);
	res.b = add_vec4f(m1.b, m2.b);
	res.c = add_vec4f(m1.c, m2.c);
	res.d = add_vec4f(m1.d, m2.d);
	return res;
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