#include <tuple>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <iostream>

// lib for saving pictures
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

struct vec3 {
    float x=0, y=0, z=0;
          float& operator[](const int i)       { return i==0 ? x : (1==i ? y : z); }
    const float& operator[](const int i) const { return i==0 ? x : (1==i ? y : z); }
    vec3  operator*(const float v) const { return {x*v, y*v, z*v};       }
    float operator*(const vec3& v) const { return x*v.x + y*v.y + z*v.z; }
    vec3  operator+(const vec3& v) const { return {x+v.x, y+v.y, z+v.z}; }
    vec3  operator-(const vec3& v) const { return {x-v.x, y-v.y, z-v.z}; }
    vec3  operator-()              const { return {-x, -y, -z};          }
    float norm() const { return std::sqrt(x*x+y*y+z*z); }
    vec3 normalized() const { return (*this)*(1.f/norm()); }
    // std::string toString() const{}
};

vec3 cross(const vec3 v1, const vec3 v2) {
    return { v1.y*v2.z - v1.z*v2.y, v1.z*v2.x - v1.x*v2.z, v1.x*v2.y - v1.y*v2.x };
}

struct Material {
    float refractive_index = 1;
    float albedo[4] = {2,0,0,0};
    vec3 diffuse_color = {0,0,0};
    float specular_exponent = 0;
    void toString() const {
        std::cout << "refractive_index "  << refractive_index << std::endl;
        std::cout << "diffuse color " << diffuse_color.x << std::endl;
        std::cout << "specular exponent " << specular_exponent << std::endl;
    }
};

struct Sphere {
    vec3 center;
    float radius;
    Material material;
};

struct Triangle {
    vec3 p1;
    vec3 p2;
    vec3 p3;
    Material material;
};

constexpr Material      ivory = {1.0, {0.9,  0.5, 0.1, 0.0}, {0.4, 0.4, 0.3},   50.};
constexpr Material      glass = {1.5, {0.0,  0.9, 0.1, 0.8}, {0.6, 0.7, 0.8},  125.};
constexpr Material red_rubber = {1.0, {1.4,  0.3, 0.0, 0.0}, {0.3, 0.1, 0.1},   10.};
constexpr Material     mirror = {1.0, {0.0, 16.0, 0.8, 0.0}, {1.0, 1.0, 1.0}, 1425.};

constexpr Sphere spheres[] = {
    // {{-3,    0,   -16}, 2,      ivory},
    // {{-1.0, -1.5, -12}, 2,      glass},
    // {{ 1.5, -0.5, -18}, 3, red_rubber},
    // {{ 7,    5,   -18}, 4,     mirror}
};

constexpr Triangle triangles[] = {
    {{-3, 0, -16}, {-1.0, -1.5, -12},{1.5, -0.5, -18}, ivory}
};

constexpr vec3 lights[] = {
    {-20, 20,  20},
    { 30, 50, -25},
    { 30, 20,  30}
};

vec3 reflect(const vec3 &I, const vec3 &N) {
    return I - N*2.f*(I*N);
}

vec3 refract(const vec3 &I, const vec3 &N, const float eta_t, const float eta_i=1.f) { // Snell's law
    float cosi = - std::max(-1.f, std::min(1.f, I*N));
    if (cosi<0) return refract(I, -N, eta_i, eta_t); // if the ray comes from the inside the object, swap the air and the media
    float eta = eta_i / eta_t;
    float k = 1 - eta*eta*(1 - cosi*cosi);
    return k<0 ? vec3{1,0,0} : I*eta + N*(eta*cosi - std::sqrt(k)); // k<0 = total reflection, no ray to refract. I refract it anyways, this has no physical meaning
}

std::tuple<bool,float> ray_sphere_intersect(const vec3 &orig, const vec3 &dir, const Sphere &s) { // ret value is a pair [intersection found, distance]
    vec3 L = s.center - orig;
    float tca = L*dir;
    float d2 = L*L - tca*tca;
    if (d2 > s.radius*s.radius) return {false, 0};
    float thc = std::sqrt(s.radius*s.radius - d2);
    float t0 = tca-thc, t1 = tca+thc;
    if (t0>.001) return {true, t0};  // offset the original point by .001 to avoid occlusion by the object itself
    if (t1>.001) return {true, t1};
    return {false, 0};
}

// return tuple of (boolean hit, float hit_time, vec normal)
std::tuple<bool, float, vec3> ray_triangle_intersect(const vec3 &orig, const vec3 &dir, const Triangle &t) {
    float u, v;  // barycentric coordinates, last one is (1 - u - v)
	float time;  // time for line pass through triangle
	vec3 e1 = t.p2 - t.p1;
	vec3 e2 = t.p3 - t.p1;
    
    vec3 p = cross(dir, e2);
	float divisor = p*e1;
	if (std::abs(divisor) < (float)1e-6) { // check if parallel
		return {false, 0, vec3()};
	}

    vec3 s = orig - t.p1;
	float cramer_factor = 1.0f / divisor;
	// u = cramer_factor * ((-1.0f) * dot(cross(s, e2), d));
	u = cramer_factor * (s * p);
	if (u < (float)1e-6 || u > 1.0f) {
		return {false, 0, vec3()};
	}

	// v = cramer_factor * dot(cross(e1, d), s);
	v = cramer_factor * (cross(s, e1) * dir);
	if (v < (float)1e-6 || u + v > 1.0f) {
		return {false, 0, vec3()};
	}

	time = cramer_factor * (e2 * cross(s, e1));
    if (time < 0) {
        return {false, 0, vec3()};
    }

    vec3 N = cross(e1, e2);   
    return {true, time, N.normalized()};
}

std::tuple<bool,vec3,vec3,Material> scene_intersect(const vec3 &orig, const vec3 &dir) {
    vec3 pt, N;
    Material material;

    float nearest_dist = 1e10;
    if (std::abs(dir.y)>.001) { // intersect the ray with the checkerboard, avoid division by zero
        float d = -(orig.y+4)/dir.y; // the checkerboard plane has equation y = -4
        vec3 p = orig + dir*d;
        if (d>.001 && d<nearest_dist && std::abs(p.x)<10 && p.z<-10 && p.z>-30) {
            nearest_dist = d;
            pt = p;
            N = {0,1,0};
            material.diffuse_color = (int(.5*pt.x+1000) + int(.5*pt.z)) & 1 ? vec3{.3, .3, .3} : vec3{.3, .2, .1};
        }
    }

    // for (const Sphere &s : spheres) { // intersect the ray with all spheres
    //     auto [intersection, d] = ray_sphere_intersect(orig, dir, s);
    //     if (!intersection || d > nearest_dist) continue;
    //     nearest_dist = d;
    //     pt = orig + dir*nearest_dist;
    //     N = (pt - s.center).normalized();
    //     material = s.material;
    // }

    for (const Triangle &t : triangles) {
        auto [intersection, d, normal] = ray_triangle_intersect(orig, dir, t);
        if (!intersection || d > nearest_dist) continue;
        nearest_dist = d;
        pt = orig + dir*nearest_dist;
        N = normal;
        material = t.material;
    }

    return { nearest_dist<1000, pt, N, material };
}

vec3 cast_ray(const vec3 &orig, const vec3 &dir, const int depth=0) {
    auto [hit, point, N, material] = scene_intersect(orig, dir);
    
    if (depth>4 || !hit)
        return {0.2, 0.7, 0.8}; // background color

    vec3 reflect_dir = reflect(dir, N).normalized();
    vec3 refract_dir = refract(dir, N, material.refractive_index).normalized();
    vec3 reflect_color = cast_ray(point, reflect_dir, depth + 1);
    vec3 refract_color = cast_ray(point, refract_dir, depth + 1);

    float diffuse_light_intensity = 0, specular_light_intensity = 0;
    for (const vec3 &light : lights) { // checking if the point lies in the shadow of the light
        vec3 light_dir = (light - point).normalized();
        auto [hit, shadow_pt, trashnrm, trashmat] = scene_intersect(point + light_dir * 0.1f, light_dir);
        if (hit && (shadow_pt-point).norm() < (light-point).norm()) continue;
        diffuse_light_intensity  += std::max(0.f, light_dir*N);
        specular_light_intensity += std::pow(std::max(0.f, -reflect(-light_dir, N)*dir), material.specular_exponent);
    }
    vec3 diffuse_color = material.diffuse_color * diffuse_light_intensity * material.albedo[0];
    vec3 specular_color = vec3{1., 1., 1.}*specular_light_intensity * material.albedo[1];
    vec3 reflect_color_res = reflect_color*material.albedo[2];
    vec3 refract_color_res =  refract_color*material.albedo[3];

    return diffuse_color + specular_color + reflect_color_res + refract_color_res;
}

int main() {
    constexpr int   width  = 1024;
    constexpr int   height = 768;
    constexpr float fov    = 1.05; // 60 degrees field of view in radians
    std::vector<vec3> framebuffer(width*height);
#pragma omp parallel for
    for (int pix = 0; pix<width*height; pix++) { // actual rendering loop
        float dir_x =  (pix%width + 0.5) -  width/2.;
        float dir_y = -(pix/width + 0.5) + height/2.; // this flips the image at the same time
        float dir_z = -height/(2.*tan(fov/2.));
        vec3 pix_color = cast_ray(vec3{0,0,0}, vec3{dir_x, dir_y, dir_z}.normalized());
        pix_color.x = std::clamp(pix_color.x, 0.0f, 1.0f);
        pix_color.y = std::clamp(pix_color.y, 0.0f, 1.0f);
        pix_color.z = std::clamp(pix_color.z, 0.0f, 1.0f);
        framebuffer[pix] = pix_color ;
    }

    std::vector<unsigned char> image(width * height * 3); // 3 bytes per pixel for RGB

    for (int i = 0; i < width * height; ++i) {
        for (int chan = 0; chan < 3; ++chan) {
            float max = std::max(1.f, std::max(framebuffer[i][0], std::max(framebuffer[i][1], framebuffer[i][2])));
            image[i * 3 + chan] = static_cast<unsigned char>(255 * std::max(0.f, std::min(1.f, framebuffer[i][chan] / max)));
        }
    }

    // Save the image using stb_image_write
    stbi_write_png("triangle.png", width, height, 3, image.data(), width * 3);
    return 0;
}

