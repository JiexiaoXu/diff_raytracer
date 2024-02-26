#include <tuple>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cmath>

// lib for saving pictures
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// include autodiff lib and eigen
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <Eigen/Dense>

// replace the vec3
using namespace autodiff;
using vec3 = Eigen::Matrix<real, 3, 1>;

// TODO: Add Loss Function 
// TODO: Add Gradient Descent 
// TODO[optional]: Add Optimizer 

struct Material {
    real refractive_index = 1;
    real albedo[4] = {2,0,0,0};
    vec3 diffuse_color = {0,0,0};
    real specular_exponent = 0;
};

struct Sphere {
    vec3 center;
    real radius;
    Material material;
};

const Material      ivory = {1.0, {0.9,  0.5, 0.1, 0.0}, {0.4, 0.4, 0.3},   50.};
const Material      glass = {1.5, {0.0,  0.9, 0.1, 0.8}, {0.6, 0.7, 0.8},  125.};
const Material red_rubber = {1.0, {1.4,  0.3, 0.0, 0.0}, {0.3, 0.1, 0.1},   10.};
const Material     mirror = {1.0, {0.0, 16.0, 0.8, 0.0}, {1.0, 1.0, 1.0}, 1425.};

const Sphere spheres[] = {
    {{-3,    0,   -16}, 2,      ivory},
    {{-1.0, -1.5, -12}, 2,      glass},
    {{ 1.5, -0.5, -18}, 3, red_rubber},
    {{ 7,    5,   -18}, 4,     mirror}
};

const vec3 lights[] = {
    {-20, 20,  20},
    { 30, 50, -25},
    { 30, 20,  30}
};

vec3 reflect(const vec3 &I, const vec3 &N) {
    return I - N*2.f*(I.dot(N));
}

vec3 refract(const vec3 &I, const vec3 &N, const real eta_t, const real eta_i=1.f) { // Snell's law
    real cosi = - max(real(-1), min(real(1), I.dot(N)));
    if (cosi<0) return refract(I, -N, eta_i, eta_t); // if the ray comes from the inside the object, swap the air and the media
    real eta = eta_i / eta_t;
    real k = 1 - eta*eta*(1 - cosi*cosi);
    return k<0 ? vec3{1,0,0} : I*eta + N*(eta*cosi - sqrt(k)); // k<0 = total reflection, no ray to refract. I refract it anyways, this has no physical meaning
}

std::tuple<bool,real> ray_sphere_intersect(const vec3 &orig, const vec3 &dir, const Sphere &s) { // ret value is a pair [intersection found, distance]
    vec3 L = s.center - orig;
    real tca = L.dot(dir);
    real d2 = L.dot(L) - tca*tca;
    if (d2 > s.radius*s.radius) return {false, 0};
    real thc = sqrt(s.radius*s.radius - d2);
    real t0 = tca-thc, t1 = tca+thc;
    if (t0>.001) return {true, t0};  // offset the original point by .001 to avoid occlusion by the object itself
    if (t1>.001) return {true, t1};
    return {false, 0};
}

std::tuple<bool,vec3,vec3,Material> scene_intersect(const vec3 &orig, const vec3 &dir) {
    vec3 pt, N;
    Material material;

    real nearest_dist = 1e10;
    if (abs(dir.y())>.001) { // intersect the ray with the checkerboard, avoid division by zero
        real d = -(orig.y() + real(4)) / dir.y(); // the checkerboard plane has equation y = -4
        vec3 p = orig + dir * d;
        if (d>real(.001) && d<nearest_dist && abs(p.x())<real(10) && p.z()<real(-10) && p.z()>real(-30)) {
            nearest_dist = d;
            pt = p;
            N = {0,1,0};
            material.diffuse_color = (int(.5*pt.x()+1000) + int(.5*pt.z())) & 1 ? vec3{.3, .3, .3} : vec3{.3, .2, .1};
        }
    }

    for (const Sphere &s : spheres) { // intersect the ray with all spheres
        auto [intersection, d] = ray_sphere_intersect(orig, dir, s);
        if (!intersection || d > nearest_dist) continue;
        nearest_dist = d;
        pt = orig + dir*nearest_dist;
        N = (pt - s.center).normalized();
        material = s.material;
    }
    return { nearest_dist<1000, pt, N, material };
}

vec3 cast_ray(const vec3 &orig, const vec3 &dir, const int depth=0) {
    // TODO: make cast_ray differentiable 

    auto [hit, point, N, material] = scene_intersect(orig, dir);
    if (depth>4 || !hit)
        return {0.2, 0.7, 0.8}; // background color

    vec3 reflect_dir = reflect(dir, N).normalized();
    vec3 refract_dir = refract(dir, N, material.refractive_index).normalized();
    vec3 reflect_color = cast_ray(point, reflect_dir, depth + 1);
    vec3 refract_color = cast_ray(point, refract_dir, depth + 1);

    real diffuse_light_intensity = 0, specular_light_intensity = 0;
    for (const vec3 &light : lights) { // checking if the point lies in the shadow of the light
        vec3 light_dir = (light - point).normalized();
        auto [hit, shadow_pt, trashnrm, trashmat] = scene_intersect(point, light_dir);
        if (hit && (shadow_pt-point).norm() < (light-point).norm()) continue;
        diffuse_light_intensity  += max(real(0), light_dir.dot(N));
        specular_light_intensity += pow(max(real(0), -reflect(-light_dir, N).dot(dir)), material.specular_exponent);
    }
    return material.diffuse_color * diffuse_light_intensity * material.albedo[0] + vec3{1., 1., 1.}*specular_light_intensity * material.albedo[1] + reflect_color*material.albedo[2] + refract_color*material.albedo[3];
}

real loss(std::vector<unsigned char> imageX, std::vector<unsigned char> imagey) {
    return 0.0f;
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
        framebuffer[pix] = cast_ray(vec3{0,0,0}, vec3{dir_x, dir_y, dir_z}.normalized());
    }

    std::vector<unsigned char> image(width * height * 3); // 3 bytes per pixel for RGB

    for (int i = 0; i < width * height; ++i) {
        for (int chan = 0; chan < 3; ++chan) {
            real max_val = max(real(1), max(framebuffer[i][0], max(framebuffer[i][1], framebuffer[i][2])));
            image[i * 3 + chan] = static_cast<unsigned char>(255 * max(real(0), min(real(1), framebuffer[i][chan] / max_val)));
        }
    }

    // Save the image using stb_image_write
    stbi_write_png("output.png", width, height, 3, image.data(), width * 3);
    return 0;
}

