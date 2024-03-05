#include <tuple>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <chrono>

// lib for saving pictures
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// include autodiff lib and eigen
#include <autodiff/reverse/var/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>
#include <Eigen/Dense>

// replace the vec3
using namespace autodiff;
using vec3 = Eigen::Matrix<var, 3, 1>;
using vec3d = Eigen::Vector3d;

constexpr int   width  = 200;
constexpr int   height = 200;
constexpr float fov    = 1.05; // 60 degrees field of view in radians

// TODO: Add Loss Function 
// TODO: Add Gradient Descent 
// TODO[optional]: Add Optimizer 


// define the primitives 
struct Material {
    double refractive_index  = 1;
    double albedo[4]         = {2,0,0,0};
    vec3d diffuse_color      = {0,0,0};
    double specular_exponent = 0;
};

struct Sphere {
    vec3 center;
    var radius;
    Material material;
};

// struct parameters {
//     Sphere spehres[];
//     vec3 lights[];
// };

const Material      ivory = {1.0, {0.9,  0.5, 0.1, 0.0}, {0.4, 0.4, 0.3},   50.};
const Material      glass = {1.5, {0.0,  0.9, 0.1, 0.8}, {0.6, 0.7, 0.8},  125.};
const Material red_rubber = {1.0, {1.4,  0.3, 0.0, 0.0}, {0.3, 0.1, 0.1},   10.};
const Material     mirror = {1.0, {0.0, 16.0, 0.8, 0.0}, {1.0, 1.0, 1.0}, 1425.};

Sphere spheres[] = {
    {{-3,    0,   -16}, 2,      ivory}
    // {{-1.0, -1.5, -12}, 2,      glass},
    // {{ 1.5, -0.5, -18}, 3, red_rubber},
    // {{ 7,    5,   -18}, 4,     mirror}
};

vec3d lights[] = {
    {-20, 20,  20},
    { 30, 50, -25},
    { 30, 20,  30}
};

// parameters = {spheres, parameters};


vec3 reflect(const vec3 &I, const vec3 &N) {
    return I - N * 2.0 * (I.dot(N));
}

vec3d reflect(const vec3d &I, const vec3d &N) {
    return I - N * 2.0 * (I.dot(N));
}


vec3 refract(const vec3 &I, const vec3 &N, const double eta_t, const double eta_i=1.f) { // Snell's law
    var cosi = - max(-1, min(1, I.dot(N)));
    if (cosi<0) return refract(I, -N, eta_i, eta_t); // if the ray comes from the inside the object, swap the air and the media
    var eta = eta_i / eta_t;
    var k = 1 - eta*eta*(1 - cosi*cosi);
    return k<0 ? vec3{1,0,0} : I*eta + N*(eta*cosi - sqrt(k)); // k<0 = total reflection, no ray to refract. I refract it anyways, this has no physical meaning
}

vec3d refract(const vec3d &I, const vec3d &N, const double eta_t, const double eta_i=1.f) { // Snell's law
    double cosi = - std::max(-1.0, std::min(1.0, I.dot(N)));
    if (cosi<0) return refract(I, -N, eta_i, eta_t); // if the ray comes from the inside the object, swap the air and the media
    double eta = eta_i / eta_t;
    double k = 1 - eta*eta*(1 - cosi*cosi);
    return k<0 ? vec3d{1,0,0} : I*eta + N*(eta*cosi - std::sqrt(k)); // k<0 = total reflection, no ray to refract. I refract it anyways, this has no physical meaning
}

std::tuple<bool,var> ray_sphere_intersect(const vec3 &orig, const vec3 &dir, const Sphere &s) { // ret value is a pair [intersection found, distance]
    vec3 L = s.center - orig;
    var tca = L.dot(dir);
    var d2 = L.dot(L) - tca*tca;
    if (d2 > s.radius*s.radius) return {false, 0};
    var thc = sqrt(s.radius*s.radius - d2);
    var t0 = tca-thc, t1 = tca+thc;
    if (t0>.001) return {true, t0};  // offset the original point by .001 to avoid occlusion by the object itself
    if (t1>.001) return {true, t1};
    return {false, 0};
}

std::tuple<bool,vec3,vec3,Material> scene_intersect(const vec3d &orig, const vec3d &dir) {
    vec3d pt, N;
    Material material;

    double nearest_dist = 1e10;
    if (abs(dir.y())>.001) { // intersect the ray with the checkerboard, avoid division by zero
        double d = -(orig.y() + 4) / dir.y(); // the checkerboard plane has equation y = -4
        vec3d p = orig + dir * d;
        if (d>.001 && d<nearest_dist && abs(p.x())<10 && p.z()<-10 && p.z()>-30) {
            nearest_dist = d;
            pt = p;
            N = {0,1,0};
            material.diffuse_color = (int(.5 * pt.x() + 1000) + int(.5 * pt.z())) & 1 ? vec3d{.3, .3, .3} : vec3d{.3, .2, .1};
        }
    }

    for (const Sphere &s : spheres) { // intersect the ray with all spheres
        auto [intersection, d] = ray_sphere_intersect(orig, dir, s);
        if (!intersection || d > nearest_dist) continue;
        nearest_dist = val(d);
        pt = orig + dir*nearest_dist;
        N = (pt - s.center).normalized();
        material = s.material;
    }

    return { nearest_dist<1000, pt, N, material };
}

vec3d cast_ray(const vec3d &orig, const vec3d &dir, const int depth=0) {
    // TODO: make cast_ray differentiable 
    if (depth == 2) return vec3d{0.2, 0.7, 0.8};

    auto [hit, point, N, material] = scene_intersect(orig, dir);
    if (!hit) return vec3d{0.2, 0.7, 0.8}; // background color

    // Calculate reflection and refraction only if necessary
    vec3d reflect_dir, refract_dir;
    vec3d reflect_color = {0, 0, 0}, refract_color = {0, 0, 0};
    if (material.albedo[2] > 0 || material.albedo[3] > 0) { // Check if reflection or refraction is needed
        reflect_dir = reflect(dir, N.cast<double>()).normalized();
        refract_dir = refract(dir, N.cast<double>(), material.refractive_index).normalized();
        if (material.albedo[2] > 0) reflect_color = cast_ray(point.cast<double>(), reflect_dir, depth + 1);
        if (material.albedo[3] > 0) refract_color = cast_ray(point.cast<double>(), refract_dir, depth + 1);
    }

    double diffuse_light_intensity = 0, specular_light_intensity = 0;
    for (const vec3d &light : lights) {
        vec3d light_dir = (light - point.cast<double>()).normalized(); // Cast point to double
        auto [hit, shadow_pt, trashnrm, trashmat] = scene_intersect(point, light_dir.cast<var>()); // Ensure scene_intersect can handle var for differentiation
        if (hit && (shadow_pt.cast<double>() - point.cast<double>()).norm() < (light - point.cast<double>()).norm()) continue;

        diffuse_light_intensity += std::max(0.0, light_dir.dot(N.cast<double>())); // Cast N to double
        vec3d specular_reflect = reflect(-light_dir, N.cast<double>()); // Reflect needs N as double
        double specular_dot = std::max(0.0, -specular_reflect.dot(dir));
        if (specular_dot > 0) specular_light_intensity += std::pow(specular_dot, material.specular_exponent);
    }

    return material.diffuse_color * diffuse_light_intensity * material.albedo[0] +
           vec3d{1.0, 1.0, 1.0} * specular_light_intensity * material.albedo[1] +
           reflect_color * material.albedo[2] +
           refract_color * material.albedo[3];
}

// compute the loss through MSE
var loss_MSE(std::vector<unsigned char> img_X, std::vector<unsigned char> img_y) {
    var loss = var(0);

    for (int i = 0; i < width * height * 3; ++i) {
        loss += pow((img_X[i] - img_y[i]), 2);
    }

    loss /= width * height * 3;
    return loss;
}

void backpropagation(Sphere &s, const std::vector<unsigned char> &img_X, const std::vector<unsigned char> &img_y, float learning_rate) {
    auto loss = loss_MSE(img_X, img_y);

    // Compute gradients
    auto grad_center = gradient(loss, s.center);
    auto grad_radius = gradient(loss, s.radius);

    // Update parameters
    s.center = s.center - learning_rate * grad_center;
    s.radius = s.radius - learning_rate * grad_radius;
}


int main() {
    auto start = std::chrono::high_resolution_clock::now();

    //////////////// Start Timing //////////////////////////
    std::vector<vec3d> framebuffer(width*height);
#pragma omp parallel for
    for (int pix = 0; pix<width*height; pix++) { // actual rendering loop
        float dir_x =  (pix%width + 0.5) -  width/2.;
        float dir_y = -(pix/width + 0.5) + height/2.; // this flips the image at the same time
        float dir_z = -height/(2.*tan(fov/2.));
        vec3 cast_res = cast_ray(vec3d{0,0,0}, vec3d{dir_x, dir_y, dir_z}.normalized());
        framebuffer[pix] = vec3d{cast_res.x(), cast_res.y(), cast_res.z()};

        if (pix % 1000 == 0) {
        #pragma omp critical
            std::cout << "Progress: " << pix << " of " << width*height << std::endl;
        }
    }

    // Load or define your target image    
    const std::vector<unsigned char> target_image = stbi_load("outone.jpg", &width, &height, 3, 0); 
    if (target_image == NULL) {
        printf("Error in loading the image\n");
        exit(1);
    }

    float learningRate = 0.01; // Setting this to 0.01, but change based on speed and accuracy
    backpropagation(spheres[0], rendered_image, target_image, learningRate); // spheres[0] is the sphere we're trying to render.

    std::vector<unsigned char> image(width * height * 3); // 3 bytes per pixel for RGB

    for (int i = 0; i < width * height; ++i) {
        for (int chan = 0; chan < 3; ++chan) {
            float max_val = std::max(1.0f, std::max(framebuffer[i][0], std::max(framebuffer[i][1], framebuffer[i][2])));
            float color_conversion = 255 * std::max(0.0f, std::min(1.0f, framebuffer[i][chan] / max_val));
            image[i * 3 + chan] = static_cast<unsigned char>(color_conversion);
        }
    }

    // Save the image using stb_image_write
    stbi_write_png("output.png", width, height, 3, image.data(), width * 3);
    ///////////////// finish up the timing ////////////////////

    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = stop - start;
    std::cout << "Time taken: " << elapsed.count() << "s\n";
    return 0;
}

