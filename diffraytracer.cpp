#include <tuple>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <chrono>

// lib for saving pictures
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// include autodiff lib and eigen
#include <autodiff/reverse/var/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>
#include <Eigen/Dense>

// replace the vec3
using namespace autodiff;
using vec3  = Eigen::Matrix<var, 3, 1>;
using vec3const = Eigen::Vector3d;

constexpr float fov    = 1.05; // 60 degrees field of view in radians

// TODO: Add Loss Function 
// TODO: Add Gradient Descent 
// TODO[optional]: Add Optimizer 


// define the primitives 
struct Material {
    double refractive_index  = 1;
    double albedo[4]         = {1,0,0,0};
    vec3const diffuse_color  = {0,0,0};
    double specular_exponent = 0;

    Material operator*(const double k) const {
        Material result = *this; // Make a copy of the current material
        result.albedo[3] = (1-k);
        return result;
    };
};

// struct Sphere {
//     vec3 center;
//     var radius;
//     Material material;
// };

struct Triangle {
    vec3 p1;
    vec3 p2;
    vec3 p3;
    Material material;

    void toString() const {
        std::cout << "p1 (" << val(p1.x()) << " " << val(p1.y()) << " " << val(p1.z()) << ")" << std::endl;
        std::cout << "p2 (" << val(p2.x()) << " " << val(p2.y()) << " " << val(p2.z()) << ")" << std::endl;
        std::cout << "p3 (" << val(p3.x()) << " " << val(p3.y()) << " " << val(p3.z()) << ")" << std::endl; 
    }
};

Material      ivory = {1.0, {1.0,  0.0, 0.0, 0.0}, {0.4, 0.4, 0.3},   50.};
// const Material      glass = {1.5, {0.0,  0.9, 0.1, 0.8}, {0.6, 0.7, 0.8},  125.};
// const Material red_rubber = {1.0, {1.4,  0.3, 0.0, 0.0}, {0.3, 0.1, 0.1},   10.};
// const Material     mirror = {1.0, {0.0, 16.0, 0.8, 0.0}, {1.0, 1.0, 1.0}, 1425.};

Triangle triangles[] = {
    {{-15, 0, -16}, {4, -6.5, -12},{3.5, -0.5, -23}, ivory},
};

vec3const lights[] = {
    {-20, 20,  20},
    { 30, 50, -25},
    { 30, 20,  30}
};

// parameters = {spheres, parameters};


vec3 reflect(const vec3 &I, const vec3 &N) {
    return I - N * 2.0 * (I.dot(N));
}

// vec3const reflect(const vec3const &I, const vec3const &N) {
//     return I - N * 2.0 * (I.dot(N));
// }


vec3 refract(const vec3 &I, const vec3 &N, const var eta_t, const var eta_i=1.f) { // Snell's law
    var cosi = - max(-1, min(1, I.dot(N)));
    if (cosi<0) return refract(I, -N, eta_i, eta_t); // if the ray comes from the inside the object, swap the air and the media
    var eta = eta_i / eta_t;
    var k = 1 - eta*eta*(1 - cosi*cosi);
    return k<0 ? vec3{1,0,0} : I*eta + N*(eta*cosi - sqrt(k)); // k<0 = total reflection, no ray to refract. I refract it anyways, this has no physical meaning
}

std::tuple<var, var, vec3> ray_triangle_intersect(const vec3 &orig, const vec3 &dir, const Triangle &t) {
    var u, v;  // barycentric coordinates, last one is (1 - u - v)
	var time;  // time for line pass through triangle
	vec3 e1 = t.p2 - t.p1;
	vec3 e2 = t.p3 - t.p1;
    
    vec3 p = dir.cross(e2);
	var divisor = p.dot(e1);
	if (abs(divisor) < 1e-6) { // check if parallel
		return {0, 0, vec3()};
	}

    vec3 s = orig - t.p1;
	var cramer_factor = 1.0f / divisor;

	u = cramer_factor * s.dot(p);
    v = cramer_factor * dir.dot(s.cross(e1));
    var uv_comp = min(u, v);
    var bary_min = min(uv_comp, (1-u-v)) * 3;
	if (u < (var)1e-6 || v < (var)1e-6 || u + v > 1.0f) {
		return {0, 0, vec3()};
	}
	

	time = cramer_factor * (e2.dot(s.cross(e1)));
    if (time < 0) {
        return {0, 0, vec3()};
    }

    vec3 N = e1.cross(e2);   
    return {bary_min, time, N.normalized()};
}

std::tuple<bool,vec3,vec3,Material> scene_intersect(const vec3 &orig, const vec3 &dir) {
    vec3 pt, N;
    Material material;

    var nearest_dist = 1e10;
    // if (abs(dir.y())>.001) { // intersect the ray with the checkerboard, avoid division by zero
    //     var d = -(orig.y() + 4) / dir.y(); // the checkerboard plane has equation y = -4
    //     vec3 p = orig + dir * d;
    //     if (d>.001 && d<nearest_dist && abs(p.x())<10 && p.z()<-10 && p.z()>-30) {
    //         nearest_dist = d;
    //         pt = p;
    //         N = {0,1,0};
    //         material.diffuse_color = (int(val(.5 * pt.x() + 1000)) + int(val(.5 * pt.z()))) & 1 ? vec3const{.3, .3, .3} : vec3const{.3, .2, .1};
    //     }
    // }

    for (const Triangle &t : triangles) {
        auto [intersection, d, normal] = ray_triangle_intersect(orig, dir, t);
        if (d < nearest_dist) nearest_dist = val(d);
        pt = orig + dir*nearest_dist;
        N = normal;
        material = t.material * val(intersection);
        if (val(intersection) > 1e-6) {
            std::cout << "intersection " << val(intersection) << std::endl;
        }
    }
    return { nearest_dist<1000, pt, N, material };
}

vec3 cast_ray(const vec3 &orig, const vec3 &dir, const int depth=0) {
    // TODO: make cast_ray differentiable 
    if (depth == 1) return vec3{0.2, 0.7, 0.8};

    auto [hit, point, N, material] = scene_intersect(orig, dir);
    if (hit <= 1e-5) return vec3{0.2, 0.7, 0.8}; // background color

    // Calculate reflection and refraction only if necessary
    vec3 reflect_dir, refract_dir;
    vec3 reflect_color = {0, 0, 0}, refract_color = {0, 0, 0};
    if (material.albedo[2] > 0 || material.albedo[3] > 0) { // Check if reflection or refraction is needed
        reflect_dir = reflect(dir, N).normalized();
        refract_dir = refract(dir, N, material.refractive_index).normalized();
        if (material.albedo[2] > 0) reflect_color = cast_ray(point, reflect_dir, depth + 1);
        if (material.albedo[3] > 0) refract_color = cast_ray(point, refract_dir, depth + 1);
    }

    var diffuse_light_intensity = 0, specular_light_intensity = 0;
    for (const vec3 &light : lights) {
        vec3 light_dir = (light - point).normalized(); // Cast point to double
        auto [hit, shadow_pt, trashnrm, trashmat] = scene_intersect(point + light_dir * 0.1f, light_dir); // Ensure scene_intersect can handle var for differentiation
        if (hit > 1e-5 && (shadow_pt - point).norm() < (light - point).norm()) continue;

        diffuse_light_intensity += max(0.0, light_dir.dot(N)); // Cast N to double
        vec3 specular_reflect = reflect(-light_dir, N); // Reflect needs N as double
        var specular_dot = max(0.0, -specular_reflect.dot(dir));
        if (specular_dot > 0) specular_light_intensity += pow(specular_dot, material.specular_exponent);
    }

    return material.diffuse_color * diffuse_light_intensity * material.albedo[0] +
           vec3{1.0, 1.0, 1.0} * specular_light_intensity * material.albedo[1] +
           reflect_color * material.albedo[2] +
           refract_color * material.albedo[3];
}

// compute the loss through MSE
var loss_MSE(std::vector<vec3> img_X, std::vector<vec3> img_y, int width, int height) {
    var loss = var(0);
    int vec_sum = 0;

    for (int i = 0; i < width * height; ++i) {
        vec3 vec_diff = img_X[i] - img_y[i];
        loss += vec_diff.squaredNorm();
    }

    loss /= width * height;
    std::cout << "loss is " << loss << std::endl;
    return loss;
}

var loss_PSNR(std::vector<vec3> img_X, std::vector<vec3> img_y, int width, int height) {
    var mse_loss = loss_MSE(img_X, img_y, width, height);
    var MAX_PSNR = 0;

    for (int i = 0; i < width * height; ++i) {
        if (img_y[i].norm() > MAX_PSNR) {
            MAX_PSNR = img_y[i].norm();
        }
    }

    return 20 * log10(MAX_PSNR) - 10 * log10(mse_loss); 
}

void backpropagation(Triangle &t, const std::vector<vec3> &img_X, const std::vector<vec3> &img_y, float learning_rate, int width, int height) {
    auto loss = loss_PSNR(img_X, img_y, width, height);

    // Compute gradients
    vec3 dp1 = gradient(loss, t.p1);
    vec3 dp2 = gradient(loss, t.p2);
    vec3 dp3 = gradient(loss, t.p3);

    // Update parameters
    t.p1 -= learning_rate * dp1;
    t.p2 -= learning_rate * dp2;
    t.p3 -= learning_rate * dp3;

    std::cout << "dp1 x " << dp1.x() << std::endl;
    // std::cout << "dp1 y " << dp1.y << std::endl;
    // std::cout << "dp1 z " << dp1.z << std::endl;
}


// inverse rendering training process
void train(std::vector<vec3>& framebuffer, const std::vector<vec3> target_fb, int width, int height, int i) {
#pragma omp parallel for
    for (int pix = 0; pix<width*height; pix++) { // actual rendering loop
        float dir_x =  (pix%width + 0.5) -  width/2.;
        float dir_y = -(pix/width + 0.5) + height/2.; // this flips the image at the same time
        float dir_z = -height/(2.*tan(fov/2.));
        vec3 cast_res = cast_ray(vec3{0,0,0}, vec3{dir_x, dir_y, dir_z}.normalized());
        framebuffer[pix] = cast_res;

        if (pix % 100 == 0) {
        #pragma omp critical
            std::cout << "Progress: " << pix << " of " << width*height << std::endl;
        }

        std::vector<unsigned char> image(width * height * 3);

        for (int i = 0; i < width * height; ++i) {
            for (int chan = 0; chan < 3; ++chan) {
                int color_conversion = (std::max(0.0, std::min(255.0,  255 * val(framebuffer[i][chan]))));
                image[i * 3 + chan] = static_cast<unsigned char>(color_conversion);
            }
        }

        // Save the image using stb_image_write
        std::string pic_name = "output_" + std::to_string(i) + ".png";
        stbi_write_png(pic_name.c_str(), width, height, 3, image.data(), width * 3);
    }

    float learningRate = 1e5; // Setting this to 0.01, but change based on speed and accuracy
    backpropagation(triangles[0], framebuffer, target_fb, learningRate, width, height); // spheres[0] is the sphere we're trying to render
}


int main() {
    auto start = std::chrono::high_resolution_clock::now();

    //////////////// Start Timing //////////////////////////
    // Load or define your target image    
    int channel, width, height;
    unsigned char* target_image = stbi_load("triangle20c.png", &width, &height, &channel, 0);
    std::cout << width << "  AND  " << height << std::endl;
    if (target_image == nullptr) {
        printf("Error in loading the image\n");
        exit(1);
    }

    std::vector<vec3> target_framebuffer(width*height);
    for (int i = 0; i < width * height; i++) {
        target_framebuffer[i] = vec3{ ((float)(*(target_image + 3*i))) / 255.0f,
                                        ((float)*(target_image + 3*i + 1)) / 255.0f,
                                        ((float)*(target_image + 3*i + 2) / 255.0f)};
    }

    std::vector<vec3> framebuffer(width*height);

    int iteration = 3;
    for (int i = 0; i < iteration; i++) {
        triangles[0].toString();
        train(framebuffer, target_framebuffer, width, height, i);
        std::cout << "iteration " << i << " out of " << iteration << std::endl;
    }
    ///////////////// finish up the timing ////////////////////

    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = stop - start;
    std::cout << "Time taken: " << elapsed.count() << "s\n";
    return 0;
}

