use std::io::Write;

// Image
const image_width: usize = 256;
const image_height: usize = 256;

fn main() {
    // println!("Hello, raytracing!");

    // Render

    print!("P3\n{} {}\n255\n", image_width, image_height);
    // for j in image_height-1..=0 {
    for j in (0..image_height).rev() {
        eprint!("\rScanlines remaining: {} ", j);
        std::io::stderr().flush().expect("some error message");
        for i in 0..image_width {
            let r: f64 = i as f64 / (image_width-1) as f64 ;
            let g: f64 = j as f64  / (image_height-1) as f64 ;
            let b: f64 = 0.25;

            let ir: usize = (255.999 * r) as usize;
            let ig: usize = (255.999 * g) as usize;
            let ib: usize = (255.999 * b) as usize;

            print!("{} {} {}\n", ir, ig, ib);
        }
    }
}

struct vec3 {
}

impl vec3 {

}
//         vec3() : e{0,0,0} {}
//         vec3(double e0, double e1, double e2) : e{e0, e1, e2} {}

//         double x() const { return e[0]; }
//         double y() const { return e[1]; }
//         double z() const { return e[2]; }

//         vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
//         double operator[](int i) const { return e[i]; }
//         double& operator[](int i) { return e[i]; }

//         vec3& operator+=(const vec3 &v) {
//             e[0] += v.e[0];
//             e[1] += v.e[1];
//             e[2] += v.e[2];
//             return *this;
//         }

//         vec3& operator*=(const double t) {
//             e[0] *= t;
//             e[1] *= t;
//             e[2] *= t;
//             return *this;
//         }

//         vec3& operator/=(const double t) {
//             return *this *= 1/t;
//         }

//         double length() const {
//             return sqrt(length_squared());
//         }

//         double length_squared() const {
//             return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
//         }

//     public:
//         double e[3];
// };

// // Type aliases for vec3
// using point3 = vec3;   // 3D point
// using color = vec3;    // RGB color
