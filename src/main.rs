use std::io::Write;
use std::ops::{Add, Div, Mul, Sub};

// // Image
// const image_width: usize = 256;
// const image_height: usize = 256;

fn main() {
    // println!("Hello, raytracing!");

    // Image
    let aspect_ratio = 16.0 / 9.0;
    let image_width = 400;
    let image_height: usize = (image_width as f64 / aspect_ratio) as usize;

    // Camera

    let viewport_height = 2.0;
    let viewport_width = aspect_ratio * viewport_height;
    let focal_length = 1.0;

    let origin = point3::new(0.0, 0.0, 0.0);
    let horizontal = vec3::new(viewport_width, 0.0, 0.0);
    let vertical = vec3::new(0.0, viewport_height, 0.0);
    let lower_left_corner =
        origin - horizontal / 2.0 - vertical / 2.0 - vec3::new(0.0, 0.0, focal_length);

    // Render
    let out = std::io::stdout();

    print!("P3\n{} {}\n255\n", image_width, image_height);
    for j in (0..image_height).rev() {
        eprint!("\rScanlines remaining: {} ", j);
        std::io::stderr().flush().expect("some error message");
        for i in 0..image_width {
            let r: f64 = i as f64 / (image_width - 1) as f64;
            let g: f64 = j as f64 / (image_height - 1) as f64;
            let b: f64 = 0.25;

            let pixel = color::new(r, g, b);

            let u = i as f64 / (image_width - 1) as f64;
            let v = j as f64 / (image_height - 1) as f64;
            let ray = Ray::new(
                origin,
                lower_left_corner + horizontal * u + vertical * v - origin,
            );
            let pixel_color = ray.ray_color();
            write_color(&out, &pixel_color)
        }
    }

    eprint!("\nDone.")
}

#[derive(Clone, Copy)]
struct vec3 {
    x: f64,
    y: f64,
    z: f64,
}

impl vec3 {
    fn zeros() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }
    fn new(e0: f64, e1: f64, e2: f64) -> Self {
        Self {
            x: e0,
            y: e1,
            z: e2,
        }
    }

    fn x(&self) -> f64 {
        self.x
    }
    fn y(&self) -> f64 {
        self.y
    }
    fn z(&self) -> f64 {
        self.z
    }

    fn minus(&self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }

    fn plus(&self, v2: &Self) -> Self {
        Self::new(self.x() + v2.x(), self.y() + v2.y(), self.z() + v2.z())
    }

    fn multiply(&self, t: f64) -> Self {
        Self::new(self.x * t, self.y * t, self.z * t)
    }

    fn divide(&self, t: f64) -> Self {
        self.multiply(1.0 / t)
    }

    fn unit_vector(&self) -> Self {
        self.divide(self.length())
    }

    fn length(&self) -> f64 {
        self.length_squared().sqrt()
    }

    fn length_squared(&self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    fn dot(&self, v: &vec3) -> f64 {
        self.x * v.x + self.y * v.y + self.z * v.z
    }

    fn cross(&self, v: &vec3) -> Self {
        vec3::new(
            self.y * v.z - self.z * v.y,
            self.z * v.x - self.x * v.z,
            self.x * v.y - self.y * v.x,
        )
    }
}

impl Add for vec3 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        self.plus(&rhs)
    }
}

impl Sub for vec3 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        vec3::new(self.x() - rhs.x(), self.y() - rhs.y(), self.z() - rhs.z())
    }
}

impl Div<f64> for vec3 {
    type Output = Self;

    fn div(self, rhs: f64) -> Self::Output {
        self.divide(rhs)
    }
}

impl Mul<f64> for vec3 {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        self.multiply(rhs)
    }
}

//         double operator[](int i) const { return e[i]; }
//         double& operator[](int i) { return e[i]; }

//         vec3& operator/=(const double t) {
//             return *this *= 1/t;
//
// };

// // Type aliases for vec3
use vec3 as point3; // 3D point
use vec3 as color; // RGB color

fn write_color(out: &std::io::Stdout, pixel_color: &color) {
    // Write the translated [0,255] value of each color component.
    // out << static_cast<int>(255.999 * pixel_color.x()) << ' '
    // << static_cast<int>(255.999 * pixel_color.y()) << ' '
    // << static_cast<int>(255.999 * pixel_color.z()) << '\n';

    print!(
        "{} {} {}\n",
        (255.999 * pixel_color.x()) as usize,
        (255.999 * pixel_color.y()) as usize,
        (255.999 * pixel_color.z()) as usize,
    );
}

struct Ray {
    origin: point3,
    direction: vec3,
}

impl Ray {
    fn zeros() -> Self {
        Ray {
            origin: point3::zeros(),
            direction: vec3::zeros(),
        }
    }
    fn new(origin: point3, direction: point3) -> Self {
        Ray { origin, direction }
    }

    fn origin(&self) -> point3 {
        self.origin
    }
    fn direction(&self) -> vec3 {
        self.direction
    }

    fn at(&self, t: f64) -> point3 {
        self.origin() + self.direction().multiply(t)
    }
}
impl Ray {
    fn ray_color(&self) -> color {
        if self.hit_sphere(point3::new(0.0,0.0,-1.0), 0.5) {
            return color::new(1.0, 0.0, 0.0)
        }
        let unit_direction = self.direction().unit_vector();
        let t = 0.5 * (unit_direction.y() + 1.0);
        color::new(1.0, 1.0, 1.0).multiply(1.0 - t) + color::new(0.5, 0.7, 1.0).multiply(t)
    }

    fn hit_sphere(&self, center: point3, radius: f64) -> bool {
        let oc = self.origin() - center;
        let dir = self.direction();
        let a = self.direction().dot(&dir);
        let b = 2.0 * oc.dot(&dir);
        let c = oc.dot(&oc) - radius*radius;
        let discriminant = b*b - a*c*4.0;
        discriminant > 0.0
    }
}
