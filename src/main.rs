use std::io::Write;
use std::ops::{Add, Div, Mul, Sub};
use std::vec::Vec;

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
        let t = self.hit_sphere(point3::new(0.0, 0.0, -1.0), 0.5);
        if t > 0.0 {
            let n = (self.at(t) - vec3::new(0.0, 0.0, -1.0)).unit_vector();
            return color::new(n.x() + 1.0, n.y() + 1.0, n.z() + 1.0) * 0.5;
        }

        let unit_direction = self.direction().unit_vector();
        // shadowing
        let t = 0.5 * (unit_direction.y() + 1.0);
        color::new(1.0, 1.0, 1.0).multiply(1.0 - t) + color::new(0.5, 0.7, 1.0).multiply(t)
    }

    fn hit_sphere(&self, center: point3, radius: f64) -> f64 {
        let oc = self.origin() - center;
        let dir = self.direction();
        let a = self.direction().length_squared();
        let half_b = oc.dot(&dir);
        let c = oc.length_squared() - radius * radius;
        let discriminant = half_b * half_b - a * c;
        if discriminant < 0.0 {
            return -1.0;
        }
        (-half_b - discriminant.sqrt()) / a
    }
}

struct HitRecord {
    point: point3,
    normal: vec3,
    t: f64,
    front_face: bool,
}

impl HitRecord {
    fn new(point: point3, normal: vec3, t: f64, front_face: bool) -> Self {
        Self {
            point,
            normal,
            t,
            front_face,
        }
    }
    fn empty() -> Self {
        Self::new(point3::zeros(), vec3::zeros(), 0.0, false)
    }
    fn set_face_normal(&mut self, ray: &Ray, outward_normal: vec3) {
        self.front_face = ray.direction().dot(&outward_normal) < 0.0;
        self.normal = if self.front_face {
            outward_normal
        } else {
            outward_normal.minus()
        }
    }
}

trait Hittable {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord>;
}

struct Sphere {
    center: point3,
    radius: f64,
}

impl Sphere {
    fn new(center: point3, radius: f64) -> Self {
        Sphere { center, radius }
    }
}

impl Hittable for Sphere {
    // TODO change hitrecord to optional return
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        let oc = ray.origin() - self.center;
        let dir = ray.direction();
        let a = dir.length_squared();
        let half_b = oc.dot(&dir);
        let c = oc.length_squared() - self.radius * self.radius;

        let discriminant = half_b * half_b - a * c;
        if discriminant < 0.0 {
            return None;
        }
        let sqrtd = discriminant.sqrt();

        // Find the nearest root that lies in the acceptable range.
        let root = (-half_b - sqrtd) / a;
        if root < t_min || t_max < root {
            let root = (-half_b + sqrtd) / a;
            if root < t_min || t_max < root {
                return None;
            }
        }

        let mut record = HitRecord::empty();
        record.t = root;
        record.point = ray.at(record.t);
        record.normal = (record.point - self.center) / self.radius;
        let outward_normal = (record.point - self.center) / self.radius;
        record.set_face_normal(ray, outward_normal);

        Some(record)
    }
}

struct HittableList {
    list: Vec<&'static dyn Hittable>,
}

impl HittableList {
    fn  hittable_list() -> Self {
        Self {
            list: Vec::new(),
        }
    }
        fn clear(&mut self) { self.list.clear() }
        fn add(&mut self, hittable: &'static Hittable) { 
            self.list.push(hittable)
         }

    //     virtual bool hit(
    //         const ray& r, double t_min, double t_max, hit_record& rec) const override;
}

impl Hittable for HittableList {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        let mut temp_rec = HitRecord::new(point3::zeros(), vec3::zeros(), 0.0, false);
        let mut hit_anything = false;
        let mut closest_so_far = t_max;

        for h in self.list.iter() {
            if let Some(record) = h.hit(ray, t_min, closest_so_far) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
            }
        }

        if !hit_anything {
            None
        } else {
            Some(temp_rec)
        }
    }
}
