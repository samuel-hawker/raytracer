use rand::random;
use std::io::Write;
use std::ops::{Add, Div, Mul, Sub, AddAssign};
use std::vec::Vec;

fn main() {
    // Image - the dimensions of the image we want to generate
    let aspect_ratio = 16.0 / 9.0;
    let image_width = 400;
    let image_height: usize = (image_width as f64 / aspect_ratio) as usize;
    let samples_per_pixel = 100;
    let max_depth = 50;

    let material_ground = Lambertian::new(Color::new(0.8, 0.8, 0.0));
    let material_center = Dielectric::new(1.5);
    let material_left = Dielectric::new(1.5);
    let material_right = Metal::new(Color::new(0.8, 0.6, 0.2), 1.0);

    let mut world = HittableList::hittable_list();
    let sphere1 = Sphere::new(
        Point3::new(0.0, -100.5, -1.0),
        100.0,
        Option::Some(Box::from(material_ground)),
    );
    let sphere2 = Sphere::new(
        Point3::new(0.0, 0.0, -1.0),
        0.5,
        Option::Some(Box::from(material_center)),
    );
    let sphere3 = Sphere::new(
        Point3::new(-1.0, 0.0, -1.0),
        0.5,
        Option::Some(Box::from(material_left)),
    );
    let sphere4 = Sphere::new(
        Point3::new(1.0, 0.0, -1.0),
        0.5,
        Option::Some(Box::from(material_right)),
    );
    world.add(&sphere1);
    world.add(&sphere2);
    world.add(&sphere3);
    world.add(&sphere4);

    // Camera - the position and angle from which we will capture an image
    let camera = Camera::new();

    // where we will output th image, currently stdout
    let out = std::io::stdout();

    // Render an image

    // the metadata at the top of th ppm file
    print!("P3\n{} {}\n255\n", image_width, image_height);
    // range across the image height and width and compute the colour of each pixel
    for j in (0..image_height).rev() {
        // debug to stderr
        eprint!("\rScanlines remaining: {} ", j);
        std::io::stderr().flush().expect("some error message");

        for i in 0..image_width {
            let mut pixel_color = Color::zeros();
            for _ in 0..samples_per_pixel {
                let u = (i as f64 + random_f64()) / (image_width - 1) as f64;
                let v = (j as f64 + random_f64()) / (image_height - 1) as f64;
                let ray = camera.get_ray(u, v);
                pixel_color += ray.ray_color(&world, max_depth);
            }

            write_color(&out, &pixel_color, samples_per_pixel);
        }
    }

    eprint!("\nDone.")
}

// example test
#[test]
fn sanity() {
    assert_eq!(2 + 2, 4);
}

#[derive(Clone, Copy, PartialEq, Debug)]
struct Vec3 {
    x: f64,
    y: f64,
    z: f64,
}

impl Vec3 {
    fn zeros() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }
    fn new(x: f64, y: f64, z: f64) -> Self {
        Self {
            x,
            y,
            z,
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

    fn unit_vector(self) -> Self {
        self / self.length()
    }

    fn length(&self) -> f64 {
        self.length_squared().sqrt()
    }

    fn length_squared(&self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    fn dot(&self, v: &Vec3) -> f64 {
        self.x * v.x + self.y * v.y + self.z * v.z
    }

    fn cross(&self, v: &Vec3) -> Self {
        Self::new(
            self.y * v.z - self.z * v.y,
            self.z * v.x - self.x * v.z,
            self.x * v.y - self.y * v.x,
        )
    }

    fn random() -> Self {
        Self::new(random_f64(), random_f64(), random_f64())
    }

    fn random_range(min: f64, max: f64) -> Self {
        Vec3::new(
            random_f64_range(min, max),
            random_f64_range(min, max),
            random_f64_range(min, max),
        )
    }

    fn random_in_unit_sphere() -> Self {
        loop {
            let p = Vec3::random_range(-1.0, 1.0);
            if p.length_squared() >= 1.0 {
                continue;
            }
            return p;
        }
    }

    fn random_unit_vector() -> Self {
        Self::random_in_unit_sphere().unit_vector()
    }

    fn near_zero(&self) -> bool {
        // Return true if the vector is close to zero in all dimensions.
        let s = 1e-8;
        self.x().abs() < s && self.y().abs() < s && self.z().abs() < s
    }

    fn reflect(self, n: Self) -> Self {
        self - (n * 2.0 * self.dot(&n))
    }

    fn refract(self, n: Self, etai_over_etat: f64) -> Self {
        // cos_theta = fmin(dot(-uv, n), 1.0);
        let cos_theta = f64::min(self.minus().dot(&n), 1.0);
        // r_out_perp =  etai_over_etat * (uv + cos_theta*n)
        let r_out_perp = etai_over_etat * (self + cos_theta * n);
        // r_out_parallel = -sqrt(fabs(1.0 - r_out_perp.length_squared())) * n
        let r_out_parallel = -1.0 * (1.0 - r_out_perp.length_squared()).abs().sqrt() * n;
        r_out_perp + r_out_parallel
    }
}

#[test]
fn test_neg() {
    assert_eq!(Vec3::new(1.0, 2.0, 3.0).minus(), Vec3::new(-1.0, -2.0, -3.0));
}

#[test]
fn test_refract() {
    let vector = Vec3::new(-1.0, -1.0, 0.0);
    let normal = Vec3::new(1.0, 1.0, 0.0);
    // no refraction
    assert_eq!(vector.refract(normal, 1.0), Vec3::new(-1.0, -1.0, 0.0));
    // refraction
    assert_eq!(vector.refract(normal, 1.5), Vec3::new(-1.118033988749895, -2.618033988749895, 0.0), "1.5 angle failed");
    assert_eq!(vector.refract(normal, 0.5), Vec3::new(-0.8660254037844386, -1.3660254037844386, 0.0), "0.5 angle failed");
}

#[test]
fn test_refract2() {
    let vector = Vec3::new(-1.0, 0.0, 0.0);
    let normal = Vec3::new(1.0, 0.0, 0.0);
    // no refraction
    assert_eq!(vector.refract(normal, 1.0), Vec3::new(-1.0, -0.0, 0.0));
    // 'refraction'
    assert_eq!(vector.refract(normal, 1.5), Vec3::new(-1.118033988749895, -2.618033988749895, 0.0), "1.5 angle failed");
    assert_eq!(vector.refract(normal, 0.5), Vec3::new(-0.8660254037844386, -1.3660254037844386, 0.0), "0.5 angle failed");
}

impl Add for Vec3 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        self.plus(&rhs)
    }
}

impl AddAssign for Vec3 {
    fn add_assign(&mut self, rhs: Self) {
        *self = self.add(rhs);
    }
}

impl Sub for Vec3 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x() - rhs.x(), self.y() - rhs.y(), self.z() - rhs.z())
    }
}

impl Div<f64> for Vec3 {
    type Output = Self;

    fn div(self, rhs: f64) -> Self::Output {
        self.divide(rhs)
    }
}

impl Mul<f64> for Vec3 {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        self.multiply(rhs)
    }
}

impl Mul<Vec3> for f64 {
    type Output = Vec3;

    fn mul(self, rhs: Vec3) -> Self::Output {
        rhs.multiply(self)
    }
}

impl Mul for Vec3 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z)
    }
}

// Type aliases for Vec3
use Vec3 as Point3; // 3D point
use Vec3 as Color; // RGB Color

// write colour in the ppm format of 'r g b'
fn write_color(out: &std::io::Stdout, pixel_Color: &Color, samples_per_pixel: usize) {
    // Divide the Color by the number of samples.
    let scale = 1.0 / samples_per_pixel as f64;
    let r = (pixel_Color.x() * scale).sqrt();
    let g = (pixel_Color.y() * scale).sqrt();
    let b = (pixel_Color.z() * scale).sqrt();

    print!(
        "{} {} {}\n",
        (256.0 * clamp_basic(r)) as usize,
        (256.0 * clamp_basic(g)) as usize,
        (256.0 * clamp_basic(b)) as usize,
    );
}

#[derive(Debug, PartialEq)]
struct Ray {
    origin: Point3,
    direction: Vec3,
}

impl Ray {
    fn zeros() -> Self {
        Ray {
            origin: Point3::zeros(),
            direction: Vec3::zeros(),
        }
    }
    fn new(origin: Point3, direction: Point3) -> Self {
        Ray { origin, direction }
    }

    fn origin(&self) -> Point3 {
        self.origin
    }
    fn direction(&self) -> Vec3 {
        self.direction
    }

    fn at(&self, t: f64) -> Point3 {
        self.origin() + self.direction().multiply(t)
    }

    fn ray_color(&self, world: &dyn Hittable, depth: usize) -> Color {
        // If we've exceeded the ray bounce limit, no more light is gathered.
        if depth <= 0 {
            return Color::zeros();
        }

        // if there is a hit then compute the colour from the hit's normal
        return if let Some(record) = world.hit(self, 0.001, INFINITY) {
            // probably overkill
            let mat = record.material.as_ref().unwrap();
            // if it scatters, then find scatter report or return no colour
            if let Some(scatter_record) = mat.scatter(self, &record) {
                let refracted_color = scatter_record.scattered.ray_color(world, depth - 1);
                let scatter_color = refracted_color * scatter_record.attenuation;
                // eprint!("{:#?} {:#?} {:#?}", scatter_record.scattered, scatter_record.attenuation, depth);
                // if scatter_Color != Color::zeros() {
                //     eprint!("{:#?}\n", scatter_Color);
                // }
                // return Color::new(0.1, 0.1, 0.1);

                scatter_color
            } else {
                // no scatter
                Color::zeros()
            }
        } else {
            // otherwise compute background
            let unit_direction = self.direction().unit_vector();
            let t = 0.5 * (unit_direction.y() + 1.0);
            Color::new(1.0, 1.0, 1.0) * (1.0 - t) + Color::new(0.5, 0.7, 1.0) * t
        } 
    }
}

struct HitRecord<'a> {
    point: Point3,
    normal: Vec3,
    material: &'a Option<Box<dyn Material>>,
    t: f64,
    front_face: bool,
}

impl<'a> HitRecord<'a> {
    fn new(
        point: Point3,
        normal: Vec3,
        material: &'a Option<Box<dyn Material>>,
        t: f64,
        front_face: bool,
    ) -> Self {
        Self {
            point,
            normal,
            material,
            t,
            front_face,
        }
    }
    fn empty() -> Self {
        Self::new(Point3::zeros(), Vec3::zeros(), &Option::None, 0.0, false)
    }
   
}

// belong to sphere?
 fn face_normal(ray: &Ray, outward_normal: Vec3) -> (Vec3, bool) {
    let front_face = ray.direction().dot(&outward_normal) < 0.0;
    let normal = if front_face {
        outward_normal
    } else {
        outward_normal.minus()
    };
    (normal, front_face)
}

#[test]
fn test_face_normal_calc() {
    let ray = Ray::new(Point3::zeros(), Point3::new(1.0, 0.0, 0.0));
    assert_eq!(face_normal(&ray, Vec3::new(-1.0, 0.0, 0.0)), (Vec3::new(-1.0, 0.0, 0.0), true) );
    assert_eq!(face_normal(&ray, Vec3::new(1.0, 0.0, 0.0)), (Vec3::new(-1.0, 0.0, 0.0), false) );
}

trait Hittable {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord>;
}

struct Sphere {
    center: Point3,
    radius: f64,
    material: Option<Box<dyn Material>>,
}

impl Sphere {
    fn new(center: Point3, radius: f64, material: Option<Box<dyn Material>>) -> Self {
        Sphere {
            center,
            radius,
            material,
        }
    }
}

impl Hittable for Sphere {
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

        let t = root;
        let point = ray.at(t);
        let outward_normal = (point - self.center) / self.radius;
        // remove face_normal?
        // let (normal, front_face) = HitRecord::face_normal(ray, outward_normal);
        let front_face = ray.direction().dot(&outward_normal) < 0.0;
        let normal = if front_face {
            outward_normal
        } else {
            outward_normal.minus()
        };
        let record = HitRecord::new(point, normal, &self.material, t, front_face);

        Some(record)
    }
}

struct HittableList<'a> {
    list: Vec<&'a dyn Hittable>,
}

impl<'a> HittableList<'a> {
    fn hittable_list() -> Self {
        Self { list: Vec::new() }
    }
    fn clear(&mut self) {
        self.list.clear()
    }
    fn add(&mut self, hittable: &'a dyn Hittable) {
        self.list.push(hittable)
    }
}

impl<'a> Hittable for HittableList<'a> {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        let mut record = Option::None;
        // set closest so far to be the max, i.e. infinity
        let mut closest_so_far = t_max;

        for h in self.list.iter() {
            if let Some(r) = h.hit(ray, t_min, closest_so_far) {
                // update closest so far, this has to be between t_min and closest so far, so an update will always be less
                closest_so_far = r.t;
                record = Option::Some(r)
            }
        }

        // return the closest record
        record
    }
}

struct Camera {
    origin: Point3,
    lower_left_corner: Point3,
    horizontal: Vec3,
    vertical: Vec3,
}

impl Camera {
    fn new() -> Self {
        let aspect_ratio = 16.0 / 9.0;
        let viewport_height = 2.0;
        let viewport_width = aspect_ratio * viewport_height;
        let focal_length = 1.0;
        let origin = Point3::new(0.0, 0.0, 0.0);
        let horizontal = Vec3::new(viewport_width, 0.0, 0.0);
        let vertical = Vec3::new(0.0, viewport_height, 0.0);
        let lower_left_corner =
            origin - horizontal / 2.0 - vertical / 2.0 - Vec3::new(0.0, 0.0, focal_length);
        Self {
            origin,
            horizontal,
            vertical,
            lower_left_corner,
        }
    }

    fn get_ray(&self, u: f64, v: f64) -> Ray {
        Ray::new(
            self.origin,
            self.lower_left_corner + self.horizontal * u + self.vertical * v - self.origin,
        )
    }
}

//the material needs to do two things:
// Produce a scattered ray (or say it absorbed the incident ray).
// If scattered, say how much the ray should be attenuated.
trait Material {
    fn scatter(&self, ray_in: &Ray, record: &HitRecord) -> Option<ScatterRecord>;
}

#[derive(Debug, PartialEq)]
struct ScatterRecord {
    attenuation: Color,
    scattered: Ray,
}

impl ScatterRecord {
    fn new(attenuation: Color, scattered: Ray) -> Self {
        Self {
            attenuation,
            scattered,
        }
    }
}

struct Lambertian {
    albedo: Color,
}

impl Lambertian {
    fn new(Color: Color) -> Self {
        Self { albedo: Color }
    }
}

impl Material for Lambertian {
    fn scatter(&self, ray_in: &Ray, record: &HitRecord) -> Option<ScatterRecord> {
        let mut scatter_direction = record.normal + Vec3::random_unit_vector();

        // Catch degenerate scatter direction
        if scatter_direction.near_zero() {
            scatter_direction = record.normal;
        }
        Option::Some(ScatterRecord::new(
            self.albedo,
            Ray::new(record.point, scatter_direction),
        ))
    }
}

struct Metal {
    albedo: Color,
    fuzz: f64,
}

impl Metal {
    fn new(Color: Color, fuzz: f64) -> Self {
        let fuzz = if fuzz > 1.0 { 1.0 } else { fuzz };
        Self {
            albedo: Color,
            fuzz,
        }
    }
}

impl Material for Metal {
    fn scatter(&self, ray_in: &Ray, record: &HitRecord) -> Option<ScatterRecord> {
        let reflected = ray_in.direction().unit_vector().reflect(record.normal);
        let scattered = Ray::new(
            record.point,
            // apply fuzzing
            reflected + Point3::random_in_unit_sphere() * self.fuzz,
        );
        let attenuation = self.albedo;
        if scattered.direction().dot(&record.normal) > 0.0 {
            Option::Some(ScatterRecord::new(attenuation, scattered))
        } else {
            Option::None
        }
    }
}

struct Dielectric {
    // Index of Refraction
    refraction_index: f64,
}

impl Dielectric {
    fn new(refraction_index: f64) -> Self {
        Self { refraction_index }
    }
}

impl Material for Dielectric {
    fn scatter(&self, ray_in: &Ray, record: &HitRecord) -> Option<ScatterRecord> {
        // white light
        let attenuation = Color::new(1.0, 1.0, 1.0);
        let refraction_ratio = if record.front_face {
            1.0 / self.refraction_index
        } else {
            self.refraction_index
        };

        let unit_direction = ray_in.direction().unit_vector();
        let refracted = unit_direction.refract(record.normal, refraction_ratio);
        // if unit_direction != refracted {
        //     // eprint!("{:?} {:?}\n", unit_direction, refracted);
        // }
        let scattered = Ray::new(record.point, refracted);
        let scatter_record = ScatterRecord::new(attenuation, scattered);
        // eprint!("{:#?}\n", scatter_record);
        Option::Some(scatter_record)
    }
}

#[test]
fn test_dielectric_scatter() {
   let material = Dielectric::new(1.5);
   let boxed_material: Option<Box<dyn Material>> = Some(Box::from(material));
   let ray = Ray::new(Point3::zeros(), Point3::new(1.0, 0.0, 0.0));
   let hit_record = HitRecord::new(Point3::new(1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), &boxed_material, 0.5, true);
   let material = hit_record.material.as_ref().unwrap();
   let scatter_record = material.scatter(&ray, &hit_record);
   let scatter_record = scatter_record.unwrap();
   // white attenuation + a new ray at the pont of collision + the direction??
   assert_eq!(scatter_record, ScatterRecord::new(Color::new(1.0, 1.0, 1.0), Ray::new(Point3::new(1.0, 0.0, 0.0), Point3::new(-1.0, 0.0, 0.0))))
}

// Constants
const INFINITY: f64 = std::f64::INFINITY;
const PI: f64 = std::f64::consts::PI;

// Utility Functions
fn degrees_to_radians(degrees: degrees) -> radians {
    return degrees * PI / 180.0;
}

// Returns a random f64 in [0,1).
fn random_f64() -> f64 {
    random::<f64>()
}

// 0-1 -> -1-2
// Returns a random f64 in [mix,max).
fn random_f64_range(min: f64, max: f64) -> f64 {
    random::<f64>() * (max - min) + min
}

fn clamp(x: f64, min: f64, max: f64) -> f64 {
    if x < min {
        return min;
    }
    if x > max {
        return max;
    }
    x
}

fn clamp_basic(x: f64) -> f64 {
    clamp(x, 0.0, 0.999)
}

use f64 as degrees;
use f64 as radians;

// need to use pointers (box)
