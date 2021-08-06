use std::io::Write;
use std::ops::Add;

// Image
const image_width: usize = 256;
const image_height: usize = 256;

fn main() {
    // println!("Hello, raytracing!");

    // Render
    let out = std::io::stdout();

    print!("P3\n{} {}\n255\n", image_width, image_height);
    // for j in image_height-1..=0 {
    for j in (0..image_height).rev() {
        eprint!("\rScanlines remaining: {} ", j);
        std::io::stderr().flush().expect("some error message");
        for i in 0..image_width {
            let r: f64 = i as f64 / (image_width - 1) as f64;
            let g: f64 = j as f64 / (image_height - 1) as f64;
            let b: f64 = 0.25;

            let pixel = color::new(r, g, b);

            // print!("{} {} {}\n", ir, ig, ib);
            write_color(&out, &pixel)
        }
    }

    eprint!("\nDone.")
}

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
        self.multiply(1.0/t)
    }

    fn unit_vector(&self) -> Self {
        self.divide(self.length())
    }

    fn length(&self) -> f64{
        self.length_squared().sqrt()
    }

    fn length_squared(&self) -> f64 {
      self.x * self.x + self.y * self.y + self.z * self.z
    }
}

impl Add for &vec3 {
    type Output = vec3;

    fn add(self, rhs: &vec3) -> vec3 {
        self.plus(&rhs)
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

    fn origin(&self) -> &point3 {
        &self.origin
    }
    fn direction(&self) -> &vec3 {
        &self.direction
    }

    fn at(&self, t: f64) -> point3  {
        self.origin() + &(self.direction().multiply(t))
    }
}
impl Ray {
    fn ray_color(&self) -> color {
        let unit_direction = self.direction().unit_vector();
        let t = 0.5 * (unit_direction.y() + 1.0);
        (&color::new(1.0, 1.0, 1.0).multiply(1.0-t)) + &(color::new(0.5, 0.7, 1.0).multiply(t))
    }
}
