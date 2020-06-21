use std::collections::HashSet;
use std::ops::{Add, AddAssign, Div, Mul, Sub};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct Point {
    pub c: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Score(pub f32, pub f32);

impl Add for Point {
    type Output = Self;

    fn add(self, b: Point) -> Point {
        let mut result: Vec<f32> = vec![0.0; b.dim()];

        for i in 0..b.dim() {
            result[i] = self.c[i] + b.c[i];
        }
        Point { c: result }
    }
}

impl AddAssign for Point {
    fn add_assign(&mut self, b: Point) {
        for i in 0..self.c.len() {
            self.c[i] += b.c[i];
        }
    }
}

impl Sub for Point {
    type Output = Self;

    fn sub(self, b: Point) -> Point {
        let mut result: Vec<f32> = vec![0.0; self.c.len()];

        for i in 0..self.c.len() {
            result[i] = self.c[i] - b.c[i];
        }
        Point { c: result }
    }

}

impl Sub<f32> for Point{
    type Output = Self;
    fn sub(self, b: f32) -> Point{
        let mut result: Vec<f32> = vec![0.0; self.dim()];

        for i in 0..self.dim() {
            result[i] = self.c[i] - b;
        }
        Point { c: result }
    }
}

impl Add<f32> for Point{
    type Output = Self;
    fn add(self, b: f32) -> Point{
        let mut result: Vec<f32> = vec![0.0; self.dim()];

        for i in 0..self.dim() {
            result[i] = self.c[i] + b;
        }
        Point { c: result }
    }
}

impl Mul<f32> for Point {
    type Output = Point;

    fn mul(self, s: f32) -> Point {
        let mut result: Vec<f32> = vec![0.0; self.c.len()];

        for i in 0..self.c.len() {
            result[i] = self.c[i] * s;
        }
        Point { c: result }
    }
}

impl Div<f32> for Point {
    type Output = Point;

    fn div(self, s: f32) -> Point {
        let mut result: Vec<f32> = vec![0.0; self.c.len()];

        for i in 0..self.c.len() {
            result[i] = self.c[i] / s;
        }
        Point { c: result }
    }
}

impl Point {
    pub fn dim(&self) -> usize {
        self.c.len()
    }

    pub fn norm(&self) -> f32 {
        self.c.iter().map(|x| x * x).sum::<f32>().sqrt()
    }
}

pub fn euclid_dist(p1: &Point, p2: &Point) -> f32 {

    assert_eq!(p1.dim(),p2.dim());
    let mut dist = 0.0;
    for i in 0..p1.dim() {
        dist += (p1.c[i as usize] - p2.c[i as usize]) * (p1.c[i as usize] - p2.c[i as usize]);
    }

    dist.sqrt()
}

pub fn calc_score(
    solution: &Vec<usize>,
    data: &Vec<Point>,
    rest: &Vec<Vec<i8>>,
    k: u32,
    l: f32,
) -> f32 {
    // calc infeasibility of the solution

    let m = data[0].dim();
    let mut nc = vec![0.0; k as usize];
    let mut inner_distance: Vec<f32> = vec![0.0; k as usize];

    let mut centroids: Vec<Point> = vec![Point { c: vec![0.0; m] }; k as usize];

    for (i, cluster) in solution.iter().enumerate() {
        nc[*cluster] += 1.0;
        centroids[*cluster] += data[i as usize].clone();
    }

    for i in 0..k {
        centroids[i as usize] = centroids[i as usize].clone() / nc[i as usize];
    }

    //println!("Centroids: {:?}", centroids);

    for (i, cluster) in solution.iter().enumerate() {
        inner_distance[*cluster] +=
            euclid_dist(&centroids[*cluster], &data[i as usize]) / nc[*cluster];
    }
    //println!("Inner dist: {:?}", inner_distance);
    let mut c_const = 0.0;

    for i in 0..k {
        c_const += inner_distance[i as usize] / k as f32;
    }

    // calc infeasibility

    let mut inf: u32 = 0;

    for (i, c1) in solution.iter().enumerate() {
        for j in i + 1..solution.len() {
            if rest[i as usize][j as usize] == -1 && *c1 == solution[j as usize] {
                inf += 1;
            } else if rest[i as usize][j as usize] == 1 && *c1 != solution[j as usize] {
                inf += 1;
            }
        }
    }

    return c_const + l * (inf as f32);
}

pub fn calc_lambda(data: &Vec<Point>, rest: &Vec<Vec<i8>>) -> f32 {
    let mut max_dist: f32 = 0.0;

    for u in data.iter() {
        for v in data.iter() {
            let d = ((*u).clone() - (*v).clone()).norm();
            if d > max_dist {
                max_dist = d
            }
        }
    }

    let mut n: f32 = rest
        .iter()
        .map(|x| (*x).iter().filter(|y| **y != 0).count())
        .sum::<usize>() as f32;

    n = (n - rest.len() as f32) / 2.0;

    max_dist / n
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct HistoryPoint(pub f32);

#[derive(Clone, Serialize, Deserialize)]
pub struct AlgResult {
    pub sol: Option<Vec<usize>>,
    pub score: f32,
    pub generations: Option<u32>,
    pub history: Option<Vec<HistoryPoint>>,
    pub time: Option<f32>,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Output {
    pub sol: Vec<usize>,
    pub score: f32,
    pub generations: Option<u32>,
    pub history: Option<Vec<HistoryPoint>>,
    pub time: f32,
    pub c: f32,
    pub inf: f32,
}

// Get out some parts of the previous calc_score function. I mantein
// them in only one function for performance reasons.

pub fn calc_c_value_inf(
    solution: &Vec<usize>,
    data: &Vec<Point>,
    rest: &Vec<Vec<i8>>,
    k: u32,
) -> Score {
    // calc infeasibility of the solution

    let mut centers: Vec<Point> = vec![
        Point {
            c: vec![0.0; data[0].dim()]
        };
        k as usize
    ];
    let mut inner: Vec<f32> = vec![0.0; k as usize];

    // create auxiliar data structures

    let mut clusters: Vec<Vec<&Point>> = Vec::new();
    let mut clusters_ids: Vec<HashSet<u32>> = Vec::new();

    for _i in 0..k {
        clusters.push(Vec::new());
        clusters_ids.push(HashSet::new());
    }

    for (i, c) in solution.iter().enumerate() {
        clusters[*c as usize].push(&data[i as usize]);
        clusters_ids[*c as usize].insert(i as u32);
    }

    // calc centers

    for i in 0..k {
        for c in clusters[i as usize].clone() {
            centers[i as usize] += (*c).clone()
        }

        centers[i as usize] =
            centers[i as usize].clone() * (1.0 / clusters[i as usize].len() as f32)
    }

    //println!("Centers good: {:?}", centers);

    // Calculate inner distance

    for i in 0..k {
        inner[i as usize] = clusters[i as usize]
            .iter()
            .map(|x| ((*x).clone() - centers[i as usize].clone()).norm())
            .sum::<f32>()
            / (clusters[i as usize].len() as f32)
    }
    //println!("Inner dist good: {:?}", inner);
    let c_const: f32 = inner.iter().sum::<f32>() / k as f32;

    let mut inf: u32 = 0;

    for (i, c) in solution.iter().enumerate() {
        for (j, rel) in rest[i as usize][i as usize..].iter().enumerate() {
            if *rel == -1 && clusters_ids[*c as usize].contains(&(j as u32 + i as u32)) {
                inf += 1
            } else if *rel == 1 && !clusters_ids[*c as usize].contains(&(j as u32 + i as u32)) {
                inf += 1
            }
        }
    }

    Score(c_const, inf as f32)
}
