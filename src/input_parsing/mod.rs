use crate::common::Point;
use std::fs::File;
use std::io::{BufRead, BufReader};

pub fn read_points(filepath: &str) -> Vec<Point> {
    let mut data: Vec<Point> = Vec::new();

    let file = File::open(filepath).unwrap();
    let reader = BufReader::new(file);

    for (_index, line) in reader.lines().enumerate() {
        let line = line.unwrap();
        let row: Vec<f32> = line
            .split(",")
            .map(|s| s.trim())
            .map(|s| s.parse().unwrap())
            .collect();

        //println!("{:?}", vec);
        data.push(Point { c: row });
    }

    data
}

pub fn read_restrictions(filepath: &str) -> Vec<Vec<i8>> {
    let mut matrix: Vec<Vec<i8>> = Vec::new();

    let file = File::open(filepath).unwrap();
    let reader = BufReader::new(file);

    for (_index, line) in reader.lines().enumerate() {
        let line = line.unwrap();
        let row: Vec<i8> = line
            .split(",")
            .map(|s| s.trim())
            .map(|s| s.parse().unwrap())
            .collect();

        //println!("{:?}", vec);
        matrix.push(row);
    }

    matrix
}
