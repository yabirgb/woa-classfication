mod common;
use common::{Point, euclid_dist};

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use rand::distributions::Uniform;

fn woa_classifier(
    data: &Vec<Point>, 
    rest: &Vec<Vec<i8>>, 
    k: u32, 
    seed: u64,
    n_agents:u32,
    max_iterations: u32
)
{

    let dim = data[0].dim();
    let mut rng = StdRng::seed_from_u64(seed);

    // determine min and max for each component on the input data

    let mut max = vec![std::f32::MIN; dim];
    let mut min = vec![std::f32::MAX; dim];

    for v in data {
        for i in 0..v.dim() {
            if max[i] < v.c[i] {
                max[i] = v.c[i]
            }
            if min[i] > v.c[i] {
                min[i] = v.c[i]
            }
        }
    }

    // vector with agents, the whales of the algorithm
    let mut whales: Vec<Point> = Vec::with_capacity(n_agents);

    // randomly initialize agents
    for _j in 0..n_agents {
        // generate dim random numbers
        let components: Vec<f32> = (0..dim).map(|i| rng.gen_range(min[i], max[i])).collect();
        // generate the centroid
        let cp = Point { c: components };
        // add the centroid to the vec of centers
        whales.push(cp);
    }

    let mut current_iteration:usize = 0;

    // Distance between point and cluster
    let mut distance; 

    while current_iteration < max_iterations{

        let mut whales_solutions: Vec<Vec<usize>> = Vec::with_capacity(n_agents);

        for whale in whales {
            let mut solution: Vec<usize> = Vec:with_capacity(n_agents);
            let mut best_distance:f32 = std::f32::MAX;
            let mut best_cluster: usize;

            for point in data{
                // find the nearest cluster
                distance = euclid_dist(whale, point);
                
                if distance < best_distance {
                    
                }

            }
            
        }

        current_iteration += 1;
    }

}