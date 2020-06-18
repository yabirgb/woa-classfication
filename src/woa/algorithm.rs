use crate::common::{Point, euclid_dist, AlgResult, calc_score};

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use rand::distributions::Uniform;

fn linear_scale(init_val:f32, final_val:f32, t: usize, total_steps: usize) -> f32{

    let step: f32 = t as f32/ total_steps as f32;

    init_val*(1.0-step) + step*final_val
}

pub fn woa_clustering(
    data: &Vec<Point>, 
    rest: &Vec<Vec<i8>>, 
    k: u32, 
    l: f32,
    seed: u64,
    n_agents:usize,
    max_iterations: usize
)->AlgResult
{

    let dim = data[0].dim();
    let mut rng = StdRng::seed_from_u64(seed);

    // Define constants of the woa algoritm
    let a_start = 2.0;
    let mut a: f32 = 2.0;
    let mut p: f32;

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
    // A list of agents
    // that cointains a list of centroids
    // that is a list of f32
    let mut whales: Vec<Vec<Vec<f32>>> = Vec::with_capacity(n_agents);

    // randomly initialize agents
    for _j in 0..n_agents {
        // generate dim random numbers
        let mut whale: Vec<Vec<f32>> = Vec::with_capacity(k as usize);
        for _i in 0..k{
            let components: Vec<f32> = (0..dim).map(|i| rng.gen_range(min[i], max[i])).collect();
            whale.push(components);
        // add the centroid to the vec of centers
        }
        whales.push(whale);
    }

    let mut current_iteration:usize = 0;

    //println!("Whales: {:?}", whales);

    while current_iteration < max_iterations{

        let mut whale_solutions: Vec<Vec<usize>> = Vec::with_capacity(n_agents);

        for whale in &whales {
            let mut solution: Vec<usize> = Vec::with_capacity(n_agents);
            let mut best_distance:f32 = std::f32::MAX;
            let mut best_cluster: usize = 0;

            for point in data{
                for (i, center) in whale.into_iter().enumerate(){
                    // find the nearest cluster
                    let distance = euclid_dist(&Point{c:center.to_vec()}, point);
                    if distance < best_distance {
                        best_cluster = i;
                        best_distance = distance;
                    }   

                }
                solution.push(best_cluster);
                best_distance = std::f32::MAX;
            }
            whale_solutions.push(solution);
        }

        // search id of the best whale

        let mut best_whale: usize = 0;
        let mut best_whale_score: f32 = std::f32::MAX;

        for (id, solution) in whale_solutions.iter().enumerate(){
            let score = calc_score(solution, &data, &rest, k, l);
            if score < best_whale_score{
                best_whale_score = score;
                best_whale = id;
            }
        }

        println!{"Best whale ({}) {:?} with score {}", best_whale, whale_solutions[best_whale], best_whale_score};

        // Update a param according to our model
        let a = linear_scale(a_start, 0.0, current_iteration, max_iterations);
        // Create r vector randomly initialized with values in [0,1]
        let r: Point = Point{c:(0..dim).map(|i| rng.gen::<f32>()).collect()};
        // Calculate movement vector
        let A: Point = r.clone()*2.0*a-a;
        println!("R: {:?} - A: {:?} - a: {}", r, A, a);
        // Compute p value
        if rng.gen::<f32>() < 0.5{

        }else{

        }


        current_iteration += 1;
    }

    AlgResult {
        sol: None,
        score: 0.0,
        generations: None,
        history: None,
        time: None,
    }

}