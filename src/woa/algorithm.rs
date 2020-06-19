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


fn find_best_whale(
    whale_solutions: &Vec<Vec<usize>>,
    data: &Vec<Point>,
    rest: &Vec<Vec<i8>>,  
    k: u32, 
    l: f32) -> (usize, f32){

    // search id of the best whale and return its index and score

    let mut best_whale: usize = 0;
    let mut best_whale_score: f32 = std::f32::MAX;

    for (id, solution) in whale_solutions.iter().enumerate(){
        let score = calc_score(solution, &data, &rest, k, l);
        if score < best_whale_score{
            best_whale_score = score;
            best_whale = id;
        }
    }

    (best_whale, best_whale_score)

}

#[allow(non_snake_case)]
fn positional_move(
    whale: &mut Vec<Vec<f32>>, 
    best_whale:&Vec<Vec<f32>>,
    A: &Point,
    C: f32,
    rng: &mut StdRng
){

    let mut D:f32 = 0.0;

    let mut whale_points: Vec<Point> = Vec::with_capacity(whale.len());
    let mut best_whale_points: Vec<Point> = Vec::with_capacity(whale.len());

    for (i, centroid) in best_whale.iter().enumerate(){
        let whale_i_cluster = Point{c: whale[i].clone()};
        let best_whale_i_cluster = Point{c: centroid.clone()};

        whale_points.push(whale_i_cluster.clone());
        best_whale_points.push(best_whale_i_cluster.clone());

        D += euclid_dist(&(best_whale_i_cluster*C)  , &whale_i_cluster);

    }

    for (i, _whale_coord) in whale_points.iter().enumerate(){
        whale[i] = (best_whale_points[i].clone()-A.clone()*D).c;
    }

    
}

#[allow(non_snake_case)]
fn spiral_move(
    whale: &mut Vec<Vec<f32>>, 
    best_whale:&Vec<Vec<f32>>,
    rng: &mut StdRng
){
    let l = rng.gen::<f32>();
    let mut D:f32 = 0.0;

    let b = 1.6;

    let mut whale_points: Vec<Point> = Vec::with_capacity(whale.len());
    let mut best_whale_points: Vec<Point> = Vec::with_capacity(whale.len());

    for (i, centroid) in best_whale.iter().enumerate(){
        let whale_i_cluster = Point{c: whale[i].clone()};
        let best_whale_i_cluster = Point{c: centroid.clone()};

        whale_points.push(whale_i_cluster.clone());
        best_whale_points.push(best_whale_i_cluster.clone());

        D += euclid_dist(&(best_whale_i_cluster)  , &whale_i_cluster);

    }

    let radius = (b*l).exp()*(2.0*l*std::f32::consts::PI).cos();

    for (i, _whale_coord) in whale_points.iter().enumerate(){
        whale[i] = (best_whale_points[i].clone() + radius*D).c;
    }
}


#[allow(non_snake_case)]
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

    let mut best_whale_solution: Vec<usize> = Vec::new();

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

        // get id of the best whale

        let (best_whale_id, best_whale_score) = find_best_whale(&whale_solutions, data, rest, k, l);

        let best_whale = whales[best_whale_id].clone();
        best_whale_solution = whale_solutions[best_whale_id].clone();
        //println!{"Best whale ({}) {:?} with score {}", best_whale, whale_solutions[best_whale], best_whale_score};

        for whale in &mut whales{

            let r = rng.gen::<f32>();
            let a_val = rng.gen::<f32>()*2.0;
            let a_vec = Point{c: vec![a_val; whale[0].len()]};
            let A = a_vec.clone()*2.0*r - a_vec.clone();
            let C = 2.0*r;

            // Compute p value
            if rng.gen::<f32>() < 0.5{
                if A.norm() < 1.0{
                    positional_move(whale,
                        &best_whale,
                        &A,
                        C,
                        &mut rng
                    );
                }
            }else{
                spiral_move(whale, &best_whale, &mut rng);
            }
        }

        current_iteration += 1;

        println!("Best solution {:?}", best_whale_solution);
        println!("Score {}", calc_score(&best_whale_solution, data, rest, k, l));

        a = linear_scale(a_start, a, current_iteration, max_iterations);
    }


    AlgResult {
        sol: None,
        score: 0.0,
        generations: None,
        history: None,
        time: None,
    }

}