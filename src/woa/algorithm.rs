use crate::common::{Point, euclid_dist, AlgResult, calc_score};

use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

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
    A: Point,
    C: Vec<f32>,
){

    // this function modifies whales by the encicling prey thecnique 
    // X(t+1) = Best solution - AD

    let mut whale_points: Vec<Point> = Vec::with_capacity(whale.len());
    let mut best_whale_points: Vec<Point> = Vec::with_capacity(whale.len());

    let mut a_vec = A.clone();

    for (i, centroid) in best_whale.iter().enumerate(){
        let whale_i_cluster = Point{c: whale[i].clone()};
        let best_whale_i_cluster = Point{c: centroid.clone()};

        whale_points.push(whale_i_cluster.clone());
        best_whale_points.push(best_whale_i_cluster.clone());

        // A*D
        a_vec.c[i] *= euclid_dist(&(best_whale_i_cluster*C[i])  , &whale_i_cluster);

    }

    for (i, _whale_coord) in whale_points.iter().enumerate(){
        whale[i] = (best_whale_points[i].clone()-a_vec.c[i]).c;
    }

    
}

#[allow(non_snake_case)]
fn random_positional_move(
    whale: &mut Vec<Vec<f32>>, 
    random: &Vec<Vec<f32>>,
    A: &Point,
){

    let mut whale_points: Vec<Point> = Vec::with_capacity(whale.len());
    let mut random_whale_points: Vec<Point> = Vec::with_capacity(whale.len());

    let mut a_vec = A.clone();

    //println!("A_vec: {:?}", a_vec );
    //println!("random: {:?}", random);
    for (i, centroid) in random.iter().enumerate(){
        let whale_i_cluster = Point{c: whale[i].clone()};
        let random_whale_i_cluster = Point{c: centroid.clone()};

        whale_points.push(whale_i_cluster.clone());
        random_whale_points.push(random_whale_i_cluster.clone());

        // A*D
        a_vec.c[i] *= euclid_dist(&random_whale_i_cluster  , &whale_i_cluster);

    }

    for (i, _whale_coord) in whale_points.iter().enumerate(){
        whale[i] = (random_whale_points[i].clone()-a_vec.c[i]).c;
    }

    
}

#[allow(non_snake_case)]
fn spiral_move(
    whale: &mut Vec<Vec<f32>>, 
    best_whale:&Vec<Vec<f32>>,
    rng: &mut StdRng
){
    let l:f32 = rng.gen_range(-1.0,1.0);

    let mut D:Vec<f32> = Vec::with_capacity(whale.len());

    let b = 1.0;

    let mut whale_points: Vec<Point> = Vec::with_capacity(whale.len());
    let mut best_whale_points: Vec<Point> = Vec::with_capacity(whale.len());

    for (i, centroid) in best_whale.iter().enumerate(){
        let whale_i_cluster = Point{c: whale[i].clone()};
        let best_whale_i_cluster = Point{c: centroid.clone()};

        whale_points.push(whale_i_cluster.clone());
        best_whale_points.push(best_whale_i_cluster.clone());

        D.push(euclid_dist(&(best_whale_i_cluster)  , &whale_i_cluster));

    }

    let radius = (b*l).exp()*(2.0*l*std::f32::consts::PI).cos();
    //println!("BEST: {:?}\nWhale: {:?}\nD: {:?}", best_whale_points,best_whale_points, D);
    for (i, _whale_coord) in whale_points.iter().enumerate(){
        whale[i] = (best_whale_points[i].clone() + D[i]*radius).c;
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
    max_evaluations: usize
)->AlgResult
{

    let dim = data[0].dim();
    let mut rng = StdRng::seed_from_u64(seed);

    // Define constants of the woa algoritm
    let a_start = 2.0;

    // set the maximum number of iterations

    let max_iterations = max_evaluations/n_agents;

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

        // Store for each whale an assignation of clusters
        let mut whale_solutions: Vec<Vec<usize>> = Vec::with_capacity(n_agents);

        // for each set of clusters centers find the best solution
        // Iterate over the points searcvhing the nearest cluster and assing
        // it to the point
        // TODO: If the solution is not valid generate a new solution
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

        // get id of the best whale beteween all the geneated solutions
        let (best_whale_id, _best_whale_score) = find_best_whale(&whale_solutions, data, rest, k, l);

        let best_whale = whales[best_whale_id].clone();
        best_whale_solution = whale_solutions[best_whale_id].clone();
        //println!{"Best whale ({}) {:?} with score {}", best_whale, whale_solutions[best_whale], best_whale_score};

        // for each sear agend

        let whales_size = whales.len();
        let whales_copy = whales.clone();

        for whale in &mut whales{

            // Update a, A, C, l and p
            let r:f32 = rng.gen_range(0.0, 1.0);

            let a = linear_scale(a_start, 0.0, current_iteration, max_iterations);

            let a_vec = Point{c: vec![a; k as usize]};
            let A = a_vec.clone()*2.0*r - a_vec.clone();
            // here C is not a vector because is a constant vector always multiplied by another
            let C:Vec<f32> = (0..k).map(|_x| 2.0*rng.gen_range(0.0, 1.0)).collect::<Vec<f32>>(); 

            //println!("Norm of A matrix: {:?}", A);
            // Compute p value
            if rng.gen::<f32>() < 0.5{
                if A.norm() < 1.0{
                    positional_move(whale,
                        &best_whale,
                        A,
                        C
                    );
                }else{
                    let selected_whale_id:usize = rng.gen_range(0, whales_size);
                    random_positional_move(whale, &whales_copy[selected_whale_id], &A);
                }
            }else{
                // l value is generated inside this function
                spiral_move(whale, &best_whale, &mut rng);
            }
        }

        // re-enter whales that have gone aoutside the boundaries

        for whale in &mut whales{
            for cluster in whale.iter_mut(){
                for (i, max_cord) in max.iter().enumerate() {
                    if cluster[i] > *max_cord{
                        cluster[i] = *max_cord;
                    }
                }
                for (i, min_cord) in min.iter().enumerate() {
                    if cluster[i] < *min_cord{
                        cluster[i] = *min_cord;
                    }
                }
            }
        }

        current_iteration += 1;

        //println!("Whales: {:?}", whales);
        //println!("Best solution {:?}", best_whale_solution);
        //println!("Score {}", calc_score(&best_whale_solution, data, rest, k, l));
        //println!("Current best score: {}", best_whale_score);
    }
    println!("Best solution {:?}", best_whale_solution);
    println!("Score {}", calc_score(&best_whale_solution, data, rest, k, l));

    AlgResult {
        sol: None,
        score: 0.0,
        generations: None,
        history: None,
        time: None,
    }

}