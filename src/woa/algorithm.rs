use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use rand::seq::SliceRandom;

use crate::common::{Point, euclid_dist, AlgResult, calc_score};
use super::{ls_solve, valid_sol};


fn linear_scale(max_val:f32, t: usize, total_steps: usize) -> f32{
    max_val - t as f32 *2.0/total_steps as f32
}

fn top_n_elements(solutions: &Vec<f32>, n:usize)->Vec<usize>{
    // given a list of elements return the index of the n elements
    // with lower value
    assert!(n < solutions.len());
    let mut sorted:Vec<(usize, &f32)> = solutions.iter().enumerate().collect::<Vec<(usize, &f32)>>();
    sorted.sort_by(|a, b| (a.1).partial_cmp(b.1).unwrap());
    let l = sorted.iter().map(|(i,_)| *i);
    let mut sol = l.collect::<Vec<usize>>();
    sol.truncate(n);
    sol
}

#[test]
fn test_top_n_elements(){
    let sol = vec![20.0, 30.0, 45.0, 30.0, 15.0, 100.0];
    assert_eq!(top_n_elements(&sol, 3), vec![4,0,1]);
    assert_eq!(top_n_elements(&sol, 4), vec![4,0,1,3]);
}

fn calc_inf_delta(
    _data: &Vec<Point>,
    rest: &Vec<Vec<i8>>,
    point_id: usize, 
    cluster_points: &Vec<usize>, 
) ->f32 {
    // calc the change on infeasibility made by assignign point_id to clust k
    // given the current solution

    let mut inf = 0.0;
    for (pair, value) in rest[point_id].iter().enumerate(){
        let contains = cluster_points.contains(&pair);
        if *value == 1 && !contains{
            inf += 1.0
        }else if *value == -1 && contains{
            inf += 1.0
        } 
    }

    inf
}

fn cluster_assignation(
    centers: &Vec<Vec<f32>>,
    data: &Vec<Point>,
    rest: &Vec<Vec<i8>>,
    k:u32,
    _id: u64,
    mut rng: &mut StdRng
)->Vec<usize>{

    // select and order to iterate over the points
    let mut order: Vec<usize> = (0..data.len()).rev().collect();
    order.shuffle(&mut rng);

    // flag variable to detect changes in the assignations

    // Store related points in a hashmap
    let mut clusters: Vec<Vec<usize>> = vec![Vec::new();k as usize];

    for point_id in order.clone().iter(){
        // select the best cluster for this point in terms of
        // infeassibility. First computy diff in infeasbility
        let mut diff_inf = Vec::new();


        for i in 0..k{
            diff_inf.push(calc_inf_delta(data, rest, *point_id, &clusters[i as usize]));
        }

        // get min value from previous list
        let min_inf_delta = diff_inf.iter().min_by(|a,b| a.partial_cmp(b).unwrap()).unwrap();

        // select the cluster or clusters that achieve this value

        let mut cluster_candidates:Vec<usize> = Vec::new();
        for (id, cluster_score) in diff_inf.iter().enumerate(){
            if clusters[id].len() == 0{
                cluster_candidates.insert(0, id);
            }else if *cluster_score == *min_inf_delta {
                cluster_candidates.push(id);
            }
            

        }


        if clusters[cluster_candidates[0]].len() == 0 || cluster_candidates.len() == 1{
            clusters[cluster_candidates[0]].push(*point_id);
        }else{
            // if we have more than one candidate choose the nearest one
            let mut distances: Vec<f32> = Vec::new();

            for cluster_id in cluster_candidates.iter(){
                let distance = euclid_dist(&Point{c:centers[*cluster_id].clone()}, &data[*point_id]);
                distances.push(distance);
            }

            // get the min distance
            let min_disntace = distances.iter().min_by(|a,b| a.partial_cmp(b).unwrap()).unwrap();

            // get the cluster with the min distance
            let mut best_cluster:usize = 0;
            for (i, cluster_dist) in distances.iter().enumerate(){
                if *cluster_dist == *min_disntace{
                    best_cluster = i;
                    break;
                }
            }

            // add the point to the cluster list
            clusters[best_cluster].push(*point_id);
            // mark the point as added
        }



    }

    let mut points_added = 0;

    let mut solution:Vec<usize> = vec![0; data.len()];
    //println!("Clusters? {:?}", clusters);
    for (i, c) in clusters.iter().enumerate() {
        assert_ne!(c.len(), 0);

        for p in c.iter() {
            solution[*p] = i;
            points_added += 1;
        }
    }

    assert_eq!(points_added, data.len());
    assert!(valid_sol(&solution, k));
    solution

}


fn find_best_whale(
    whale_solutions: &Vec<Vec<usize>>,
    data: &Vec<Point>,
    rest: &Vec<Vec<i8>>,  
    k: u32, 
    l: f32,
    evaluations: &mut usize
) -> (usize, f32, Vec<f32>){

    // search id of the best whale and return its index and score

    let mut best_whale: usize = 0;
    let mut best_whale_score: f32 = std::f32::MAX;
    let mut scores = Vec::with_capacity(whale_solutions.len());

    for (id, solution) in whale_solutions.iter().enumerate(){
        let score = calc_score(solution, &data, &rest, k, l);
        
        scores.push(score);

        *evaluations += 1;
        if score < best_whale_score{
            best_whale_score = score;
            best_whale = id;
        }
    }

    (best_whale, best_whale_score, scores)

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
    C: Vec<f32>,
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
        a_vec.c[i] *= euclid_dist(&(random_whale_i_cluster*C[i])  , &whale_i_cluster);

    }

    for (i, _whale_coord) in whale_points.iter().enumerate(){
        whale[i] = (random_whale_points[i].clone()-a_vec.c[i]).c;
    }

    
}

#[allow(non_snake_case)]
fn spiral_move(
    whale: &mut Vec<Vec<f32>>, 
    best_whale:&Vec<Vec<f32>>,
    l: f32
){
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
        whale[i] = (best_whale_points[i].clone() + D[i].abs()*radius).c;
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

    let mut best_solution_ever: Vec<usize> = Vec::new();
    let mut best_score_ever: f32 = std::f32::MAX;

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

    let mut current_evaluations:usize = 0;

    //println!("Whales: {:?}", whales);

    let mut best_whale_solution: Vec<usize>;

    while current_evaluations < max_evaluations{

        // Store for each whale an assignation of clusters
        let mut whale_solutions: Vec<Vec<usize>> = Vec::with_capacity(n_agents);

        // for each set of clusters centers find the best solution
        // Iterate over the points searcvhing the nearest cluster and assing
        // it to the point
        for (id, whale) in whales.iter().enumerate() {
            whale_solutions.push(cluster_assignation(
                whale,
                data,
                rest,
                k,
                id as u64,
                &mut rng
            ));
        }

        // get id of the best whale beteween all the geneated solutions
        let (best_whale_id, best_whale_score, step_scores) = find_best_whale(&whale_solutions, data, rest, k, l, &mut current_evaluations);

        let best_whale = whales[best_whale_id].clone();
        best_whale_solution = whale_solutions[best_whale_id].clone();
        //println!{"Best whale ({}) {:?} with score {}", best_whale, whale_solutions[best_whale], best_whale_score};

        // for each sear agend

        let whales_size = whales.len();
        let whales_copy = whales.clone();

        for whale in &mut whales{

            // Update a, A, C, l and p
            let r:f32 = rng.gen_range(0.0, 1.0);

            let a = linear_scale(2.0, current_evaluations, max_evaluations);
            //println!("a val: {}", a);

            let a_vec = Point{c: vec![a; k as usize]};
            let A = a_vec.clone()*2.0*r - a_vec.clone();
            // here C is not a vector because is a constant vector always multiplied by another
            let C:Vec<f32> = (0..k).map(|_x|  2.0*rng.gen_range(0.0, 1.0)).collect::<Vec<f32>>(); 

            //println!("Norm of A matrix: {:?}", A.norm());
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
                    random_positional_move(whale, &whales_copy[selected_whale_id], &A, C);
                }
            }else{
                // l value is generated inside this function
                let a2 =  -1.0 - current_evaluations as f32 * (-1.0/max_evaluations as f32);
                let l = (a2 - 1.0) *rng.gen_range(0.0,1.0) + 1.0;

                spiral_move(whale, &best_whale, l);
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

        //println!("Whales: {:?}", whales);
        //println!("Best solution {:?}", best_whale_solution);
        //println!("Score {}", calc_score(&best_whale_solution, data, rest, k, l));
        //println!("Best whale id: {}", best_whale_id);
        if best_whale_score < best_score_ever {
            best_score_ever = best_whale_score;
            best_solution_ever = best_whale_solution.clone();
            println!("Best score ever {}", best_score_ever);
        }
        
    }
    //println!("Best solution {:?}", best_whale_solution);
    //println!("Score {}", calc_score(&best_whale_solution, data, rest, k, l));

    //println!("Best solution ever {:?}", best_solution_ever.clone());
    //println!("Best score ever {}", best_score_ever);

    AlgResult {
        sol: Some(best_solution_ever),
        score: best_score_ever,
        generations: None,
        history: None,
        time: None,
    }

}

#[allow(non_snake_case)]
pub fn woa_clustering_ls(
    data: &Vec<Point>, 
    rest: &Vec<Vec<i8>>, 
    k: u32, 
    l: f32,
    seed: u64,
    n_agents:usize,
    max_evaluations: usize
)->AlgResult
{

    let MAX_LS_EVALS:usize = 5000;

    let dim = data[0].dim();
    let mut rng = StdRng::seed_from_u64(seed);

    let mut best_solution_ever: Vec<usize> = Vec::new();
    let mut best_score_ever: f32 = std::f32::MAX;

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

    let mut current_evaluations:usize = 0;

    //println!("Whales: {:?}", whales);

    let mut best_whale_solution: Vec<usize>;

    while current_evaluations < max_evaluations{

        // Store for each whale an assignation of clusters
        let mut whale_solutions: Vec<Vec<usize>> = Vec::with_capacity(n_agents);

        // for each set of clusters centers find the best solution
        // Iterate over the points searcvhing the nearest cluster and assing
        // it to the point
        for (id, whale) in whales.iter().enumerate() {
            whale_solutions.push(cluster_assignation(
                whale,
                data,
                rest,
                k,
                id as u64,
                &mut rng
            ));
        }

        // get id of the best whale beteween all the geneated solutions
        let (mut best_whale_id, mut best_whale_score, step_scores) = find_best_whale(&whale_solutions, data, rest, k, l, &mut current_evaluations);

        let best_whale = whales[best_whale_id].clone();
        best_whale_solution = whale_solutions[best_whale_id].clone();
        //println!{"Best whale ({}) {:?} with score {}", best_whale, whale_solutions[best_whale], best_whale_score};

        // for each sear agend

        let whales_size = whales.len();
        let whales_copy = whales.clone();

        for whale in &mut whales{

            // Update a, A, C, l and p
            let r:f32 = rng.gen_range(0.0, 1.0);

            let a = linear_scale(2.0, current_evaluations, max_evaluations);
            //println!("a val: {}", a);

            let a_vec = Point{c: vec![a; k as usize]};
            let A = a_vec.clone()*2.0*r - a_vec.clone();
            // here C is not a vector because is a constant vector always multiplied by another
            let C:Vec<f32> = (0..k).map(|_x|  2.0*rng.gen_range(0.0, 1.0)).collect::<Vec<f32>>(); 

            //println!("Norm of A matrix: {:?}", A.norm());
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
                    random_positional_move(whale, &whales_copy[selected_whale_id], &A, C);
                }
            }else{
                // l value is generated inside this function
                let a2 =  -1.0 - current_evaluations as f32 * (-1.0/max_evaluations as f32);
                let l = (a2 - 1.0) *rng.gen_range(0.0,1.0) + 1.0;
                //let l = rng.gen_range(-1.0, 1.0);
                spiral_move(whale, &best_whale, l);
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

        //println!("Current best score prev ls {}", best_whale_score);
        //println!("Whales: {:?}", whales);

        let mut new_whales: Vec<Vec<Vec<f32>>> = Vec::new();

        //println!("{:?}", whales);

        for (whale_id, _whale) in whales.iter_mut().enumerate(){
            let (ls_centroid, ls_solution, ls_score) = ls_solve(
                &whale_solutions[whale_id],
                step_scores[whale_id],
                data,
                rest,
                k,
                l,
                MAX_LS_EVALS,
                &mut rng
            );

            current_evaluations += MAX_LS_EVALS;

            new_whales.push(ls_centroid);
            //whale_solutions[whale_id] = ls_solution.clone();
            if ls_score < best_whale_score{
                best_whale_score = ls_score;
                best_whale_solution = ls_solution.clone();
            }
        }

        whales = new_whales.clone();

        //println!("Best solution {:?}", best_whale_solution);
        //println!("Score {}", calc_score(&best_whale_solution, data, rest, k, l));
        //println!("Best whale id: {}", best_whale_id);
        if best_whale_score < best_score_ever {
            best_score_ever = best_whale_score;
            best_solution_ever = best_whale_solution.clone();
            println!("Best score ever {}", best_score_ever);

        }


        //println!("Current best score after ls {}", best_whale_score);
        //println!("Whales: {:?}", whales);

        
    }
    //println!("Best solution {:?}", best_whale_solution);
    //println!("Score {}", calc_score(&best_whale_solution, data, rest, k, l));

    //println!("Best solution ever {:?}", best_solution_ever.clone());
    //println!("Best score ever {}", best_score_ever);
    assert!(valid_sol(&best_solution_ever, k));
    AlgResult {
        sol: Some(best_solution_ever),
        score: best_score_ever,
        generations: Some(current_evaluations as u32),
        history: None,
        time: None,
    }

}

#[allow(non_snake_case)]
pub fn woa_clustering_best_pool(
    data: &Vec<Point>, 
    rest: &Vec<Vec<i8>>, 
    k: u32, 
    l: f32,
    seed: u64,
    n_agents:usize,
    max_evaluations: usize
)->AlgResult
{

    let MAX_LS_EVALS:usize = 5000;
    let EL_SIZE = 5;

    let dim = data[0].dim();
    let mut rng = StdRng::seed_from_u64(seed);

    let mut best_solution_ever: Vec<usize> = Vec::new();
    let mut best_score_ever: f32 = std::f32::MAX;

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

    let mut current_evaluations:usize = 0;
    // when was the last evaluation where we saw an imrpovement in solution
    let mut last_improvement: usize = 0;

    // vector with the best solutions idss
    let mut elite_solutions: Vec<usize>;
    let mut best_whale_solution: Vec<usize>;

    while current_evaluations < max_evaluations{

        // Store for each whale an assignation of clusters
        let mut whale_solutions: Vec<Vec<usize>> = Vec::with_capacity(n_agents);

        // for each set of clusters centers find the best solution
        // Iterate over the points searcvhing the nearest cluster and assing
        // it to the point
        for (id, whale) in whales.iter().enumerate() {
            whale_solutions.push(cluster_assignation(
                whale,
                data,
                rest,
                k,
                id as u64,
                &mut rng
            ));
        }

        // get id of the best whale beteween all the geneated solutions
        let (best_whale_id, mut best_whale_score, step_scores) = find_best_whale(&whale_solutions, data, rest, k, l, &mut current_evaluations);

        // update the list of elite solutions
        elite_solutions = top_n_elements(&step_scores, EL_SIZE);

        let best_whale = whales[best_whale_id].clone();
        best_whale_solution = whale_solutions[best_whale_id].clone();

        // for each search agent
        let whales_size = whales.len();
        let whales_copy = whales.clone();

        for whale in &mut whales{

            // Update a, A, C, l and p
            let r:f32 = rng.gen_range(0.0, 1.0);

            let a = linear_scale(2.0, current_evaluations, max_evaluations);
            //println!("a val: {}", a);

            let a_vec = Point{c: vec![a; k as usize]};
            let A = a_vec.clone()*2.0*r - a_vec.clone();
            // here C is not a vector because is a constant vector always multiplied by another
            let C:Vec<f32> = (0..k).map(|_x|  2.0*rng.gen_range(0.0, 1.0)).collect::<Vec<f32>>(); 

            //println!("Norm of A matrix: {:?}", A.norm());
            // Compute p value
            if rng.gen::<f32>() < 0.5{
                if A.norm() < 1.0{
                    // select a whale from the EL pool
                    let selected_whale_id:usize = rng.gen_range(0, EL_SIZE);
                    positional_move(whale,
                        &whales_copy[selected_whale_id],
                        A,
                        C
                    );
                }else{
                    let selected_whale_id:usize = rng.gen_range(0, whales_size);
                    random_positional_move(whale, &whales_copy[selected_whale_id], &A, C);
                }
            }else{
                // l value is generated inside this function
                let a2 =  -1.0 - current_evaluations as f32 * (-1.0/max_evaluations as f32);
                let l = (a2 - 1.0) *rng.gen_range(0.0,1.0) + 1.0;
                let selected_whale_id:usize = rng.gen_range(0, whales_size);
                spiral_move(whale, &whales_copy[selected_whale_id], l);
                //let l = rng.gen_ra.enumerate()
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

        //println!("Current best score prev ls {}", best_whale_score);
        //println!("Whales: {:?}", whales);

        // apply intensification to the search if there has been no improvement in a long term
        // but only to the top list

        if current_evaluations - last_improvement > max_evaluations/10000{
            let mut new_whales: Vec<Vec<Vec<f32>>> = Vec::new();

            //println!("{:?}", whales);
    
            for whale_id in elite_solutions.iter(){
                let (ls_centroid, ls_solution, ls_score) = ls_solve(
                    &whale_solutions[*whale_id],
                    step_scores[*whale_id],
                    data,
                    rest,
                    k,
                    l,
                    MAX_LS_EVALS,
                    &mut rng
                );
    
                current_evaluations += MAX_LS_EVALS;
    
                new_whales.push(ls_centroid);
                //whale_solutions[whale_id] = ls_solution.clone();
                if ls_score < best_whale_score{
                    best_whale_score = ls_score;
                    best_whale_solution = ls_solution.clone();
                }
            }

            let mut w = 0;
            for whale_id in elite_solutions.iter(){    
                whales[*whale_id] = new_whales[w].clone();
                w += 1;
            }
        }

        if best_whale_score < best_score_ever {
            best_score_ever = best_whale_score;
            best_solution_ever = best_whale_solution.clone();
            last_improvement = current_evaluations;
            println!("Best score ever {}", best_score_ever);

        }
    }

    assert!(valid_sol(&best_solution_ever, k));
    AlgResult {
        sol: Some(best_solution_ever),
        score: best_score_ever,
        generations: Some(current_evaluations as u32),
        history: None,
        time: None,
    }

}