use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use super::{valid_sol};
use crate::common::{calc_score, Point};

// Description of a change in the solution. First coordinate is the
// position and second the new cluster
#[derive(Debug)]
struct Change(usize, usize);

pub fn ls_solve(
    solu: &Vec<usize>,
    score_provided: f32,
    data: &Vec<Point>,
    rest: &Vec<Vec<i8>>,
    k: u32,
    l: f32,
    max_iters: usize,
    mut rng: &mut StdRng,
) -> (Vec<Vec<f32>>, Vec<usize>, f32){

    // given a whale solution and its score improve the solution
    // and return a new improved set of coordinates

    let mut solution = solu.clone();
    let mut score = score_provided;

    // create a list of all possible changes to make
    let mut changes: Vec<Change> = Vec::with_capacity(solution.len() * (k - 1) as usize);

    for i in 0..solution.len() {
        for c in 0..k {
            if c as usize != solution[i] {
                changes.push(Change(i as usize, c as usize))
            }
        }
    }

    // create an order to read the changes
    let mut order: Vec<usize> = (0..changes.len()).collect();
    order.shuffle(&mut rng);

    // count the number of evaluations made
    let mut evaluations = 0;
    let mut finished = false;

    while evaluations < max_iters && !finished {
        finished = true;
        for (_i, turn) in order.iter().enumerate() {
            let change = &changes[*turn];
            // Store the old cluster just in case
            let mut sol = solution.clone();
            // make the change
            sol[change.0] = change.1;

            // check the condition that every cluster has an element
            if valid_sol(&sol, k) {
                let new_score = calc_score(&sol, &data, &rest, k, l);
                evaluations += 1;

                // if it is a good change
                if new_score < score {
                    // update the score
                    //println!("{}", score);
                    score = new_score;

                    // update the solution
                    solution = sol.clone();

                    // update the changes
                    changes = Vec::with_capacity(sol.len() * (k - 1) as usize);
                    for i in 0..sol.len() {
                        for c in 0..k {
                            if c as usize != sol[i] {
                                changes.push(Change(i as usize, c as usize))
                            }
                        }
                    }


                    // New order on the changes
                    order = (0..changes.len()).collect();
                    order.shuffle(&mut rng);
                    finished = false;
                    break;
                }
            }
        }

    }

    // now given the improved version return the whale centroids
    let mut centroids:Vec<Vec<f32>> = vec![vec![0.0; data[0].dim() as usize]; k as usize];
    let mut n_points_in_cluster = vec![0; k as usize];

    for (point_id, cluster) in solution.iter().enumerate(){
        for (j, coord_value) in data[point_id].c.iter().enumerate(){
            centroids[*cluster][j] += coord_value;
        }
        n_points_in_cluster[*cluster] += 1;
    }

    for (k, centroid) in centroids.iter_mut().enumerate(){
        for i in 0..centroid.len(){
            centroid[i] *= 1.0/(n_points_in_cluster[k] as f32);
        }
    }

    (centroids, solution,score)

}