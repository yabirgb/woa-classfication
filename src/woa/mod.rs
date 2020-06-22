mod algorithm;
mod soft_ls;
pub use algorithm::{woa_clustering, woa_clustering_ls, woa_clustering_best_pool};
pub use soft_ls::{ls_solve};

pub fn valid_sol(sol: &Vec<usize>, k: u32) -> bool {
    let mut clusters: Vec<usize> = vec![0; k as usize];
    let mut valid = true;

    for p in sol.iter() {
        clusters[*p as usize] += 1;
        for i in clusters.iter() {
            if *i == 0 {
                valid = false;
                break;
            }
        }

        if valid {
            return true;
        } else {
            valid = true;
        }
    }

    return false;
}