extern crate rand;
extern crate serde_json;

mod common;
mod input_parsing;
mod woa;

use std::env;
use std::time::Instant;

use std::fs::OpenOptions;

use common::{calc_c_value_inf, calc_lambda, calc_score, AlgResult, Output, Point};
use input_parsing::{read_points, read_restrictions};
use woa::woa_clustering;

fn main() {
    //let data: Vec<Array1<f32>> = read_points("/home/yabirgb/Documents/data/iris_set.dat");
    //let restrictions: Array2<f32> = read_restrictions("/home/yabir/Documents/uni/mh/data/iris_set_const_10.const");
    //println!("{:?} {}", restrictions, data.len());

    //println!("{}", Norm::norm_l2(&data[0]));

    let args: Vec<String> = env::args().collect();
    if args.len() < 6 {
        println!("Número incorrecto de argumentos. Ejemplo de ejecución: ");
        println!("./par path_datos:str path_restricciones:str algoritmo:str n_clusters:u32 seed:u32 history:u32");
        return;
    }

    let data: Vec<Point> = read_points(&args[1]);
    let restrictions: Vec<Vec<i8>> = read_restrictions(&args[2]);

    let k: u32;
    let seed: u64;
    let mut history: bool = false;

    match env::args().nth(3).and_then(|a| a.parse().ok()) {
        Some(x) => k = x,
        None => {
            println!("No se ha podido leer correctamente el número de clusters");
            println!("./par path_datos:str path_restricciones:str algoritmo:str n_clusters:u32 seed:u32 history:bool");
            return;
        }
    }

    match env::args().nth(4).and_then(|a| a.parse().ok()) {
        Some(x) => seed = x,
        None => {
            println!("No se ha podido leer correctamente la semilla");
            println!("./par path_datos:str path_restricciones:str algoritmo:str n_clusters:u32 seed:u32 history:bool");
            return;
        }
    }

    match env::args().nth(5).and_then(|a| a.parse().ok()) {
        Some(x) => match x {
            1 => history = true,
            _ => {}
        },
        None => {}
    }

    let l = calc_lambda(&data, &restrictions);
    let mut result: AlgResult;


    let mut nrestrictions: u32 = 10;
    let start = Instant::now();
    
    let time = start.elapsed().as_secs_f32();

    let mut dataset_name = "unknow".to_string();

    if args[1][..].to_string().contains("iris") {
        dataset_name = "iris".to_string();
    } else if args[1][..].to_string().contains("ecoli") {
        dataset_name = "ecoli".to_string();
    } else if args[1][..].to_string().contains("rand") {
        dataset_name = "rand".to_string();
    } else if args[1][..].to_string().contains("thyroid") {
        dataset_name = "newthyroid".to_string();
    }

    if args[2][..].to_string().contains("20") {
        nrestrictions = 20;
    }

    result = woa_clustering(&data, &restrictions, k, l, seed, 5, 10);

    let mut print = false;

    match result.sol {
        Some(x) => {
            let score = calc_score(&x, &data, &restrictions, k, l);
            let coefs = calc_c_value_inf(&x, &data, &restrictions, k);
            //println!("Score: {}", score);
            //println!("{:?}", x)

            let output = Output {
                sol: x.clone(),
                score,
                generations: result.generations,
                history: result.history,
                time: time,
                c: coefs.0,
                inf: coefs.1,
            };

            result.score = score;
            result.time = Some(time);
            result.sol = Some(x.clone());
            print = true;

            let path = format!(
                "outputs/sol_{}_{}_{}_{}.json",
                "woa", dataset_name, nrestrictions, seed
            );
            //println!("{}", serde_json::to_string_pretty(&result).unwrap());
            let file = OpenOptions::new()
                .create(true)
                .write(true)
                .append(false)
                .open(path)
                .unwrap();

            serde_json::to_writer_pretty(file, &output).expect("Fail");
        }
        None => println!("No solution found for {} with seed {}", dataset_name, seed),
    }

    if print {}
}
