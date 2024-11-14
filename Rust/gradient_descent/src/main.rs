use std::error::Error;
use csv::{Reader};
use nalgebra::{DMatrix, DVector,  DVectorView};
use rand::random;
use std::io;

#[allow(non_camel_case_types)]


fn get_data(path: &str) -> Result<(DMatrix<f64>, DMatrix<f64>), Box<dyn Error>> {
    //get data from csv and return X = Vec<x(i)> :: Normalized
    let mut rdr = Reader::from_path(path)?;

    let headers = rdr.headers()?;
    let column_count = headers.len();
    let mut row_count = 0; // set in loop to avoid emptying rdr, with the .records().count() method
    
    let mut x_data: Vec<f64> = Vec::new();
    let mut y_data: Vec<f64> = Vec::new();


    for result in rdr.records(){
        let record = result?;

        if let Some(first) = record.get(0){
            y_data.push(first.parse::<f64>()?);
        }

        for s in record.iter().skip(1){
            x_data.push(s.parse::<f64>()?);
        }

        row_count += 1
    }


    let x_train = DMatrix::from_vec(column_count - 1, row_count, x_data).transpose();
    let y_train = DMatrix::from_vec(row_count, 1, y_data);

    Ok((x_train, y_train))
}





fn compute_model_prediction(x: &DMatrix<f64>, w: &DVector<f64> , b: &f64) -> DVector<f64>{

    (x * w).add_scalar(*b)
}


fn update_params(x: &DMatrix<f64>,mut w: DVector<f64>,mut b: f64, diff: &DVector<f64>,  learning_rate: &f64, m: &usize) -> (DVector<f64>, f64, DVector<f64>) {
    //take in weights and biasis calculate the gradient and update w and b, w -= alpha * gradient_matrix, return w, b, gradient_matrix
    let n = w.len();
    let mut w_change = DVector::<f64>::zeros(n);

    for i in 0..n{
        w_change[i] = (x.transpose() * diff).sum() / *m as f64;
    }

    w -= *learning_rate* &w_change;
    b -= *learning_rate * (diff.sum()/ *m as f64);

    (w, b, w_change)

}


fn gradient_descent(x: &DMatrix<f64>, y: &DMatrix<f64>, w: &DVector<f64>, b: &f64, alpha: &f64, iterations: i64) -> (DVector<f64>, f64, Vec<f64>) {
    //update parametrs w and b until convergane min J(w,b)
    let mut w_in = w.clone();
    let mut b_in = b.clone();

    let tolerance: f64 = 1e-10;
    let mut cost: f64 = 0.0;
    let mut cost_vec: Vec<f64> = Vec::new();

    let m = y.len();
    let n = w.len();

    let mut w_change = DVector::<f64>::zeros(n);
    for i in 0..iterations{
        
        let pred = compute_model_prediction(x,&w_in, &b_in);
        let diff = &pred - y;

        let diff_squared = diff.map(|x| x.powi(2));
        cost = (diff_squared).sum() / (2.0*m as f64);

        if i % 10 == 0{
            println!("{cost}");
            cost_vec.push(cost);
        }

        (w_in, b_in, w_change) = update_params(&x, w_in, b_in, &diff, &alpha, &m);

        let mut is_finished = true;

        for w in &w_change{
            if w.abs() > tolerance{
                is_finished = false;
                break
            }
        }

        if is_finished{
            println!("Finished at {} iterations", i);
            return (w_in, b_in, cost_vec);
        }
        

        
    }
    (w_in, b_in, cost_vec)

}




trait NormalizeZ{

    fn normalize_z(&mut self) -> (DVector<f64>, DVector<f64>);
} 

impl NormalizeZ for DMatrix<f64>{

    fn normalize_z(&mut self) -> (DVector<f64>, DVector<f64>){


        let mut mean_vec: Vec<f64> = Vec::new();
        let mut std_vec: Vec<f64>  = Vec::new();

        for col in self.column_iter(){
            mean_vec.push(col.mean());
            std_vec.push(get_std(col));
        }

        let mean_vec = DVector::from_vec(mean_vec);
        let std_vec = DVector::from_vec(std_vec);

        for (j, (mean, std)) in mean_vec.iter().zip(std_vec.iter()).enumerate() {
            if *std != 0.0 { // Avoid division by zero
                let mut col = self.column_mut(j);
                col.iter_mut().for_each(|val| *val = (*val - mean) / std);
            }
        }

        (mean_vec, std_vec)

    }
}




fn get_std(mx:  DVectorView<f64>) -> f64{

    let mean = mx.iter().sum::<f64>() / mx.len() as f64;

    let variance = mx
        .iter()
        .map(|&value| (value - mean).powi(2))
        .sum::<f64>()
        / mx.len() as f64;

        variance.sqrt()
}






fn main() -> Result<(), Box<dyn Error>>{

    let mut x_train = DMatrix::zeros(0,0);
    let mut y_train = DMatrix::zeros(0,0);

    match get_data("../../datasets/environments.csv"){
        Ok((x,y)) => {

                (x_train, y_train) = (x,y); 
            }

        Err(e) => {println!("Error occurred: {}", e)},
    }
    let (mean_vec, std_vec)= x_train.normalize_z();

    let b_init =  y_train.mean() + 20.0;
    let w_init = DVector::from(vec![random::<f64>(), random::<f64>(), random::<f64>()]);


    let alpha = 0.007;
    let iters = 10000;

    let (w, b, cost_vec) = gradient_descent(&x_train, &y_train, &w_init, &b_init, &alpha, iters);

    println!("{:?} {}\n\n" ,w, b);

    predict(mean_vec, std_vec, &w, &b);

    Ok(())
}


fn predict(mean_vec: DVector<f64>, std_vec: DVector<f64>, w: &DVector<f64> , b: &f64) {

    let mut x = DMatrix::<f64>::zeros(3, 1);
    let vec = ["the age ", "the weight", "the HorsepPower"];

    for i in 0..3{
    
        println!("Please enter {} car: ", vec[i]);

        let mut input = String::new();
        io::stdin().read_line(&mut input).expect("Failed to read line");
        let number: f64 = input.trim().parse().expect("Please enter a valid number");

        x[i] = number;
    }

    for (idx, value) in x.iter_mut().enumerate(){

        *value = (*value - mean_vec[idx]) / std_vec[idx];
        println!("{}", value);
    }

    println!("{:?} {:?} {:?}", x, mean_vec, std_vec);

    let pred = compute_model_prediction(&x.transpose(), w, b);

    println!("Excepected Km per Gallon {}", pred);


}
