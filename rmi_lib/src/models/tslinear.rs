// < begin copyright > 
// Copyright Ryan Marcus 2020
// 
// See root directory of this project for license terms.
// 
// < end copyright > 
 
 

use crate::models::*;
use crate::models::linear::slr;
use rand::Rng;
use std::cmp;

#[inline]
fn apply_error_fn(prediction : f64, y : f64, e : i8) -> f64{
    let err = (prediction - y).abs();
    if e == 0{
        return (1.0 + err).log2();
    }
    if e == 1{
        return (1.0 + err).log2().ceil();
    }
    if e == 2{
        return err;
    }
    if e == 3{
        let u_err : u64 = (1.0 + err) as u64;
        return (63 as u32 - u_err.leading_zeros()) as f64;
    }
    return 0.0;
}

fn calculate_error(loc_data: &Vec<(f64, f64)>, a : f64, b : f64, i : usize, e : i8) -> f64{
    let prediction : f64 = a.mul_add(loc_data[i].0, b);
    return apply_error_fn(prediction, loc_data[i].1, e);
}

fn _calculate_error_all(loc_data: &Vec<(f64, f64)>, a : f64, b : f64, e : i8) -> f64{
    let mut err : f64 = 0.0;
    for i in 0..loc_data.len(){
        let prediction : f64 = a.mul_add(loc_data[i].0, b);
        err += apply_error_fn(prediction, loc_data[i].1, e);
    }
    return err;
}

fn fit_line(loc_data: &Vec<(f64, f64)>, fr : usize, to : usize) -> (f64, f64) {
    let a = (loc_data[to].1 - loc_data[fr].1) / (loc_data[to].0 - loc_data[fr].0);
    let b = loc_data[fr].1 - loc_data[fr].0 * a;
    return (a,b);
}

fn log_regression_fast(loc_data: &Vec<(f64, f64)>, h : usize, e : i8) -> (f64, f64, f64){
    let mut rng = rand::thread_rng();
    if h == 0 {
        let fr = rng.gen_range(0,loc_data.len());
        let mut to = rng.gen_range(0,loc_data.len());
        while to == fr {
            to = rng.gen_range(0,loc_data.len());
        }
        let (a,b) = fit_line(&loc_data, fr, to);

        return (a, b, 0.0);
    }
    
    let (a_l, b_l, mut err_l) = log_regression_fast(&loc_data, h-1, e);
    let (a_r, b_r, mut err_r) = log_regression_fast(&loc_data, h-1, e);
    
    let test_range = cmp::min(1 << h, loc_data.len());
    for _ in 0..test_range {
        let element = rng.gen_range(0,loc_data.len());
        err_l += calculate_error(&loc_data, a_l, b_l, element, e);
        err_r += calculate_error(&loc_data, a_r, b_r, element, e);
    }

    if err_l < err_r {
        return (a_l,b_l,err_l);
    }else{
        return (a_r,b_r,err_r);
    }
}

fn _log_regression_rec(loc_data: &Vec<(f64, f64)>, h : usize, e : i8) -> (usize, usize){
    let mut rng = rand::thread_rng();
    if h == 0 {
        let fr = rng.gen_range(0,loc_data.len());
        let mut to = rng.gen_range(0,loc_data.len());
        while to == fr {
            to = rng.gen_range(0,loc_data.len());
        }
        return (fr, to);
    }
    
    let (fr_l, to_l) = _log_regression_rec(&loc_data, h-1, e);
    let (fr_r, to_r) = _log_regression_rec(&loc_data, h-1, e);
    
    let (a_l,b_l) = fit_line(&loc_data, fr_l, to_l);
    let (a_r,b_r) = fit_line(&loc_data, fr_r, to_r);
    let mut err_l : f64 = 0.0;
    //err_l += calculate_error(&loc_data, a_l, b_l, fr_r, e);
    //err_l += calculate_error(&loc_data, a_l, b_l, to_r, e);
    
    let mut err_r : f64 = 0.0;
    //err_r += calculate_error(&loc_data, a_r, b_r, fr_l, e);
    //err_r += calculate_error(&loc_data, a_r, b_r, to_l, e);
    
    let test_range = cmp::min(2 << h, loc_data.len());
    for _ in 0..test_range {
        let element = rng.gen_range(0,loc_data.len());
        err_l += calculate_error(&loc_data, a_l, b_l, element, e);
        err_r += calculate_error(&loc_data, a_r, b_r, element, e);
    }

    if err_l < err_r {
        return (fr_l, to_l);
    }else{
        return (fr_r, to_r);
    }
}

fn log_regression(loc_data: &Vec<(f64, f64)>, e : i8) -> (f64, f64) {
    if loc_data.len() == 0 {
        return (0.0,0.0);
    }
    let mut h = (1.0 + loc_data.len() as f64).log2().ceil() as usize;
    //h += 1;
    h = cmp::min(h, 20);
    let (a,b,_err) = log_regression_fast(loc_data, h, e);
    return (b,a);
}

pub struct TsLinearModel {
    params: (f64, f64),
}

impl TsLinearModel {
    pub fn new<T: TrainingKey>(data: &RMITrainingData<T>) -> TsLinearModel {
        let mut data_vector : Vec<(f64, f64)> = Vec::new();
        data_vector.reserve(data.len());


        for val in data.iter() {
            data_vector.push((val.0.as_float(), val.1 as f64));
        }

        let mut params = log_regression(&data_vector, 3);
        if !params.0.is_finite() || !params.1.is_finite(){
            params = slr(data.iter().map(|(inp, offset)| (inp.as_float(), offset as f64)));
            //println!(" {} {}", params.1, params.0);
        }
        return TsLinearModel { params };
    }
}

impl Model for TsLinearModel {
    fn predict_to_float(&self, inp: &ModelInput) -> f64 {
        let (intercept, slope) = self.params;
        return slope.mul_add(inp.as_float(), intercept);
    }

    fn input_type(&self) -> ModelDataType {
        return ModelDataType::Float;
    }
    fn output_type(&self) -> ModelDataType {
        return ModelDataType::Float;
    }

    fn params(&self) -> Vec<ModelParam> {
        return vec![self.params.0.into(), self.params.1.into()];
    }

    fn code(&self) -> String {
        return String::from(
            "
inline double tslinear(double alpha, double beta, double inp) {
    return std::fma(beta, inp, alpha);
}",
        );
    }

    fn function_name(&self) -> String {
        return String::from("tslinear");
    }

    fn set_to_constant_model(&mut self, constant: u64) -> bool {
        self.params = (constant as f64, 0.0);
        return true;
    }
}
