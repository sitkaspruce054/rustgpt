use std::collections::HashMap;
use rayon::prelude::*;
use crate::tensor::Tensor;

pub struct AdamW {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub step: u32,
    pub m_buffers: HashMap<usize, Vec<f32>>,
    pub v_buffers: HashMap<usize, Vec<f32>>,
}


impl AdamW {

    pub fn new(lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            step: 0,
            m_buffers: HashMap::new(),
            v_buffers: HashMap::new(),
        }
    }


    pub fn step(&mut self, weights: &mut Vec<&mut Tensor>) {
        self.step += 1;
        let t = self.step as f32;
        let bc1 = 1.0 - self.beta1.powf(t);
        let bc2 = 1.0 - self.beta2.powf(t);
        let step_size = self.lr / bc1;

        for (idx, weight) in weights.iter_mut().enumerate() {

            if let Some(grad) = &weight.grad {
                let data = &mut weight.data;
                // first moment estimate
                let m = self.m_buffers.entry(idx).or_insert_with(|| vec![0.0; data.len()]);
                // the second raw moment estimate
                let v = self.v_buffers.entry(idx).or_insert_with(|| vec![0.0; data.len()]);
                
                data.par_iter_mut()
                    .zip(grad.par_iter())
                    .zip(m.par_iter_mut())
                    .zip(v.par_iter_mut())
                    .for_each(|(((w, &g), mi), vi)| {
                        *w -= self.lr * self.weight_decay * (*w);
                        *mi = self.beta1 * (*mi) + (1.0 - self.beta1) * g;
                        *vi = self.beta2 * (*vi) + (1.0 - self.beta2) * (g * g);

                        let v_hat = *vi / bc2;
                        *w -= step_size * (*mi) / (v_hat.sqrt() + self.eps);
                    });
            }
        }
    }
}