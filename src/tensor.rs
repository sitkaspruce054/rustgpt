use std::f32::consts::PI;
use rand::{rng, Rng};
use rayon::prelude::*;
use safetensors::SafeTensors;
/**
Represents a Tensor
*/
#[derive(Clone)]
pub struct Tensor {
    pub data: Vec<f32>, // The data
    pub shape: Vec<usize>, // The shape of the tensor
    pub strides: Vec<usize>, // The strides
    pub grad: Option<Vec<f32>> // Used for storing gradients
}

///Internal routine to compute the strides of a vector,
/// given its shape
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i+1] * shape[i+1];
    }
    strides
}

impl Tensor {
    ///Creates a new Tensor object with the given shape and data
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Self {
        let expected_len: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected_len,
            "Mismatched data/shape! Shape {:?} needs {} elements, but got {}",
            shape, expected_len, data.len()
        );

        // Compute strides for the new shape
        let strides = compute_strides(&shape);

        Tensor {
            data,
            shape,
            strides,
            grad: None,
        }
    }

    /// Initialized with a normal distribution (used for weights)
    pub fn random_init(shape: Vec<usize>, std: f32) -> Self {
        let size: usize = shape.iter().product();
        let mut data = vec![0.0; size];
        let mut r_gen = rng(); // Updated for rand 0.9

        // Generate data in pairs using Box-Muller
        for chunk in data.chunks_mut(2) {
            let u1: f32 = r_gen.random(); // .gen() is now .random()
            let u2: f32 = r_gen.random();

            let r = (-2.0 * u1.ln()).sqrt() * std;
            let theta = 2.0 * std::f32::consts::PI * u2;

            chunk[0] = r * theta.cos();
            if chunk.len() > 1 {
                chunk[1] = r * theta.sin();
            }
        }

        Tensor::new(shape, data)
    }



    pub fn layernorm_for_train(
        x: &Tensor,
        gamma: &Tensor,
        beta: &Tensor,
        eps: f32
    ) -> (Tensor, Tensor, Vec<f32>) {
        let shape_len = x.shape.len();

        // We want the last dimension (Hidden size)
        let cols = x.shape[shape_len - 1];

        // The number of rows is everything else multiplied together (Batch * Seq)
        let rows = x.data.len() / cols;

        // Explicitly check the weights
        assert_eq!(gamma.data.len(), cols, "Gamma must be size {}", cols);
        assert_eq!(beta.data.len(), cols, "Beta must be size {}", cols);

        let mut out_data = vec![0.0; x.data.len()];
        let mut x_hat_data = vec![0.0; x.data.len()];
        let mut inv_stds = vec![0.0; rows];

        // We chunk the input and output into rows
        out_data.par_chunks_mut(cols)
            .zip(x_hat_data.par_chunks_mut(cols))
            .zip(inv_stds.par_iter_mut())
            .zip(x.data.par_chunks(cols))
            .for_each(|(((out_row, x_hat_row), inv_std), x_row)| {
                // 1. Calculate Statistics
                let mean = x_row.iter().sum::<f32>() / cols as f32;
                let var = x_row.iter()
                    .map(|&val| (val - mean).powi(2))
                    .sum::<f32>() / cols as f32;

                let i_std = 1.0 / (var + eps).sqrt();
                *inv_std = i_std;

                // 2. Normalize and Scale
                for (i, (((o, xh), &xi), (&g, &b))) in out_row.iter_mut()
                    .zip(x_hat_row.iter_mut())
                    .zip(x_row.iter())
                    .zip(gamma.data.iter().zip(beta.data.iter()))
                    .enumerate()
                {
                    let normalized = (xi - mean) * i_std;
                    *xh = normalized;
                    *o = normalized * g + b;
                }
            });

        (
            Tensor::new(x.shape.clone(), out_data),
            Tensor::new(x.shape.clone(), x_hat_data),
            inv_stds
        )
    }

    pub fn sum_rows(&self) -> Tensor {
        let cols = self.shape[self.shape.len() - 1];
        let mut sums = vec![0.0; cols];

        // Sum every "feature" across all "rows" (Batch * Seq)
        for chunk in self.data.chunks_exact(cols) {
            for (i, &v) in chunk.iter().enumerate() {
                sums[i] += v;
            }
        }
        Tensor::new(vec![cols], sums)
    }

    pub fn multiply_inplace(&mut self, other: &Tensor) {
        assert_eq!(self.shape, other.shape, "Shape mismatch in multiply_inplace");

        // Use Rayon for parallel element-wise multiplication
        self.data.par_iter_mut()
            .zip(other.data.par_iter())
            .for_each(|(a, &b)| {
                *a *= b;
            });
    }

    pub fn gelu_derivative_inplace(&mut self) {
        let sqrt_2_over_pi = (2.0 / std::f32::consts::PI).sqrt();
        self.data.par_iter_mut().for_each(|x| {
            let x3 = *x * *x * *x;
            let inner = sqrt_2_over_pi * (*x + 0.044715 * x3);
            let tanh_inner = inner.tanh();
            let sech2_inner = 1.0 - tanh_inner * tanh_inner;
            *x = 0.5 * (1.0 + tanh_inner) + (0.5 * *x * sech2_inner * sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * *x * *x));
        });
    }

    pub fn accumulate_grad(&mut self, grad: Tensor) {
        if self.grad.is_none() {
            self.grad = Some(vec![0.0; self.data.len()]);
        }
        let self_grad = self.grad.as_mut().unwrap();
        // Use Rayon for high-speed accumulation on M4
        self_grad.par_iter_mut().zip(grad.data.par_iter()).for_each(|(sg, g)| {
            *sg += g;
        });
    }

    pub fn layernorm_backward(
        grad_out: &Tensor,    // dL/dy: [Batch, Seq, Hidden]
        cache_x_hat: &Tensor, // The normalized x from forward: [Batch, Seq, Hidden]
        gamma: &Tensor,       // The weights: [Hidden]
        inv_std: &[f32],      // Cached 1/sqrt(var + eps): [Batch * Seq]
    ) -> (Tensor, Tensor, Tensor) {
        // 1. Correctly extract dimensions for 3D/2D
        let last_dim_idx = grad_out.shape.len() - 1;
        let cols = grad_out.shape[last_dim_idx]; // Hidden (768)
        let rows = grad_out.data.len() / cols;    // Batch * Seq (e.g., 1024)

        let mut grad_x = vec![0.0; grad_out.data.len()];
        let mut grad_gamma = vec![0.0; cols];
        let mut grad_beta = vec![0.0; cols];

        // 2. Weight Gradients (Gamma and Beta)
        // These must be summed across all tokens in the batch/sequence
        for i in 0..rows {
            let row_offset = i * cols;
            for j in 0..cols {
                let idx = row_offset + j;
                grad_beta[j] += grad_out.data[idx];
                grad_gamma[j] += grad_out.data[idx] * cache_x_hat.data[idx];
            }
        }

        // 3. Input Gradient (dL/dx)
        // Parallelize across tokens (rows)
        grad_x.par_chunks_mut(cols).enumerate().for_each(|(i, row_grad_x)| {
            let row_offset = i * cols;
            let row_grad_out = &grad_out.data[row_offset..row_offset + cols];
            let row_x_hat = &cache_x_hat.data[row_offset..row_offset + cols];
            let i_std = inv_std[i];

            let mut sum_dout = 0.0;
            let mut sum_dout_xhat = 0.0;

            for j in 0..cols {
                let d_normalized = row_grad_out[j] * gamma.data[j];
                sum_dout += d_normalized;
                sum_dout_xhat += d_normalized * row_x_hat[j];
            }

            let mean_dout = sum_dout / cols as f32;
            let mean_dout_xhat = sum_dout_xhat / cols as f32;

            for j in 0..cols {
                row_grad_x[j] = i_std * (
                    (row_grad_out[j] * gamma.data[j]) - mean_dout - (row_x_hat[j] * mean_dout_xhat)
                );
            }
        });

        (
            Tensor::new(grad_out.shape.clone(), grad_x),
            Tensor::new(vec![cols], grad_gamma),
            Tensor::new(vec![cols], grad_beta)
        )
    }

    pub fn from_safetensors(name: &str, st: &SafeTensors) -> Self {
        let view = st.tensor(name).expect(&format!("Weight {} not found", name));
        let shape = view.shape().to_vec();

        // Convert raw bytes to f32
        let data = view.data()
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();

        Tensor::new(shape, data)
    }



    /// Initialized with a constant value (used for biases and LayerNorm)
    pub fn constant(shape: Vec<usize>, val: f32) -> Self {
        let size: usize = shape.iter().product();
        Tensor::new(shape, vec![val; size])
    }



    pub fn matmul_backward(a: &Tensor, b: &Tensor, grad_out: &Tensor) -> (Tensor, Tensor) {
        // 1. grad_a = grad_out @ B^T
        // This is the gradient flowing back to the input 'x'
        // grad_out: [1, 1024, 50257], B: [768, 50257] -> Result: [1, 1024, 768]
        let grad_a = Tensor::matmul(grad_out, &b.transpose());

        // 2. grad_b = A^T @ grad_out
        // This is the gradient for the weights (e.g., LM Head or MLP)
        // We must flatten the leading dimensions so we are doing:
        // [768, 1024] @ [1024, 50257] -> [768, 50257]

        let a_cols = a.shape[a.shape.len() - 1];
        let a_rows = a.data.len() / a_cols;
        let a_flat = Tensor::new(vec![a_rows, a_cols], a.data.clone());

        let go_cols = grad_out.shape[grad_out.shape.len() - 1];
        let go_rows = grad_out.data.len() / go_cols;
        let go_flat = Tensor::new(vec![go_rows, go_cols], grad_out.data.clone());


        let grad_b = Tensor::matmul(&a_flat.transpose(), &go_flat);

        (grad_a, grad_b)
    }


    pub fn get_index(&self,coords: &[usize]) -> usize {
        coords.iter().zip(&self.strides).map(|(c,s)| c * s).sum()
    }

    pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
        let a_rank = a.shape.len();
        let b_rank = b.shape.len();


        // For A [1, 1024, 768], K is 768 (the last dimension)
        let k_a = a.shape[a_rank - 1];
        // For B [768, 2304], K is 768 (the first dimension)
        let k_b = b.shape[0];

        assert_eq!(k_a, k_b, "MatMul Inner Dimension Mismatch: A is {:?}, B is {:?}", a.shape, b.shape);


        // M is the product of all dimensions of A except the last one (e.g., 1 * 1024)
        let m = a.data.len() / k_a;
        let n = b.shape[1];


        // Result should keep A's prefix but swap the last dimension for B's width
        // Input [1, 1024, 768] @ [768, 2304] -> [1, 1024, 2304]
        let mut out_shape = a.shape.clone();
        let last_idx = out_shape.len() - 1;
        out_shape[last_idx] = n;

        // 4. Perform the Math
        let mut res_data = vec![0.0; m * n];

        // Parallelize over M (total rows)
        res_data.par_chunks_mut(n).enumerate().for_each(|(i, cur_row)| {
            let a_row_start = i * k_a;
            for j in 0..k_a {
                let val_a = a.data[a_row_start + j];
                let b_row_start = j * n;
                let b_row = &b.data[b_row_start..b_row_start + n];

                // Inner loop: multiply-accumulate across the N dimension
                for l in 0..n {
                    cur_row[l] += val_a * b_row[l];
                }
            }
        });

        Tensor::new(out_shape, res_data)
    }
    
    pub fn layernorm(x: &Tensor, gamma: &Tensor, beta: &Tensor, eps: f32) -> Tensor {
        let mut out_data = x.data.clone();
        let n = x.shape[x.shape.len() - 1];

        out_data.par_chunks_mut(n).for_each(|row| {
            let mean = row.iter().sum::<f32>() / n as f32;
            let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n as f32;
            let inv_std = 1.0 / (var + eps).sqrt();

            for i in 0..n {
                row[i] = (row[i] - mean) * inv_std * gamma.data[i] + beta.data[i];
            }
        });

        Tensor::new(x.shape.clone(),out_data)
    }

    pub fn gelu_inplace(&mut self) {
        let sqrt_2_over_pi = (2.0 / PI).sqrt();

        const COEFF: f32 = 0.044715;


        self.data.par_iter_mut().for_each(|x| {
            let x3 = *x * *x * *x;
            let inner = sqrt_2_over_pi * (*x + COEFF * x3);
            *x = 0.5 * *x * (1.0 + inner.tanh());
        });

    }

    pub fn gelu(&self) -> Tensor {
        let mut new_tensor: Tensor = self.clone();
        new_tensor.gelu_inplace();
        new_tensor
    }

    pub fn matmul_transposed(a: &Tensor, b_t: &Tensor) -> Tensor {
        // a is [Batch, Seq, Emb] -> e.g., [1, 8, 128]
        // b_t is [Vocab, Emb]    -> e.g., [5000, 128]
        let batch_size = a.shape[0];
        let seq_len = a.shape[1];
        let k_dim = a.shape[2];      // Must be n_embd (128)
        let out_dim = b_t.shape[0];  // Results in Vocab (5000)

        assert_eq!(k_dim, b_t.shape[1], "Inner dimensions (n_embd) must match");

        let mut out_data = vec![0.0; batch_size * seq_len * out_dim];

        // Use Rayon to parallelize across the Batch * Seq dimensions
        // Each thread calculates the "vocab scores" for one specific token
        out_data.par_chunks_mut(out_dim).enumerate().for_each(|(idx, row_out)| {
            let b = idx / seq_len;
            let s = idx % seq_len;

            // Offset into the hidden state for this specific token
            let a_start = (b * seq_len + s) * k_dim;
            let a_row = &a.data[a_start..a_start + k_dim];

            for i in 0..out_dim {
                let mut sum = 0.0;
                // Offset into the weight matrix for this specific vocab word
                let b_start = i * k_dim;
                let b_row = &b_t.data[b_start..b_start + k_dim];

                // Manual unrolling or SIMD would happen here for more speed
                for k in 0..k_dim {
                    sum += a_row[k] * b_row[k];
                }
                row_out[i] = sum;
            }
        });

        Tensor::new(vec![batch_size, seq_len, out_dim], out_data)
    }
    pub fn matmul_transposed_backward(
        x: &Tensor,       // [Batch*Seq, Emb]
        w: &Tensor,       // [Vocab, Emb] (The tied weights)
        d_logits: &Tensor // [Batch*Seq, Vocab]
    ) -> (Tensor, Tensor) {
        // 1. d_x = d_logits @ w  -> [Batch*Seq, Vocab] @ [Vocab, Emb] = [Batch*Seq, Emb]
        let d_x = Tensor::matmul(d_logits, w);

        // 2. d_w = d_logits^T @ x -> [Vocab, Batch*Seq] @ [Batch*Seq, Emb] = [Vocab, Emb]
        // Since we want d_w to match the shape of w [Vocab, Emb]:
        let d_w = Tensor::matmul(&d_logits.transpose(), x);

        (d_x, d_w)
    }


    pub fn gather(&self, indices: &[u32]) -> Tensor {
        let n_embd = self.shape[1];
        let num_indices = indices.len();
        let mut data = Vec::with_capacity(num_indices * n_embd);

        for &idx in indices {
            let start = idx as usize * n_embd;
            let end = start + n_embd;
            data.extend_from_slice(&self.data[start..end]);
        }

        Tensor::new(vec![num_indices, n_embd], data)
    }

    pub fn transpose(&self) -> Tensor {
        let rank = self.shape.len();
        assert!(rank >= 2, "Transpose requires at least 2 dimensions");

        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();

        // Swap the last two dimensions
        new_shape.swap(rank - 1, rank - 2);
        new_strides.swap(rank - 1, rank - 2);


        let mut new_data = vec![0.0; self.data.len()];

        // Total number of elements in each [M x N] matrix after the Batch dim
        let m = new_shape[rank - 2];
        let n = new_shape[rank - 1];
        let batch_stride = if rank > 2 { self.strides[rank - 3] } else { self.data.len() };
        let num_batches = self.data.len() / batch_stride;

        for b in 0..num_batches {
            let b_offset = b * batch_stride;
            for i in 0..m {
                for j in 0..n {
                    // Read from original (swap i and j logic)
                    // Original was [Batch, N, M]
                    let old_idx = b_offset + (j * m) + i;
                    let new_idx = b_offset + (i * n) + j;
                    new_data[new_idx] = self.data[old_idx];
                }
            }
        }
        Tensor::new(new_shape,new_data)
    }
    ///Computes the softmax of the tensor in place
    pub fn softmax_inplace(&mut self) {
        let last_dim = *self.shape.last().unwrap();
        self.data.par_chunks_mut(last_dim).for_each(|row| {
            // 1. Find max, but ignore the mask (-1e9) if possible
            let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

            // 2. If max_val is still extremely low, this row is likely fully masked
            if max_val < -1e8 {
                row.fill(0.0);
                row[0] = 1.0;
                return;
            }

            let mut sum = 0.0;
            for v in row.iter_mut() {
                // Stability: subtract max_val
                *v = (*v - max_val).exp();
                sum += *v;
            }

            let inv_sum = 1.0 / sum;
            for val in row.iter_mut() {
                *val *= inv_sum;
            }
        });
    }

    ///Computes the softmax of the tensor,
    /// allocating a new tensor containing the
    /// softmax of the tensor
    pub fn softmax(self) -> Tensor {
        let mut tensor = self.clone();
        tensor.softmax_inplace();
        tensor
    }
    ///Adds two tensors together, returning a new Tensor representing
    /// the sum of the two tensors
    pub fn add(&self, other: &Tensor) -> Tensor {
        // 1. Total element check (e.g., 786,432 == 786,432)
        assert_eq!(
            self.data.len(),
            other.data.len(),
            "Total element mismatch for add: {:?} vs {:?}",
            self.shape,
            other.shape
        );

        // 2. Prepare output data
        let mut res_data = vec![0.0; self.data.len()];


        res_data.par_iter_mut()
            .zip(self.data.par_iter())
            .zip(other.data.par_iter())
            .for_each(|((res, &a), &b)| {
                *res = a + b;
            });

        // 4. Return using the shape of 'self' to maintain the model's preferred rank
        Tensor::new(self.shape.clone(), res_data)
    }
    ///zeros out the gradients
    pub fn zero_grad(&mut self) {
        if let Some(ref mut g) = self.grad {
            g.fill(0.0);
        }
    }

    ///creates a TensorView for compatibility with the safetensors library
    pub fn to_view(&self) -> safetensors::tensor::TensorView {
        // We cast the f32 slice to a u8 slice.
        // This is safe because f32 is 4 bytes and we aren't changing the data.
        let data_u8: &[u8] = unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr() as *const u8,
                self.data.len() * std::mem::size_of::<f32>(),
            )
        };

        // Map our Vec<usize> shape to Vec<usize> (safetensors format)
        safetensors::tensor::TensorView::new(
            safetensors::Dtype::F32,
            self.shape.clone(),
            data_u8,
        ).unwrap()
    }


    ///adds the bias tensor, returning a new tensor
    pub fn add_bias(&self, bias: &Tensor) -> Tensor {
        // 1. Get the width of the features (the last dimension)
        let last_dim_idx = self.shape.len() - 1;
        let cols = self.shape[last_dim_idx];

        // Sanity check
        assert_eq!(
            bias.data.len(),
            cols,
            "Bias size {} must match tensor feature width {}",
            bias.data.len(),
            cols
        );

        let mut data = self.data.clone();

        // 2. Add bias to every row in parallel
        data.par_chunks_mut(cols).for_each(|row| {
            for (row_val, bias_val) in row.iter_mut().zip(bias.data.iter()) {
                *row_val += *bias_val;
            }
        });

        Tensor::new(self.shape.clone(), data)
    }
}