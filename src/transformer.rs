use crate::tensor::Tensor;
use rayon::prelude::*;
pub struct TransformerBlock {
    pub n_head: usize, // Added to make the block dynamic
    // Layer Norms
    pub ln_1_gamma: Tensor,
    pub ln_1_beta: Tensor,
    pub ln_2_gamma: Tensor,
    pub ln_2_beta: Tensor,

    // Attention Weights
    pub c_attn_weight: Tensor,
    pub c_attn_bias: Tensor,
    pub c_proj_weight: Tensor,
    pub c_proj_bias: Tensor,

    // FFN Weights
    pub mlp_fc_weight: Tensor,
    pub mlp_fc_bias: Tensor,
    pub mlp_proj_weight: Tensor,
    pub mlp_proj_bias: Tensor,
}

pub struct LayerCache {
    pub ln1_x_hat: Tensor,
    pub ln1_inv_std: Vec<f32>,
    pub ln1_out: Tensor,

    pub qkv_combined: Tensor,
    pub q: Tensor,              // [Batch, 12, Seq, 64]
    pub k: Tensor,              // [Batch, 12, Seq, 64]
    pub v: Tensor,              // [Batch, 12, Seq, 64]
    pub attn_probs: Tensor,     // [Batch, 12, Seq, Seq]
    pub merged_ctx: Tensor,

    pub ln2_x_hat: Tensor,
    pub ln2_inv_std: Vec<f32>,
    pub ln2_out: Tensor,
    pub mlp_fc_out: Tensor,
    pub mlp_h_act: Tensor,
}

pub struct ForwardCache {
    pub token_ids: Vec<u32>,
    pub initial_embeddings: Tensor,
    pub block_caches: Vec<LayerCache>,

    // --- Final Head Path ---
    pub pre_ln_f_x: Tensor,      // The residual stream before final LayerNorm
    pub ln_f_x_hat: Tensor,      // Normalized x (mean 0, std 1)
    pub ln_f_inv_std: Vec<f32>,  // 1/sqrt(var + eps)
    pub final_norm_out: Tensor,  // Input to the LM Head matmul
}

impl TransformerBlock {


    fn split_heads(&self, x: &Tensor, batch_size: usize, seq_len: usize) -> Tensor {
        let n_head = self.n_head; // Use dynamic field
        let head_dim = x.shape[x.shape.len() - 1] / n_head;

        let new_shape = vec![batch_size, n_head, seq_len, head_dim];
        Tensor::new(new_shape, x.data.clone())
    }

    fn merge_heads(&self, x: &Tensor) -> Tensor {
        let batch = x.shape[0];
        let n_head = x.shape[1];
        let seq_len = x.shape[2];
        let head_dim = x.shape[3];
        let n_embd = n_head * head_dim;

        Tensor::new(vec![batch, seq_len, n_embd], x.data.clone())
    }

    // Utility to slice a specific head from a 4D batch
    fn get_head_slice_4d<'a>(&self, tensor: &'a Tensor, b: usize, h: usize) -> &'a [f32] {
        let n_head = tensor.shape[1];
        let seq_len = tensor.shape[2];
        let head_dim = tensor.shape[3];

        let head_size = seq_len * head_dim;
        let batch_stride = n_head * head_size;

        let start = (b * batch_stride) + (h * head_size);
        let end = start + head_size;
        &tensor.data[start..end]
    }

    pub fn new_random(n_embd: usize, n_head: usize, res_scale: f32) -> Self {
        let std = 0.02; // Standard GPT-2 initialization
        let n_inner = n_embd * 4;

        Self {
            n_head, // Initialize the dynamic head count
            ln_1_gamma: Tensor::constant(vec![n_embd], 1.0),
            ln_1_beta: Tensor::constant(vec![n_embd], 0.0),
            c_attn_weight: Tensor::random_init(vec![n_embd, 3 * n_embd], std),
            c_attn_bias: Tensor::constant(vec![3 * n_embd], 0.0),
            c_proj_weight: Tensor::random_init(vec![n_embd, n_embd], std * res_scale),
            c_proj_bias: Tensor::constant(vec![n_embd], 0.0),
            ln_2_gamma: Tensor::constant(vec![n_embd], 1.0),
            ln_2_beta: Tensor::constant(vec![n_embd], 0.0),
            mlp_fc_weight: Tensor::random_init(vec![n_embd, n_inner], std),
            mlp_fc_bias: Tensor::constant(vec![n_inner], 0.0),
            mlp_proj_weight: Tensor::random_init(vec![n_inner, n_embd], std * res_scale),
            mlp_proj_bias: Tensor::constant(vec![n_embd], 0.0),
        }
    }


    pub fn zero_grad(&mut self) {
        // Layer Norms
        self.ln_1_gamma.zero_grad();
        self.ln_1_beta.zero_grad();
        self.ln_2_gamma.zero_grad();
        self.ln_2_beta.zero_grad();

        // Attention
        self.c_attn_weight.zero_grad();
        self.c_attn_bias.zero_grad();
        self.c_proj_weight.zero_grad();
        self.c_proj_bias.zero_grad();

        // MLP
        self.mlp_fc_weight.zero_grad();
        self.mlp_fc_bias.zero_grad();
        self.mlp_proj_weight.zero_grad();
        self.mlp_proj_bias.zero_grad();
    }

    fn unmerge_heads(&self, d_ctx: &Tensor) -> Tensor {
        let n_embd = d_ctx.shape[d_ctx.shape.len() - 1];
        let n_head = self.n_head; // Use dynamic field
        let head_dim = n_embd / n_head;
        let seq_len = d_ctx.data.len() / (d_ctx.shape[0] * n_embd); // Fix for batches

        let mut out_data = vec![0.0; d_ctx.data.len()];
        // Note: Simplified logic assumes batch=1 for unmerge; if training with
        // batches, ensure your loop iterates over Batch * Heads * Seq.
        // ... (Transposition logic) ...
        Tensor::new(vec![d_ctx.shape[0], n_head, seq_len, head_dim], out_data)
    }

    pub fn clip_gradients(&mut self, max_norm: f32) {
        let mut total_norm: f32 = 0.0;

        // 1. Calculate the global norm across ALL weights
        for weight in self.get_weights_mut() {
            if let Some(g) = &weight.grad {
                total_norm += g.iter().map(|x| x * x).sum::<f32>();
            }
        }
        total_norm = total_norm.sqrt();

        // 2. Scale down if it exceeds max_norm
        if total_norm > max_norm {
            let scale = max_norm / total_norm;
            for weight in self.get_weights_mut() {
                if let Some(g) = &mut weight.grad {
                    for val in g.iter_mut() { *val *= scale; }
                }
            }
        }
    }

    pub fn get_weights_mut(&mut self) -> Vec<&mut Tensor> {
        vec![
            &mut self.ln_1_gamma, &mut self.ln_1_beta,
            &mut self.ln_2_gamma, &mut self.ln_2_beta,
            &mut self.c_attn_weight, &mut self.c_attn_bias,
            &mut self.c_proj_weight, &mut self.c_proj_bias,
            &mut self.mlp_fc_weight, &mut self.mlp_fc_bias,
            &mut self.mlp_proj_weight, &mut self.mlp_proj_bias,
        ]
    }





    fn combine_qkv(&self, d_q: Vec<f32>, d_k: Vec<f32>, d_v: Vec<f32>, seq_len: usize, n_embd: usize, batch_size: usize) -> Tensor {
        let n_head = self.n_head; // Use dynamic field
        let head_dim = n_embd / n_head;
        let mut d_qkv_data = vec![0.0; batch_size * seq_len * n_embd * 3];

        for b in 0..batch_size {
            let batch_offset_qkv = b * seq_len * (n_embd * 3);
            let batch_offset_heads = b * n_head * seq_len * head_dim;

            for t in 0..seq_len {
                let row_start = batch_offset_qkv + t * (n_embd * 3);
                for h in 0..n_head {
                    let head_offset = batch_offset_heads + h * (seq_len * head_dim) + t * head_dim;
                    let q_dest = row_start + (h * head_dim);
                    let k_dest = row_start + n_embd + (h * head_dim);
                    let v_dest = row_start + (2 * n_embd) + (h * head_dim);

                    d_qkv_data[q_dest..q_dest + head_dim].copy_from_slice(&d_q[head_offset..head_offset + head_dim]);
                    d_qkv_data[k_dest..k_dest + head_dim].copy_from_slice(&d_k[head_offset..head_offset + head_dim]);
                    d_qkv_data[v_dest..v_dest + head_dim].copy_from_slice(&d_v[head_offset..head_offset + head_dim]);
                }
            }
        }
        Tensor::new(vec![batch_size, seq_len, n_embd * 3], d_qkv_data)
    }

    pub fn backward(&mut self, grad_out: Tensor, cache: &LayerCache) -> Tensor {
        // --- 1. MLP BACKWARD ---
        let d_mlp_out = grad_out.clone();
        let d_norm2 = self.mlp_backward(&d_mlp_out, cache);

        // 3. LayerNorm 2 Backward - Returns (d_x, d_gamma, d_beta)
        let (d_x_mid_from_mlp, d_gamma2, d_beta2) = Tensor::layernorm_backward(
            &d_norm2,
            &cache.ln2_out,
            &self.ln_2_gamma,
            &cache.ln2_inv_std
        );

        // Accumulate LayerNorm 2 weight gradients
        self.ln_2_gamma.accumulate_grad(d_gamma2);
        self.ln_2_beta.accumulate_grad(d_beta2);

        // Sum the gradients at the first residual junction (Skip connection)
        let d_x_mid = grad_out.add(&d_x_mid_from_mlp);

        // --- 2. ATTENTION BACKWARD ---
        let d_attn_out = d_x_mid.clone();
        let d_norm1 = self.mha_backward(&d_attn_out, cache);

        // 6. LayerNorm 1 Backward - Returns (d_x, d_gamma, d_beta)
        let (d_x_in_from_attn, d_gamma1, d_beta1) = Tensor::layernorm_backward(
            &d_norm1,
            &cache.ln1_out,
            &self.ln_1_gamma,
            &cache.ln1_inv_std
        );

        // Accumulate LayerNorm 1 weight gradients
        self.ln_1_gamma.accumulate_grad(d_gamma1);
        self.ln_1_beta.accumulate_grad(d_beta1);

        // Final gradient for this block's input
        d_x_mid.add(&d_x_in_from_attn)
    }


    pub fn mha_for_train(&self, x: &Tensor) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) {
        let batch_size = x.shape[0];
        let seq_len = x.shape[1];
        let n_embd = x.shape[2];
        let head_dim = n_embd / self.n_head;

        let qkv_combined = Tensor::matmul(x, &self.c_attn_weight).add_bias(&self.c_attn_bias);
        let (q_full, k_full, v_full) = self.split_qkv(&qkv_combined, n_embd);

        let q = self.split_heads(&q_full, batch_size, seq_len);
        let k = self.split_heads(&k_full, batch_size, seq_len);
        let v = self.split_heads(&v_full, batch_size, seq_len);

        let mut scores = self.scaled_dot_product(&q, &k, head_dim);
        self.apply_causal_mask(&mut scores);
        scores.softmax_inplace();
        let attn_probs = scores.clone();

        let ctx_heads = self.attend_values(&scores, &v);
        let merged_ctx = self.merge_heads(&ctx_heads);

        let out = Tensor::matmul(&merged_ctx, &self.c_proj_weight).add_bias(&self.c_proj_bias);
        (out, q, k, v, attn_probs, merged_ctx, qkv_combined)
    }


    pub fn forward_train(&self, x: &Tensor) -> (Tensor, LayerCache) {
        // --- 1. Attention Path ---
        let (norm_1, ln1_x_hat, ln1_inv_std) = Tensor::layernorm_for_train(x, &self.ln_1_gamma, &self.ln_1_beta, 1e-5);
        let (attn_out, q, k, v, attn_probs, merged_ctx, qkv_combined) = self.mha_for_train(&norm_1);

        let x_mid = x.add(&attn_out);

        // --- 2. MLP Path ---
        let (norm_2, ln2_x_hat, ln2_inv_std) = Tensor::layernorm_for_train(&x_mid, &self.ln_2_gamma, &self.ln_2_beta, 1e-5);

        // Feed Forward internals
        let mlp_fc_out = Tensor::matmul(&norm_2, &self.mlp_fc_weight).add_bias(&self.mlp_fc_bias);
        let mut mlp_h_act = mlp_fc_out.clone();
        mlp_h_act.gelu_inplace();

        let mlp_out = Tensor::matmul(&mlp_h_act, &self.mlp_proj_weight).add_bias(&self.mlp_proj_bias);
        let x_out = x_mid.add(&mlp_out);

        // --- 3. Construct Cache ---
        let cache = LayerCache {
            ln1_x_hat,
            ln1_inv_std,
            ln1_out: norm_1,
            qkv_combined,
            q, k, v,
            attn_probs,
            merged_ctx,
            ln2_x_hat,
            ln2_inv_std,
            ln2_out: norm_2,
            mlp_fc_out,
            mlp_h_act,
        };

        (x_out, cache)
    }



    fn mlp_backward(&mut self, d_mlp_out: &Tensor, cache: &LayerCache) -> Tensor {
        // 1. dL/d_proj_weight = h^T @ d_mlp_out
        // 2. dL/d_h = d_mlp_out @ proj_weight^T
        let (d_h, d_proj_w) = Tensor::matmul_backward(&cache.mlp_h_act, &self.mlp_proj_weight, d_mlp_out);

        // 3. dL/d_gelu = d_h * GELU_prime(cache.mlp_fc_out)
        let mut d_gelu = cache.mlp_fc_out.clone();
        d_gelu.gelu_derivative_inplace(); // Custom method to multiply by activation derivative
        d_gelu.multiply_inplace(&d_h);

        // 4. dL/d_fc_weight = x^T @ d_gelu
        let (d_norm2, d_fc_w) = Tensor::matmul_backward(&cache.ln2_out, &self.mlp_fc_weight, &d_gelu);

        // Accumulate gradients (to be used by Optimizer later)
        self.mlp_proj_weight.accumulate_grad(d_proj_w);
        self.mlp_fc_weight.accumulate_grad(d_fc_w);

        d_norm2
    }

    pub fn load_from_st(i: usize, st: &safetensors::SafeTensors) -> Self {
        // 1. Determine the prefix used in the safetensors file
        let test_key = format!("h.{}.ln_1.weight", i);
        let prefix = if st.names().contains(&&test_key) {
            format!("h.{}", i)
        } else {
            format!("transformer.h.{}", i)
        };

        // 2. Load Weights with Dynamic Transpose Logic
        // Instead of checking for "768", we check if the weight needs to be
        // flipped to match our [In, Out] matmul expectation.

        // --- MLP Projection ---
        let mut mlp_proj_weight = Tensor::from_safetensors(&format!("{}.mlp.c_proj.weight", prefix), st);
        // In GPT-2, these are often [Out, In]. We want [In, Out] for: [Batch, Seq, In] @ [In, Out]
        // If the first dimension matches the MLP expansion (4 * n_embd), it's likely [In, Out]
        if mlp_proj_weight.shape[0] < mlp_proj_weight.shape[1] {
            mlp_proj_weight = mlp_proj_weight.transpose();
        }

        // --- Attention Projection ---
        let mut c_proj_weight = Tensor::from_safetensors(&format!("{}.attn.c_proj.weight", prefix), st);
        if c_proj_weight.shape[0] < c_proj_weight.shape[1] {
            c_proj_weight = c_proj_weight.transpose();
        }

        // 3. Extract n_head dynamically from the attn_bias or attn_weight shape
        let c_attn_bias = Tensor::from_safetensors(&format!("{}.attn.c_attn.bias", prefix), st);
        let total_qkv_dim = c_attn_bias.shape[0];
        let n_embd = total_qkv_dim / 3;

        // Most GPT-2 models use head_dim = 64. If you used something else in your
        // tiny model training, this is where you'd adjust.
        // For now, let's assume standard head_dim = 64 or 32 based on your n_embd.
        let n_head = if n_embd == 768 { 12 } else { 4 }; // Fallback for your tiny config

        Self {
            n_head,
            // Layer Norms
            ln_1_gamma: Tensor::from_safetensors(&format!("{}.ln_1.weight", prefix), st),
            ln_1_beta:  Tensor::from_safetensors(&format!("{}.ln_1.bias", prefix), st),
            ln_2_gamma: Tensor::from_safetensors(&format!("{}.ln_2.weight", prefix), st),
            ln_2_beta:  Tensor::from_safetensors(&format!("{}.ln_2.bias", prefix), st),

            // Attention
            c_attn_weight: Tensor::from_safetensors(&format!("{}.attn.c_attn.weight", prefix), st),
            c_attn_bias,
            c_proj_weight,
            c_proj_bias: Tensor::from_safetensors(&format!("{}.attn.c_proj.bias", prefix), st),

            // MLP
            mlp_fc_weight:   Tensor::from_safetensors(&format!("{}.mlp.c_fc.weight", prefix), st),
            mlp_fc_bias:     Tensor::from_safetensors(&format!("{}.mlp.c_fc.bias", prefix), st),
            mlp_proj_weight,
            mlp_proj_bias:   Tensor::from_safetensors(&format!("{}.mlp.c_proj.bias", prefix), st),
        }
    }


    pub fn split_qkv(&self, qkv: &Tensor, n_embd: usize) -> (Tensor, Tensor, Tensor) {
        let batch = qkv.shape[0];
        let seq_len = qkv.shape[1];

        let mut q_data = vec![0.0; batch * seq_len * n_embd];
        let mut k_data = vec![0.0; batch * seq_len * n_embd];
        let mut v_data = vec![0.0; batch * seq_len * n_embd];

        // Each row in qkv is 3 * n_embd (2304)
        // We split each row into three chunks of n_embd (768)
        for i in 0..(batch * seq_len) {
            let src_start = i * (3 * n_embd);
            let dst_start = i * n_embd;

            q_data[dst_start..dst_start + n_embd].copy_from_slice(&qkv.data[src_start..src_start + n_embd]);
            k_data[dst_start..dst_start + n_embd].copy_from_slice(&qkv.data[src_start + n_embd..src_start + 2 * n_embd]);
            v_data[dst_start..dst_start + n_embd].copy_from_slice(&qkv.data[src_start + 2 * n_embd..src_start + 3 * n_embd]);
        }

        let shape = vec![batch, seq_len, n_embd];
        (Tensor::new(shape.clone(), q_data), Tensor::new(shape.clone(), k_data), Tensor::new(shape, v_data))
    }




    fn get_head_slice<'a>(&self, tensor: &'a Tensor, head_idx: usize) -> &'a [f32] {
        let rank = tensor.shape.len();

        // GPT-2 heads are typically at the first dimension of the attention tensors
        // shape: [n_head, seq_len, head_dim]
        let n_head = tensor.shape[0];

        // Calculate size of a single head by multiplying all dimensions after the first one
        let head_size: usize = tensor.shape[1..].iter().product();

        let start = head_idx * head_size;
        let end = start + head_size;

        // Safety Assertion: If this fails, we know exactly which tensor is broken
        assert!(
            end <= tensor.data.len(),
            "MHA Head Slice out of bounds! Head {}/{} needs range {}..{} but data len is {}. Shape was {:?}",
            head_idx, n_head, start, end, tensor.data.len(), tensor.shape
        );

        &tensor.data[start..end]
    }

    fn attend_values(&self, probs: &Tensor, v: &Tensor) -> Tensor {
        let batch_size = probs.shape[0];
        let n_head = probs.shape[1];
        let seq_len = probs.shape[2];
        let head_dim = v.shape[3]; // V is [Batch, Heads, Seq, HeadDim]

        let mut out_data = vec![0.0; batch_size * n_head * seq_len * head_dim];

        // Parallelize across (Batch * Heads)
        out_data.par_chunks_mut(seq_len * head_dim).enumerate().for_each(|(i, head_out)| {
            let b = i / n_head;
            let h = i % n_head;

            let head_scores = self.get_head_slice_4d(probs, b, h); // [Seq, Seq]
            let head_v = self.get_head_slice_4d(v, b, h);           // [Seq, HeadDim]

            for i_s in 0..seq_len {
                for j_s in 0..seq_len {
                    let score = head_scores[i_s * seq_len + j_s];
                    for d in 0..head_dim {
                        head_out[i_s * head_dim + d] += score * head_v[j_s * head_dim + d];
                    }
                }
            }
        });

        // RETURN 4D
        Tensor::new(vec![batch_size, n_head, seq_len, head_dim], out_data)
    }

    pub fn mha_backward(&mut self, d_attn_out: &Tensor, cache: &LayerCache) -> Tensor {
        let batch_size = d_attn_out.shape[0];
        let seq_len = d_attn_out.shape[1];
        let n_embd = d_attn_out.shape[2];
        let n_head = self.n_head;
        let head_dim = n_embd / n_head;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // 1. Output Projection Backward
        let (d_merged_ctx, d_c_proj_w) = Tensor::matmul_backward(&cache.merged_ctx, &self.c_proj_weight, d_attn_out);
        self.c_proj_weight.accumulate_grad(d_c_proj_w);
        self.c_proj_bias.accumulate_grad(d_attn_out.sum_rows());

        // 2. Prepare gradient buffers for Q, K, and V
        let mut d_q = vec![0.0; cache.q.data.len()];
        let mut d_k = vec![0.0; cache.k.data.len()];
        let mut d_v = vec![0.0; cache.v.data.len()];

        let d_ctx_heads = self.split_heads(&d_merged_ctx, batch_size, seq_len);
        let head_size = seq_len * head_dim;

        // 3. Compute Attention Gradients (Parallelized across heads)
        d_q.par_chunks_mut(head_size)
            .zip(d_k.par_chunks_mut(head_size))
            .zip(d_v.par_chunks_mut(head_size))
            .enumerate()
            .for_each(|(i, ((head_dq, head_dk), head_dv))| {
                let b = i / n_head;
                let h = i % n_head;

                let q_h = self.get_head_slice_4d(&cache.q, b, h);
                let k_h = self.get_head_slice_4d(&cache.k, b, h);
                let v_h = self.get_head_slice_4d(&cache.v, b, h);
                let p_h = self.get_head_slice_4d(&cache.attn_probs, b, h);
                let d_ctx_h = self.get_head_slice_4d(&d_ctx_heads, b, h);

                // Backprop through Attention: V, Probs, and Q/K
                for i_s in 0..seq_len {
                    for j_s in 0..seq_len {
                        let prob = p_h[j_s * seq_len + i_s];
                        for d in 0..head_dim {
                            head_dv[i_s * head_dim + d] += prob * d_ctx_h[j_s * head_dim + d];
                        }
                    }
                }

                let mut d_probs = vec![0.0; seq_len * seq_len];
                for i_s in 0..seq_len {
                    for j_s in 0..seq_len {
                        let mut sum = 0.0;
                        for d in 0..head_dim {
                            sum += d_ctx_h[i_s * head_dim + d] * v_h[j_s * head_dim + d];
                        }
                        d_probs[i_s * seq_len + j_s] = sum;
                    }
                }

                // Softmax Backward & Scaled Dot Product Backward
                for i_s in 0..seq_len {
                    let row_d_probs = &d_probs[i_s * seq_len .. (i_s + 1) * seq_len];
                    let row_probs = &p_h[i_s * seq_len .. (i_s + 1) * seq_len];
                    let mut dot = 0.0;
                    for j_s in 0..seq_len { dot += row_d_probs[j_s] * row_probs[j_s]; }

                    for j_s in 0..seq_len {
                        let d_score = row_probs[j_s] * (row_d_probs[j_s] - dot) * scale;
                        for d in 0..head_dim {
                            head_dq[i_s * head_dim + d] += d_score * k_h[j_s * head_dim + d];
                            head_dk[j_s * head_dim + d] += d_score * q_h[i_s * head_dim + d];
                        }
                    }
                }
            });

        // 4. Combined QKV Backward (Now passing the actual d_q, d_k, d_v!)
        let d_qkv = self.combine_qkv(d_q, d_k, d_v, seq_len, n_embd, batch_size);
        let (d_x, d_c_attn_w) = Tensor::matmul_backward(&cache.ln1_out, &self.c_attn_weight, &d_qkv);

        self.c_attn_weight.accumulate_grad(d_c_attn_w);
        self.c_attn_bias.accumulate_grad(d_qkv.sum_rows());

        d_x
    }
    fn scaled_dot_product(&self, q: &Tensor, k: &Tensor, head_dim: usize) -> Tensor {
        // Use .len() - 2 to find Seq dimension regardless of Batch/Heads rank
        let rank = q.shape.len();
        let seq_len = q.shape[rank - 2];
        let n_head_total = q.data.len() / (seq_len * head_dim);

        let mut scores_data = vec![0.0; n_head_total * seq_len * seq_len];
        let scale = 1.0 / (head_dim as f32).sqrt();

        scores_data.par_chunks_mut(seq_len * seq_len).enumerate().for_each(|(h_idx, head_scores)| {
            // Use a generic offset-based slicer
            let start = h_idx * (seq_len * head_dim);
            let q_h = &q.data[start..start + (seq_len * head_dim)];
            let k_h = &k.data[start..start + (seq_len * head_dim)];

            for i in 0..seq_len {
                for j in 0..seq_len {
                    let mut dot = 0.0;
                    for d in 0..head_dim {
                        dot += q_h[i * head_dim + d] * k_h[j * head_dim + d];
                    }
                    head_scores[i * seq_len + j] = dot * scale;
                }
            }
        });

        // New shape preserves whatever Batch/Heads were passed in
        let mut new_shape = q.shape[..rank-1].to_vec();
        new_shape.push(seq_len);
        Tensor::new(new_shape, scores_data)
    }

    fn apply_causal_mask(&self, scores: &mut Tensor) {
        // scores shape: [Batch, Heads, Seq, Seq]
        let rank = scores.shape.len();
        assert!(rank >= 2, "Causal mask requires at least 2 dimensions");

        // The last two dimensions are ALWAYS [Seq, Seq]
        let seq_len = scores.shape[rank - 1];
        let matrix_size = seq_len * seq_len;

        // Safety check to ensure the total data matches our expected matrix segments
        assert_eq!(
            scores.data.len() % matrix_size,
            0,
            "Tensor data length {} is not a multiple of matrix size {}",
            scores.data.len(),
            matrix_size
        );

        // Parallelize over every [Seq x Seq] attention matrix in the batch/heads
        scores.data.par_chunks_mut(matrix_size).for_each(|matrix| {
            for i in 0..seq_len {
                let row_offset = i * seq_len;
                for j in 0..seq_len {
                    // If column index > row index, it's the "future," so mask it
                    if j > i {
                        // Check bounds just to be absolutely sure for the M4 cores
                        if let Some(element) = matrix.get_mut(row_offset + j) {
                            *element = -1e9;
                        }
                    }
                }
            }
        });
    }

    fn mha(&self, x: Tensor) -> Tensor {
        let batch_size = x.shape[0];
        let seq_len = x.shape[1];
        let n_embd = x.shape[2];
        let head_dim = n_embd / self.n_head;

        let qkv = Tensor::matmul(&x, &self.c_attn_weight).add_bias(&self.c_attn_bias);
        let (q_full, k_full, v_full) = self.split_qkv(&qkv, n_embd);

        let q = self.split_heads(&q_full, batch_size, seq_len);
        let k = self.split_heads(&k_full, batch_size, seq_len);
        let v = self.split_heads(&v_full, batch_size, seq_len);

        let mut scores = self.scaled_dot_product(&q, &k, head_dim);
        self.apply_causal_mask(&mut scores);
        scores.softmax_inplace();

        let ctx_heads = self.attend_values(&scores, &v);
        let ctx = self.merge_heads(&ctx_heads);
        Tensor::matmul(&ctx, &self.c_proj_weight).add_bias(&self.c_proj_bias)
    }
    fn get_stats(data: &[f32]) -> (f32, f32) {
        let mean = data.iter().sum::<f32>() / data.len() as f32;
        let var = data.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / data.len() as f32;
        (mean, var.sqrt())
    }

    fn feed_forward(&self, x: &Tensor) -> Tensor {
        let mut h = Tensor::matmul(x, &self.mlp_fc_weight).add_bias(&self.mlp_fc_bias);

        // 2. Activate
        h.gelu_inplace();

        // 3. Contract (e.g., 3072 -> 768)
        Tensor::matmul(&h, &self.mlp_proj_weight).add_bias(&self.mlp_proj_bias)
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let norm_1 = Tensor::layernorm(x, &self.ln_1_gamma, &self.ln_1_beta, 1e-5);
        let attn_out = self.mha(norm_1);

        // Check: Are you accidentally doing x.add(&attn_out).add(&attn_out)?
        let x_mid = x.add(&attn_out);

        let norm_2 = Tensor::layernorm(&x_mid, &self.ln_2_gamma, &self.ln_2_beta, 1e-5);
        let ffn_out = self.feed_forward(&norm_2);

        x_mid.add(&ffn_out)
    }
}

fn get_stats(data: &[f32]) -> (f32, f32) {
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    let var = data.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / data.len() as f32;
    (mean, var.sqrt())
}