use std::fs::File;
use std::io;
use std::io::Write;
use memmap2::MmapOptions;
use rand::Rng;
use safetensors::SafeTensors;
use crate::tensor::Tensor;
use rayon::prelude::*;
use crate::tokenizer::Tokenizer;
use crate::transformer::{ForwardCache, TransformerBlock};

pub struct LLM {

    pub wte: Tensor,
    pub wpe: Tensor,

    pub blocks: Vec<TransformerBlock>,
    //used for the final layernorm
    pub ln_f_gamma: Tensor,
    pub ln_f_beta: Tensor,

    // 4. The Output Head (Logits)
    pub lm_head: Tensor,
    config: (usize,usize)
}


impl LLM {

    pub fn sync_tied_weights(&mut self) {
        // 1. Combine Gradients

        if let Some(lm_head_grad) = self.lm_head.grad.as_ref() {
            if self.wte.grad.is_none() {
                self.wte.grad = Some(vec![0.0; self.wte.data.len()]);
            }

            let wte_grad = self.wte.grad.as_mut().unwrap();
            // Since they are the same shape now, we can just add them directly
            for i in 0..wte_grad.len() {
                wte_grad[i] += lm_head_grad[i];
            }

            // Clear the lm_head grad so it's not double-counted by the optimizer
            if let Some(g) = self.lm_head.grad.as_mut() { g.fill(0.0); }
        }

        // 2. Sync Data
        // After optimizer.step(), wte has moved. lm_head must be updated.
        // Since lm_head is a clone of wte in our new random setup, we keep it that way.
        self.lm_head.data.copy_from_slice(&self.wte.data);
    }

    pub fn all_weights_mut(&mut self) -> Vec<&mut Tensor> {
        let mut weights = Vec::new();

        // 1. Embeddings
        weights.push(&mut self.wte);
        weights.push(&mut self.wpe);

        // 2. Transformer Blocks
        for block in &mut self.blocks {
            weights.extend(block.get_weights_mut());
        }

        // 3. Final LayerNorm
        weights.push(&mut self.ln_f_gamma);
        weights.push(&mut self.ln_f_beta);

        // Note: lm_head is excluded due to weight tying with wte
        weights
    }



    pub fn clip_gradients(&mut self, max_norm: f32) {

        let mut total_norm_sq: f64 = 0.0;

        let weights = self.all_weights_mut();

        for weight in &weights {
            if let Some(grad) = &weight.grad {
                // Parallelize the sum of squares across the M4 cores
                let sum_sq: f32 = grad.par_iter()
                    .map(|&g| g * g)
                    .sum();
                total_norm_sq += sum_sq as f64;
            }
        }

        let total_norm = total_norm_sq.sqrt() as f32;

        //Scaling down
        if total_norm > max_norm {
            let scale = max_norm / (total_norm + 1e-6); // 1e-6 prevents div by zero

            for weight in weights {
                if let Some(grad) = &mut weight.grad {
                    grad.par_iter_mut().for_each(|g| {
                        *g *= scale;
                    });
                }
            }
        }
    }

    pub fn embedding_forward(&self, input_ids: &[u32], seq_len: usize, batch_size: usize) -> (Tensor, Tensor) {
        let n_embd = self.wte.shape[1];
        let mut x_data = vec![0.0; batch_size * seq_len * n_embd];

        for b in 0..batch_size {
            for t in 0..seq_len {
                // Index into the flat input_ids [Batch * Seq]
                let token_id = input_ids[b * seq_len + t] as usize;

                //Get WTE (Token)
                let wte_row = &self.wte.data[token_id * n_embd .. (token_id + 1) * n_embd];

                //Get WPE (Position)
                let wpe_row = &self.wpe.data[t * n_embd .. (t + 1) * n_embd];

                //Combine and save to result buffer
                let out_idx = (b * seq_len * n_embd) + (t * n_embd);
                for i in 0..n_embd {
                    x_data[out_idx + i] = wte_row[i] + wpe_row[i];
                }
            }
        }

        let x = Tensor::new(vec![batch_size, seq_len, n_embd], x_data);
        (x.clone(), x) // Return x and a copy for the cache
    }

    pub fn from_pretrained(path: &str) -> Self {
        let file = File::open(path).expect("Failed to open weights file");
        let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };
        let st = SafeTensors::deserialize(&mmap).unwrap();

        // 1. Identify layers
        let mut n_layer = 0;
        while st.names().contains(&&format!("h.{}.ln_1.weight", n_layer)) ||
            st.names().contains(&&format!("transformer.h.{}.ln_1.weight", n_layer)) {
            n_layer += 1;
        }

        // 2. Load global weights
        let wte = Tensor::from_safetensors("wte.weight", &st);
        let wpe = Tensor::from_safetensors("wpe.weight", &st);
        let ln_f_gamma = Tensor::from_safetensors("ln_f.weight", &st);
        let ln_f_beta = Tensor::from_safetensors("ln_f.bias", &st);

        // 3. Load blocks
        let blocks: Vec<TransformerBlock> = (0..n_layer).into_par_iter().map(|i| {
            TransformerBlock::load_from_st(i, &st)
        }).collect();


        let lm_head = if st.names().contains(&&String::from("lm_head.weight")) {
            let mut head = Tensor::from_safetensors("lm_head.weight", &st);
            // Safety check: Ensure it's not accidentally stored transposed in the file
            if head.shape[0] != wte.shape[0] {
                head = head.transpose();
            }
            head
        } else {
            wte.clone() // Correctly tied
        };

        let n_embd = wte.shape[1];
        let block_size = wpe.shape[0];

        // Final sanity check before construction
        assert_eq!(lm_head.shape[1], n_embd, "LM Head width must match n_embd");

        LLM {
            wte,
            wpe,
            blocks,
            ln_f_gamma,
            ln_f_beta,
            lm_head,
            config: (n_embd, block_size)
        }
    }

    pub fn audit_weights(&self) {
        println!("--- Model Weight Audit ---");
        let weights = [
            ("wte", &self.wte),
            ("wpe", &self.wpe),
            ("lm_head", &self.lm_head),
            ("ln_f_gamma", &self.ln_f_gamma),
        ];

        for (name, tensor) in weights {
            let mut nan_count = 0;
            let mut inf_count = 0;
            let mut max_val = f32::NEG_INFINITY;

            for &v in &tensor.data {
                if v.is_nan() { nan_count += 1; }
                if v.is_infinite() { inf_count += 1; }
                if v.abs() > max_val { max_val = v.abs(); }
            }

            println!("{}: MaxAbs={:.4}, NaNs={}, Infs={}", name, max_val, nan_count, inf_count);
        }
        println!("--------------------------");
    }

    pub fn forward(&self, tokens: &[u32]) -> Tensor {
        let seq_len = tokens.len();
        let batch_size = 1;

        // 1. Embeddings
        let (mut x, _) = self.embedding_forward(tokens, seq_len, batch_size);

        // 2. Transformer Blocks
        for block in &self.blocks {
            x = block.forward(&x);
        }

        // 3. Final LayerNorm
        let x_norm = Tensor::layernorm(&x, &self.ln_f_gamma, &self.ln_f_beta, 1e-5);

        // 4. FIX: Use matmul_transposed for the Tied Head
        // OLD: Tensor::matmul(&x_norm, &self.lm_head)
        Tensor::matmul_transposed(&x_norm, &self.lm_head)
    }

    // pub fn new_random(cfg: Config) -> Self {
    //     let mut rng = rand::thread_rng();
    //
    //     // 1. WTE: Standard Normal (mu=0, sigma=0.02)
    //     let wte = Tensor::random_init(vec![cfg.vocab_size, cfg.n_embd], 0.02);
    //
    //     // 2. WPE: Standard Normal (mu=0, sigma=0.02)
    //     let wpe = Tensor::random_init(vec![cfg.block_size, cfg.n_embd], 0.02);
    //
    //     let mut blocks = Vec::new();
    //     for i in 0..cfg.n_layer {
    //         // Scale weights by 1/sqrt(N_layer) to keep residual variance stable
    //         let res_scale = 1.0 / (cfg.n_layer as f32).sqrt();
    //         blocks.push(TransformerBlock::new_random(i, cfg, res_scale));
    //     }
    //
    //     LLM {
    //         wte,
    //         wpe,
    //         blocks,
    //         ln_f_gamma: Tensor::constant(vec![cfg.n_embd], 1.0),
    //         ln_f_beta: Tensor::constant(vec![cfg.n_embd], 0.0),
    //         lm_head: Tensor::constant(vec![cfg.n_embd, cfg.vocab_size], 0.0), // Tying usually happens later
    //     }
    // }

    pub fn save_checkpoint(&self, path: &str) {
        let mut map = std::collections::HashMap::new();

        // 1. Map Global Weights
        map.insert("wte.weight".to_string(), self.wte.to_view());
        map.insert("wpe.weight".to_string(), self.wpe.to_view());
        map.insert("ln_f.weight".to_string(), self.ln_f_gamma.to_view());
        map.insert("ln_f.bias".to_string(), self.ln_f_beta.to_view());

        // Only save lm_head if it's not tied, or save it anyway for compatibility
        map.insert("lm_head.weight".to_string(), self.lm_head.to_view());

        // 2. Map Block Weights
        for (i, block) in self.blocks.iter().enumerate() {
            let p = format!("h.{}", i); // Prefix
            map.insert(format!("{}.ln_1.weight", p), block.ln_1_gamma.to_view());
            map.insert(format!("{}.ln_1.bias", p), block.ln_1_beta.to_view());
            map.insert(format!("{}.ln_2.weight", p), block.ln_2_gamma.to_view());
            map.insert(format!("{}.ln_2.bias", p), block.ln_2_beta.to_view());

            map.insert(format!("{}.attn.c_attn.weight", p), block.c_attn_weight.to_view());
            map.insert(format!("{}.attn.c_attn.bias", p), block.c_attn_bias.to_view());
            map.insert(format!("{}.attn.c_proj.weight", p), block.c_proj_weight.to_view());
            map.insert(format!("{}.attn.c_proj.bias", p), block.c_proj_bias.to_view());

            map.insert(format!("{}.mlp.c_fc.weight", p), block.mlp_fc_weight.to_view());
            map.insert(format!("{}.mlp.c_fc.bias", p), block.mlp_fc_bias.to_view());
            map.insert(format!("{}.mlp.c_proj.weight", p), block.mlp_proj_weight.to_view());
            map.insert(format!("{}.mlp.c_proj.bias", p), block.mlp_proj_bias.to_view());
        }

        // 3. Serialize to disk
        // Note: we convert the HashMap into a Vec of tuples as required by the crate
        let metadata = None;
        let tensors_vec: Vec<(String, safetensors::tensor::TensorView)> =
            map.into_iter().collect();

        match safetensors::serialize_to_file(tensors_vec, &metadata, std::path::Path::new(path)) {
            Ok(_) => println!("💾 Checkpoint saved to: {}", path),
            Err(e) => eprintln!("❌ Failed to save checkpoint: {:?}", e),
        }
    }

    pub fn zero_grad(&mut self) {
        // Clear main model weights
        if let Some(g) = self.wte.grad.as_mut() { g.fill(0.0); }
        if let Some(g) = self.wpe.grad.as_mut() { g.fill(0.0); }
        if let Some(g) = self.ln_f_gamma.grad.as_mut() { g.fill(0.0); }
        if let Some(g) = self.ln_f_beta.grad.as_mut() { g.fill(0.0); }
        if let Some(g) = self.lm_head.grad.as_mut() { g.fill(0.0); }

        // Clear every block
        for block in &mut self.blocks {
            block.zero_grad(); // You'll need to implement this in TransformerBlock too
        }
    }

    pub fn embedding_backward(&mut self, d_x: &Tensor, token_ids: &[u32], n_embd: usize, block_size: usize) {
        // d_x shape is [Batch * Seq, n_embd]
        let total_tokens = token_ids.len();

        // 1. Ensure gradients are initialized and the correct size
        if self.wte.grad.is_none() {
            self.wte.grad = Some(vec![0.0; self.wte.data.len()]);
        }
        if self.wpe.grad.is_none() {
            self.wpe.grad = Some(vec![0.0; self.wpe.data.len()]);
        }

        let wte_grad = self.wte.grad.as_mut().unwrap();
        let wpe_grad = self.wpe.grad.as_mut().unwrap();

        // 2. Accumulate gradients
        // We iterate through every token in the batch
        for t_idx in 0..total_tokens {
            let token_id = token_ids[t_idx] as usize;

            // Calculate which position in the context window this token occupies (0 to block_size-1)
            let pos_idx = t_idx % block_size;

            // Slice of gradient coming from the transformer blocks for this specific token
            let d_row = &d_x.data[t_idx * n_embd .. (t_idx + 1) * n_embd];

            // Update WTE (Token Embedding)
            let wte_offset = token_id * n_embd;
            for i in 0..n_embd {
                wte_grad[wte_offset + i] += d_row[i];
            }

            // Update WPE (Position Embedding)
            let wpe_offset = pos_idx * n_embd;
            for i in 0..n_embd {
                wpe_grad[wpe_offset + i] += d_row[i];
            }
        }
    }
    pub fn n_embd(&self) -> usize {
        self.wte.shape[1]
    }

    pub fn calculate_loss(&self, logits: &Tensor, targets: &[u32]) -> (f32, Tensor) {
        let vocab_size = logits.shape[logits.shape.len() - 1];
        let n_tokens = targets.len();
        let mut total_loss = 0.0;
        let mut d_logits_data = vec![0.0; logits.data.len()];

        // --- Hyperparameters for Label Smoothing ---
        // 0.1 is standard. It tells the model: "Be 90% sure of the target,
        // and spread 10% across everything else."
        let smoothing = 0.1;
        let confidence = 1.0 - smoothing;
        let low_prob = smoothing / (vocab_size as f32);

        for i in 0..n_tokens {
            let row_offset = i * vocab_size;
            let target_id = targets[i] as usize;

            let row = &logits.data[row_offset..row_offset + vocab_size];

            // 1. Stable Softmax
            let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut sum_exp = 0.0;
            let mut probs = vec![0.0; vocab_size];
            for j in 0..vocab_size {
                probs[j] = (row[j] - max_val).exp();
                sum_exp += probs[j];
            }
            for j in 0..vocab_size {
                probs[j] /= sum_exp;
            }

            // 2. Cross Entropy Loss with Safety Clamp
            // .max(1e-10) ensures we never take ln(0), which causes -inf/NaN crashes.
            total_loss -= probs[target_id].max(1e-10).ln();

            // 3. Gradient Calculation with Label Smoothing
            // Formula: (Probs - Smoothed_Targets) / N
            for j in 0..vocab_size {
                let target_prob = if j == target_id { confidence } else { low_prob };
                d_logits_data[row_offset + j] = (probs[j] - target_prob) / n_tokens as f32;
            }
        }

        (total_loss / n_tokens as f32, Tensor::new(logits.shape.clone(), d_logits_data))
    }

    pub fn forward_train(&self, input_ids: &[u32]) -> (Tensor, ForwardCache) {
        // 1. Correctly derive sequence length and batch size from metadata
        let block_size = self.wpe.shape[0]; // Model's max context (e.g. 128)
        let total_tokens = input_ids.len();

        // We assume the input is a flat vector of [Batch * Context]
        // If your loader is giving you 2048 tokens for a 128-context model,
        // then batch_size must be 16.
        let batch_size = total_tokens / block_size;
        let seq_len = block_size;

        // 2. Call embedding_forward with the correct dimensions
        let (x_emb, _wte_out) = self.embedding_forward(input_ids, seq_len, batch_size);
        //println!("Start Shape: {:?}", x_emb.shape); // Expect [seq_len, 768]
        let mut x = x_emb.clone();

        let mut block_caches = Vec::with_capacity(self.blocks.len());
        for (i, block) in self.blocks.iter().enumerate() {
            let (x_next, b_cache) = block.forward_train(&x);
            x = x_next;
            //println!("Block {} output shape: {:?}", i, x.shape); // If this says 1024, the bug is in the block
            block_caches.push(b_cache);
        }

        // 3. Capture the state before Final Normalization
        // This is the gradient destination for the final LayerNorm backward pass
        let pre_ln_f_x = x.clone();

        // 4. Final LayerNorm
        let (ln_f_out, ln_f_x_hat, ln_f_inv_std) = Tensor::layernorm_for_train(
            &x, &self.ln_f_gamma, &self.ln_f_beta, 1e-5
        );

        // 5. LM Head (Logits)
        let logits = Tensor::matmul(&x, &self.lm_head.transpose());
        let cache = ForwardCache {
            token_ids: input_ids.to_vec(),
            initial_embeddings: x_emb,
            block_caches,
            pre_ln_f_x,    // <--- Added
            ln_f_x_hat,
            ln_f_inv_std,
            final_norm_out: ln_f_out,
        };

        (logits, cache)
    }





    pub fn backward(&mut self, d_logits: Tensor, mut caches: ForwardCache) {
        // 1. Identify dimensions
        // d_logits is [Batch, Seq, Vocab] -> e.g., [64, 32, 5000]
        let batch = d_logits.shape[0];
        let seq = d_logits.shape[1];
        let vocab_size = d_logits.shape[2];
        let n_embd = self.lm_head.shape[1];
        let total_tokens = batch * seq;

        // --- 1. LM HEAD BACKWARD (Flattened to 2D) ---
        // Reshape to 2D to ensure matrix multiplication inner-dimensions match
        let d_logits_2d = Tensor::new(vec![total_tokens, vocab_size], d_logits.data);
        let final_norm_2d = Tensor::new(vec![total_tokens, n_embd], caches.final_norm_out.data);

        // Calculate gradients using the specialized transposed logic
        let (d_x_2d, d_lm_head_w) = Tensor::matmul_transposed_backward(
            &final_norm_2d,
            &self.lm_head,
            &d_logits_2d
        );

        // Accumulate weights gradient (matching lm_head's [Vocab, Emb] shape)
        self.lm_head.accumulate_grad(d_lm_head_w);

        // Reshape d_x back to 3D [Batch, Seq, Emb] for the rest of the network
        let d_final_norm = Tensor::new(vec![batch, seq, n_embd], d_x_2d.data);

        // --- 2. FINAL LAYERNORM BACKWARD ---
        let (mut d_x, d_gamma_f, d_beta_f) = Tensor::layernorm_backward(
            &d_final_norm,
            &caches.ln_f_x_hat,
            &self.ln_f_gamma,
            &caches.ln_f_inv_std
        );

        self.ln_f_gamma.accumulate_grad(d_gamma_f);
        self.ln_f_beta.accumulate_grad(d_beta_f);

        // --- 3. TRANSFORMER BLOCKS BACKWARD (Reverse Order) ---
        for block in self.blocks.iter_mut().rev() {
            let block_cache = caches.block_caches.pop()
                .expect("Cache/Block count mismatch during backward pass");

            // Pass d_x through the transformer blocks
            d_x = block.backward(d_x, &block_cache);
        }

        // --- 4. EMBEDDING BACKWARD ---
        // Use the actual n_embd and context_len (block_size) from your config
        self.embedding_backward(
            &d_x,
            &caches.token_ids,
            n_embd,
            seq
        );
    }

    fn get_last_row(&self, logits: &Tensor) -> Vec<f32> {
            // logits shape: [seq_len, vocab_size]
            let seq_len = logits.shape[0];
            let vocab_size = logits.shape[1];

            let start = (seq_len - 1) * vocab_size;
            let end = start + vocab_size;

            // Return just the last row as a flat vector
            logits.data[start..end].to_vec()
        }

    pub fn new_random(
        vocab_size: usize,
        n_embd: usize,
        n_head: usize,  // Added this argument
        n_layer: usize,
        block_size: usize
    ) -> Self {
        let wte = Tensor::random_init(vec![vocab_size, n_embd], 0.02);
        let wpe = Tensor::random_init(vec![block_size, n_embd], 0.02);

        let mut blocks = Vec::with_capacity(n_layer);
        for _ in 0..n_layer {
            blocks.push(TransformerBlock::new_random(n_embd, n_head, n_layer));
        }

        let ln_f_gamma = Tensor::constant(vec![n_embd], 1.0);
        let ln_f_beta = Tensor::constant(vec![n_embd], 0.0);

        // Weight Tying: The LM Head usually shares the same weights as WTE.
        // In our custom library, we clone the reference so they stay in sync.
        let lm_head = wte.clone();

        LLM {
            wte,
            wpe,
            blocks,
            ln_f_gamma,
            ln_f_beta,
            lm_head,
            config: (1, block_size)
        }
    }


        fn softmax_logits(&self, mut logits: Vec<f32>) -> Vec<f32> {
            // 1. Find max for numerical stability
            let max_val = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

            // 2. Exp and Sum
            let mut sum = 0.0;
            for val in logits.iter_mut() {
                *val = (*val - max_val).exp();
                sum += *val;
            }

            // 3. Normalize
            for val in logits.iter_mut() {
                *val /= sum;
            }

            logits
        }

        fn weighted_random_choice(&self, probs: Vec<f32>) -> u32 {
            let mut rng = rand::rng(); // In Rand 0.9, use rng() instead of thread_rng()
            let roll: f32 = rng.random(); // In Rand 0.9, .gen() is now .random()

            let mut cumulative_prob = 0.0;
            for (idx, &prob) in probs.iter().enumerate() {
                cumulative_prob += prob;
                if roll < cumulative_prob {
                    return idx as u32;
                }
            }

            // Fallback for rounding errors
            (probs.len() - 1) as u32
        }

    pub fn sample(
        &self,
        mut logits: Vec<f32>,
        tokens_so_far: &[u32],
        frequency_penalty: f32, // Penalizes based on total count (e.g. 0.1)
        presence_penalty: f32,  // Penalizes if exists at all (e.g. 0.1)
        temperature: f32,
        top_p: f32
    ) -> u32 {
        // 1. Apply Penalties FIRST
        // This discourages the "years years years" loop at the source.
        if !tokens_so_far.is_empty() {
            let mut counts = std::collections::HashMap::new();
            for &t in tokens_so_far {
                *counts.entry(t).or_insert(0) += 1;
            }

            for (&token_id, &count) in counts.iter() {
                let idx = token_id as usize;
                if idx < logits.len() {
                    // Presence penalty is a flat deduction
                    logits[idx] -= presence_penalty;
                    // Frequency penalty scales with how many times the word appeared
                    logits[idx] -= count as f32 * frequency_penalty;
                }
            }
        }

        // 2. Apply Temperature SECOND
        // Note: If temperature is 1.0, this does nothing (which is correct)
        if temperature != 1.0 {
            let inv_temp = 1.0 / temperature;
            for val in logits.iter_mut() {
                *val *= inv_temp;
            }
        }

        // 3. Softmax to get probabilities
        let probs = self.softmax_logits(logits);

        // 4. Top-P (Nucleus) Filtering (Your existing logic)
        let mut indices: Vec<usize> = (0..probs.len()).collect();
        indices.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());

        let mut cumulative_prob = 0.0;
        let mut last_idx = indices.len() - 1;
        for (i, &idx) in indices.iter().enumerate() {
            cumulative_prob += probs[idx];
            if cumulative_prob > top_p {
                last_idx = i;
                break;
            }
        }

        // Re-normalize the nucleus
        let mut filtered_probs = vec![0.0; probs.len()];
        let mut sum = 0.0;
        for &idx in &indices[..=last_idx] {
            filtered_probs[idx] = probs[idx];
            sum += probs[idx];
        }

        for p in filtered_probs.iter_mut() {
            *p /= sum;
        }

        self.weighted_random_choice(filtered_probs)
    }

    fn extract_last_logits(&self, logits: &Tensor) -> Vec<f32> {
        let vocab_size = logits.shape[logits.shape.len() - 1];
        let total_len = logits.data.len();

        // The last vocab_size elements represent the logits for the most recent token
        let start = total_len - vocab_size;
        logits.data[start..total_len].to_vec()
    }

    pub fn generate(&self, prompt_tokens: Vec<u32>, max_new_tokens: usize, temperature: f32, top_p: f32, tokenizer: &Tokenizer) -> Vec<u32> {
        let mut tokens = prompt_tokens;

        for _ in 0..max_new_tokens {
            let block_size = self.config.1;
            let start_idx = tokens.len().saturating_sub(block_size);
            let input_slice = &tokens[start_idx..];

            let logits_tensor = self.forward(input_slice);
            let last_token_logits = self.extract_last_logits(&logits_tensor);


            let next_token = self.sample(last_token_logits, &tokens, 2.2, 4.0, temperature, top_p);

            if next_token == 0 {
                println!("\n[EOT Detected]");
                break;
            }

            tokens.push(next_token);
            let raw_text = tokenizer.decode(&[next_token]);
            print!("{}", raw_text);
            io::stdout().flush().unwrap();
        }
        println!("\n[Generation Finished]");
        tokens
    }
    }