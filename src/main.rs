mod tokenizer;
mod tensor;
mod transformer;
mod llm;
mod dataloader;
mod adam;

use std::{env, fs};
use std::path::Path;
use std::time::Instant;
use crate::llm::LLM;
use crate::tokenizer::Tokenizer;
use crate::dataloader::DataLoader;
use crate::adam::AdamW;

fn main() -> Result<(), Box<dyn std::error::Error>> {

    let args: Vec<String> = env::args().collect();
    let is_inference = args.iter().any(|arg| arg == "--infer");

    // --- 1. CONFIGURATION ---

    let checkpoint_to_load = "stories2_d512_e1.safetensors";
    // --- 1. CONFIGURATION (STORYTELLER V1: d=512, L=8, ctx=128) ---
    let tokenizer_path = "stories50_tokenizer.json";
    let training_data_path = "TinyStoriesV2-GPT4-train.txt";
    let token_cache_path = "stories_ctx256.bin";

    let target_vocab_size = 5000;
    let n_embd = 512;
    let n_layer = 8;
    let n_head = 8;
    let context_len = 256;
    let batch_size = 16;

    let gpt2_regex = r#"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#;
    let re = fancy_regex::Regex::new(gpt2_regex).unwrap();


    // --- 2. TOKENIZER ---
    let mut tokenizer = if Path::new(tokenizer_path).exists() {
        Tokenizer::load(tokenizer_path)?
    } else {
        println!("📂 Training new byte-level BPE tokenizer...");
        let raw_text = fs::read_to_string(training_data_path).expect("TinyStories file not found");
        let mut new_tokenizer = Tokenizer::new(target_vocab_size, re.clone());
        new_tokenizer.train(target_vocab_size, raw_text.as_bytes());
        new_tokenizer.save(tokenizer_path)?;
        new_tokenizer
    };
    tokenizer.set_regex(re);

    // --- 3. DATA LOADING ---
    let all_tokens = if Path::new(token_cache_path).exists() {
        Tokenizer::load_tokens_from_bin(token_cache_path)?
    } else {
        println!("✍️ Tokenizing for ctx=256...");
        let full_text = fs::read_to_string(training_data_path)?;
        let tokens = tokenizer.encode(&full_text);

        let limit = 50_000_000;
        let subset_tokens = if tokens.len() > limit {
            println!("Taking subset");
            tokens[..limit].to_vec()
        } else {
            tokens
        };
        Tokenizer::save_tokens_to_bin(&subset_tokens, token_cache_path)?;
        subset_tokens
    };
    if is_inference {
        run_inference_mode(&tokenizer, checkpoint_to_load)?;
        return Ok(());
    }

    let split_idx = (all_tokens.len() as f32 * 0.9) as usize;
    let train_tokens = all_tokens[..split_idx].to_vec();
    let val_tokens = all_tokens[split_idx..].to_vec();

    let mut train_loader = DataLoader::new(train_tokens, batch_size, context_len);
    let mut val_loader = DataLoader::new(val_tokens, batch_size, context_len);

    // --- 4. MODEL RECOVERY LOGIC ---
    // We revert to Epoch 1 because Epochs 2-4 diverged.
    //let checkpoint_to_load = "stories_d512_e1.safetensors";
    let mut start_epoch = 0;
    let total_epochs = 50;

    let mut model = if Path::new(checkpoint_to_load).exists() {
        println!("RECOVERY MODE: Loading stable checkpoint {}...", checkpoint_to_load);
        start_epoch = 3;
        LLM::from_pretrained(checkpoint_to_load)
    } else {
        println!("Initializing Fresh Storyteller V2 (d=512, L=8, ctx=256)...");
        LLM::new_random(target_vocab_size as usize, n_embd, n_head, n_layer, context_len)
    };
    
    let max_lr = 6e-5;
    let min_lr = 1e-5;
    let warmup_steps = 1000;

    let steps_per_epoch = train_loader.tokens.len() / (batch_size * context_len);
    let remaining_epochs = total_epochs - start_epoch;
    let total_steps = steps_per_epoch * remaining_epochs;

    // Reset global_step to 0 to recalibrate Adam for the new speed
    let mut global_step = 0;
    let mut optimizer = AdamW::new(max_lr, 0.9, 0.95, 1e-8, 0.1);

    println!("⚙️ Resuming Training: Epoch {}/{} | Max LR: {}", start_epoch, total_epochs, max_lr);
    let start_time = Instant::now();

    for epoch in start_epoch..total_epochs {
        train_loader.reset();
        let mut step = 0;

        while let Some((x, y)) = train_loader.next_batch() {
            // New Cosine Schedule for the recovery phase
            let lr = if global_step < warmup_steps {
                max_lr * (global_step as f32 / warmup_steps as f32)
            } else {
                let progress = (global_step - warmup_steps) as f32 / (total_steps - warmup_steps) as f32;
                let cosine_scale = 0.5 * (1.0 + (std::f32::consts::PI * progress.min(1.0)).cos());
                min_lr + (max_lr - min_lr) * cosine_scale
            };
            optimizer.lr = lr;

            model.zero_grad();
            let (logits, cache) = model.forward_train(&x);
            let (loss_val, d_logits) = model.calculate_loss(&logits, &y);

            // Stability Emergency Brake
            if loss_val.is_nan() || loss_val > 9.0 {
                println!("🛑 Divergence Alert: Loss is {}. System exiting to prevent weight corruption.", loss_val);
                return Ok(());
            }

            model.backward(d_logits, cache);

            // Ultra-tight gradient clipping for deep architecture stability
            model.clip_gradients(0.05);

            optimizer.step(&mut model.all_weights_mut());
            model.sync_tied_weights();

            if step % 50 == 0 {
                let elapsed = start_time.elapsed().as_secs_f32();
                let tokens_per_sec = (global_step as f32 * batch_size as f32 * context_len as f32) / elapsed;
                println!(
                    "E{:02} | S{:04} | Loss {:.4} | LR {:.6} | {:.0} tok/s",
                    epoch, step, loss_val, lr, tokens_per_sec
                );
            }
            step += 1;
            global_step += 1;
        }

        // --- 6. EVALUATION & SAMPLING ---
        let val_loss = evaluate(&model, &mut val_loader, 10);
        println!("\n📊 EPOCH {} COMPLETE | Val Loss: {:.4}", epoch, val_loss);

        let prompt_tokens = tokenizer.encode("Once upon a time, there was a");
        let res = model.generate(prompt_tokens, 256, 0.7, 0.9, &tokenizer);
        println!("\n[SAMPLE]: {}\n", tokenizer.decode(&res));

        let checkpoint_name = format!("stories2_d512_e{}.safetensors", epoch);
        model.save_checkpoint(&checkpoint_name);
    }

    Ok(())
}

fn run_inference_mode(tokenizer: &Tokenizer, checkpoint_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Loading model for Inference: {}...", checkpoint_path);
    let model = LLM::from_pretrained(checkpoint_path);

    use std::io::{self, Write};
    let stdin = io::stdin();

    loop {
        print!("\n📝 Enter prompt (or 'exit'): ");
        io::stdout().flush()?;

        let mut input = String::new();
        stdin.read_line(&mut input)?;
        let prompt = input.trim();

        if prompt == "exit" { break; }
        if prompt.is_empty() { continue; }

        let prompt_tokens = tokenizer.encode(prompt);

        // Settings for higher quality: Lower temperature (0.7) and Top-P (0.9)
        let res = model.generate(prompt_tokens, 256, 0.75, 0.9, &tokenizer);

        println!("\n🤖 Model says:\n{}", tokenizer.decode(&res));
        println!("\n---");
    }

    Ok(())
}

fn evaluate(model: &LLM, loader: &mut DataLoader, num_batches: usize) -> f32 {
    loader.reset();
    let mut total_loss = 0.0;
    let mut count = 0;
    for _ in 0..num_batches {
        if let Some((x, y)) = loader.next_batch() {
            let (logits, _) = model.forward_train(&x);
            let (loss, _) = model.calculate_loss(&logits, &y);
            total_loss += loss;
            count += 1;
        }
    }
    if count == 0 { 0.0 } else { total_loss / count as f32 }
}