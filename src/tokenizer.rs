use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use rayon::prelude::*;
use fancy_regex::Regex;
use serde::{Deserialize, Serialize};
use serde_with::serde_as;

type Word = Vec<u32>;
type MergeType = (u32, u32);

pub fn bytes_to_unicode() -> HashMap<u8, char> {
    let mut bs: Vec<u8> = (b'!'..=b'~').collect();
    bs.extend(b'\xA1'..=b'\xAC');
    bs.extend(b'\xAE'..=b'\xFF');

    let mut cs: Vec<u32> = bs.iter().map(|&b| b as u32).collect();
    let mut n = 0;
    for b in 0..=255 {
        if !bs.contains(&(b as u8)) {
            bs.push(b as u8);
            cs.push(256 + n);
            n += 1;
        }
    }

    let mut mapping = HashMap::new();
    for (b, c) in bs.into_iter().zip(cs.into_iter()) {
        mapping.insert(b, char::from_u32(c).unwrap());
    }
    mapping
}

#[serde_as]
#[derive(Serialize, Deserialize, Debug)]
pub struct Tokenizer {
    #[serde(skip)]
    word_counts: HashMap<Word, usize>,

    #[serde_as(as = "Vec<(_, _)>")]
    vocab: HashMap<u32, Vec<u8>>,

    merges: Vec<(u32, u32)>,
    #[serde_as(as = "Vec<(_, _)>")]
    merge_map: HashMap<MergeType, u32>,
    vocab_size: u32,
    #[serde(skip)]
    regex: Option<Regex>,
}

impl Tokenizer {
    pub fn new(vocab_size: u32, regex: Regex) -> Self {
        Tokenizer {
            word_counts: HashMap::new(),
            vocab: HashMap::new(),
            merges: Vec::new(),
            merge_map: HashMap::new(),
            regex: Some(regex),
            vocab_size,
        }
    }


    pub fn train(&mut self, target_size: u32, text: &[u8]) {
        let byte_encoder = bytes_to_unicode();

        // 1. Reserve ID 0 for <|endoftext|>
        // We store it as a special byte sequence that won't occur in normal text
        self.vocab.insert(0, b"<|endoftext|>".to_vec());

        // 2. Map standard bytes to IDs 1 through 256 (Shifted by 1)
        for i in 0..256 {
            let b = i as u8;
            let c = byte_encoder[&b];
            self.vocab.insert((i + 1) as u32, c.to_string().into_bytes());
        }

        self.word_counts = self.aggregate_counts(text);

        // 3. Start merging from ID 257 onwards
        let mut cur_id = 257;
        while cur_id < target_size {
            let stats = self.get_stats();
            let next_merge = stats.par_iter()
                .max_by(|a, b| a.1.cmp(b.1).then_with(|| b.0.cmp(a.0)));

            if let Some((&pair, _)) = next_merge {
                let mut new_bytes = self.vocab[&pair.0].clone();
                new_bytes.extend(&self.vocab[&pair.1]);
                self.vocab.insert(cur_id, new_bytes);
                self.merges.push(pair);
                self.merge_map.insert(pair, cur_id);
                self.word_counts = self.update_corpus(pair, cur_id);
                cur_id += 1;
            } else {
                break;
            }
            if cur_id % 100 == 0 { println!("Vocab size: {}", cur_id); }
        }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut encoded_text = Vec::new();

        // 1. Split text into chunks by the special token
        // This ensures <|endoftext|> is handled as ID 0 regardless of regex
        let parts: Vec<&str> = text.split("<|endoftext|>").collect();
        let last_idx = parts.len() - 1;

        for (i, part) in parts.into_iter().enumerate() {
            // Encode the regular text part
            if !part.is_empty() {
                let re = self.regex.as_ref().expect("Regex not set");
                for mat in re.find_iter(part) {
                    if let Ok(m) = mat {
                        let mut tokens: Word = m.as_str()
                            .as_bytes()
                            .iter()
                            .map(|&b| (b as u32) + 1)
                            .collect();

                        for &pair in &self.merges {
                            if tokens.len() < 2 { break; }
                            tokens = self.merge_internal(tokens, pair);
                        }
                        encoded_text.extend(tokens);
                    }
                }
            }

            // If we split on the token, insert ID 0 (except after the very last part)
            if i < last_idx {
                encoded_text.push(0);
            }
        }
        encoded_text
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut raw_byte_buffer = Vec::new();

        // Move this outside the loop (done)
        let byte_decoder: HashMap<char, u8> = bytes_to_unicode()
            .into_iter()
            .map(|(b, c)| (c, b))
            .collect();

        for &id in ids {
            if id == 0 {
                raw_byte_buffer.extend_from_slice(b"\n[END]\n");
                continue;
            }

            if let Some(token_bytes) = self.vocab.get(&id) {
                // String::from_utf8_lossy handles partial UTF-8 fragments safely
                let decorated_str = String::from_utf8_lossy(token_bytes);
                for c in decorated_str.chars() {
                    if let Some(&original_byte) = byte_decoder.get(&c) {
                        raw_byte_buffer.push(original_byte);
                    } else {
                        raw_byte_buffer.extend(c.to_string().as_bytes());
                    }
                }
            }
        }
        String::from_utf8_lossy(&raw_byte_buffer).into_owned()
    }

    pub fn set_regex(&mut self, regex: Regex) {
        self.regex = Some(regex);
    }

    // --- Training Helpers ---
    fn get_stats(&self) -> HashMap<(u32, u32), usize> {
        self.word_counts.par_iter().fold(HashMap::new, |mut acc, (word, &freq)| {
            for wnd in word.windows(2) {
                let pair = (wnd[0], wnd[1]);
                *acc.entry(pair).or_insert(0) += freq;
            }
            acc
        }).reduce(HashMap::new, |mut a, b| {
            for (pair, ct) in b {
                *a.entry(pair).or_insert(0) += ct;
            }
            a
        })
    }

    fn update_corpus(&self, pair: (u32, u32), new_token_id: u32) -> HashMap<Word, usize> {
        self.word_counts.par_iter()
            .map(|(word, &freq)| {
                let mut new_word = Vec::new();
                let mut i = 0;
                while i < word.len() {
                    if i < word.len() - 1 && word[i] == pair.0 && word[i+1] == pair.1 {
                        new_word.push(new_token_id);
                        i += 2;
                    } else {
                        new_word.push(word[i]);
                        i += 1;
                    }
                }
                (new_word, freq)
            }).collect()
    }

    fn aggregate_counts(&self, input_text: &[u8]) -> HashMap<Word, usize> {
        input_text.par_chunks(1024 * 1024).map(|chunk| {
            let mut local_map = HashMap::new();
            let txt = String::from_utf8_lossy(chunk);

            if let Some(re) = &self.regex {
                for mat in re.find_iter(&txt) {
                    if let Ok(m) = mat {
                        // CRITICAL FIX: Add + 1 here to match the reserved EOT shift
                        let word_idx: Vec<u32> = m.as_str().as_bytes()
                            .iter()
                            .map(|&b| (b as u32) + 1)
                            .collect();
                        *local_map.entry(word_idx).or_insert(0) += 1;
                    }
                }
            }
            local_map
        }).reduce(HashMap::new, |mut main, local| {
            for (w, c) in local {
                *main.entry(w).or_insert(0) += c;
            }
            main
        })
    }



    fn merge_internal(&self, tokens: Word, pair: MergeType) -> Word {
        let mut new_tokens = Vec::with_capacity(tokens.len());
        let mut i = 0;
        while i < tokens.len() {
            if i < tokens.len() - 1 && tokens[i] == pair.0 && tokens[i+1] == pair.1 {
                if let Some(&new_id) = self.merge_map.get(&pair) {
                    new_tokens.push(new_id);
                    i += 2;
                    continue;
                }
            }
            new_tokens.push(tokens[i]);
            i += 1;
        }
        new_tokens
    }



    // --- Persistence ---
    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let writer = BufWriter::new(File::create(path)?);
        serde_json::to_writer(writer, &self)?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let reader = BufReader::new(File::open(path)?);
        Ok(serde_json::from_reader(reader)?)
    }

    pub fn save_tokens_to_bin(tokens: &[u32], path: &str) -> std::io::Result<()> {
        let mut file = File::create(path)?;
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(tokens.as_ptr() as *const u8, tokens.len() * 4)
        };
        file.write_all(bytes)
    }

    pub fn load_tokens_from_bin(path: &str) -> std::io::Result<Vec<u32>> {
        let mut file = File::open(path)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        let tokens: Vec<u32> = unsafe {
            std::slice::from_raw_parts(bytes.as_ptr() as *const u32, bytes.len() / 4).to_vec()
        };
        Ok(tokens)
    }
}