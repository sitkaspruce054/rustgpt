use rand::seq::SliceRandom;

pub struct DataLoader {
    pub tokens: Vec<u32>,
    batch_size: usize,
    block_size: usize,
    current_idx: usize,
    indices: Vec<usize>,
}

impl DataLoader {
    pub fn new(tokens: Vec<u32>, batch_size: usize, block_size: usize) -> Self {
        //sanity check
        if tokens.len() <= block_size {
            panic!("Dataset too small for block_size {}", block_size);
        }

        let num_possible_starts = tokens.len() - block_size;

        //sampling and shuffling
        let mut indices: Vec<usize> = (0..num_possible_starts)
            .step_by(block_size)
            .collect();

        let mut rng = rand::rng();
        indices.shuffle(&mut rng);

        DataLoader {
            tokens,
            batch_size,
            block_size,
            current_idx: 0,
            indices,
        }
    }

    pub fn reset(&mut self) {
        let mut rng = rand::rng();
        self.indices.shuffle(&mut rng);
        self.current_idx = 0;
    }

    pub fn next_batch(&mut self) -> Option<(Vec<u32>, Vec<u32>)> {
        if self.current_idx + self.batch_size > self.indices.len() {
            return None;
        }

        let mut batch_x = Vec::with_capacity(self.batch_size * self.block_size);
        let mut batch_y = Vec::with_capacity(self.batch_size * self.block_size);

        for i in 0..self.batch_size {
            let start = self.indices[self.current_idx + i];
            let end = start + self.block_size;

            // X is [start ... end-1]
            batch_x.extend_from_slice(&self.tokens[start..end]);

            // Y is [start+1 ... end]
            batch_y.extend_from_slice(&self.tokens[start + 1..end + 1]);
        }

        self.current_idx += self.batch_size;
        Some((batch_x, batch_y))
    }
}