//! Training module.

pub mod cpu;
pub mod gpu;

/// History for plotting training progress.
pub struct TrainingHistory {
    pub scores: Vec<f32>,     // Average scores per checkpoint
    pub losses: Vec<f32>,     // Average losses per checkpoint  
    pub episodes: Vec<usize>, // Episode numbers
}

impl TrainingHistory {
    pub fn new() -> Self {
        Self {
            scores: Vec::new(),
            losses: Vec::new(),
            episodes: Vec::new(),
        }
    }

    pub fn record(&mut self, episode: usize, avg_score: f32, avg_loss: f32) {
        self.episodes.push(episode);
        self.scores.push(avg_score);
        self.losses.push(avg_loss);
    }

    /// Prints ASCII graph of training progress.
    pub fn print_graph(&self) {
        if self.scores.is_empty() {
            return;
        }

        let width = 60;
        let height = 15;

        // Score graph
        println!();
        println!("📈 Score Progress:");
        println!("─────────────────────────────────────────────────────────────");
        self.print_single_graph(&self.scores, width, height, "Score");

        // Loss graph
        println!();
        println!("📉 Loss Progress:");
        println!("─────────────────────────────────────────────────────────────");
        self.print_single_graph(&self.losses, width, height, "Loss");
    }

    fn print_single_graph(&self, data: &[f32], width: usize, height: usize, label: &str) {
        if data.is_empty() {
            return;
        }

        let min_val = data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = (max_val - min_val).max(0.001);

        // Resample data to fit width
        let resampled: Vec<f32> = if data.len() <= width {
            data.to_vec()
        } else {
            (0..width)
                .map(|i| {
                    let idx = i * data.len() / width;
                    data[idx]
                })
                .collect()
        };

        // Print graph
        for row in 0..height {
            let threshold = max_val - (row as f32 / height as f32) * range;
            
            // Y-axis label
            if row == 0 {
                print!("{:>7.0} │", max_val);
            } else if row == height - 1 {
                print!("{:>7.0} │", min_val);
            } else if row == height / 2 {
                print!("{:>7.0} │", (min_val + max_val) / 2.0);
            } else {
                print!("        │");
            }

            // Data points
            for &val in &resampled {
                if val >= threshold {
                    print!("█");
                } else {
                    print!(" ");
                }
            }
            println!();
        }

        // X-axis
        print!("        └");
        for _ in 0..resampled.len() {
            print!("─");
        }
        println!();

        // X-axis labels
        let first_ep = self.episodes.first().unwrap_or(&0);
        let last_ep = self.episodes.last().unwrap_or(&0);
        println!("         {:^width$}", format!("{} (ep {} → {})", label, first_ep, last_ep), width = resampled.len());
    }
}
