//! 2048 Game AI using ArKan KAN network.
//!
//! This example demonstrates training a KAN network to play 2048 using DQN.
//!
//! # Usage
//!
//! ```bash
//! # CPU training (recommended to start - most stable)
//! cargo run --release -- --mode train-cpu --episodes 1000
//!
//! # GPU hybrid training (GPU forward/backward + CPU optimizer)
//! cargo run --release -- --mode train-gpu-hybrid --episodes 1000
//!
//! # GPU native training (all on GPU - fastest, but no gradient clipping)
//! cargo run --release -- --mode train-gpu-native --episodes 1000
//!
//! # Random agent baseline
//! cargo run --release -- --mode random --episodes 100
//!
//! # Heuristic agent (corner strategy)
//! cargo run --release -- --mode heuristic --episodes 100
//! ```

mod game;
mod env;
mod utils;
mod agents;
mod train;

use clap::Parser;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "game2048")]
#[command(about = "Train KAN network to play 2048")]
struct Args {
    /// Mode: random, heuristic, train-cpu, train-gpu-hybrid, train-gpu-native
    #[arg(long, default_value = "train-cpu")]
    mode: String,

    /// Number of training episodes
    #[arg(long, default_value = "1000")]
    episodes: usize,

    /// Batch size for training
    #[arg(long, default_value = "32")]
    batch_size: usize,

    /// Learning rate
    #[arg(long, default_value = "0.001")]
    lr: f32,

    /// Replay buffer capacity
    #[arg(long, default_value = "10000")]
    replay_capacity: usize,

    /// Target network update frequency
    #[arg(long, default_value = "100")]
    target_update: usize,

    /// Initial epsilon for exploration
    #[arg(long, default_value = "1.0")]
    epsilon_start: f32,

    /// Final epsilon for exploration
    #[arg(long, default_value = "0.01")]
    epsilon_end: f32,

    /// Epsilon decay rate
    #[arg(long, default_value = "0.995")]
    epsilon_decay: f32,

    /// Discount factor (gamma)
    #[arg(long, default_value = "0.99")]
    gamma: f32,

    /// Print board during demo
    #[arg(long, default_value = "false")]
    verbose: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║           2048 AI Training with ArKan KAN                 ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!();
    println!("Configuration:");
    println!("  Mode:           {}", args.mode);
    println!("  Episodes:       {}", args.episodes);
    println!("  Batch size:     {}", args.batch_size);
    println!("  Learning rate:  {}", args.lr);
    println!("  Gamma:          {}", args.gamma);
    println!();

    let start = Instant::now();

    match args.mode.as_str() {
        "random" => {
            println!("🎲 Running random agent baseline...");
            agents::random::run_episodes(args.episodes, args.verbose);
        }
        "heuristic" => {
            println!("🧠 Running heuristic agent...");
            agents::heuristic::run_episodes(args.episodes, args.verbose);
        }
        "train-cpu" => {
            println!("🖥️  Training DQN agent on CPU...");
            println!("   (Most stable - has gradient clipping)");
            train::cpu::train(
                args.episodes,
                args.batch_size,
                args.lr,
                args.gamma,
                args.replay_capacity,
                args.target_update,
                args.epsilon_start,
                args.epsilon_end,
                args.epsilon_decay,
            )?;
        }
        "train-gpu-hybrid" => {
            println!("🎮 Training DQN agent on GPU (Hybrid mode)...");
            println!("   (GPU forward/backward + CPU optimizer - stable)");
            train::gpu::train_hybrid(
                args.episodes,
                args.batch_size,
                args.lr,
                args.gamma,
                args.replay_capacity,
                args.target_update,
                args.epsilon_start,
                args.epsilon_end,
                args.epsilon_decay,
            )?;
        }
        "train-gpu-native" => {
            println!("⚡ Training DQN agent on GPU (Native mode)...");
            println!("   (All on GPU - fastest, NO gradient clipping!)");
            train::gpu::train_native(
                args.episodes,
                args.batch_size,
                args.lr,
                args.gamma,
                args.replay_capacity,
                args.target_update,
                args.epsilon_start,
                args.epsilon_end,
                args.epsilon_decay,
            )?;
        }
        _ => {
            eprintln!("Unknown mode: {}.", args.mode);
            eprintln!("Available modes:");
            eprintln!("  random          - Random agent baseline");
            eprintln!("  heuristic       - Heuristic corner strategy");
            eprintln!("  train-cpu       - CPU training (stable)");
            eprintln!("  train-gpu-hybrid - GPU hybrid training");
            eprintln!("  train-gpu-native - GPU native training (fast)");
            std::process::exit(1);
        }
    }

    println!();
    println!("Total time: {:.1}s", start.elapsed().as_secs_f64());

    Ok(())
}
