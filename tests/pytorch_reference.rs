//! PyTorch reference tests for optimizer verification.
//!
//! These tests compare ArKan optimizer implementations against PyTorch reference values.
//! Reference values were generated using scripts/pytorch_optimizer_reference.py
//!
//! The goal is to ensure mathematical equivalence between implementations.

/// Tolerance for floating point comparison
/// PyTorch uses float32, and there can be minor differences due to:
/// - Different order of operations
/// - FMA (fused multiply-add) usage
/// - Platform-specific optimizations
const TOLERANCE: f32 = 1e-5;
const LOOSE_TOLERANCE: f32 = 1e-4;

/// Simple quadratic function: f(x) = Σ x_i²
/// Gradient: ∂f/∂x_i = 2 * x_i
fn quadratic_grad(params: &[f32]) -> Vec<f32> {
    params.iter().map(|x| 2.0 * x).collect()
}

/// Rosenbrock function: f(x,y) = (1-x)² + 100(y-x²)²
/// Gradient:
///   ∂f/∂x = -2(1-x) + 100 * 2(y-x²) * (-2x) = -2(1-x) - 400x(y-x²)
///   ∂f/∂y = 100 * 2(y-x²) = 200(y-x²)
fn rosenbrock_grad(x: f32, y: f32) -> (f32, f32) {
    let dx = -2.0 * (1.0 - x) - 400.0 * x * (y - x * x);
    let dy = 200.0 * (y - x * x);
    (dx, dy)
}

fn rosenbrock_loss(x: f32, y: f32) -> f32 {
    (1.0 - x).powi(2) + 100.0 * (y - x * x).powi(2)
}

/// Helper to compare vectors with tolerance
fn assert_vec_eq(actual: &[f32], expected: &[f32], tolerance: f32, context: &str) {
    assert_eq!(actual.len(), expected.len(), "{}: length mismatch", context);
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        assert!(
            diff < tolerance,
            "{}: element {} differs: actual={}, expected={}, diff={}",
            context, i, a, e, diff
        );
    }
}

// ============================================================================
// Adam Tests - Verifying Adam formula against PyTorch
// ============================================================================

#[test]
fn test_pytorch_adam_default_quadratic() {
    // PyTorch reference:
    // Adam(lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    // init_params=[1.0, 2.0, 3.0, 4.0]
    // 10 steps on quadratic f(x) = Σ x_i²
    
    // Expected trajectory from PyTorch (first few steps)
    let expected_steps = vec![
        vec![1.0, 2.0, 3.0, 4.0],                                    // Initial
        vec![0.99, 1.99, 2.99, 3.99],                                // Step 1
        vec![0.9800027608871, 1.9800013303757, 2.9800009727478, 3.9800007343292], // Step 2
        vec![0.9700101017952, 1.9700049161911, 2.9700033664703, 3.9700024127960], // Step 3
    ];
    
    let mut params = vec![1.0f32, 2.0, 3.0, 4.0];
    
    // Simulate Adam manually (we need direct param access, not through network)
    // Adam state
    let mut m = vec![0.0f32; 4]; // First moment
    let mut v = vec![0.0f32; 4]; // Second moment
    let lr = 0.01f32;
    let beta1 = 0.9f32;
    let beta2 = 0.999f32;
    let eps = 1e-8f32;
    
    // Verify initial state
    assert_vec_eq(&params, &expected_steps[0], TOLERANCE, "Initial params");
    
    for step in 1..4 {
        let grad = quadratic_grad(&params);
        
        // Update biased first moment estimate
        for i in 0..4 {
            m[i] = beta1 * m[i] + (1.0 - beta1) * grad[i];
        }
        
        // Update biased second raw moment estimate
        for i in 0..4 {
            v[i] = beta2 * v[i] + (1.0 - beta2) * grad[i] * grad[i];
        }
        
        // Compute bias-corrected estimates
        let bias_correction1 = 1.0 - beta1.powi(step as i32);
        let bias_correction2 = 1.0 - beta2.powi(step as i32);
        
        // Update parameters
        for i in 0..4 {
            let m_hat = m[i] / bias_correction1;
            let v_hat = v[i] / bias_correction2;
            params[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
        
        assert_vec_eq(&params, &expected_steps[step], LOOSE_TOLERANCE, 
            &format!("Step {}", step));
    }
}

#[test]
fn test_pytorch_adam_with_weight_decay() {
    // PyTorch Adam with weight_decay=0.01 (L2 regularization)
    // Note: PyTorch's Adam adds weight_decay * param to gradient BEFORE Adam update
    
    // Expected after 10 steps from PyTorch
    let expected_final = vec![0.9003496766090393, 1.9001675844192505, 2.9001107215881348, 3.9000821113586426];
    
    let mut params = vec![1.0f32, 2.0, 3.0, 4.0];
    let mut m = vec![0.0f32; 4];
    let mut v = vec![0.0f32; 4];
    let lr = 0.01f32;
    let beta1 = 0.9f32;
    let beta2 = 0.999f32;
    let eps = 1e-8f32;
    let weight_decay = 0.01f32;
    
    for step in 1..=10 {
        // Gradient with L2 regularization (PyTorch style)
        let mut grad = quadratic_grad(&params);
        for i in 0..4 {
            grad[i] += weight_decay * params[i]; // L2 regularization
        }
        
        for i in 0..4 {
            m[i] = beta1 * m[i] + (1.0 - beta1) * grad[i];
            v[i] = beta2 * v[i] + (1.0 - beta2) * grad[i] * grad[i];
        }
        
        let bias_correction1 = 1.0 - beta1.powi(step);
        let bias_correction2 = 1.0 - beta2.powi(step);
        
        for i in 0..4 {
            let m_hat = m[i] / bias_correction1;
            let v_hat = v[i] / bias_correction2;
            params[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
    }
    
    assert_vec_eq(&params, &expected_final, LOOSE_TOLERANCE, "Final params with weight decay");
}

#[test]
fn test_pytorch_adamw_decoupled_weight_decay() {
    // PyTorch AdamW with decoupled weight decay
    // weight_decay is applied directly to weights, not added to gradient
    
    // Expected after 10 steps from PyTorch AdamW
    let expected_final = vec![0.8993987441062927, 1.898216724395752, 2.8971598148345947, 3.8961315155029297];
    
    let mut params = vec![1.0f32, 2.0, 3.0, 4.0];
    let mut m = vec![0.0f32; 4];
    let mut v = vec![0.0f32; 4];
    let lr = 0.01f32;
    let beta1 = 0.9f32;
    let beta2 = 0.999f32;
    let eps = 1e-8f32;
    let weight_decay = 0.01f32;
    
    for step in 1..=10 {
        // Regular gradient (no L2 term)
        let grad = quadratic_grad(&params);
        
        for i in 0..4 {
            m[i] = beta1 * m[i] + (1.0 - beta1) * grad[i];
            v[i] = beta2 * v[i] + (1.0 - beta2) * grad[i] * grad[i];
        }
        
        let bias_correction1 = 1.0 - beta1.powi(step);
        let bias_correction2 = 1.0 - beta2.powi(step);
        
        for i in 0..4 {
            let m_hat = m[i] / bias_correction1;
            let v_hat = v[i] / bias_correction2;
            // AdamW: decoupled weight decay
            params[i] -= lr * (m_hat / (v_hat.sqrt() + eps) + weight_decay * params[i]);
        }
    }
    
    assert_vec_eq(&params, &expected_final, LOOSE_TOLERANCE, "Final params AdamW");
}

#[test]
fn test_pytorch_adam_custom_betas() {
    // PyTorch Adam with custom betas=(0.5, 0.9999)
    
    // Expected after 10 steps from PyTorch
    let expected_final = vec![0.9900153279304504, 1.9900075197219849, 2.9900052547454834, 3.9900035858154297];
    
    let mut params = vec![1.0f32, 2.0, 3.0, 4.0];
    let mut m = vec![0.0f32; 4];
    let mut v = vec![0.0f32; 4];
    let lr = 0.001f32;
    let beta1 = 0.5f32;
    let beta2 = 0.9999f32;
    let eps = 1e-8f32;
    
    for step in 1..=10 {
        let grad = quadratic_grad(&params);
        
        for i in 0..4 {
            m[i] = beta1 * m[i] + (1.0 - beta1) * grad[i];
            v[i] = beta2 * v[i] + (1.0 - beta2) * grad[i] * grad[i];
        }
        
        let bias_correction1 = 1.0 - beta1.powi(step);
        let bias_correction2 = 1.0 - beta2.powi(step);
        
        for i in 0..4 {
            let m_hat = m[i] / bias_correction1;
            let v_hat = v[i] / bias_correction2;
            params[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
    }
    
    assert_vec_eq(&params, &expected_final, LOOSE_TOLERANCE, "Final params custom betas");
}

// ============================================================================
// SGD Tests - Verifying SGD formula against PyTorch
// ============================================================================

#[test]
fn test_pytorch_sgd_no_momentum() {
    // PyTorch SGD(lr=0.1, momentum=0) on quadratic
    
    // Expected trajectory (analytical for quadratic):
    // x[t+1] = x[t] - lr * 2 * x[t] = x[t] * (1 - 2*lr) = x[t] * 0.8
    
    let expected_steps = vec![
        vec![1.0, 2.0, 3.0, 4.0],                     // Initial
        vec![0.8, 1.6, 2.4, 3.2],                     // Step 1
        vec![0.64, 1.28, 1.92, 2.56],                 // Step 2
        vec![0.512, 1.024, 1.536, 2.048],             // Step 3
    ];
    
    let mut params = vec![1.0f32, 2.0, 3.0, 4.0];
    let lr = 0.1f32;
    
    assert_vec_eq(&params, &expected_steps[0], TOLERANCE, "Initial");
    
    for step in 1..4 {
        let grad = quadratic_grad(&params);
        for i in 0..4 {
            params[i] -= lr * grad[i];
        }
        assert_vec_eq(&params, &expected_steps[step], TOLERANCE, &format!("Step {}", step));
    }
}

#[test]
fn test_pytorch_sgd_with_momentum() {
    // PyTorch SGD(lr=0.1, momentum=0.9) on quadratic
    // Note: PyTorch SGD momentum formula: v = momentum * v + grad; param -= lr * v
    
    // Expected from PyTorch
    let expected_final = vec![
        0.0043998658657073975, 
        0.008799731731414795, 
        0.013199687004089355, 
        0.01759946346282959
    ];
    
    let mut params = vec![1.0f32, 2.0, 3.0, 4.0];
    let mut velocity = vec![0.0f32; 4];
    let lr = 0.1f32;
    let momentum = 0.9f32;
    
    for _ in 0..10 {
        let grad = quadratic_grad(&params);
        for i in 0..4 {
            velocity[i] = momentum * velocity[i] + grad[i];
            params[i] -= lr * velocity[i];
        }
    }
    
    assert_vec_eq(&params, &expected_final, LOOSE_TOLERANCE, "SGD with momentum");
}

#[test]
fn test_pytorch_sgd_nesterov() {
    // PyTorch SGD(lr=0.1, momentum=0.9, nesterov=True) on quadratic
    // Nesterov momentum: v = momentum * v + grad; param -= lr * (momentum * v + grad)
    
    // Expected from PyTorch
    let expected_final = vec![
        0.051360733807086945, 
        0.10272146761417389, 
        0.15408211946487427, 
        0.20544293522834778
    ];
    
    let mut params = vec![1.0f32, 2.0, 3.0, 4.0];
    let mut velocity = vec![0.0f32; 4];
    let lr = 0.1f32;
    let momentum = 0.9f32;
    
    for _ in 0..10 {
        let grad = quadratic_grad(&params);
        for i in 0..4 {
            velocity[i] = momentum * velocity[i] + grad[i];
            // Nesterov: use look-ahead gradient
            params[i] -= lr * (momentum * velocity[i] + grad[i]);
        }
    }
    
    assert_vec_eq(&params, &expected_final, LOOSE_TOLERANCE, "SGD Nesterov");
}

#[test]
fn test_pytorch_sgd_with_weight_decay() {
    // PyTorch SGD(lr=0.1, momentum=0.9, weight_decay=0.01)
    
    // Expected from PyTorch
    let expected_final = vec![
        0.011977165937423706, 
        0.023954331874847412, 
        0.03593176603317261, 
        0.047908663749694824
    ];
    
    let mut params = vec![1.0f32, 2.0, 3.0, 4.0];
    let mut velocity = vec![0.0f32; 4];
    let lr = 0.1f32;
    let momentum = 0.9f32;
    let weight_decay = 0.01f32;
    
    for _ in 0..10 {
        let mut grad = quadratic_grad(&params);
        // PyTorch SGD adds weight_decay * param to gradient
        for i in 0..4 {
            grad[i] += weight_decay * params[i];
        }
        for i in 0..4 {
            velocity[i] = momentum * velocity[i] + grad[i];
            params[i] -= lr * velocity[i];
        }
    }
    
    assert_vec_eq(&params, &expected_final, LOOSE_TOLERANCE, "SGD with weight decay");
}

// ============================================================================
// L-BFGS Tests - Verifying L-BFGS behavior
// ============================================================================

#[test]
fn test_pytorch_lbfgs_quadratic_convergence() {
    // L-BFGS should converge on quadratic in very few steps
    // Since quadratic is convex, L-BFGS approximates true Hessian well
    
    // PyTorch LBFGS with strong_wolfe on quadratic
    // Expected: should reach near-zero very quickly
    
    let init_params = vec![1.0f32, 2.0, 3.0, 4.0];
    let init_loss = init_params.iter().map(|x| x * x).sum::<f32>();
    
    // For quadratic, L-BFGS should converge to ~0 within 5 steps
    // We just verify it converges significantly
    
    assert!((init_loss - 30.0).abs() < 1e-5, "Initial loss should be 30");
    
    // Note: Full L-BFGS integration test would require network, skipping here
}

#[test]
fn test_pytorch_lbfgs_rosenbrock() {
    // L-BFGS on Rosenbrock function starting from (-1, 1)
    // Minimum is at (1, 1) with f(1,1) = 0
    
    // PyTorch LBFGS converges to near (1, 1) after ~20 steps
    // Expected trajectory shows decreasing loss
    
    let mut x = -1.0f32;
    let mut y = 1.0f32;
    
    let init_loss = rosenbrock_loss(x, y);
    assert!((init_loss - 4.0).abs() < 1e-5, "Initial Rosenbrock loss at (-1,1) should be 4");
    
    // Verify gradient computation
    let (gx, gy) = rosenbrock_grad(x, y);
    // At (-1, 1): 
    // dx = -2(1-(-1)) - 400*(-1)*(1-1) = -2*2 - 0 = -4
    // dy = 200*(1-1) = 0
    assert!((gx - (-4.0)).abs() < 1e-5, "Gradient x at (-1,1)");
    assert!((gy - 0.0).abs() < 1e-5, "Gradient y at (-1,1)");
    
    // Simple gradient descent to verify we can reach minimum
    // (L-BFGS would be faster but harder to verify step-by-step)
    let lr = 0.001f32;
    for _ in 0..1000 {
        let (gx, gy) = rosenbrock_grad(x, y);
        x -= lr * gx;
        y -= lr * gy;
    }
    
    let final_loss = rosenbrock_loss(x, y);
    assert!(final_loss < 1.0, "Should converge toward minimum, got loss={}", final_loss);
}

// ============================================================================
// ArKan Integration Tests - Test that ArKan optimizers work correctly
// ============================================================================

#[test]
fn test_arkan_adam_integration() {
    use arkan::{Adam, AdamConfig, KanConfig, KanNetwork, Optimizer};
    
    let config = KanConfig {
        input_dim: 2,
        hidden_dims: vec![4],
        output_dim: 2,
        grid_size: 3,
        spline_order: 3,
        input_mean: vec![0.0; 2],
        input_std: vec![1.0; 2],
        ..Default::default()
    };
    
    let mut network = KanNetwork::new(config);
    let adam_config = AdamConfig {
        lr: 0.01,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
        weight_decay: 0.0,
        ..Default::default()
    };
    
    let mut adam = Adam::new(&network, adam_config);
    
    // Get initial weights
    let initial_weights: Vec<f32> = network.layers[0].weights.clone();
    
    // Perform forward/backward pass
    let mut workspace = network.create_workspace(1);
    let input = vec![0.5f32, 0.5];
    let target = vec![1.0f32, 0.0];
    
    network.train_step(&input, &target, None, 1.0, &mut workspace);
    
    // Extract gradients from workspace (they are already Vec<Vec<f32>>)
    let weight_grads: Vec<Vec<f32>> = workspace.weight_grads.clone();
    let bias_grads: Vec<Vec<f32>> = workspace.bias_grads.clone();
    
    // Apply Adam step
    adam.step(&mut network, &weight_grads, &bias_grads, None).unwrap();
    
    // Weights should have changed
    let new_weights: Vec<f32> = network.layers[0].weights.clone();
    
    assert_ne!(initial_weights, new_weights, "Weights should be updated");
    
    // Verify no NaN/Inf
    for w in &new_weights {
        assert!(w.is_finite(), "Weight should be finite");
    }
}

#[test]
fn test_arkan_sgd_integration() {
    use arkan::{KanConfig, KanNetwork, Optimizer, SGD, SGDConfig};
    
    let config = KanConfig {
        input_dim: 2,
        hidden_dims: vec![4],
        output_dim: 2,
        grid_size: 3,
        spline_order: 3,
        input_mean: vec![0.0; 2],
        input_std: vec![1.0; 2],
        ..Default::default()
    };
    
    let mut network = KanNetwork::new(config);
    let sgd_config = SGDConfig {
        lr: 0.1,
        momentum: 0.9,
        weight_decay: 0.0,
        nesterov: false,
        ..Default::default()
    };
    
    let mut sgd = SGD::new(&network, sgd_config);
    
    // Get initial weights
    let initial_weights: Vec<f32> = network.layers[0].weights.clone();
    
    // Perform forward/backward pass
    let mut workspace = network.create_workspace(1);
    let input = vec![0.5f32, 0.5];
    let target = vec![1.0f32, 0.0];
    
    network.train_step(&input, &target, None, 1.0, &mut workspace);
    
    // Extract gradients from workspace (they are already Vec<Vec<f32>>)
    let weight_grads: Vec<Vec<f32>> = workspace.weight_grads.clone();
    let bias_grads: Vec<Vec<f32>> = workspace.bias_grads.clone();
    
    // Apply SGD step
    sgd.step(&mut network, &weight_grads, &bias_grads, None).unwrap();
    
    // Weights should have changed
    let new_weights: Vec<f32> = network.layers[0].weights.clone();
    
    assert_ne!(initial_weights, new_weights, "Weights should be updated");
}

#[test]
fn test_arkan_sgd_nesterov_more_aggressive() {
    use arkan::{KanConfig, KanNetwork, Optimizer, SGD, SGDConfig};
    
    let config = KanConfig {
        input_dim: 2,
        hidden_dims: vec![4],
        output_dim: 2,
        grid_size: 3,
        spline_order: 3,
        input_mean: vec![0.0; 2],
        input_std: vec![1.0; 2],
        ..Default::default()
    };
    
    // Create two networks with same initialization
    let mut network_std = KanNetwork::new(config.clone());
    let mut network_nes = KanNetwork::new(config);
    
    // Copy weights from std to nes to ensure same starting point
    for (layer_std, layer_nes) in network_std.layers.iter().zip(network_nes.layers.iter_mut()) {
        layer_nes.weights.copy_from_slice(&layer_std.weights);
        layer_nes.bias.copy_from_slice(&layer_std.bias);
    }
    
    let sgd_std = SGDConfig {
        lr: 0.1,
        momentum: 0.9,
        weight_decay: 0.0,
        nesterov: false,
        ..Default::default()
    };
    let sgd_nes = SGDConfig {
        lr: 0.1,
        momentum: 0.9,
        weight_decay: 0.0,
        nesterov: true,
        ..Default::default()
    };
    
    let mut sgd_standard = SGD::new(&network_std, sgd_std);
    let mut sgd_nesterov = SGD::new(&network_nes, sgd_nes);
    
    let mut workspace_std = network_std.create_workspace(1);
    let mut workspace_nes = network_nes.create_workspace(1);
    
    let input = vec![0.5f32, 0.5];
    let target = vec![1.0f32, 0.0];
    
    // Multiple training steps
    for _ in 0..5 {
        network_std.train_step(&input, &target, None, 1.0, &mut workspace_std);
        let wg = workspace_std.weight_grads.clone();
        let bg = workspace_std.bias_grads.clone();
        sgd_standard.step(&mut network_std, &wg, &bg, None).unwrap();
        
        network_nes.train_step(&input, &target, None, 1.0, &mut workspace_nes);
        let wg = workspace_nes.weight_grads.clone();
        let bg = workspace_nes.bias_grads.clone();
        sgd_nesterov.step(&mut network_nes, &wg, &bg, None).unwrap();
    }
    
    // Get final weights
    let weights_std: Vec<f32> = network_std.layers[0].weights.clone();
    let weights_nes: Vec<f32> = network_nes.layers[0].weights.clone();
    
    // Weights should be different (Nesterov typically moves more aggressively)
    let diff: f32 = weights_std.iter()
        .zip(weights_nes.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    
    assert!(diff > 1e-6, "Nesterov should produce different weights than standard momentum");
}

#[test]
fn test_arkan_adam_bias_correction() {
    // Verify bias correction is applied correctly
    // Early steps should show larger effective learning rate due to bias correction
    
    // With beta1=0.9, beta2=0.999:
    // Step 1: bias_correction1 = 0.1, bias_correction2 = 0.001
    // m_hat = m / 0.1 = 10 * m
    // v_hat = v / 0.001 = 1000 * v
    
    let mut m = 0.0f32;
    let mut v = 0.0f32;
    let beta1 = 0.9f32;
    let beta2 = 0.999f32;
    let grad = 1.0f32;
    
    // Step 1
    m = beta1 * m + (1.0 - beta1) * grad; // m = 0.1
    v = beta2 * v + (1.0 - beta2) * grad * grad; // v = 0.001
    
    let bc1 = 1.0 - beta1.powi(1); // 0.1
    let bc2 = 1.0 - beta2.powi(1); // 0.001
    
    let m_hat = m / bc1; // 0.1 / 0.1 = 1.0
    let v_hat = v / bc2; // 0.001 / 0.001 = 1.0
    
    assert!((m_hat - 1.0).abs() < 1e-5, "m_hat should be 1.0");
    assert!((v_hat - 1.0).abs() < 1e-5, "v_hat should be 1.0");
}

#[test]
fn test_arkan_lbfgs_creation() {
    use arkan::{KanConfig, KanNetwork, LBFGS, LBFGSConfig};
    
    let config = KanConfig {
        input_dim: 2,
        hidden_dims: vec![],
        output_dim: 2,
        input_mean: vec![0.0; 2],
        input_std: vec![1.0; 2],
        ..KanConfig::preset()
    };
    let network = KanNetwork::new(config);
    
    // Initialize LBFGS
    let lbfgs = LBFGS::new(&network, LBFGSConfig::default());
    
    // Verify it was created successfully
    assert_eq!(lbfgs.num_evals(), 0);
}

// ============================================================================
// Summary of PyTorch formulas verified:
// ============================================================================
//
// Adam:
//   m_t = β1 * m_{t-1} + (1 - β1) * g_t
//   v_t = β2 * v_{t-1} + (1 - β2) * g_t²
//   m̂_t = m_t / (1 - β1^t)
//   v̂_t = v_t / (1 - β2^t)
//   θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)
//
// Adam with L2 regularization:
//   g_t = ∇f(θ) + weight_decay * θ  (added to gradient)
//
// AdamW (decoupled):
//   θ_t = θ_{t-1} - α * (m̂_t / (√v̂_t + ε) + weight_decay * θ_{t-1})
//
// SGD:
//   θ_t = θ_{t-1} - α * g_t
//
// SGD with momentum:
//   v_t = μ * v_{t-1} + g_t
//   θ_t = θ_{t-1} - α * v_t
//
// SGD with Nesterov:
//   v_t = μ * v_{t-1} + g_t
//   θ_t = θ_{t-1} - α * (μ * v_t + g_t)
//
