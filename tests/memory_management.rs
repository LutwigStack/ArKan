//! Memory Management Correctness Tests
//!
//! Tests for GPU memory management: async downloads, large tensors,
//! alignment requirements, and stress testing.
//!
//! Closes dead zones from FUNCTIONALITY_AUDIT.md:
//! - Async download correctness
//! - Large tensor handling (100MB+)
//! - Alignment requirements
//!
//! Run with: cargo test --features gpu --test memory_management -- --ignored

#![cfg(feature = "gpu")]

use arkan::gpu::{GpuTensor, WgpuBackend, WgpuOptions};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

// =============================================================================
// ASYNC DOWNLOAD TESTS
// =============================================================================

/// Test that async download returns correct data.
/// This is the critical dead zone - function exists but no test!
#[test]
#[ignore = "Requires GPU"]
fn test_async_download_correctness() {
    let backend = WgpuBackend::init(WgpuOptions::default()).expect("GPU init");

    // Create tensor with known data
    let data: Vec<f32> = (0..1024).map(|i| i as f32 * 0.5).collect();
    let tensor = GpuTensor::upload(&backend.device, &backend.queue, &data, vec![1024])
        .expect("Upload failed");

    // Use channel to receive async result
    let (tx, rx) = std::sync::mpsc::channel();

    tensor.download_async(&backend.device, &backend.queue, move |result| {
        tx.send(result).expect("Send failed");
    });

    // Poll device until callback fires
    let timeout = Instant::now();
    loop {
        backend.device.poll(wgpu::Maintain::Poll);

        if let Ok(result) = rx.try_recv() {
            let downloaded = result.expect("Async download failed");

            // Verify data integrity
            assert_eq!(downloaded.len(), data.len(), "Length mismatch");
            for (i, (&expected, &got)) in data.iter().zip(downloaded.iter()).enumerate() {
                assert_eq!(
                    expected.to_bits(),
                    got.to_bits(),
                    "Mismatch at index {}: expected {}, got {}",
                    i,
                    expected,
                    got
                );
            }

            println!(
                "✅ Async download returned correct data ({} elements)",
                data.len()
            );
            return;
        }

        if timeout.elapsed() > Duration::from_secs(5) {
            panic!("Async download timeout after 5 seconds");
        }

        std::thread::sleep(Duration::from_millis(1));
    }
}

/// Test async download with multiple concurrent downloads.
#[test]
#[ignore = "Requires GPU"]
fn test_async_download_multiple_concurrent() {
    let backend = WgpuBackend::init(WgpuOptions::default()).expect("GPU init");

    // Create multiple tensors with different data
    let tensors: Vec<_> = (0..5)
        .map(|i| {
            let data: Vec<f32> = (0..256).map(|j| (i * 1000 + j) as f32).collect();
            let tensor = GpuTensor::upload(&backend.device, &backend.queue, &data, vec![256])
                .expect("Upload failed");
            (tensor, data)
        })
        .collect();

    // Start all async downloads
    let results: Vec<_> = tensors
        .iter()
        .map(|(tensor, _)| {
            let (tx, rx) = std::sync::mpsc::channel();
            tensor.download_async(&backend.device, &backend.queue, move |result| {
                tx.send(result).expect("Send failed");
            });
            rx
        })
        .collect();

    // Poll and collect results
    let timeout = Instant::now();
    let mut collected = vec![None; 5];

    while collected.iter().any(|r| r.is_none()) {
        backend.device.poll(wgpu::Maintain::Poll);

        for (i, rx) in results.iter().enumerate() {
            if collected[i].is_none() {
                if let Ok(result) = rx.try_recv() {
                    collected[i] = Some(result.expect("Download failed"));
                }
            }
        }

        if timeout.elapsed() > Duration::from_secs(10) {
            panic!("Multiple async downloads timeout");
        }

        std::thread::sleep(Duration::from_millis(1));
    }

    // Verify all results
    for (i, ((_, expected), got)) in tensors.iter().zip(collected.iter()).enumerate() {
        let got = got.as_ref().unwrap();
        assert_eq!(expected.len(), got.len(), "Tensor {} length mismatch", i);
        for (j, (&e, &g)) in expected.iter().zip(got.iter()).enumerate() {
            assert_eq!(
                e.to_bits(),
                g.to_bits(),
                "Tensor {} element {} mismatch",
                i,
                j
            );
        }
    }

    println!("✅ Multiple concurrent async downloads correct (5 tensors)");
}

/// Test async download vs sync download produce same results.
#[test]
#[ignore = "Requires GPU"]
fn test_async_download_vs_sync_parity() {
    let backend = WgpuBackend::init(WgpuOptions::default()).expect("GPU init");

    let data: Vec<f32> = (0..4096).map(|i| (i as f32 * 0.123).sin()).collect();
    let tensor = GpuTensor::upload(&backend.device, &backend.queue, &data, vec![4096])
        .expect("Upload failed");

    // Sync download
    let sync_result = tensor
        .download(&backend.device, &backend.queue)
        .expect("Sync download failed");

    // Async download
    let (tx, rx) = std::sync::mpsc::channel();
    tensor.download_async(&backend.device, &backend.queue, move |result| {
        tx.send(result).expect("Send failed");
    });

    let timeout = Instant::now();
    loop {
        backend.device.poll(wgpu::Maintain::Poll);

        if let Ok(result) = rx.try_recv() {
            let async_result = result.expect("Async download failed");

            // Must be bit-identical
            assert_eq!(sync_result.len(), async_result.len());
            for (i, (&s, &a)) in sync_result.iter().zip(async_result.iter()).enumerate() {
                assert_eq!(
                    s.to_bits(),
                    a.to_bits(),
                    "Sync vs async mismatch at {}: sync={}, async={}",
                    i,
                    s,
                    a
                );
            }

            println!("✅ Async download matches sync download (4096 elements)");
            return;
        }

        if timeout.elapsed() > Duration::from_secs(5) {
            panic!("Async download timeout");
        }

        std::thread::sleep(Duration::from_millis(1));
    }
}

/// Test async download callback is called exactly once.
#[test]
#[ignore = "Requires GPU"]
fn test_async_download_callback_called_once() {
    let backend = WgpuBackend::init(WgpuOptions::default()).expect("GPU init");

    let data = vec![1.0f32; 128];
    let tensor = GpuTensor::upload(&backend.device, &backend.queue, &data, vec![128])
        .expect("Upload failed");

    let call_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let count_clone = Arc::clone(&call_count);
    let completed = Arc::new(AtomicBool::new(false));
    let completed_clone = Arc::clone(&completed);

    tensor.download_async(&backend.device, &backend.queue, move |_result| {
        count_clone.fetch_add(1, Ordering::SeqCst);
        completed_clone.store(true, Ordering::SeqCst);
    });

    // Poll until completed
    let timeout = Instant::now();
    while !completed.load(Ordering::SeqCst) {
        backend.device.poll(wgpu::Maintain::Poll);
        if timeout.elapsed() > Duration::from_secs(5) {
            panic!("Callback not called within timeout");
        }
        std::thread::sleep(Duration::from_millis(1));
    }

    // Wait a bit more to ensure no double-callback
    std::thread::sleep(Duration::from_millis(100));
    backend.device.poll(wgpu::Maintain::Poll);

    let count = call_count.load(Ordering::SeqCst);
    assert_eq!(
        count, 1,
        "Callback should be called exactly once, got {}",
        count
    );

    println!("✅ Async download callback called exactly once");
}

// =============================================================================
// LARGE TENSOR STRESS TESTS
// =============================================================================

/// Test tensor upload/download with 10MB data.
#[test]
#[ignore = "Requires GPU"]
fn test_large_tensor_10mb() {
    let backend = WgpuBackend::init(WgpuOptions::default()).expect("GPU init");

    // 10MB = 10 * 1024 * 1024 bytes = 2.5M f32 elements
    let num_elements = 10 * 1024 * 1024 / 4;
    let data: Vec<f32> = (0..num_elements).map(|i| (i as f32).sin()).collect();

    let start = Instant::now();
    let tensor = GpuTensor::upload(&backend.device, &backend.queue, &data, vec![num_elements])
        .expect("Upload failed");
    let upload_time = start.elapsed();

    let start = Instant::now();
    let downloaded = tensor
        .download(&backend.device, &backend.queue)
        .expect("Download failed");
    let download_time = start.elapsed();

    // Verify data integrity (spot check)
    assert_eq!(downloaded.len(), data.len());
    for i in (0..data.len()).step_by(10000) {
        assert_eq!(
            data[i].to_bits(),
            downloaded[i].to_bits(),
            "Mismatch at index {}",
            i
        );
    }

    println!(
        "✅ 10MB tensor: upload={:?}, download={:?}",
        upload_time, download_time
    );
}

/// Test tensor upload/download with 100MB data.
#[test]
#[ignore = "Requires GPU - may OOM on low-VRAM GPUs"]
fn test_large_tensor_100mb() {
    let backend = WgpuBackend::init(WgpuOptions::default()).expect("GPU init");

    // 100MB = 100 * 1024 * 1024 bytes = 25M f32 elements
    let num_elements = 100 * 1024 * 1024 / 4;
    let data: Vec<f32> = (0..num_elements)
        .map(|i| (i as f32 * 0.0001).cos())
        .collect();

    let start = Instant::now();
    let tensor = GpuTensor::upload(&backend.device, &backend.queue, &data, vec![num_elements])
        .expect("Upload failed");
    let upload_time = start.elapsed();

    let start = Instant::now();
    let downloaded = tensor
        .download(&backend.device, &backend.queue)
        .expect("Download failed");
    let download_time = start.elapsed();

    // Verify data integrity (spot check every 100k elements)
    assert_eq!(downloaded.len(), data.len());
    for i in (0..data.len()).step_by(100000) {
        assert_eq!(
            data[i].to_bits(),
            downloaded[i].to_bits(),
            "Mismatch at index {}",
            i
        );
    }

    let throughput_upload = 100.0 / upload_time.as_secs_f64();
    let throughput_download = 100.0 / download_time.as_secs_f64();

    println!(
        "✅ 100MB tensor: upload={:?} ({:.1} MB/s), download={:?} ({:.1} MB/s)",
        upload_time, throughput_upload, download_time, throughput_download
    );
}

/// Test tensor upload/download with maximum allowed buffer size.
/// wgpu has a limit of 256MB per buffer, so we test near that limit.
#[test]
#[ignore = "Requires GPU with 512MB+ VRAM"]
fn test_large_tensor_near_max_buffer() {
    let backend = WgpuBackend::init(WgpuOptions::default()).expect("GPU init");

    // wgpu max buffer size is typically 256MB = 64M f32 elements
    // Test at 200MB = 50M f32 elements to stay safely below limit
    let num_elements = 200 * 1024 * 1024 / 4;

    println!("Allocating 200MB CPU buffer...");
    let data: Vec<f32> = (0..num_elements).map(|i| (i % 1000) as f32).collect();

    println!("Uploading to GPU...");
    let start = Instant::now();
    let tensor = GpuTensor::upload(&backend.device, &backend.queue, &data, vec![num_elements])
        .expect("Upload failed");
    let upload_time = start.elapsed();

    println!("Downloading from GPU...");
    let start = Instant::now();
    let downloaded = tensor
        .download(&backend.device, &backend.queue)
        .expect("Download failed");
    let download_time = start.elapsed();

    // Verify data integrity (sample check)
    assert_eq!(downloaded.len(), data.len());
    let samples = [0, 1000, 1000000, 25000000, num_elements - 1];
    for &i in &samples {
        assert_eq!(
            data[i].to_bits(),
            downloaded[i].to_bits(),
            "Mismatch at index {}",
            i
        );
    }

    println!(
        "✅ 200MB tensor: upload={:?}, download={:?}",
        upload_time, download_time
    );
}

/// Document max buffer size - with use_adapter_limits=true (default),
/// this should be much larger than 256MB on desktop GPUs.
#[test]
#[ignore = "Requires GPU"]
fn test_max_buffer_size_documented() {
    let backend = WgpuBackend::init(WgpuOptions::default()).expect("GPU init");

    // Query device limits
    let limits = backend.device.limits();
    let max_buffer_size = limits.max_buffer_size;

    println!(
        "ℹ️  Device max_buffer_size = {} bytes ({} MB)",
        max_buffer_size,
        max_buffer_size / 1024 / 1024
    );

    // With use_adapter_limits=true (default), desktop GPUs should have much more
    // than the default 256MB. Integrated/mobile might still have 256MB.
    if max_buffer_size > 256 * 1024 * 1024 {
        println!("   ✓ Using full adapter limits (>256MB)");
    } else {
        println!("   ⚠️ Using default limits (256MB) - integrated/mobile GPU?");
    }

    // Minimum requirement: at least 256MB for any GPU
    assert!(
        max_buffer_size >= 256 * 1024 * 1024,
        "Expected max_buffer_size >= 256MB, got {} bytes",
        max_buffer_size
    );

    println!("✅ Max buffer size: {} MB", max_buffer_size / 1024 / 1024);
}

/// Test tensor upload/download with 500MB data.
/// With use_adapter_limits=true (default), this should work on desktop GPUs.
#[test]
#[ignore = "Requires GPU with sufficient VRAM"]
fn test_large_tensor_500mb() {
    let backend = WgpuBackend::init(WgpuOptions::default()).expect("GPU init");

    // Check device limits first
    let limits = backend.device.limits();
    let max_buffer_size = limits.max_buffer_size;
    let required_size = 500 * 1024 * 1024u64;

    if max_buffer_size < required_size {
        println!(
            "⚠️  SKIPPED: wgpu max_buffer_size ({} MB) < required ({} MB)",
            max_buffer_size / 1024 / 1024,
            required_size / 1024 / 1024
        );
        println!("   This is a known wgpu limitation, not a bug");
        return;
    }

    // 500MB = 500 * 1024 * 1024 bytes = 125M f32 elements
    let num_elements = 500 * 1024 * 1024 / 4;

    println!("Allocating 500MB CPU buffer...");
    let data: Vec<f32> = (0..num_elements).map(|i| (i % 1000) as f32).collect();

    println!("Uploading to GPU...");
    let start = Instant::now();
    let tensor = GpuTensor::upload(&backend.device, &backend.queue, &data, vec![num_elements])
        .expect("Upload failed");
    let upload_time = start.elapsed();

    println!("Downloading from GPU...");
    let start = Instant::now();
    let downloaded = tensor
        .download(&backend.device, &backend.queue)
        .expect("Download failed");
    let download_time = start.elapsed();

    // Verify data integrity (sample check)
    assert_eq!(downloaded.len(), data.len());
    let samples = [0, 1000, 1000000, 50000000, num_elements - 1];
    for &i in &samples {
        assert_eq!(
            data[i].to_bits(),
            downloaded[i].to_bits(),
            "Mismatch at index {}",
            i
        );
    }

    println!(
        "✅ 500MB tensor: upload={:?}, download={:?}",
        upload_time, download_time
    );
}

/// Test tensor upload/download with 2GB data.
/// This tests real VRAM limits on RTX 4070 SUPER (12GB).
#[test]
#[ignore = "Requires GPU with 12+ GB VRAM (RTX 4070 SUPER)"]
fn test_large_tensor_2gb() {
    let backend = WgpuBackend::init(WgpuOptions::default()).expect("GPU init");

    // 2GB = 2 * 1024 * 1024 * 1024 bytes = 512M f32 elements
    let num_elements = 2 * 1024 * 1024 * 1024 / 4;
    let required_bytes = (num_elements * 4) as u64;

    println!(
        "Testing 2GB tensor ({} MB, {} elements)...",
        required_bytes / 1024 / 1024,
        num_elements
    );

    println!("Allocating 2GB CPU buffer...");
    let start = Instant::now();
    let data: Vec<f32> = (0..num_elements).map(|i| (i % 10000) as f32).collect();
    println!("  CPU allocation: {:?}", start.elapsed());

    println!("Uploading to GPU...");
    let start = Instant::now();
    let tensor = GpuTensor::upload(&backend.device, &backend.queue, &data, vec![num_elements])
        .expect("Upload failed");
    let upload_time = start.elapsed();
    println!(
        "  Upload: {:?} ({:.1} GB/s)",
        upload_time,
        (required_bytes as f64 / 1e9) / upload_time.as_secs_f64()
    );

    println!("Downloading from GPU...");
    let start = Instant::now();
    let downloaded = tensor
        .download(&backend.device, &backend.queue)
        .expect("Download failed");
    let download_time = start.elapsed();
    println!(
        "  Download: {:?} ({:.1} GB/s)",
        download_time,
        (required_bytes as f64 / 1e9) / download_time.as_secs_f64()
    );

    // Verify data integrity (sample check)
    assert_eq!(downloaded.len(), data.len());
    let samples = [
        0,
        1000,
        1_000_000,
        100_000_000,
        500_000_000,
        num_elements - 1,
    ];
    for &i in &samples {
        assert_eq!(
            data[i].to_bits(),
            downloaded[i].to_bits(),
            "Mismatch at index {}",
            i
        );
    }

    println!(
        "✅ 2GB tensor: upload={:?}, download={:?}",
        upload_time, download_time
    );
}

/// Test tensor upload/download with 3GB data using custom VRAM limit.
/// This tests ~1/4 of RTX 4070 SUPER VRAM (12GB).
///
/// Uses `WgpuOptions::with_max_vram(8)` to set 8GB limit.
#[test]
#[ignore = "Requires GPU with 12+ GB VRAM and 8+ GB free RAM"]
fn test_large_tensor_3gb() {
    // Use 8GB max VRAM limit for RTX 4070 SUPER (12GB)
    let backend = WgpuBackend::init(WgpuOptions::with_max_vram(8)).expect("GPU init");

    println!(
        "ℹ️  Backend max_vram_alloc = {} GB",
        backend.max_vram_alloc() / 1024 / 1024 / 1024
    );

    // 3GB = 3 * 1024 * 1024 * 1024 bytes = 768M f32 elements
    let num_elements: usize = 3 * 1024 * 1024 * 1024 / 4;
    let required_bytes = (num_elements * 4) as u64;

    println!(
        "Testing 3GB tensor ({} MB, {} elements)...",
        required_bytes / 1024 / 1024,
        num_elements
    );

    println!("Allocating 3GB CPU buffer...");
    let start = Instant::now();
    let data: Vec<f32> = (0..num_elements).map(|i| (i % 10000) as f32).collect();
    println!("  CPU allocation: {:?}", start.elapsed());

    println!("Uploading to GPU with custom limit...");
    let start = Instant::now();
    // Use upload_with_limit with backend's configured max
    let tensor = GpuTensor::upload_with_limit(
        &backend.device,
        &data,
        vec![num_elements],
        Some(backend.max_vram_alloc()),
    )
    .expect("Upload failed");
    let upload_time = start.elapsed();
    println!(
        "  Upload: {:?} ({:.1} GB/s)",
        upload_time,
        (required_bytes as f64 / 1e9) / upload_time.as_secs_f64()
    );

    println!("Downloading from GPU...");
    let start = Instant::now();
    let downloaded = tensor
        .download(&backend.device, &backend.queue)
        .expect("Download failed");
    let download_time = start.elapsed();
    println!(
        "  Download: {:?} ({:.1} GB/s)",
        download_time,
        (required_bytes as f64 / 1e9) / download_time.as_secs_f64()
    );

    // Verify data integrity (sample check)
    assert_eq!(downloaded.len(), data.len());
    let samples = [
        0,
        1000,
        1_000_000,
        500_000_000,
        700_000_000,
        num_elements - 1,
    ];
    for &i in &samples {
        assert_eq!(
            data[i].to_bits(),
            downloaded[i].to_bits(),
            "Mismatch at index {}",
            i
        );
    }

    println!(
        "✅ 3GB tensor: upload={:?}, download={:?}",
        upload_time, download_time
    );
}

/// Test VramLimit::Percent configuration.
/// Uses 30% of device max_buffer_size as recommended.
#[test]
#[ignore = "Requires GPU"]
fn test_vram_limit_percent() {
    use arkan::gpu::VramLimit;

    // Test with 30% limit (recommended)
    let backend = WgpuBackend::init(WgpuOptions::with_max_vram_percent(30)).expect("GPU init");

    let device_max = backend.limits().max_buffer_size;
    // Same calculation as in VramLimit::resolve (divide first to avoid overflow)
    let expected_30_percent = (device_max / 100) * 30;

    println!(
        "ℹ️  Device max_buffer_size = {} MB",
        device_max / 1024 / 1024
    );
    println!(
        "ℹ️  30% limit = {} MB",
        backend.max_vram_alloc() / 1024 / 1024
    );
    println!("ℹ️  Expected = {} MB", expected_30_percent / 1024 / 1024);

    assert_eq!(backend.max_vram_alloc(), expected_30_percent);

    // Test VramLimit::resolve directly (with small values to avoid overflow concerns)
    assert_eq!(VramLimit::Percent(30).resolve(1000), 300);
    assert_eq!(VramLimit::Percent(100).resolve(1000), 1000);
    assert_eq!(VramLimit::Percent(0).resolve(1000), 0);
    assert_eq!(
        VramLimit::Gigabytes(2).resolve(1000),
        2 * 1024 * 1024 * 1024
    );
    assert_eq!(VramLimit::Bytes(12345).resolve(1000), 12345);
    assert_eq!(VramLimit::Unlimited.resolve(1000), 1000);

    println!("✅ VramLimit::Percent(30) works correctly");
}

/// Test that 30% limit allows reasonable tensor size.
/// RTX 4070 SUPER (12GB): 30% of max (~unlimited) ≈ ~3.6GB.
#[test]
#[ignore = "Requires GPU with 12+ GB VRAM"]
fn test_large_tensor_with_percent_limit() {
    // Use 30% limit - safe for most operations
    let backend = WgpuBackend::init(WgpuOptions::with_max_vram_percent(30)).expect("GPU init");

    let max_alloc = backend.max_vram_alloc();
    println!(
        "ℹ️  30% limit = {} MB ({} GB)",
        max_alloc / 1024 / 1024,
        max_alloc / 1024 / 1024 / 1024
    );

    // Try allocating 1GB tensor (should work within 30% of 12GB)
    let num_elements: usize = 1024 * 1024 * 1024 / 4; // 1GB = 256M elements
    let required_bytes = (num_elements * 4) as u64;

    if required_bytes > max_alloc {
        println!(
            "⚠️  SKIPPED: 1GB > 30% limit ({} MB)",
            max_alloc / 1024 / 1024
        );
        return;
    }

    println!("Allocating 1GB tensor...");
    let data: Vec<f32> = (0..num_elements).map(|i| (i % 10000) as f32).collect();

    let start = Instant::now();
    let tensor =
        GpuTensor::upload_with_limit(&backend.device, &data, vec![num_elements], Some(max_alloc))
            .expect("Upload failed");
    let upload_time = start.elapsed();

    let start = Instant::now();
    let downloaded = tensor
        .download(&backend.device, &backend.queue)
        .expect("Download failed");
    let download_time = start.elapsed();

    assert_eq!(downloaded.len(), data.len());
    assert_eq!(downloaded[0].to_bits(), data[0].to_bits());
    assert_eq!(
        downloaded[num_elements - 1].to_bits(),
        data[num_elements - 1].to_bits()
    );

    println!(
        "✅ 1GB tensor with 30% limit: upload={:?}, download={:?}",
        upload_time, download_time
    );
}

/// Test async download with large tensor (100MB).
#[test]
#[ignore = "Requires GPU - may OOM on low-VRAM GPUs"]
fn test_async_download_large_tensor() {
    let backend = WgpuBackend::init(WgpuOptions::default()).expect("GPU init");

    // 100MB
    let num_elements = 100 * 1024 * 1024 / 4;
    let data: Vec<f32> = (0..num_elements).map(|i| (i % 10000) as f32).collect();

    let tensor = GpuTensor::upload(&backend.device, &backend.queue, &data, vec![num_elements])
        .expect("Upload failed");

    let (tx, rx) = std::sync::mpsc::channel();
    let start = Instant::now();

    tensor.download_async(&backend.device, &backend.queue, move |result| {
        tx.send(result).expect("Send failed");
    });

    // Poll with timeout
    let timeout = Instant::now();
    loop {
        backend.device.poll(wgpu::Maintain::Poll);

        if let Ok(result) = rx.try_recv() {
            let async_time = start.elapsed();
            let downloaded = result.expect("Async download failed");

            // Spot check
            assert_eq!(downloaded.len(), data.len());
            for i in (0..data.len()).step_by(100000) {
                assert_eq!(data[i].to_bits(), downloaded[i].to_bits());
            }

            println!("✅ 100MB async download completed in {:?}", async_time);
            return;
        }

        if timeout.elapsed() > Duration::from_secs(30) {
            panic!("100MB async download timeout after 30 seconds");
        }

        std::thread::sleep(Duration::from_millis(10));
    }
}

// =============================================================================
// ALIGNMENT TESTS
// =============================================================================

/// Test that tensors with non-4-byte-aligned element counts work correctly.
/// wgpu requires 4-byte alignment for buffer copies.
#[test]
#[ignore = "Requires GPU"]
fn test_alignment_odd_element_counts() {
    let backend = WgpuBackend::init(WgpuOptions::default()).expect("GPU init");

    // Test various odd element counts that might cause alignment issues
    let test_sizes = [1, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 127, 255, 1023];

    for &size in &test_sizes {
        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();

        let tensor = GpuTensor::upload(&backend.device, &backend.queue, &data, vec![size])
            .expect(&format!("Upload failed for size {}", size));

        let downloaded = tensor
            .download(&backend.device, &backend.queue)
            .expect(&format!("Download failed for size {}", size));

        assert_eq!(
            downloaded.len(),
            data.len(),
            "Size {} length mismatch",
            size
        );
        for (i, (&e, &g)) in data.iter().zip(downloaded.iter()).enumerate() {
            assert_eq!(
                e.to_bits(),
                g.to_bits(),
                "Size {} mismatch at index {}",
                size,
                i
            );
        }
    }

    println!("✅ All odd element counts work correctly: {:?}", test_sizes);
}

/// Test 2D tensor shapes with non-aligned dimensions.
#[test]
#[ignore = "Requires GPU"]
fn test_alignment_2d_shapes() {
    let backend = WgpuBackend::init(WgpuOptions::default()).expect("GPU init");

    // Test shapes where total elements or row size might cause issues
    let test_shapes: Vec<(usize, usize)> = vec![
        (1, 1),
        (1, 3),
        (3, 1),
        (3, 3),
        (7, 11),
        (11, 7),
        (13, 17),
        (31, 33),
        (127, 129),
    ];

    for (rows, cols) in test_shapes {
        let total = rows * cols;
        let data: Vec<f32> = (0..total).map(|i| i as f32).collect();

        let tensor = GpuTensor::upload(&backend.device, &backend.queue, &data, vec![rows, cols])
            .expect(&format!("Upload failed for shape [{}, {}]", rows, cols));

        let downloaded = tensor
            .download(&backend.device, &backend.queue)
            .expect(&format!("Download failed for shape [{}, {}]", rows, cols));

        assert_eq!(downloaded.len(), data.len());
        for (i, (&e, &g)) in data.iter().zip(downloaded.iter()).enumerate() {
            assert_eq!(
                e.to_bits(),
                g.to_bits(),
                "Shape [{}, {}] mismatch at index {}",
                rows,
                cols,
                i
            );
        }
    }

    println!("✅ All 2D shapes work correctly");
}

/// Test that f32 alignment is maintained (wgpu requires 4-byte alignment).
#[test]
#[ignore = "Requires GPU"]
fn test_alignment_f32_natural() {
    let backend = WgpuBackend::init(WgpuOptions::default()).expect("GPU init");

    // f32 is 4 bytes, so any f32 slice is naturally aligned
    // This test verifies the buffer creation respects this

    let sizes = [1, 2, 4, 8, 15, 16, 17, 63, 64, 65, 255, 256, 257];

    for &size in &sizes {
        let data: Vec<f32> = (0..size).map(|i| i as f32 * 1.5).collect();

        let tensor = GpuTensor::upload(&backend.device, &backend.queue, &data, vec![size])
            .expect(&format!("Upload failed for size {}", size));

        // Verify buffer size is correct (should be size * 4 bytes)
        let expected_bytes = size * 4;
        assert!(
            tensor.capacity_bytes as usize >= expected_bytes,
            "Size {}: capacity {} < expected {}",
            size,
            tensor.capacity_bytes,
            expected_bytes
        );

        let downloaded = tensor
            .download(&backend.device, &backend.queue)
            .expect(&format!("Download failed for size {}", size));

        assert_eq!(downloaded, data, "Size {} data mismatch", size);
    }

    println!("✅ f32 natural alignment maintained for all sizes");
}

// =============================================================================
// STRESS TESTS
// =============================================================================

/// Stress test: many small tensor allocations.
#[test]
#[ignore = "Requires GPU"]
fn test_stress_many_small_tensors() {
    let backend = WgpuBackend::init(WgpuOptions::default()).expect("GPU init");

    let num_tensors = 1000;
    let tensor_size = 256;

    let start = Instant::now();
    let mut tensors = Vec::with_capacity(num_tensors);

    for i in 0..num_tensors {
        let data: Vec<f32> = (0..tensor_size)
            .map(|j| (i * tensor_size + j) as f32)
            .collect();
        let tensor = GpuTensor::upload(&backend.device, &backend.queue, &data, vec![tensor_size])
            .expect(&format!("Upload {} failed", i));
        tensors.push(tensor);
    }

    let alloc_time = start.elapsed();

    // Verify random samples
    let samples = [0, 100, 500, 999];
    for &i in &samples {
        let expected: Vec<f32> = (0..tensor_size)
            .map(|j| (i * tensor_size + j) as f32)
            .collect();
        let downloaded = tensors[i]
            .download(&backend.device, &backend.queue)
            .expect(&format!("Download {} failed", i));
        assert_eq!(downloaded, expected, "Tensor {} data mismatch", i);
    }

    // Drop all tensors
    drop(tensors);

    println!(
        "✅ {} small tensors allocated/verified in {:?}",
        num_tensors, alloc_time
    );
}

/// Stress test: rapid upload/download cycles.
#[test]
#[ignore = "Requires GPU"]
fn test_stress_rapid_upload_download() {
    let backend = WgpuBackend::init(WgpuOptions::default()).expect("GPU init");

    let iterations = 100;
    let tensor_size = 10000;

    let start = Instant::now();

    for i in 0..iterations {
        let data: Vec<f32> = (0..tensor_size).map(|j| (i * j) as f32).collect();

        let tensor = GpuTensor::upload(&backend.device, &backend.queue, &data, vec![tensor_size])
            .expect("Upload failed");

        let downloaded = tensor
            .download(&backend.device, &backend.queue)
            .expect("Download failed");

        // Spot check
        assert_eq!(downloaded[0].to_bits(), data[0].to_bits());
        assert_eq!(
            downloaded[tensor_size / 2].to_bits(),
            data[tensor_size / 2].to_bits()
        );
        assert_eq!(
            downloaded[tensor_size - 1].to_bits(),
            data[tensor_size - 1].to_bits()
        );
    }

    let total_time = start.elapsed();
    let per_iteration = total_time / iterations as u32;

    println!(
        "✅ {} upload/download cycles in {:?} ({:?}/iter)",
        iterations, total_time, per_iteration
    );
}

/// Stress test: mixed sync and async downloads.
#[test]
#[ignore = "Requires GPU"]
fn test_stress_mixed_sync_async() {
    let backend = WgpuBackend::init(WgpuOptions::default()).expect("GPU init");

    let num_ops = 50;
    let tensor_size = 1024;

    let start = Instant::now();

    for i in 0..num_ops {
        let data: Vec<f32> = (0..tensor_size).map(|j| (i * j) as f32).collect();

        let tensor = GpuTensor::upload(&backend.device, &backend.queue, &data, vec![tensor_size])
            .expect("Upload failed");

        if i % 2 == 0 {
            // Sync download
            let downloaded = tensor
                .download(&backend.device, &backend.queue)
                .expect("Sync download failed");
            assert_eq!(downloaded[0].to_bits(), data[0].to_bits());
        } else {
            // Async download
            let (tx, rx) = std::sync::mpsc::channel();
            let data_clone = data.clone();

            tensor.download_async(&backend.device, &backend.queue, move |result| {
                let downloaded = result.expect("Async download failed");
                assert_eq!(downloaded[0].to_bits(), data_clone[0].to_bits());
                tx.send(()).expect("Send failed");
            });

            // Poll until complete
            let timeout = Instant::now();
            loop {
                backend.device.poll(wgpu::Maintain::Poll);
                if rx.try_recv().is_ok() {
                    break;
                }
                if timeout.elapsed() > Duration::from_secs(5) {
                    panic!("Async download {} timeout", i);
                }
                std::thread::sleep(Duration::from_millis(1));
            }
        }
    }

    let total_time = start.elapsed();

    println!(
        "✅ {} mixed sync/async operations in {:?}",
        num_ops, total_time
    );
}

// =============================================================================
// EDGE CASES
// =============================================================================

/// Test single element tensor.
#[test]
#[ignore = "Requires GPU"]
fn test_single_element_tensor() {
    let backend = WgpuBackend::init(WgpuOptions::default()).expect("GPU init");

    let data = vec![42.5f32];
    let tensor =
        GpuTensor::upload(&backend.device, &backend.queue, &data, vec![1]).expect("Upload failed");

    let downloaded = tensor
        .download(&backend.device, &backend.queue)
        .expect("Download failed");

    assert_eq!(downloaded.len(), 1);
    assert_eq!(downloaded[0], 42.5f32);

    println!("✅ Single element tensor works");
}

/// Test tensor with special float values.
#[test]
#[ignore = "Requires GPU"]
fn test_special_float_values() {
    let backend = WgpuBackend::init(WgpuOptions::default()).expect("GPU init");

    let data = vec![
        0.0f32,
        -0.0f32,
        f32::MIN,
        f32::MAX,
        f32::MIN_POSITIVE,
        f32::EPSILON,
        std::f32::consts::PI,
        std::f32::consts::E,
        1.0e-38f32, // Near subnormal
        1.0e38f32,  // Large but valid
    ];

    let tensor = GpuTensor::upload(&backend.device, &backend.queue, &data, vec![data.len()])
        .expect("Upload failed");

    let downloaded = tensor
        .download(&backend.device, &backend.queue)
        .expect("Download failed");

    for (i, (&expected, &got)) in data.iter().zip(downloaded.iter()).enumerate() {
        assert_eq!(
            expected.to_bits(),
            got.to_bits(),
            "Special float {} mismatch: expected {} ({:#x}), got {} ({:#x})",
            i,
            expected,
            expected.to_bits(),
            got,
            got.to_bits()
        );
    }

    println!("✅ Special float values preserved correctly");
}

/// Test that NaN and Inf values are preserved (but may cause issues in computation).
#[test]
#[ignore = "Requires GPU"]
fn test_nan_inf_preservation() {
    let backend = WgpuBackend::init(WgpuOptions::default()).expect("GPU init");

    let data = vec![
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        1.0f32, // Normal value for comparison
    ];

    let tensor =
        GpuTensor::upload(&backend.device, &backend.queue, &data, vec![4]).expect("Upload failed");

    let downloaded = tensor
        .download(&backend.device, &backend.queue)
        .expect("Download failed");

    assert!(downloaded[0].is_nan(), "NaN not preserved");
    assert!(
        downloaded[1].is_infinite() && downloaded[1].is_sign_positive(),
        "Inf not preserved"
    );
    assert!(
        downloaded[2].is_infinite() && downloaded[2].is_sign_negative(),
        "-Inf not preserved"
    );
    assert_eq!(downloaded[3], 1.0f32, "Normal value corrupted");

    println!("✅ NaN and Inf values preserved in memory transfer");
}

// =============================================================================
// GPU WORKSPACE VRAM LIMIT TESTS
// =============================================================================

/// Test that GpuWorkspace respects max_vram_alloc from backend.
#[test]
#[ignore = "Requires GPU"]
fn test_workspace_inherits_vram_limit() {
    use arkan::gpu::GpuNetwork;
    use arkan::{KanConfig, KanNetwork};

    // Create backend with 4GB limit
    let backend = WgpuBackend::init(WgpuOptions::with_max_vram(4)).expect("GPU init");

    let config = KanConfig::preset();
    let cpu_network = KanNetwork::new(config.clone());

    let gpu_network = GpuNetwork::from_cpu(&backend, &cpu_network).expect("GPU network");

    // Workspace created from gpu_network should inherit the limit
    let workspace = gpu_network.create_workspace(64).expect("Workspace");

    let expected_limit = 4 * 1024 * 1024 * 1024u64;
    assert_eq!(workspace.max_vram_alloc(), expected_limit);
    assert_eq!(gpu_network.max_vram_alloc(), expected_limit);

    println!("✅ GpuWorkspace inherits max_vram_alloc from GpuNetwork (4GB)");
}

/// Test that GpuWorkspace::new_with_limit works correctly.
#[test]
#[ignore = "Requires GPU"]
fn test_workspace_new_with_limit() {
    use arkan::gpu::GpuWorkspace;

    let backend = WgpuBackend::init(WgpuOptions::default()).expect("GPU init");

    // Create workspace with custom 8GB limit
    let limit_8gb = 8 * 1024 * 1024 * 1024u64;
    let workspace =
        GpuWorkspace::new_with_limit(&backend.device, 64, 128, 64, limit_8gb).expect("Workspace");

    assert_eq!(workspace.max_vram_alloc(), limit_8gb);

    println!("✅ GpuWorkspace::new_with_limit(8GB) works correctly");
}
