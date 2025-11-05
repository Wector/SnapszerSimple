"""Test GPU setup and JAX device configuration."""

import jax
import jax.numpy as jnp


def test_gpu_setup():
    """Check JAX GPU configuration."""
    print("="*60)
    print("JAX GPU SETUP TEST")
    print("="*60)

    # Check JAX version
    print(f"\nJAX version: {jax.__version__}")

    # Check available devices
    print(f"\nAvailable devices:")
    devices = jax.devices()
    for i, device in enumerate(devices):
        print(f"  [{i}] {device}")

    # Check default backend
    print(f"\nDefault backend: {jax.default_backend()}")

    # Check if GPU is available
    try:
        gpu_devices = jax.devices('gpu')
        print(f"\nGPU devices found: {len(gpu_devices)}")
        for i, gpu in enumerate(gpu_devices):
            print(f"  GPU {i}: {gpu}")
    except RuntimeError as e:
        print(f"\nNo GPU devices available: {e}")
        return False

    # Test simple GPU computation
    print("\n" + "="*60)
    print("Testing GPU computation...")
    print("="*60)

    # Create test array
    x = jnp.ones((1000, 1000))

    # Check which device it's on
    print(f"\nArray device: {x.device}")

    # Do a simple computation
    @jax.jit
    def matmul_test(x):
        return jnp.dot(x, x.T)

    # Warm up
    result = matmul_test(x)
    result.block_until_ready()

    # Time it
    import time
    n_iters = 10
    start = time.time()
    for _ in range(n_iters):
        result = matmul_test(x)
        result.block_until_ready()
    elapsed = time.time() - start

    print(f"\nMatrix multiplication (1000x1000):")
    print(f"  {n_iters} iterations in {elapsed:.4f}s")
    print(f"  Average: {elapsed/n_iters*1000:.2f}ms per iteration")

    print("\n" + "="*60)
    print("✓ GPU setup successful!")
    print("="*60)

    return True


if __name__ == "__main__":
    success = test_gpu_setup()
    exit(0 if success else 1)
