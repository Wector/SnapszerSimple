import subprocess
import re
import time

# Grid search space
NUM_ENVS_OPTIONS = [1024, 2048, 4096, 8192, 16384]
HIDDEN_SIZE_OPTIONS = [512, 1024]

best_sps = 0
best_config = None

print(f"Starting Auto-Tuner for RTX 4090...")
print(f"Search Space: Num Envs={NUM_ENVS_OPTIONS}, Hidden Size={HIDDEN_SIZE_OPTIONS}")
print("-" * 60)

for num_envs in NUM_ENVS_OPTIONS:
    for hidden_size in HIDDEN_SIZE_OPTIONS:
        print(f"Testing Config: Num Envs={num_envs}, Hidden Size={hidden_size}...")
        
        cmd = [
            "python", "train_ppo_selfplay.py",
            "--num-envs", str(num_envs),
            "--hidden-size", str(hidden_size),
            "--benchmark"
        ]
        
        try:
            # Run for a short burst
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300) # 300s timeout
            output = result.stdout
            
            # Parse SPS from output
            # Look for table row: "1        | 1400     | ..."
            # Regex: Start of line, digits, pipe, whitespace, CAPTURE digits, whitespace, pipe
            sps_values = []
            for line in output.splitlines():
                # Match "Update: SPS=" (Old) OR "   1 | 1400 |" (New)
                # Try new format first
                match = re.search(r"^\s*\d+\s+\|\s+(\d+)\s+\|", line)
                if match:
                    sps_values.append(int(match.group(1)))
                    continue
                
                # Fallback to old format just in case
                match = re.search(r"SPS=(\d+)", line)
                if match:
                    sps_values.append(int(match.group(1)))
            
            if sps_values:
                avg_sps = sum(sps_values) / len(sps_values)
                print(f"  -> Avg SPS: {avg_sps:.0f}")
                
                # Metric: Throughput = SPS * BatchSize? 
                # Actually SPS is steps per second, so it already accounts for batch size in terms of data generation?
                # Wait, PPO SPS usually means "Total environment steps per second".
                # So higher SPS is better.
                
                if avg_sps > best_sps:
                    best_sps = avg_sps
                    best_config = (num_envs, hidden_size)
            else:
                print("  -> No SPS data found (Did it crash?)")
                print(result.stderr[:200]) # Print error snippet
                
        except subprocess.TimeoutExpired:
            print("  -> Timed out!")
        except Exception as e:
            print(f"  -> Error: {e}")

print("-" * 60)
if best_config:
    print(f"BEST CONFIGURATION FOUND:")
    print(f"  Num Envs: {best_config[0]}")
    print(f"  Hidden Size: {best_config[1]}")
    print(f"  Peak SPS: {best_sps:.0f}")
    print(f"\nTo run this config:\npython train_ppo_selfplay.py --num-envs {best_config[0]} --hidden-size {best_config[1]}")
else:
    print("No valid configuration found.")
