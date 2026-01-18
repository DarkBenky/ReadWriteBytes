#!/usr/bin/env python3
import wandb
import sys
import os
import json
import time

def main():
    # Initialize wandb
    wandb.init(
        project="image-super-resolution",
        config={
            "architecture": "encoder-middle-resize2d-decoder",
            "model_params": "20M",
            "input_size": "800x600",
            "output_size": "1200x900",
            "loss": "MAE(0.1) + SSIM(0.5) + FFT(0.4)",
            "optimizer": "SGD",
        }
    )
    
    fifo_path = "/tmp/ml_metrics.fifo"
    
    # Create FIFO if it doesn't exist
    if not os.path.exists(fifo_path):
        os.mkfifo(fifo_path)
        print(f"Created FIFO at {fifo_path}")
    
    print(f"Waiting for metrics from C program...")
    sys.stdout.flush()
    
    # Open FIFO for reading (blocks until writer connects)
    with open(fifo_path, 'r') as fifo:
        print("Connected to C program, logging to wandb...")
        sys.stdout.flush()
        
        while True:
            line = fifo.readline()
            if not line:
                # Writer closed, wait a bit and try to reconnect
                time.sleep(0.1)
                continue
            
            line = line.strip()
            if not line:
                continue
            
            try:
                # Parse JSON metrics
                data = json.loads(line)
                
                # Log to wandb
                wandb.log(data)
                
                # Print confirmation
                if 'epoch' in data:
                    print(f"Logged epoch {data['epoch']}: loss={data.get('loss', 'N/A'):.6f}")
                    sys.stdout.flush()
                    
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON: {line}", file=sys.stderr)
                print(f"Error: {e}", file=sys.stderr)
                continue
            except Exception as e:
                print(f"Error logging to wandb: {e}", file=sys.stderr)
                continue
    
    wandb.finish()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nShutting down wandb logger...")
        wandb.finish()
        sys.exit(0)
