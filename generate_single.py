import os
import torch
import numpy as np
from PIL import Image
from diffusers import DDPMScheduler, UNet2DModel
import matplotlib.pyplot as plt
import argparse

# Function to generate a single image
def generate_single_image(checkpoint_path, steps=100, epoch_num=None):
    # Device setup - support for CUDA, MPS (Mac GPU), or CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Load the trained model
    model = UNet2DModel(
        sample_size=64,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 256),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        norm_num_groups=32,
    )

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()  # Set to evaluation mode

    # Set up the noise scheduler
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    # Create a directory to save generated images
    os.makedirs("generated_pong_frames", exist_ok=True)
    
    # Generate a single image
    with torch.no_grad():
        print("Generating image...")
        
        # Start from random noise
        image = torch.randn(1, 3, 64, 64).to(device)
        print(f"Initial noise shape: {image.shape}, range: [{image.min().item():.2f}, {image.max().item():.2f}]")
        
        # Use fewer steps for faster generation
        timesteps = torch.linspace(0, scheduler.config.num_train_timesteps - 1, steps, dtype=torch.long).flip(0).to(device)
        
        # Sampling loop for diffusion
        for step, t in enumerate(timesteps):
            # Progress indicator for long generation
            if step % 10 == 0:
                print(f"  Diffusion step {step}/{len(timesteps)}")
            
            # Get model prediction
            noise_pred = model(image, t.unsqueeze(0)).sample
            
            # Update sample with scheduler
            image = scheduler.step(noise_pred, t.item(), image).prev_sample
        
        # Process the generated image
        image = (image / 2 + 0.5).clamp(0, 1)
        print(f"Final image range: [{image.min().item():.2f}, {image.max().item():.2f}]")
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        
        # Create a unique filename based on the epoch and steps
        epoch_str = f"epoch{epoch_num}_" if epoch_num is not None else ""
        filename = f"pong_{epoch_str}steps{steps}"
        
        # Save the image with matplotlib (with larger size for better visibility)
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"Generated Pong Frame (Steps: {steps})")
        plt.savefig(f"generated_pong_frames/{filename}.png", dpi=100)
        plt.close()
        
        # Also save as regular image
        img = Image.fromarray((image * 255).astype(np.uint8))
        img.save(f"generated_pong_frames/{filename}_raw.png")
        
        # Save a larger version for better visibility
        img_large = img.resize((256, 256), Image.NEAREST)
        img_large.save(f"generated_pong_frames/{filename}_large.png")
        
        print(f"Saved image to generated_pong_frames/{filename}.png")
        return f"generated_pong_frames/{filename}.png"

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate a single image using trained diffusion model")
    parser.add_argument("--checkpoint", type=str, help="Path to specific checkpoint to use")
    parser.add_argument("--epoch", type=int, help="Specific epoch to use (e.g., 5, 10, 15)")
    parser.add_argument("--steps", type=int, default=100, help="Number of diffusion steps for inference")
    args = parser.parse_args()
    
    # Check if model checkpoints exist
    if not os.path.exists("model_checkpoints"):
        print("Warning: No model checkpoints found. Please train the model first.")
        exit(1)
    
    # Determine which checkpoint to use
    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoint_path = args.checkpoint
        epoch_num = None
    elif args.epoch is not None:
        checkpoint_path = f"model_checkpoints/diffusion_model_epoch_{args.epoch}.pth"
        epoch_num = args.epoch
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint for epoch {args.epoch} not found.")
            exit(1)
    else:
        # Find the latest checkpoint
        checkpoints = [f for f in os.listdir("model_checkpoints") if f.endswith(".pth")]
        if not checkpoints:
            print("No checkpoint files found in model_checkpoints directory.")
            exit(1)
        
        # Sort checkpoints by epoch number
        checkpoints.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        checkpoint_path = os.path.join("model_checkpoints", checkpoints[-1])
        epoch_num = int(checkpoints[-1].split("_")[-1].split(".")[0])
    
    print(f"Using checkpoint: {checkpoint_path}")
    print(f"Generating Pong frame with {args.steps} diffusion steps...")
    output_path = generate_single_image(checkpoint_path, args.steps, epoch_num)
    print(f"Done! Image saved to {output_path}") 