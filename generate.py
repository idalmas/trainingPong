import os
import torch
import numpy as np
from PIL import Image
from diffusers import DDPMScheduler, UNet2DModel
from torchvision import transforms
import matplotlib.pyplot as plt

# Function to generate images
def generate_images(num_images=4, checkpoint_path="model_checkpoints/diffusion_model_epoch_95.pth"):
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
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()  # Set to evaluation mode

    # Set up the noise scheduler
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    # Create a directory to save generated images
    os.makedirs("generated_pong_frames", exist_ok=True)
    
    # Generate multiple images
    with torch.no_grad():
        for i in range(num_images):
            # Start from random noise
            image = torch.randn(1, 3, 64, 64).to(device)
            
            # Sampling loop for diffusion
            for t in reversed(range(0, scheduler.num_train_timesteps)):
                # Convert t to tensor
                timestep = torch.tensor([t], device=device)
                
                # Get model prediction
                noise_pred = model(image, timestep).sample
                
                # Update sample with scheduler
                image = scheduler.step(noise_pred, t, image).prev_sample
            
            # Process the generated image
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            
            # Save the image
            plt.figure(figsize=(4, 4))
            plt.imshow(image)
            plt.axis('off')
            plt.savefig(f"generated_pong_frames/pong_generated_{i}.png")
            plt.close()
            
            # Also save as regular image
            img = Image.fromarray((image * 255).astype(np.uint8))
            img.save(f"generated_pong_frames/pong_generated_{i}_raw.png")
            
            print(f"Generated image {i+1}/{num_images}")

if __name__ == "__main__":
    # Check if model checkpoints exist
    if not os.path.exists("model_checkpoints"):
        print("Warning: No model checkpoints found. Please train the model first.")
        exit(1)
    
    # Find the latest checkpoint
    checkpoints = [f for f in os.listdir("model_checkpoints") if f.endswith(".pth")]
    if not checkpoints:
        print("No checkpoint files found in model_checkpoints directory.")
        exit(1)
    
    # Sort checkpoints by epoch number
    checkpoints.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    latest_checkpoint = os.path.join("model_checkpoints", checkpoints[-1])
    
    print(f"Using latest checkpoint: {latest_checkpoint}")
    print("Generating Pong frames...")
    generate_images(num_images=8, checkpoint_path=latest_checkpoint)
    print("Done! Check the 'generated_pong_frames' folder for results.") 