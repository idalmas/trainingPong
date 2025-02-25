import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import DDPMScheduler, UNet2DModel

# Define a dataset for Pong screenshots
class PongDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

# Preprocessing: resize images and normalize
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Changed from 128x128 to 64x64 for faster training
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Option 1: Using os.path.expanduser to get your home directory
home_dir = os.path.expanduser("~")  # This gets your home directory
dataset = PongDataset(os.path.join(home_dir, "Documents/Github/PyClassicPong/frames"), transform=transform)

# Option 2: Using absolute path (example for Windows)
# dataset = PongDataset("C:/Users/YourUsername/Documents/Github/PyClassicPong/frames", transform=transform)

# Create a directory to save models
os.makedirs("model_checkpoints", exist_ok=True)

# Initialize a UNet-based diffusion model (for 64x64 RGB images)
# For Pong which has simple graphics, we can use a slightly smaller model
model = UNet2DModel(
    sample_size=64,  # Changed from 128 to 64
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(64, 128, 256, 256),  # Simplified architecture
    down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    norm_num_groups=32,
)

# Set up the noise scheduler for diffusion (DDPM)
scheduler = DDPMScheduler(num_train_timesteps=1000)

# Define optimizer and training parameters
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 100  # Running 100 epochs for overnight training

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

model.to(device)

# Main training function
def train_model():
    # Enable mixed precision training if using CUDA GPU
    use_amp = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    print(f"Starting training for {num_epochs} epochs...")
    print(f"Dataset size: {len(dataset)} images")
    print(f"Mixed precision training: {'Enabled' if use_amp else 'Disabled'}")
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for step, batch in enumerate(dataloader):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Sample random timesteps for each image in the batch
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (batch.size(0),), device=device).long()
            
            # Sample noise and add it to the images according to the scheduler
            noise = torch.randn_like(batch)
            noisy_images = scheduler.add_noise(batch, noise, timesteps)
            
            # Mixed precision training for CUDA GPUs
            if use_amp:
                with torch.cuda.amp.autocast():
                    # The model predicts the noise added to the images
                    noise_pred = model(noisy_images, timesteps).sample
                    loss = nn.functional.mse_loss(noise_pred, noise)
                
                # Scale the loss and do backward pass
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training for CPU and MPS
                noise_pred = model(noisy_images, timesteps).sample
                loss = nn.functional.mse_loss(noise_pred, noise)
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if step % 10 == 0:
                print(f"Epoch {epoch} | Step {step}/{len(dataloader)} | Loss: {loss.item():.4f}")
        
        # Calculate and print average loss for the epoch
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch} completed | Average Loss: {avg_loss:.4f}")
        
        # Save model weights after each epoch
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            checkpoint_path = f"model_checkpoints/diffusion_model_epoch_{epoch}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    print("Training completed!")

# This is the fix for the multiprocessing issue
if __name__ == '__main__':
    # Set num_workers=0 to avoid multiprocessing issues on macOS
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    train_model()
