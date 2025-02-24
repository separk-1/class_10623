
import wandb
import torch

from trainer import Trainer
from unet import Unet

# fix random seed for reproducibility
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
torch.cuda.manual_seed_all(2024)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def setup_diffusion(train_steps, save_and_sample_every, fid, 
                    save_folder, data_path,
                    load_path=None,
                    unet_dim_mults=[1, 2, 4, 8], 
                    image_size=32, 
                    batch_size=32, 
                    data_class="cat", 
                    time_steps=50, 
                    unet_dim=16, 
                    learning_rate=1e-3, 
                    dataloader_workers=16, 
                    Diffusion=None):
    
    if Diffusion is None:
        from diffusion import Diffusion
    
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    model = Unet(
        dim=unet_dim,
        dim_mults=unet_dim_mults,
    ).to(device)

    diffusion = Diffusion(
        model,
        image_size=image_size,
        channels=3,
        timesteps=time_steps,  # number of steps
    ).to(device)

    trainer = Trainer(
        diffusion,
        data_path,
        image_size=image_size,
        train_batch_size=batch_size,
        train_lr=learning_rate,
        train_num_steps=train_steps,  # total training steps
        gradient_accumulate_every=2,  # gradient accumulation steps
        results_folder=save_folder,
        load_path=load_path,
        dataset="train",
        data_class=data_class,
        device=device,
        save_and_sample_every=save_and_sample_every,
        fid=fid,
        dataloader_workers=dataloader_workers
    )

    return model, diffusion, trainer


def train_diffusion(train_steps, save_and_sample_every, fid, 
                    save_folder, data_path,
                    load_path=None,
                    unet_dim_mults=[1, 2, 4, 8], 
                    image_size=32, 
                    batch_size=32, 
                    data_class="cat", 
                    time_steps=50, 
                    unet_dim=16, 
                    learning_rate=1e-3, 
                    dataloader_workers=16, 
                    Diffusion=None):
    wandb.login()
    wandb.init(
        project="DDPM_AFHQ",
        config={
            "image_size": image_size,
            "batch_size": batch_size,
            "data_class": data_class,
            "train_steps": train_steps,
            "time_steps": time_steps,
            "unet_dim": unet_dim,
            "learning_rate": learning_rate,
            "save_and_sample_every": save_and_sample_every,
            "fid": fid,
            "dataloader_workers": dataloader_workers,
            "unet_dim_mults": unet_dim_mults,
        },
        reinit=True,
        name=save_folder.split("/")[-1],
    )

    _, _, trainer = setup_diffusion(train_steps=train_steps, 
                                    save_and_sample_every=save_and_sample_every, 
                                    fid=fid, 
                                    save_folder=save_folder,
                                    data_path=data_path,
                                    load_path=load_path,
                                    unet_dim_mults=unet_dim_mults, 
                                    image_size=image_size, 
                                    batch_size=batch_size, 
                                    data_class=data_class, 
                                    time_steps=time_steps, 
                                    unet_dim=unet_dim, 
                                    learning_rate=learning_rate, 
                                    dataloader_workers=dataloader_workers, 
                                    Diffusion=Diffusion)

    trainer.train()
    wandb.finish()


def visualize_diffusion(train_steps, save_and_sample_every, fid, 
                    save_folder, data_path,
                    load_path=None,
                    unet_dim_mults=[1, 2, 4, 8], 
                    image_size=32, 
                    batch_size=32, 
                    data_class="cat", 
                    time_steps=50, 
                    unet_dim=16, 
                    learning_rate=1e-3, 
                    dataloader_workers=16, 
                    Diffusion=None):
    wandb.login()
    wandb.init(
        project="DDPM_AFHQ",
        config={
            "image_size": image_size,
            "batch_size": batch_size,
            "data_class": data_class,
            "train_steps": train_steps,
            "time_steps": time_steps,
            "unet_dim": unet_dim,
            "learning_rate": learning_rate,
            "save_and_sample_every": save_and_sample_every,
            "fid": fid,
            "dataloader_workers": dataloader_workers,
            "unet_dim_mults": unet_dim_mults,
        },
        reinit=True,
        name=save_folder.split("/")[-1],
    )

    _, _, trainer = setup_diffusion(train_steps=train_steps, 
                                    save_and_sample_every=save_and_sample_every, 
                                    fid=fid, 
                                    save_folder=save_folder,
                                    data_path=data_path,
                                    load_path=load_path,
                                    unet_dim_mults=unet_dim_mults, 
                                    image_size=image_size, 
                                    batch_size=batch_size, 
                                    data_class=data_class, 
                                    time_steps=time_steps, 
                                    unet_dim=unet_dim, 
                                    learning_rate=learning_rate, 
                                    dataloader_workers=dataloader_workers, 
                                    Diffusion=Diffusion)
    
    trainer.visualize_diffusion()
    wandb.finish()