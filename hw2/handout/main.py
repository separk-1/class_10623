from utils import train_diffusion, visualize_diffusion
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--time_steps",
        default=50,
        type=int,
        help="The number of steps the scheduler takes to go from clean image to an isotropic gaussian. This is also the number of steps of diffusion.",
    )
    parser.add_argument(
        "--train_steps",
        default=10000,
        type=int,
        help="The number of iterations for training.",
    )
    parser.add_argument("--save_folder", default="./results", type=str)
    parser.add_argument("--data_path", default="./data/train/", type=str)
    parser.add_argument("--load_path", default=None, type=str)
    parser.add_argument(
        "--data_class", choices=["all", "cat", "dog", "wild"], default="cat", type=str
    )
    parser.add_argument("--image_size", default=32, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--unet_dim", default=16, type=int)
    parser.add_argument("--unet_dim_mults", nargs="+", default=[1, 2, 4, 8], type=int)
    parser.add_argument("--fid", action="store_true")
    parser.add_argument(
        "--save_and_sample_every",
        default=1000,
        type=int,
        help="The number of steps between periodically saving the model state, " + \
             "sampling example images, and optional calculating FID"
    )

    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--dataloader_workers", default=16, type=int)

    args = parser.parse_args()
    print(args)

    if args.visualize:
        if args.load_path is None:
            print("No model to visualize, Please provide a load path.")
            exit(0)
        visualize_diffusion(vars(args))
    else:
        train_diffusion(**vars(args))