import argparse
import os
from contextlib import nullcontext
import math
import numpy as np

import torch
from PIL import Image
from tqdm import tqdm
from transparent_background import Remover

from spar3d.models.mesh import QUAD_REMESH_AVAILABLE, TRIANGLE_REMESH_AVAILABLE
from spar3d.system import SPAR3D
from spar3d.utils import foreground_crop, get_device, remove_background


def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue


def extract_structured_camera_positions(images, args):
    """Extract camera positions from image filenames if they follow a structured naming convention.
    
    Looks for filenames like img_X_Y_Z.jpg where X, Y, Z are camera coordinates.
    """
    # Check if filenames follow the expected format
    position_pattern = r'.*_(-?\d+\.?\d*)_(-?\d+\.?\d*)_(-?\d+\.?\d*)\.(png|jpg|jpeg)$'
    
    import re
    positions = []
    valid_indices = []
    
    for i, image_path in enumerate(images):
        filename = os.path.basename(image_path)
        match = re.match(position_pattern, filename)
        
        if match:
            x, y, z = float(match.group(1)), float(match.group(2)), float(match.group(3))
            positions.append([x, y, z])
            valid_indices.append(i)
    
    # If we found positions for all images, return them
    if len(positions) == len(images):
        print(f"Extracted camera positions from {len(positions)} image filenames")
        return positions
    elif len(positions) > 0:
        print(f"Warning: Could only extract camera positions from {len(positions)} of {len(images)} images")
        print(f"Using default camera positions for all images")
    
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-dir", type=str, required=True, help="Directory containing multi-view images."
    )
    parser.add_argument(
        "--device",
        default=get_device(),
        type=str,
        help=f"Device to use. If no CUDA/MPS-compatible device is found, the baking will fail. Default: '{get_device()}'",
    )
    parser.add_argument(
        "--pretrained-model",
        default="stabilityai/stable-point-aware-3d",
        type=str,
        help="Path to the pretrained model. Could be either a huggingface model id is or a local path. Default: 'stabilityai/stable-point-aware-3d'",
    )
    parser.add_argument(
        "--foreground-ratio",
        default=1.3,
        type=float,
        help="Ratio of the foreground size to the image size. Only used when --no-remove-bg is not specified. Default: 0.85",
    )
    parser.add_argument(
        "--output-dir",
        default="output_multiview/",
        type=str,
        help="Output directory to save the results. Default: 'output_multiview/'",
    )
    parser.add_argument(
        "--texture-resolution",
        default=1024,
        type=int,
        help="Texture atlas resolution. Default: 1024",
    )
    parser.add_argument(
        "--low-vram-mode",
        action="store_true",
        help=(
            "Use low VRAM mode. SPAR3D consumes 10.5GB of VRAM by default. "
            "This mode will reduce the VRAM consumption to roughly 7GB but in exchange "
            "the model will be slower. Default: False"
        ),
    )
    
    # New arguments for enhanced multi-view reconstruction
    parser.add_argument(
        "--hemisphere-distribution",
        action="store_true",
        help="Use a hemisphere distribution for camera positions instead of a circle."
    )
    parser.add_argument(
        "--camera-distance",
        type=float,
        default=None,
        help="Custom distance for camera positions. If not specified, the model default is used."
    )
    parser.add_argument(
        "--detect-camera-positions",
        action="store_true",
        help="Try to detect camera positions from image filenames (format: img_X_Y_Z.jpg)."
    )
    parser.add_argument(
        "--point-density",
        type=int,
        default=1024,
        help="Density of the point cloud used for reconstruction. Default: 1024"
    )

    remesh_choices = ["none"]
    if TRIANGLE_REMESH_AVAILABLE:
        remesh_choices.append("triangle")
    if QUAD_REMESH_AVAILABLE:
        remesh_choices.append("quad")
    parser.add_argument(
        "--remesh_option",
        choices=remesh_choices,
        default="none",
        help="Remeshing option",
    )
    if TRIANGLE_REMESH_AVAILABLE or QUAD_REMESH_AVAILABLE:
        parser.add_argument(
            "--reduction_count_type",
            choices=["keep", "vertex", "faces"],
            default="keep",
            help="Vertex count type",
        )
        parser.add_argument(
            "--target_count",
            type=check_positive,
            help="Selected target count.",
            default=2000,
        )
    
    args = parser.parse_args()

    # Ensure args.device contains cuda
    devices = ["cuda", "mps", "cpu"]
    if not any(args.device in device for device in devices):
        raise ValueError("Invalid device. Use cuda, mps or cpu")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    device = args.device
    if not (torch.cuda.is_available() or torch.backends.mps.is_available()):
        device = "cpu"

    print("Device used: ", device)

    # Load the model
    model = SPAR3D.from_pretrained(
        args.pretrained_model,
        config_name="config.yaml",
        weight_name="model.safetensors",
        low_vram_mode=args.low_vram_mode,
    )
    model.to(device)
    model.eval()

    # Find all images in the directory
    image_paths = []
    for file in sorted(os.listdir(args.image_dir)):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            image_paths.append(os.path.join(args.image_dir, file))
    
    if len(image_paths) < 2:
        raise ValueError(f"Found only {len(image_paths)} images in {args.image_dir}. Need at least 2 images for multi-view reconstruction.")
    
    # Check if we should extract camera positions from filenames
    camera_positions = None
    if args.detect_camera_positions:
        camera_positions = extract_structured_camera_positions(image_paths, args)
    
    print(f"Found {len(image_paths)} images for multi-view reconstruction.")
    
    # Process all images
    bg_remover = Remover(device=device)
    images = []
    
    print("Preprocessing images...")
    for image_path in tqdm(image_paths):
        image = remove_background(Image.open(image_path).convert("RGBA"), bg_remover)
        image = foreground_crop(image, args.foreground_ratio)
        images.append(image)
    
    # Save input images
    for i, image in enumerate(images):
        os.makedirs(os.path.join(output_dir, "inputs"), exist_ok=True)
        image.save(os.path.join(output_dir, "inputs", f"input_{i:03d}.png"))
    
    # Set vertex count for remeshing
    vertex_count = (
        -1
        if args.reduction_count_type == "keep"
        else (
            args.target_count
            if args.reduction_count_type == "vertex"
            else args.target_count // 2
        )
    )

    # Override camera distance if specified
    if args.camera_distance is not None:
        model.cfg.default_distance = args.camera_distance

    # Process multi-view images
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    print("Processing multi-view reconstruction...")
    
    with torch.no_grad():
        with (
            torch.autocast(device_type=device, dtype=torch.bfloat16)
            if "cuda" in device
            else nullcontext()
        ):
            mesh, glob_dict = model.run_multiview_image(
                images,
                bake_resolution=args.texture_resolution,
                remesh=args.remesh_option,
                vertex_count=vertex_count,
                return_points=True,
                camera_positions=camera_positions,
            )
    
    if torch.cuda.is_available():
        print("Peak Memory:", torch.cuda.max_memory_allocated() / 1024 / 1024, "MB")
    elif torch.backends.mps.is_available():
        print("Peak Memory:", torch.mps.driver_allocated_memory() / 1024 / 1024, "MB")

    # Save output mesh and point cloud
    out_mesh_path = os.path.join(output_dir, "mesh.glb")
    mesh.export(out_mesh_path, include_normals=True)
    
    out_points_path = os.path.join(output_dir, "points.ply")
    glob_dict["point_clouds"][0].export(out_points_path)
    
    print(f"\nReconstruction complete! Results saved to {output_dir}")
    print(f"- 3D Mesh: {out_mesh_path}")
    print(f"- Point Cloud: {out_points_path}")
    print("\nTips for better results:")
    print("1. Make sure your input images cover the object from different angles")
    print("2. Try different camera distance values with --camera-distance")
    print("3. Include camera position information in filenames (e.g., img_1.0_2.0_3.0.jpg)")
    print("4. Experiment with different point cloud densities using --point-density")
