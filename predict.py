import sys

sys.path.append("core")

import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from raft_stereo import RAFTStereo
from utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
from pipelime.sequences.readers.filesystem import UnderfolderReader
from pipelime.sequences.writers.filesystem import UnderfolderWriter
import cv2
from pipelime.sequences.samples import PlainSample, SamplesSequence

DEVICE = "cpu"


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def demo(args):
    model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt, map_location=DEVICE))

    model = model.module
    model.to(DEVICE)
    model.eval()
    # model = torch.jit.trace(
    #     model, (torch.rand(1, 3, 256, 256), torch.rand(1, 3, 256, 256))
    # )

    # output_directory = Path(args.output_directory)
    # output_directory.mkdir(exist_ok=True)

    dataset = UnderfolderReader(folder=args.dataset)
    NP_TO_TORCH = (
        lambda x: torch.from_numpy(x).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)
    )
    TORCH_TO_NP = lambda x: x.cpu().permute(0, 2, 3, 1)[0, ::].numpy()

    downsample = 1

    writer = UnderfolderWriter(folder=args.output_directory)
    with torch.no_grad():

        for sample_idx in range(900, len(dataset)):
            sample = dataset[sample_idx]
            image1 = sample["centerrect"]
            image2 = sample["rightrect"]

            cv2.imshow("image", image1)
            cv2.waitKey(1)

            image1 = cv2.resize(image1, (0, 0), fx=1 / downsample, fy=1 / downsample)
            image2 = cv2.resize(image2, (0, 0), fx=1 / downsample, fy=1 / downsample)
            # image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2RGB)
            image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2RGB)

            # image1 = load_image(imfile1)
            # image2 = load_image(imfile2)

            image1 = NP_TO_TORCH(image1)
            image2 = NP_TO_TORCH(image2)
            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)

            _, flow_up = model(image1, image2, iters=args.valid_iters, test_mode=True)
            print(flow_up.shape)

            flow_up = TORCH_TO_NP(flow_up)
            flow_up = (-flow_up).astype(np.uint8)
            flow_up = cv2.normalize(flow_up, None, 0, 255, cv2.NORM_MINMAX)
            flow_up = cv2.applyColorMap(flow_up, cv2.COLORMAP_MAGMA)

            out_sample = PlainSample(data={"image": flow_up}, id=sample_idx)
            writer.write(SamplesSequence(samples=[out_sample]))

            # print(flow_up.min(), flow_up.max())
            # cv2.imshow("flow_up", flow_up)
            # cv2.waitKey(0)
            # file_stem = imfile1.split("/")[-2]
            # if args.save_numpy:
            #     np.save(
            #         output_directory / f"{file_stem}.npy",
            #         flow_up.cpu().numpy().squeeze(),
            #     )
            # plt.imsave(
            #     output_directory / f"{file_stem}.png",
            #     -flow_up.cpu().numpy().squeeze(),
            #     cmap="jet",
            # )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_ckpt", help="restore checkpoint", required=True)
    parser.add_argument(
        "--save_numpy", action="store_true", help="save output as numpy arrays"
    )
    parser.add_argument(
        "--dataset",
        help="input dataset",
        default="/Users/danieledegregorio/Downloads/Pepsitest/dataset2",
    )

    parser.add_argument(
        "-l",
        "--left_imgs",
        help="path to all first (left) frames",
        default="datasets/Middlebury/MiddEval3/testH/*/im0.png",
    )
    parser.add_argument(
        "-r",
        "--right_imgs",
        help="path to all second (right) frames",
        default="datasets/Middlebury/MiddEval3/testH/*/im1.png",
    )
    parser.add_argument(
        "--output_directory", help="directory to save output", default="demo_output"
    )
    parser.add_argument(
        "--mixed_precision", action="store_true", help="use mixed precision"
    )
    parser.add_argument(
        "--valid_iters",
        type=int,
        default=32,
        help="number of flow-field updates during forward pass",
    )

    # Architecture choices
    parser.add_argument(
        "--hidden_dims",
        nargs="+",
        type=int,
        default=[128] * 3,
        help="hidden state and context dimensions",
    )
    parser.add_argument(
        "--corr_implementation",
        choices=["reg", "alt", "reg_cuda", "alt_cuda"],
        default="reg",
        help="correlation volume implementation",
    )
    parser.add_argument(
        "--shared_backbone",
        action="store_true",
        help="use a single backbone for the context and feature encoders",
    )
    parser.add_argument(
        "--corr_levels",
        type=int,
        default=4,
        help="number of levels in the correlation pyramid",
    )
    parser.add_argument(
        "--corr_radius", type=int, default=4, help="width of the correlation pyramid"
    )
    parser.add_argument(
        "--n_downsample",
        type=int,
        default=2,
        help="resolution of the disparity field (1/2^K)",
    )
    parser.add_argument(
        "--slow_fast_gru",
        action="store_true",
        help="iterate the low-res GRUs more frequently",
    )
    parser.add_argument(
        "--n_gru_layers", type=int, default=3, help="number of hidden GRU levels"
    )

    args = parser.parse_args()

    demo(args)
