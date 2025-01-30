import argparse
import os
from training import train
from prediction import test
from pathlib import Path

def get_args_parser():
    parser = argparse.ArgumentParser('Gloss-free Sign Language Translation script')
    
    # Add mode and config arguments
    parser.add_argument("mode", choices=["train", "test"], help="train a model or test")
    parser.add_argument("config_path", type=str, help="path to YAML config file")
    parser.add_argument("--ckpt", type=str, help="checkpoint for prediction")
    parser.add_argument("--output_path", type=str, help="path for saving translation output")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu to run your job on")
    
    # Add data processing arguments
    parser.add_argument('--input-size', default=224, type=int)
    parser.add_argument('--resize', default=256, type=int)
 
    return parser

def main():
    parser = get_args_parser()
    args = parser.parse_args()
    
    cfg_file = args.config_path
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if args.mode == "train":
        train(cfg_file=cfg_file, args=args)
    elif args.mode == "test":
        test(cfg_file=args.config_path, ckpt=args.ckpt, output_path=args.output_path)
    else:
        raise ValueError("Unknown mode")

if __name__ == "__main__":
    main()

'''
To train: python main.py train configs/sign_volta.yaml
python -m main train [CONFIG PATH]

To test: python main.py test configs/sign.yaml --ckpt checkpoints/best.IT_00000000.ckpt --output_path outputs/test.txt
python -m main test [CONFIG PATH] --ckpt [CHECKPOINT PATH]
'''