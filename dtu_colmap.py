import os
import sys
import shutil
import argparse
from utils.colmap2poses import gen_poses

def copy_images(data_dir, out_dir, n_images=49, light_cond='3_r5000'):
    out_path = os.path.join(out_dir, 'input/')
    os.makedirs(out_path, exist_ok=True)
    
    for i in range(1, n_images+1):
        fname = os.path.join(data_dir, f'rect_{i:03d}_{light_cond}.png')
        print(f'Copying {fname} to {out_path}')
        
        if os.path.exists(fname):
            shutil.copy(fname, out_path)
        else:
            print(f'Warning: {fname} does not exist!')

def main():
    parser = argparse.ArgumentParser(description="Copy images from data directory to output directory")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the original DTU dataset directory")
    parser.add_argument('--out_dir', type=str, required=True, help="Path to the output directory")
    parser.add_argument('--n_images', type=int, default=49, help="Number of images for all views")
    parser.add_argument('--light_cond', type=str, default='3_r5000', help="Lighting condition (default: '3_r5000')")
    parser.add_argument('--match_type', type=str, 
                        default='exhaustive_matcher', help='type of matcher used.  Valid options: \
                        exhaustive_matcher sequential_matcher.  Other matchers not supported at this time')
    
    args = parser.parse_args()
    if args.match_type != 'exhaustive_matcher' and args.match_type != 'sequential_matcher':
        print('ERROR: matcher type ' + args.match_type + ' is not valid.  Aborting')
        sys.exit()

    # copy images
    copy_images(args.data_dir, args.out_dir, args.n_images, args.light_cond)

    # convert images
    print('Begin converting ', args.out_dir)
    os.system('python convert.py -s ' + args.out_dir + ' --camera PINHOLE')
    print('Finished converting ', args.out_dir)

    shutil.rmtree(os.path.join(args.out_dir, 'input'))

    # generate puses_bounds.npy
    print('Generating pose ', args.out_dir)
    gen_poses(args.out_dir, args.match_type)
    print('Finished ', args.out_dir)


if __name__ == "__main__":
    main()
