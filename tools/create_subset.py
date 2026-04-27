import pickle
import random
import argparse
import os

def create_subset(input_pkl, output_pkl, percentage=0.01):
    print(f"Loading {input_pkl}...")
    if not os.path.exists(input_pkl):
        print(f"Error: {input_pkl} not found.")
        return

    with open(input_pkl, 'rb') as f:
        infos = pickle.load(f)
    
    # Subsample uniformly to maintain diverse coverage
    k = int(len(infos) * percentage)
    
    random.seed(42)
    subset = random.sample(infos, k)
    
    print(f"Original size: {len(infos)}")
    print(f"Subset size: {len(subset)} ({percentage*100}%)")
    
    with open(output_pkl, 'wb') as f:
        pickle.dump(subset, f)
    print(f"Saved to {output_pkl}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/home/it4i-andreaam/UniTR/data/nuscenes/nuscenes_infos_10sweeps_train.pkl')
    parser.add_argument('--output', type=str, default='/home/it4i-andreaam/UniTR/data/nuscenes_1pct/nuscenes_infos_10sweeps_train_1percent.pkl')
    parser.add_argument('--percent', type=float, default=0.01)
    args = parser.parse_args()
    
    # Also check if it's in the root of data/nuscenes instead
    if not os.path.exists(args.input):
        alt_input = '../data/nuscenes/nuscenes_infos_10sweeps_train.pkl'
        if os.path.exists(alt_input):
            print(f"Found at {alt_input} instead of {args.input}")
            args.input = alt_input
            args.output = '../data/nuscenes/nuscenes_infos_10sweeps_train_1percent.pkl'
            
    create_subset(args.input, args.output, args.percent)
