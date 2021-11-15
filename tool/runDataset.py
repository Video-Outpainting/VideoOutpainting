from PIL import Image
import os, sys
import subprocess
import argparse


def run(args):
    dirs = os.listdir( args.pathToDataset )
    print(len(dirs))
    i = 0
    for item in sorted(dirs):        
        print(item)
        i+=1
        if(args.vertical):
            subprocess.call([sys.executable, 'video_outpaint.py', '--path', args.pathToDataset+item, '--outroot', args.outroot+item, '--Width', '0.333', '--replace']) #Vertical to horizontal
        else:
            subprocess.call([sys.executable, 'video_outpaint.py', '--path', args.pathToDataset+item, '--outroot', args.outroot+item, '--Width', '0.125', '--replace']) #Landscape to ultra-wide
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pathToDataset', default='/home/user/Documents/Youtube-VOS/JPEGImages/', help="dataset for evaluation")
    parser.add_argument('--outroot', default='../result/', help="output directory") 
    parser.add_argument('--vertical', action='store_true', help="vertical to horizontal or horizontal to ultrawide")
    run(parser.parse_args())
