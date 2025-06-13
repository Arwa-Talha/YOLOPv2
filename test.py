import sys
import torch
from demo import make_parser, detect  

# Simulate command line arguments (optional if running from CLI)
sys.argv = [
    'test.py',
    '--weights', 'yolopv2.pt',
    '--source', 'example.mp4',
    '--device', '0',
    '--project', 'runs/detect',
    '--name', 'my_output'
]

if __name__ == '__main__':
    opt = make_parser().parse_args()
    print(opt)

    with torch.no_grad():
        detect(opt)
