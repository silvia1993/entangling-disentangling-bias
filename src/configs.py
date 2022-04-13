import argparse

parser = argparse.ArgumentParser()

### COMMON
parser.add_argument('--dataset', default='celeba', type=str, help='mnist, celeba')
parser.add_argument('--dataset_path', default='/home/silvia/data/celeba', type=str, help='mnist, celeba')
parser.add_argument('--model', default='resnet50', type=str)
parser.add_argument('--attribute', type=int, default=27)
parser.add_argument('--protected_attribute', type=int, default=20)

parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--alpha', default=6, type=float, help='EnD alpha')
parser.add_argument('--beta', default=1, type=float, help='EnD beta')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--batch_size', default=256, type=int, help='batch size')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--local', dest='local', action='store_true', help='disable wandb')

### REBIAS MNIST
parser.add_argument('--rho', type=float, default=0.997, help='rho for biased mnist (.999, .997, .995, .990)')

### COMMONS
parser.set_defaults(local=False)
config = parser.parse_args()
