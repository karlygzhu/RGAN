import argparse

parser = argparse.ArgumentParser(description='Pytorch 1.8.1: Model Example')

# param
parser.add_argument('--batch_size', type=int, default=8, help="Training batch size")
parser.add_argument('--clip', type=int, default=20, help="Gradient clipping range")
parser.add_argument('--num_frames', type=int, default=5, help="Number of input images at a time")
parser.add_argument('--scale', type=int, default=4, help="Upsampling multiplier")
parser.add_argument('--seed', type=int, default=6, help="Random seed")
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--weight_decay', type=float, default=0, help="Weight decay")
parser.add_argument('--loss_weight', type=float, default=0.01, help="Loss weight")

# epochs and learning rate
parser.add_argument('--gamma', type=int, default=0.5, help="Learning rate decay")
parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate")
parser.add_argument('--num_epochs', type=int, default=75, help="Number of epochs to train")
parser.add_argument('--step_size', type=int, default=25, help='Learning rate decay period')


# Image processing
parser.add_argument('--crop_size_LR', type=int, default=64, help="Low resolution image crop size")
parser.add_argument('--crop_size_HR', type=int, default=256, help="High resolution image cropping size")


# save
parser.add_argument('--image_out', type=str, default='./results_out', help="Storage path for generated images")
parser.add_argument('--save_model_path', type=str, default='./checkpoints', help="Storage path for model parameters")


# path
parser.add_argument('--test_data', type=str, default='./data_test/filelist_test.txt', help="Test data")
parser.add_argument('--test_datasets', type=str, default='Vid4', help="Select testing set")
parser.add_argument('--train_Vimeo', type=str, default='./data81/sep_trainlist.txt', help="data81 datasets")
parser.add_argument('--train_datasets', type=str, default='Vimeo-90K', help="Selecting the Vimeo-90K dataset")


opt = parser.parse_args()

