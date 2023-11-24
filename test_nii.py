import argparse
import os
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
import Config as config
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
from networks.vit_seg_modeling import VisionTransformer
import glob
parser = argparse.ArgumentParser()
# parser.add_argument('--root_path', type=str,
#                     default='D:/data/Synapse_npy/Synapse_npy/npy_nii/images', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='transunet', help='model_name')

parser.add_argument('--labeled_num', type=int, default=12,
                    help='labeled data')

parser.add_argument('--root_path', type=str,
                    default='D:/data/Synapse_npy/Synapse_npy/train/images', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=8, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    #asd = metric.binary.asd(pred, gt)
    #hd95 = metric.binary.hd95(pred, gt)
    return dice


def test_single_volume(case, net, test_save_path, FLAGS):

    label_path = r'D:\data\Synapse_npy\Synapse_npy\npy_nii\labels'
    #nii
    image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(FLAGS.root_path,case[-15:])))
    label = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(label_path,case[-15:])))
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (224 / x, 224 / y), order=0)
        inputs = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            if FLAGS.model == "unet_urds":
                out_main, _, _, _ = net(inputs)
            else:
                out_main = net(inputs.cpu())
            out = torch.argmax(torch.softmax(
                out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 224, y / 224), order=0)
            prediction[ind] = pred

    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    sitk.WriteImage(prd_itk, test_save_path + case[-15:])

    label_itk = sitk.GetImageFromArray(label.astype(np.float32))
    label_itk.SetSpacing((1, 1, 10))
    sitk.WriteImage(label_itk, test_save_path+'label/' + case[-15:])
    #nib.save(prediction, test_save_path + case[-15:])



def Inference(FLAGS):
    image_list = sorted(glob.glob(os.path.join(FLAGS.root_path, '*nii.gz')))
    test_save_path = r'D:\data\Synapse_npy\Synapse_npy\show/'
    #config_vit = config.get_CTranS_config()
    # model = UCTransNet3Plus(config_vit,n_channels=config.n_channels,n_classes=config.n_labels)
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    net = VisionTransformer(config_vit, img_size=224, num_classes=9, zero_head=False, vis=False)

    save_mode_path = r'C:\code\TransUNet-main\model\TU_Synapse224\TU_pretrain_R50-ViT-B_16_skip3_epo150_bs8_224\epoch_99.pth'
    checkpoint = torch.load(save_mode_path, map_location='cuda')
    net.load_state_dict(checkpoint)
    print("init weight from {}".format(save_mode_path))
    net.eval()
    for case in tqdm(image_list):
        test_single_volume(case, net, test_save_path, FLAGS)




if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric = Inference(FLAGS)
    print(metric)
    print((metric[0]+metric[1]+metric[2])/3)
