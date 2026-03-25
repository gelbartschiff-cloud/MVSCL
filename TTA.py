import torch
import torch.nn as nn
import torch.nn.functional as F
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from networks.ResUnet import ResUnet
from config import *
from utils.metrics import calculate_metrics
import numpy as np
import argparse
import sys, datetime
from torch.utils.data import DataLoader
from dataloaders.OPTIC_dataloader import OPTIC_dataset
from dataloaders.convert_csv_to_list import convert_labeled_list
from dataloaders.transform import collate_fn_wo_transform

torch.set_num_threads(1)


def collect_params(model, mode: str = "bn_all"):
    """
    Collect parameters for TTA adaptation.

    mode:
      - 'bn_all'           : all BatchNorm affine (original behavior)
      - 'bn_decoder'       : BatchNorm affine in decoder / upsampling path
      - 'bn_head'          : BatchNorm affine in segmentation head only
      - 'bn_decoder_head'  : BatchNorm affine in decoder + head

    Returns:
      params, names
    """
    params = []
    names = []

    # name heuristics (work well for ResUnet-like architectures)
    decoder_keywords = ['decoder', 'up', 'upsample', 'decode']
    head_keywords = ['head', 'final', 'out', 'seg', 'logit', 'pred']

    def is_decoder(name):
        n = name.lower()
        return any(k in n for k in decoder_keywords)

    def is_head(name):
        n = name.lower()
        return any(k in n for k in head_keywords)

    for module_name, m in model.named_modules():
        if not isinstance(m, nn.BatchNorm2d):
            continue

        pick = False
        if mode == "bn_all":
            pick = True
        elif mode == "bn_decoder":
            pick = is_decoder(module_name)
        elif mode == "bn_head":
            pick = is_head(module_name)
        elif mode == "bn_decoder_head":
            pick = is_decoder(module_name) or is_head(module_name)
        else:
            raise ValueError(f"Unknown adapt_mode: {mode}")

        if not pick:
            continue

        for pn, p in m.named_parameters(recurse=False):
            if pn in ['weight', 'bias']:
                params.append(p)
                names.append(f"{module_name}.{pn}")

    if len(params) == 0:
        raise RuntimeError(f"No parameters collected for adapt_mode={mode}")

    return params, names


def apply_flip(x: torch.Tensor, do_h: bool, do_v: bool) -> torch.Tensor:
    """x: (B,C,H,W)"""
    if do_h:
        x = torch.flip(x, dims=[3])
    if do_v:
        x = torch.flip(x, dims=[2])
    return x


def sobel_edge_mag(prob: torch.Tensor) -> torch.Tensor:
    """
    prob: (B, C, H, W) in [0,1]
    return: (B, C, H, W) edge magnitude (differentiable)
    """
    # Sobel kernels
    kx = torch.tensor([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=prob.dtype, device=prob.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]], dtype=prob.dtype, device=prob.device).view(1, 1, 3, 3)

    B, C, H, W = prob.shape
    # depthwise conv
    prob_ = prob.view(B * C, 1, H, W)
    gx = F.conv2d(prob_, kx, padding=1)
    gy = F.conv2d(prob_, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
    mag = mag.view(B, C, H, W)
    return mag


def l2_normalize_channel(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    x: (B, C, H, W)
    normalize over channel dim per-pixel
    """
    norm = torch.sqrt(torch.sum(x * x, dim=1, keepdim=True) + eps)
    return x / norm


class TrainTTA:
    def __init__(self, config):
        config.time_now = datetime.datetime.now().__format__("%Y%m%d_%H%M%S_%f")
        self.load_model = os.path.join(config.path_save_model, str(config.Source_Dataset))
        self.adapt_mode = config.adapt_mode

        self.log_path = os.path.join(config.path_save_log, 'TrainTTA')

        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        self.log_path = os.path.join(self.log_path, config.time_now + '.log')
        sys.stdout = Logger(self.log_path, sys.stdout)

        # Data Loading
        target_test_csv = []
        if config.Target_Dataset != 'REFUGE_Valid':
            target_test_csv.append(config.Target_Dataset + '_train.csv')
            target_test_csv.append(config.Target_Dataset + '_test.csv')
        else:
            target_test_csv.append(config.Target_Dataset + '.csv')
        ts_img_list, ts_label_list = convert_labeled_list(config.dataset_root, target_test_csv)

        target_test_dataset = OPTIC_dataset(
            config.dataset_root, ts_img_list, ts_label_list,
            config.image_size, img_normalize=True
        )
        self.target_test_loader = DataLoader(
            dataset=target_test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn_wo_transform,
            num_workers=config.num_workers
        )

        # Model
        self.backbone = config.backbone
        self.out_ch = config.out_ch
        self.image_size = config.image_size
        self.model_type = config.model_type

        # Optimizer
        self.optim = config.optimizer
        self.lr = config.lr
        self.momentum = config.momentum
        self.betas = (config.beta1, config.beta2)

        # Keep original args for compatibility (not used here)
        self.aux = config.aux_loss
        self.pse = config.pse_loss

        # Consistency hyperparams
        self.scales = config.scales  # list of floats
        self.w_prob = config.w_prob
        self.w_bound = config.w_bound
        self.w_fea = config.w_fea
        self.t_steps = config.tta_steps

        self.device = config.device
        self.build_model()
        self.print_network()

    def build_model(self):
        self.model = ResUnet(
            resnet=self.backbone,
            num_classes=self.out_ch,
            pretrained=False
        ).to(self.device)

        checkpoint = torch.load(
            self.load_model + '/' + 'last-' + self.model_type + '.pth',
            map_location=self.device
        )
        self.model.load_state_dict(checkpoint, strict=False)

        # ===== Improvement 1: adaptive parameter scope =====
        params, names = collect_params(self.model, mode=self.adapt_mode)

        num_tensors = len(params)
        num_scalars = sum(p.numel() for p in params)
        print(f"[TTA] adapt_mode={self.adapt_mode}, tensors={num_tensors}, scalars={num_scalars}")

        if self.optim == 'SGD':
            self.optimizer = torch.optim.SGD(
                params, lr=self.lr, momentum=self.momentum, nesterov=True
            )
        elif self.optim == 'Adam':
            self.optimizer = torch.optim.Adam(
                params, lr=self.lr, betas=self.betas
            )
        elif self.optim == 'AdamW':
            self.optimizer = torch.optim.AdamW(
                params, lr=self.lr, betas=self.betas
            )
        else:
            raise NotImplementedError(f"ERROR: no such optimizer {self.optim}!")


    def print_network(self):
        num_params = sum(p.numel() for p in self.model.parameters())
        print("The number of total parameters: {}".format(num_params))

    def _reset_bn_like_original(self):
        """
        Keep your ORIGINAL BN behavior (no improvement2):
        each step: BN trainable, and wipe running stats to use batch-stat (bs=1 may be noisy).
        """
        self.model.train()
        self.model.requires_grad_(False)
        for nm, m in self.model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None

    def _consistency_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Stronger consistency:
          - base view (teacher detach): prob0, edge0, fea0
          - augmented flip view: enforce prob/edge/fea consistency after inverse flip
          - multi-scale views: enforce prob consistency (and boundary optionally) after upsample
        """
        B, C, H, W = x.shape

        # ---- Base view (teacher) ----
        logit0, fea0 = self.model(x)
        prob0 = torch.sigmoid(logit0)  # (B, out_ch, H, W)
        edge0 = sobel_edge_mag(prob0)

        # teacher (stop-grad) for stability
        prob0_t = prob0.detach()
        edge0_t = edge0.detach()
        fea0_t = fea0.detach() if isinstance(fea0, torch.Tensor) else None

        loss_prob = prob0.mean() * 0.0
        loss_bound = prob0.mean() * 0.0
        loss_fea = prob0.mean() * 0.0

        # ---- Flip augmentation view ----
        # random h/v flips per batch (same for all images in batch)
        do_h = bool(torch.randint(0, 2, (1,), device=x.device).item())
        do_v = bool(torch.randint(0, 2, (1,), device=x.device).item())

        x1 = apply_flip(x, do_h, do_v)
        logit1, fea1 = self.model(x1)
        prob1 = torch.sigmoid(logit1)

        # invert flip back to base space
        prob1 = apply_flip(prob1, do_h, do_v)
        edge1 = sobel_edge_mag(prob1)

        # prob consistency (L2)
        loss_prob = loss_prob + F.mse_loss(prob1, prob0_t)

        # boundary consistency (L1 on edge mag tends to be stable)
        loss_bound = loss_bound + F.l1_loss(edge1, edge0_t)

        # feature consistency (normalize channel-wise then L2)
        if isinstance(fea1, torch.Tensor) and isinstance(fea0_t, torch.Tensor):
            # invert flip for feature too
            fea1 = apply_flip(fea1, do_h, do_v)

            # match spatial size if needed
            if fea1.shape[-2:] != fea0_t.shape[-2:]:
                fea1 = F.interpolate(fea1, size=fea0_t.shape[-2:], mode='bilinear', align_corners=False)

            fea0n = l2_normalize_channel(fea0_t)
            fea1n = l2_normalize_channel(fea1)
            loss_fea = loss_fea + F.mse_loss(fea1n, fea0n)

        # ---- Multi-scale consistency ----
        # compare all other scales to base teacher prob0_t (and optionally edge0_t)
        for s in self.scales:
            if abs(s - 1.0) < 1e-6:
                continue
            new_h = max(16, int(round(H * s)))
            new_w = max(16, int(round(W * s)))
            xs = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
            logits, _ = self.model(xs)
            probs = torch.sigmoid(logits)
            probs = F.interpolate(probs, size=(H, W), mode='bilinear', align_corners=False)

            loss_prob = loss_prob + F.mse_loss(probs, prob0_t)

            # optional: boundary consistency on multi-scale
            if self.w_bound > 0:
                edges = sobel_edge_mag(probs)
                loss_bound = loss_bound + F.l1_loss(edges, edge0_t)

        total = self.w_prob * loss_prob + self.w_bound * loss_bound + self.w_fea * loss_fea
        return total

    def run(self):
        metric_dict = ['Disc_Dice', 'Disc_ASSD', 'Cup_Dice', 'Cup_ASSD']
        metrics_test = [[], [], [], []]

        for batch, data in enumerate(self.target_test_loader):
            x, y = data['data'], data['mask']
            x = torch.from_numpy(x).to(dtype=torch.float32).to(self.device)
            y = torch.from_numpy(y).to(dtype=torch.float32).to(self.device)

            # ---- TTA: do a few gradient steps using consistency loss ----
            for _ in range(self.t_steps):
                self._reset_bn_like_original()
                loss = self._consistency_loss(x)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # ---- Evaluate after adaptation ----
            with torch.no_grad():
                self.model.eval()
                pred_logit, _ = self.model(x)
                seg_output = torch.sigmoid(pred_logit)

            metrics = calculate_metrics(seg_output.detach().cpu(), y.detach().cpu())
            for i in range(len(metrics)):
                assert isinstance(metrics[i], list), "The metrics value is not list type."
                metrics_test[i] += metrics[i]

            if (batch + 1) % 50 == 0:
                print(f"[TTA] batch={batch+1}/{len(self.target_test_loader)} last_consis_loss={float(loss.item()):.6f}")

        test_metrics_y = np.mean(metrics_test, axis=1)
        print_test_metric_mean = {metric_dict[i]: test_metrics_y[i] for i in range(len(test_metrics_y))}
        print("Test Metrics Mean: ", print_test_metric_mean)

        test_metrics_y = np.std(metrics_test, axis=1)
        print_test_metric_std = {metric_dict[i]: test_metrics_y[i] for i in range(len(test_metrics_y))}
        print("Test Metrics Std: ", print_test_metric_std)
        return print_test_metric_mean


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # keep your original args so run.sh won't break
    parser.add_argument('--aux_loss', type=str, default='ent', help='consis/ent/recon/rotate/supres/denoise')
    parser.add_argument('--pse_loss', type=str, default='consis', help='consis/ent/recon/rotate/supres/denoise')
    parser.add_argument(
        '--adapt_mode',
        type=str,
        default='bn_all',
        help='bn_all | bn_decoder | bn_head | bn_decoder_head'
    )

    parser.add_argument('--Source_Dataset', type=str,
                        help='RIM_ONE_r3/REFUGE/ORIGA/REFUGE_Valid/Drishti_GS')

    parser.add_argument('--optimizer', type=str, required=False, default='Adam', help='SGD/Adam/AdamW')
    parser.add_argument('--lr', type=float, required=False, default=1e-4)
    parser.add_argument('--momentum', type=float, required=False, default=0.99)
    parser.add_argument('--beta1', type=float, required=False, default=0.9)
    parser.add_argument('--beta2', type=float, required=False, default=0.999)
    parser.add_argument('--weight_decay', type=float, required=False, default=0.00)
    parser.add_argument('--batch_size', type=int, required=False, default=1)

    parser.add_argument('--model_type', type=str, required=False, default='Res_Unet')
    parser.add_argument('--backbone', type=str, required=False, default='resnet34')

    parser.add_argument('--in_ch', type=int, required=False, default=3)
    parser.add_argument('--out_ch', type=int, required=False, default=2)

    parser.add_argument('--image_size', type=int, required=False, default=512)
    parser.add_argument('--num_workers', type=int, required=False, default=8)

    parser.add_argument('--path_save_model', type=str)
    parser.add_argument('--dataset_root', type=str)
    parser.add_argument('--path_save_log', type=str, required=False, default='./logs/')

    if torch.cuda.is_available():
        parser.add_argument('--device', type=str, required=False, default='cuda:0')
    else:
        parser.add_argument('--device', type=str, required=False, default='cpu')

    # ===== Improvement 4 hyperparams =====
    parser.add_argument('--tta_steps', type=int, default=1, help='TTA gradient steps per image (1~3 usually)')
    parser.add_argument('--scales', type=float, nargs='+', default=[1, 1.0, 1],
                        help='multi-scale factors, e.g., --scales 0.75 1.0 1.25')

    parser.add_argument('--w_prob', type=float, default=0, help='weight of prob/logit consistency')
    parser.add_argument('--w_bound', type=float, default=0., help='weight of boundary consistency (Sobel)')
    parser.add_argument('--w_fea', type=float, default=0., help='weight of feature consistency (decoder feature)')

    config = parser.parse_args()

    targets = ['RIM_ONE_r3', 'REFUGE', 'ORIGA', 'REFUGE_Valid', 'Drishti_GS']
    targets.remove(config.Source_Dataset)

    dice_score = 0
    for config.Target_Dataset in targets:
        TTA = TrainTTA(config)
        metric = TTA.run()
        mean_dice = (metric['Disc_Dice'] + metric['Cup_Dice']) / 2
        dice_score += mean_dice

    print(config.Source_Dataset + ': Dice Mean=' + str(dice_score / len(targets)))
    print('\n\n\n')
