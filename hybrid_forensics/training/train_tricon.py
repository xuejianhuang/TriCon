import os
os.environ["PYTHONWARNINGS"] = "ignore"

import argparse
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import roc_auc_score

from ..models.tricon_net import ImprovedTriCon, SpeechOnlyBaseline, DynamicOnlyBaseline
from ..datasets.feature_dataset import FeatureDataset, create_dataloader


class TriConModule(pl.LightningModule):

    def __init__(
        self,
        model_type: str = "TriCon",
        sem_dim: int = 2048,
        dyn_dim: int = 768,
        hidden_dim: int = 512,
        num_classes: int = 2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        use_frequent: bool = False,
        contrast_weight: float = 0.5,
        contrast_margin: float = 0.5,
        seed: int = 42,
        wavelet_mlp_hidden: int = 256,
        use_per_clip: bool = True,
        class_weights: torch.Tensor = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model_type = model_type
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_frequent = use_frequent
        self.wavelet_mlp_hidden = wavelet_mlp_hidden
        self.contrast_weight = contrast_weight
        self.contrast_margin = contrast_margin
        self.use_per_clip = use_per_clip

        # Class-weighted BCE
        if class_weights is not None:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.ce_loss = nn.CrossEntropyLoss()

        if model_type in ("TriCon", "improved"):
            self.model = ImprovedTriCon(
                sem_dim=sem_dim,
                dyn_dim=dyn_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                use_frequent=use_frequent,
                wavelet_mlp_hidden=getattr(self, 'wavelet_mlp_hidden', 256),
            )
        elif model_type == "speech_only":
            self.model = SpeechOnlyBaseline(
                sem_dim=sem_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
            )
        elif model_type == "dynamic_only":
            self.model = DynamicOnlyBaseline(
                dyn_dim=dyn_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def forward(self, sem_feat, dyn_feat, freq_feat=None, mask=None,
                return_projections=False):
        return self.model(
            sem=sem_feat, dyn=dyn_feat, freq=freq_feat, mask=mask,
            return_projections=return_projections,
        )

    @staticmethod
    def _group_mean_logits(logits: torch.Tensor, sample_ids: torch.Tensor) -> torch.Tensor:
        B = int(sample_ids.max().item()) + 1
        grouped = torch.zeros(B, logits.shape[1], device=logits.device)
        counts = torch.zeros(B, device=logits.device)
        sample_ids_exp = sample_ids.unsqueeze(-1).expand(-1, logits.shape[1]).long()
        grouped.scatter_add_(0, sample_ids_exp, logits)
        ones = torch.ones(logits.shape[0], device=logits.device)
        counts.scatter_add_(0, sample_ids.long(), ones)
        grouped = grouped / counts.unsqueeze(-1).clamp(min=1)
        return grouped

    @staticmethod
    def _group_labels(labels: torch.Tensor, sample_ids: torch.Tensor) -> torch.Tensor:
        B = int(sample_ids.max().item()) + 1
        video_labels = torch.zeros(B, dtype=labels.dtype, device=labels.device)
        for i in range(B):
            mask = (sample_ids == i)
            video_labels[i] = labels[mask][0]
        return video_labels

    def _contrastive_residual_loss(self, proj_dict, labels):
      
        real_mask = (labels == 0).float()
        fake_mask = (labels == 1).float()
        n_real = real_mask.sum() + 1e-8
        n_fake = fake_mask.sum() + 1e-8
        n_total = len(labels)

        loss = torch.tensor(0.0, device=labels.device)
        stats = {}

        for key in ['r_dyn_raw', 'r_freq_raw']:
            if key not in proj_dict:
                continue
            r = proj_dict[key]
            r_mag = r.abs().mean(dim=-1).mean(dim=-1)          # per-clip magnitude

            stats[f'{key}_real'] = (real_mask * r_mag).sum() / n_real
            stats[f'{key}_fake'] = (fake_mask * r_mag).sum() / n_fake

            real_term = (real_mask * r_mag).sum() / n_total
            fake_term = (
                fake_mask * torch.clamp(self.contrast_margin - r_mag, min=0)
            ).sum() / n_total

            loss = loss + real_term + fake_term

        dyn_real = stats.get('r_dyn_raw_real', torch.tensor(0.0))
        dyn_fake = stats.get('r_dyn_raw_fake', torch.tensor(0.0))
        freq_real = stats.get('r_freq_raw_real', torch.tensor(0.0))
        freq_fake = stats.get('r_freq_raw_fake', torch.tensor(0.0))
        stats['res_real'] = dyn_real + freq_real
        stats['res_fake'] = dyn_fake + freq_fake

        return loss, stats

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs,
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        sem_feat = batch['sem_feat']
        dyn_feat = batch['dyn_feat']
        labels = batch['label']
        mask = batch.get('mask')
        freq_feat = batch.get('freq_feat')
        sample_ids = batch.get('sample_ids')

        output, proj_dict = self(
            sem_feat, dyn_feat, freq_feat, mask,
            return_projections=True,
        )

        # Clip-level cross-entropy loss
        ce_loss = self.ce_loss(output.logits, labels)

        # Per-clip grouping
        if self.use_per_clip and sample_ids is not None:
            video_logits = self._group_mean_logits(output.logits, sample_ids)
            video_labels = self._group_labels(labels, sample_ids)
        else:
            video_logits = output.logits
            video_labels = labels

        contrast_loss, res_stats = self._contrastive_residual_loss(
            proj_dict, labels
        )
        loss = (
            ce_loss
            + self.contrast_weight * contrast_loss
        )

        self.log('train_ce', ce_loss, prog_bar=False,
                 on_step=True, on_epoch=True)
        self.log('train_contrast', contrast_loss, prog_bar=False,
                 on_step=True, on_epoch=True)
        self.log('res_real', res_stats['res_real'], prog_bar=True,
                 on_step=True, on_epoch=True)
        self.log('res_fake', res_stats['res_fake'], prog_bar=True,
                 on_step=True, on_epoch=True)
        self.log('r_dyn_real', res_stats.get('r_dyn_raw_real', 0.0),
                 prog_bar=False, on_step=False, on_epoch=True)
        self.log('r_dyn_fake', res_stats.get('r_dyn_raw_fake', 0.0),
                 prog_bar=False, on_step=False, on_epoch=True)
        if 'r_freq_raw_real' in res_stats:
            self.log('r_freq_real', res_stats['r_freq_raw_real'],
                     prog_bar=False, on_step=False, on_epoch=True)
            self.log('r_freq_fake', res_stats['r_freq_raw_fake'],
                     prog_bar=False, on_step=False, on_epoch=True)

        self.log('train_loss', loss, prog_bar=True,
                 on_step=True, on_epoch=True)
        self.log('train_acc',
                 (video_logits.argmax(dim=-1) == video_labels).float().mean(),
                 prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def on_validation_epoch_start(self):
        self.val_logits = []
        self.val_labels = []

    def validation_step(self, batch, batch_idx):
        sem_feat = batch['sem_feat']
        dyn_feat = batch['dyn_feat']
        labels = batch['label']
        mask = batch.get('mask')
        freq_feat = batch.get('freq_feat')
        sample_ids = batch.get('sample_ids')

        output = self(sem_feat, dyn_feat, freq_feat, mask)

        # CE loss
        loss = self.ce_loss(output.logits, labels)

        # Per-clip grouping for accuracy / AUC
        if self.use_per_clip and sample_ids is not None:
            video_logits = self._group_mean_logits(output.logits, sample_ids)
            video_labels = self._group_labels(labels, sample_ids)
        else:
            video_logits = output.logits
            video_labels = labels

        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc',
                 (video_logits.argmax(dim=-1) == video_labels).float().mean(),
                 prog_bar=True, on_step=False, on_epoch=True)

        # Accumulate video-level logits for epoch-level AUC
        self.val_logits.append(video_logits.detach())
        self.val_labels.append(video_labels.detach())
        return loss

    def on_validation_epoch_end(self):
        all_logits = torch.cat(self.val_logits)
        all_labels = torch.cat(self.val_labels)
        probs = all_logits.softmax(dim=-1)[:, 1]
        try:
            auc = roc_auc_score(
                all_labels.cpu().numpy(), probs.cpu().numpy(),
            )
            self.log('val_auc', auc, prog_bar=True)
        except ValueError:
            # AUC is undefined when only one class is present
            self.log('val_auc', 0.5, prog_bar=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train TriCon for deepfake detection"
    )

    parser.add_argument(
        '--train_feature_dir', type=str, required=True,
        help='Directory containing training features',
    )
    parser.add_argument(
        '--val_feature_dir', type=str, required=True,
        help='Directory containing validation features',
    )
    parser.add_argument(
        '--train_file_list', type=str, default=None,
        help='Path to training file list '
             '(optional, uses all features if not provided)',
    )
    parser.add_argument(
        '--val_file_list', type=str, default=None,
        help='Path to validation file list '
             '(optional, uses all features if not provided)',
    )

    parser.add_argument(
        '--use_frequent', action='store_true', default=False,
        help='Incorporate frequency features with residual projection',
    )
    parser.add_argument(
        '--train_frequent_dir', type=str, default=None,
        help='Directory containing training frequency features '
             '(auto-derived from --train_feature_dir if omitted)',
    )
    parser.add_argument(
        '--val_frequent_dir', type=str, default=None,
        help='Directory containing validation frequency features '
             '(auto-derived from --val_feature_dir if omitted)',
    )

    parser.add_argument(
        '--model_type', type=str, default='TriCon',
        choices=['TriCon', 'improved', 'speech_only', 'dynamic_only'],
        help='Model variant to use',
    )

    parser.add_argument(
        '--epochs', type=int, default=10,
        help='Number of training epochs',
    )
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='Batch size',
    )
    parser.add_argument(
        '--lr', type=float, default=1e-3,
        help='Learning rate',
    )
    parser.add_argument(
        '--weight_decay', type=float, default=1e-4,
        help='Weight decay',
    )
    parser.add_argument(
        '--contrast_weight', type=float, default=0.5,
        help='Weight of contrastive residual loss. '
             'Higher values push fake residuals further from real residuals. '
             '(default: 0.5)',
    )
    parser.add_argument(
        '--contrast_margin', type=float, default=0.5,
        help='Margin for contrastive residual loss. Fake residuals are pushed '
             'to exceed this value. (default: 0.5)',
    )
    parser.add_argument(
        '--seq_len', type=int, default=25,
        help='Temporal sequence length (default: 25, standard MS-TCN output). '
             'Semantic features are resampled to match.',
    )
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='Number of data loader workers',
    )
    parser.add_argument(
        '--sem_truncate', type=int, default=0,
        help='Truncate semantic features to first N dimensions (0 = no truncation). '
             'Use with matching --sem_dim to build weaker baselines.',
    )
    parser.add_argument(
        '--dyn_truncate', type=int, default=0,
        help='Truncate dynamic (and frequency) features to first N dimensions. '
             'Use with matching --dyn_dim.',
    )

    parser.add_argument(
        '--sem_dim', type=int, default=2048,
        help='Semantic feature dimension',
    )
    parser.add_argument(
        '--dyn_dim', type=int, default=768,
        help='Dynamic feature dimension',
    )
    parser.add_argument(
        '--hidden_dim', type=int, default=512,
        help='Hidden dimension',
    )
    parser.add_argument(
        '--num_classes', type=int, default=2,
        help='Number of classes',
    )

    parser.add_argument(
        '--wavelet_mlp_hidden', type=int, default=256,
        help='Hidden dimension of the MLP in the wavelet frequency stream. '
             'Only used when --use_frequent is set.',
    )

    parser.add_argument(
        '--class_weight', type=str, default='balanced',
        help='Class weights for BCE. '
             '"balanced": compute from training-set distribution; '
             '"none": unweighted CE; '
             'or "w0,w1" for custom weights (e.g. "0.3,0.7").',
    )
    parser.add_argument(
        '--use_per_clip', action='store_true', default=True,
        help='Enable per-clip prediction and averaging. '
             'Each clip is predicted independently, '
             'then averaged to produce the video score.',
    )
    parser.add_argument(
        '--no_per_clip', action='store_false', dest='use_per_clip',
        help='Disable per-clip prediction (use legacy concatenation).',
    )

    parser.add_argument(
        '--save_dir', type=str, default='./checkpoints',
        help='Directory to save checkpoints',
    )
    parser.add_argument(
        '--log_dir', type=str, default='./logs',
        help='Directory for tensorboard logs',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (set to 0 to disable deterministic mode)',
    )
    parser.add_argument(
        '--devices', type=int, default=1,
        help='Number of GPUs (0 for CPU)',
    )

    return parser.parse_args()


def setup_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args):
    print("=" * 60)
    print("TriCon Training")
    if args.use_frequent:
        print("  Branches: r_dyn + r_freq  (dual conflict residual)")
        print("  Wavelet stream: DWT → trainable MLP ψ → IDWT (end-to-end)")
    print(f"  Loss: L_CE + {args.contrast_weight} * L_contrast")
    print(f"  Contrast margin: {args.contrast_margin}")
    print("=" * 60)

    # Model computes ffreq internally when use_frequent=True
    dataset_use_frequent = False

    train_dataset, train_loader = create_dataloader(
        file_list=args.train_file_list,
        feature_root=args.train_feature_dir,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        seq_len=args.seq_len,
        require_dynamic=True,
        use_concat=True,
        use_frequent=dataset_use_frequent,
        frequent_root=args.train_frequent_dir,
        sem_truncate=args.sem_truncate,
        dyn_truncate=args.dyn_truncate,
        use_per_clip=args.use_per_clip,
    )
    val_dataset, val_loader = create_dataloader(
        file_list=args.val_file_list,
        feature_root=args.val_feature_dir,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seq_len=args.seq_len,
        require_dynamic=True,
        use_concat=True,
        use_frequent=dataset_use_frequent,
        frequent_root=args.val_frequent_dir,
        sem_truncate=args.sem_truncate,
        dyn_truncate=args.dyn_truncate,
        use_per_clip=args.use_per_clip,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Class weights
    class_weights = None
    if args.class_weight == 'balanced':
        labels = [s["label"] for s in train_dataset.samples]
        n_real = sum(1 for l in labels if l == 0)
        n_fake = sum(1 for l in labels if l == 1)
        n_total = len(labels)
        if n_real > 0 and n_fake > 0:
            w_real = n_total / (2.0 * n_real)
            w_fake = n_total / (2.0 * n_fake)
            class_weights = torch.tensor([w_real, w_fake], dtype=torch.float32)
            print(f"  Class weights: real={w_real:.3f}, fake={w_fake:.3f} "
                  f"(ratio real:fake = {n_real}:{n_fake})")
        else:
            print(f"  Warning: cannot compute balanced class weights "
                  f"(real={n_real}, fake={n_fake}); using unweighted CE.")
    elif args.class_weight is not None:
        # Custom weights: "w0,w1"
        parts = args.class_weight.split(",")
        class_weights = torch.tensor([float(p) for p in parts], dtype=torch.float32)
        print(f"  Custom class weights: {class_weights.tolist()}")

    model = TriConModule(
        model_type=args.model_type,
        sem_dim=args.sem_dim,
        dyn_dim=args.dyn_dim,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_frequent=args.use_frequent,
        contrast_weight=args.contrast_weight,
        contrast_margin=args.contrast_margin,
        seed=args.seed,
        wavelet_mlp_hidden=getattr(args, 'wavelet_mlp_hidden', 256),
        use_per_clip=args.use_per_clip,
        class_weights=class_weights,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    best_model_callback = ModelCheckpoint(
        dirpath=args.save_dir,
        filename='best_TriCon',
        monitor='val_auc',
        mode='max',
        save_top_k=1,
        auto_insert_metric_name=False,
        verbose=True,
    )

    trainer = Trainer(
        max_epochs=args.epochs,
        callbacks=[best_model_callback],
        devices=args.devices,
        accelerator=(
            'gpu' if torch.cuda.is_available() and args.devices > 0 else 'cpu'
        ),
        enable_checkpointing=True,
        enable_progress_bar=True,
    )

    trainer.fit(model, train_loader, val_loader)

    print(f"\nTraining complete! "
          f"Best model saved to {args.save_dir}/best_TriCon.ckpt")


def main():
    args = parse_args()

    if args.seed != 0:
        actual_seed = args.seed
        setup_seed(actual_seed)
    else:
        import time
        actual_seed = int(time.time() * 1e6) % (2**31)
        print(f"Seed=0 — using time-based random seed: {actual_seed}")
        setup_seed(actual_seed)
        torch.backends.cudnn.benchmark = True

    args.seed = actual_seed

    train(args)


if __name__ == '__main__':
    main()
