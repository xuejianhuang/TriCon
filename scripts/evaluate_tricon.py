import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn import metrics
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from hybrid_forensics.models.tricon_net import (
    ImprovedTriCon,
    SpeechOnlyBaseline,
    DynamicOnlyBaseline,
)
from hybrid_forensics.datasets import create_dataloader


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


def _group_labels(labels: torch.Tensor, sample_ids: torch.Tensor) -> torch.Tensor:
    """Extract per-video labels from per-clip labels."""
    B = int(sample_ids.max().item()) + 1
    video_labels = torch.zeros(B, dtype=labels.dtype, device=labels.device)
    for i in range(B):
        mask = (sample_ids == i)
        video_labels[i] = labels[mask][0]
    return video_labels


def evaluate_from_cache(
    feature_dir: str,
    file_list: str,
    checkpoint_path: str,
    output_dir: str,
    batch_size: int = 16,
    seq_len: int = 25,
    num_workers: int = 4,
    threshold: float = 0.5,
    frequent_dir: str = None,
    sem_truncate: int = 0,
    dyn_truncate: int = 0,
    real_file_list: str = None,
    use_per_clip: bool = True,
) -> dict:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(checkpoint_path, map_location=device)
    hp = ckpt.get("hyper_parameters", {})
    train_seed = hp.get("seed", "unknown")

    if frequent_dir is None:
        stored = hp.get("train_frequent_dir") or hp.get("frequent_root")
        if stored:
            frequent_dir = stored.replace("/train", "/test").replace("\\train", "\\test")
        else:
            frequent_dir = feature_dir.replace("cached_features", "frequent_features")

    print("=" * 60)
    print("TriCon Evaluation")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Training seed: {train_seed}")
    print(f"Feature directory: {feature_dir}")
    print(f"Frequency directory: {frequent_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Decision threshold: {threshold}")

    model_type = hp.get("model_type", "TriCon")
    use_frequent = hp.get("use_frequent", False)
    sem_dim = hp.get("sem_dim", 2048)
    dyn_dim = hp.get("dyn_dim", 768)
    hidden_dim = hp.get("hidden_dim", 512)
    num_classes = hp.get("num_classes", 2)

    print(f"\nLoading TriCon...")
    print(f"  Architecture: {model_type}")

    if model_type in ("TriCon", "improved"):
        model = ImprovedTriCon(
            sem_dim=sem_dim, dyn_dim=dyn_dim,
            hidden_dim=hidden_dim, num_classes=num_classes,
            use_frequent=use_frequent,
        ).to(device)
    elif model_type == "speech_only":
        model = SpeechOnlyBaseline(
            sem_dim=sem_dim,
            hidden_dim=hidden_dim, num_classes=num_classes,
        ).to(device)
    elif model_type == "dynamic_only":
        model = DynamicOnlyBaseline(
            dyn_dim=dyn_dim,
            hidden_dim=hidden_dim, num_classes=num_classes,
        ).to(device)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    sd = ckpt["state_dict"]
    sd = {k.replace("model.", ""): v for k, v in sd.items()
          if not k.startswith("ce_loss.")}
    model.load_state_dict(sd, strict=False)
    model.eval()

    if 'epoch' in ckpt:
        print(f"  Checkpoint: Epoch {ckpt['epoch']}")
    if 'metrics' in ckpt and 'val_acc' in ckpt['metrics']:
        print(f"  Validation accuracy: {ckpt['metrics']['val_acc']:.4f}")

    print("Model loaded successfully")

    print(f"\nLoading test data...")
    _, test_loader = create_dataloader(
        file_list=file_list,
        feature_root=feature_dir,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        seq_len=seq_len,
        require_dynamic=True,
        use_concat=True,
        use_frequent=use_frequent,
        frequent_root=frequent_dir,
        sem_truncate=sem_truncate,
        dyn_truncate=dyn_truncate,
        real_file_list=real_file_list,
        use_per_clip=use_per_clip,
    )
    print(f"  Loaded {len(test_loader.dataset)} test samples")
    if use_frequent:
        print(f"  Frequency features: enabled")

    print(f"\nEvaluating...")
    all_labels = []
    all_scores = []
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Processing batches"):
            labels = batch['label'].to(device)
            sem = batch['sem_feat'].to(device)
            dyn = batch['dyn_feat'].to(device)
            freq = batch.get('freq_feat')
            if freq is not None:
                freq = freq.to(device)
            mask = batch.get('mask', None)
            if mask is not None:
                mask = mask.to(device)
            sample_ids = batch.get('sample_ids')

            output = model(sem=sem, dyn=dyn, freq=freq, mask=mask)

            # Per-clip grouping: average clip predictions → video predictions
            if sample_ids is not None:
                sample_ids = sample_ids.to(device)
                video_logits = _group_mean_logits(output.logits, sample_ids)
                video_labels = _group_labels(labels, sample_ids)
                probs = torch.softmax(video_logits, dim=-1)
                fake_probs = probs[:, 1]
                preds = (fake_probs >= threshold).long()
                all_labels.extend(video_labels.cpu().numpy())
            else:
                probs = torch.softmax(output.logits, dim=-1)
                fake_probs = probs[:, 1]
                preds = (fake_probs >= threshold).long()
                all_labels.extend(labels.cpu().numpy())

            all_scores.extend(fake_probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    labels = np.array(all_labels)
    scores = np.array(all_scores)
    preds = np.array(all_preds)

    fpr, tpr, _ = metrics.roc_curve(labels, scores)
    auc = metrics.auc(fpr, tpr)
    acc = metrics.accuracy_score(labels, preds)
    precision = metrics.precision_score(labels, preds, zero_division=0)
    recall = metrics.recall_score(labels, preds, zero_division=0)
    f1 = metrics.f1_score(labels, preds, zero_division=0)

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"AUC:       {auc:.4f} ({auc*100:.2f}%)")
    print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("=" * 60)

    cm = metrics.confusion_matrix(labels, preds)
    print("\nConfusion Matrix:")
    print("              Predicted")
    print("              Real  Fake")
    print(f"Actual Real   {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"       Fake   {cm[1,0]:4d}  {cm[1,1]:4d}")

    os.makedirs(output_dir, exist_ok=True)
    results = {
        'seed': train_seed,
        'metrics': {
            'auc': float(auc),
            'accuracy': float(acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
        },
        'confusion_matrix': cm.tolist(),
        'threshold': threshold,
        'checkpoint': checkpoint_path,
        'num_samples': len(labels),
        'num_real': int((labels == 0).sum()),
        'num_fake': int((labels == 1).sum()),
        'predictions': {
            'labels': labels.tolist(),
            'scores': scores.tolist(),
            'preds': preds.tolist(),
        },
    }

    output_file = os.path.join(output_dir, 'TriCon_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate TriCon using cached features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--feature_dir',
        help='Directory containing cached test features (alias for --test_feature_dir)'
    )
    parser.add_argument(
        '--test_feature_dir',
        help='Directory containing cached test features (overrides --feature_dir)'
    )
    parser.add_argument(
        '--file_list',
        required=True,
        help='Path to test file_list.txt'
    )
    parser.add_argument(
        '--checkpoint',
        required=True,
        help='Path to trained TriCon checkpoint (.pt or .ckpt file)'
    )
    parser.add_argument(
        '--output_dir',
        default='results/TriCon',
        help='Output directory for evaluation results'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--seq_len',
        type=int,
        default=25,
        help='Temporal sequence length (default: 25)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Decision threshold for classification (default: 0.5)'
    )
    parser.add_argument(
        '--frequent_dir',
        default=None,
        help='Directory containing frequency features. '
             'Auto-detected from checkpoint or derived from --feature_dir when omitted.'
    )
    parser.add_argument(
        '--sem_truncate', type=int, default=0,
        help='Truncate semantic features to first N dims (must match training).'
    )
    parser.add_argument(
        '--dyn_truncate', type=int, default=0,
        help='Truncate dynamic/freq features to first N dims (must match training).'
    )
    parser.add_argument(
        '--use_per_clip', action='store_true', default=True,
        help='Enable per-clip prediction averaging (default behavior).',
    )
    parser.add_argument(
        '--no_per_clip', action='store_false', dest='use_per_clip',
        help='Disable per-clip prediction (legacy mode).',
    )
    parser.add_argument(
        '--real_file_list',
        default=None,
        help='Optional second file_list.txt with real samples (label=0). '
             'Use this when --file_list only contains fake samples.'
    )

    args = parser.parse_args()

    test_feature_dir = args.test_feature_dir if args.test_feature_dir else args.feature_dir
    if not test_feature_dir:
        parser.error("Either --test_feature_dir or --feature_dir must be specified")

    evaluate_from_cache(
        feature_dir=test_feature_dir,
        file_list=args.file_list,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_workers=args.num_workers,
        threshold=args.threshold,
        frequent_dir=args.frequent_dir,
        sem_truncate=args.sem_truncate,
        dyn_truncate=args.dyn_truncate,
        real_file_list=args.real_file_list,
        use_per_clip=args.use_per_clip,
    )


if __name__ == '__main__':
    main()
