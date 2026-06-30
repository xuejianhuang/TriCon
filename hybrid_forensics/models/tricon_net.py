import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import pywt
except ImportError:
    pywt = None


class WaveletFrequencyStream(nn.Module):
    def __init__(self, dim: int = 768, wavelet: str = "db4", mlp_hidden: int = 256):
        super().__init__()
        if pywt is None:
            raise ImportError(
                "PyWavelets is required for WaveletFrequencyStream. "
                "Install with: pip install PyWavelets"
            )
        self.dim = dim

        w = pywt.Wavelet(wavelet)

        dec_lo = torch.tensor(w.dec_lo[::-1].copy(), dtype=torch.float32)
        dec_hi = torch.tensor(w.dec_hi[::-1].copy(), dtype=torch.float32)
        rec_lo = torch.tensor(w.rec_lo.copy(), dtype=torch.float32)
        rec_hi = torch.tensor(w.rec_hi.copy(), dtype=torch.float32)

        self.register_buffer("dec_lo", dec_lo)
        self.register_buffer("dec_hi", dec_hi)
        self.register_buffer("rec_lo", rec_lo)
        self.register_buffer("rec_hi", rec_hi)
        self.filter_len = len(w.dec_lo)

        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.Tanh(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.Tanh(),
            nn.Linear(mlp_hidden, dim),
            nn.Tanh(),
        )

    def _dwt_channels(self, x: torch.Tensor):
        """Apply 1-D DWT along temporal axis for all channels."""
        pad = self.filter_len - 1
        x_padded = F.pad(x, (pad, pad), mode="replicate")

        k_lo = self.dec_lo.view(1, 1, -1)
        k_hi = self.dec_hi.view(1, 1, -1)

        cA = F.conv1d(x_padded, k_lo, stride=2)
        cD = F.conv1d(x_padded, k_hi, stride=2)
        return cA, cD

    def _idwt_channels(self, cA: torch.Tensor, cD: torch.Tensor,
                        target_len: int):
        """Apply 1-D IDWT returning target_len samples."""
        B_flat, _, T_h = cA.shape

        T_up = T_h * 2
        cA_up = torch.zeros(B_flat, 1, T_up, device=cA.device)
        cD_up = torch.zeros(B_flat, 1, T_up, device=cA.device)
        cA_up[:, :, 0::2] = cA
        cD_up[:, :, 0::2] = cD

        pad = self.filter_len - 1
        cA_padded = F.pad(cA_up, (pad, pad), mode="replicate")
        cD_padded = F.pad(cD_up, (pad, pad), mode="replicate")

        k_lo = self.rec_lo.view(1, 1, -1)
        k_hi = self.rec_hi.view(1, 1, -1)

        xA = F.conv1d(cA_padded, k_lo, stride=1)
        xD = F.conv1d(cD_padded, k_hi, stride=1)

        x = xA + xD

        out_len = x.shape[-1]
        if out_len > target_len:
            start = (out_len - target_len) // 2
            x = x[:, :, start:start + target_len]
        elif out_len < target_len:
            x = F.pad(x, (0, target_len - out_len), mode="reflect")

        return x

    def forward(self, fdyn: torch.Tensor) -> torch.Tensor:
        """Forward pass: DWT -> frequency MLP -> IDWT."""
        B, T, D = fdyn.shape
        T_orig = T

        if T % 2 != 0:
            fdyn = F.pad(fdyn, (0, 0, 0, 1), mode="replicate")
            T = T + 1

        x = fdyn.permute(0, 2, 1).reshape(B * D, 1, T).contiguous()
        cA, cD = self._dwt_channels(x)
        T_h = cA.shape[-1]

        # Frequency MLP
        cA_t = cA.reshape(B, D, T_h).permute(0, 2, 1)
        cA_mod = self.mlp(cA_t)
        cA_mod_r = cA_mod.permute(0, 2, 1).reshape(B * D, 1, T_h).contiguous()

        # IDWT reconstruction
        x_recon = self._idwt_channels(cA_mod_r, cD, T)
        ffreq = x_recon.reshape(B, D, T).permute(0, 2, 1).contiguous()

        if T != T_orig:
            ffreq = ffreq[:, :T_orig, :]

        return ffreq


class Projector(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SemanticConditionedGating(nn.Module):

    def __init__(
        self,
        sem_dim: int = 2048,
        dyn_dim: int = 768,
        hidden_dim: int = 512,
        use_frequent: bool = False,
    ):
        super().__init__()
        self.use_frequent = use_frequent

        in_dim = sem_dim + dyn_dim * 2
        if use_frequent:
            in_dim += dyn_dim * 2
            out_dim = dyn_dim * 2
        else:
            out_dim = dyn_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim),
            nn.Sigmoid(),
        )

        self.alpha_d = nn.Parameter(torch.ones(1))
        if use_frequent:
            self.alpha_f = nn.Parameter(torch.ones(1))

    def forward(
        self,
        sem: torch.Tensor,
        dyn: torch.Tensor,
        r_dyn: torch.Tensor,
        freq: torch.Tensor = None,
        r_freq: torch.Tensor = None,
    ):
        x = torch.cat([sem, dyn, r_dyn], dim=-1)

        if self.use_frequent and freq is not None and r_freq is not None:
            x = torch.cat([x, freq, r_freq], dim=-1)
            gates = self.net(x)
            g_dyn, g_freq = gates.chunk(2, dim=-1)
            fused = dyn + self.alpha_d * g_dyn * r_dyn + self.alpha_f * g_freq * r_freq
            return fused, g_dyn, g_freq
        else:
            g_dyn = self.net(x)
            fused = dyn + self.alpha_d * g_dyn * r_dyn
            return fused, g_dyn, None


class TemporalAttention(nn.Module):

    def __init__(self, dim: int = 768, attention_dim: int = 256):
        super().__init__()
        self.W = nn.Linear(dim, attention_dim)
        self.v = nn.Linear(attention_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        e = self.v(torch.tanh(self.W(x)))

        if mask is not None:
            e = e.masked_fill(~mask.unsqueeze(-1), float('-inf'))

        beta = torch.softmax(e, dim=1)
        z = (beta * x).sum(dim=1)
        return z, beta


class TriConInterface(nn.Module):

    def __init__(
        self,
        sem_dim: int = 2048,
        dyn_dim: int = 768,
        hidden_dim: int = 512,
        num_classes: int = 2,
        use_frequent: bool = False,
        wavelet_mlp_hidden: int = 256,
    ):
        super().__init__()
        self.sem_dim = sem_dim
        self.dyn_dim = dyn_dim
        self.hidden_dim = hidden_dim
        self.use_frequent = use_frequent

        self.sem_to_dyn = Projector(sem_dim, hidden_dim, dyn_dim)

        if use_frequent:
            self.sem_to_freq = Projector(sem_dim, hidden_dim, dyn_dim)
            self.wavelet_stream = WaveletFrequencyStream(
                dim=dyn_dim, wavelet="db4", mlp_hidden=wavelet_mlp_hidden,
            )

        self.gating = SemanticConditionedGating(
            sem_dim=sem_dim,
            dyn_dim=dyn_dim,
            hidden_dim=hidden_dim,
            use_frequent=use_frequent,
        )

        self.temporal_attention = TemporalAttention(
            dim=dyn_dim,
            attention_dim=hidden_dim // 2,
        )

        self.classifier = nn.Sequential(
            nn.Linear(dyn_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        sem: torch.Tensor,
        dyn: torch.Tensor,
        freq: torch.Tensor = None,
        mask: torch.Tensor = None,
        return_projections: bool = False,
    ):
        dyn_pred = self.sem_to_dyn(sem)
        r_dyn_raw = dyn - dyn_pred
        r_dyn = r_dyn_raw.abs()

        freq_pred = None
        r_freq_raw = None
        r_freq = None
        freq_arg = None
        if self.use_frequent:
            freq_arg = self.wavelet_stream(dyn)
            freq_pred = self.sem_to_freq(sem)
            r_freq_raw = freq_arg - freq_pred
            r_freq = r_freq_raw.abs()

        fused, g_dyn, g_freq = self.gating(
            sem=sem, dyn=dyn, r_dyn=r_dyn,
            freq=freq_arg, r_freq=r_freq,
        )

        if mask is not None:
            fused = fused * mask.unsqueeze(-1)

        z, beta = self.temporal_attention(fused, mask=mask)

        logits = self.classifier(z)

        output = torch.zeros_like(logits)
        output[:] = logits
        output.logits = logits

        if return_projections:
            proj_dict = {
                "dyn_pred": dyn_pred,
                "r_dyn_raw": r_dyn_raw,
                "r_dyn": r_dyn,
                "g_dyn": g_dyn,
                "g_freq": g_freq,
                "attention_weights": beta,
            }
            if freq_pred is not None:
                proj_dict["freq_pred"] = freq_pred
                proj_dict["r_freq_raw"] = r_freq_raw
                proj_dict["r_freq"] = r_freq
            return output, proj_dict

        return output


class ImprovedTriCon(TriConInterface):

    def __init__(
        self,
        sem_dim: int = 2048,
        dyn_dim: int = 768,
        hidden_dim: int = 512,
        num_classes: int = 2,
        use_frequent: bool = False,
        wavelet_mlp_hidden: int = 256,
    ):
        super().__init__(
            sem_dim=sem_dim,
            dyn_dim=dyn_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            use_frequent=use_frequent,
            wavelet_mlp_hidden=wavelet_mlp_hidden,
        )


class ConcatBaseline(TriConInterface):

    def __init__(
        self,
        sem_dim: int = 2048,
        dyn_dim: int = 768,
        hidden_dim: int = 512,
        num_classes: int = 2,
        use_frequent: bool = False,
        wavelet_mlp_hidden: int = 256,
    ):
        super().__init__(
            sem_dim=sem_dim,
            dyn_dim=dyn_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            use_frequent=use_frequent,
            wavelet_mlp_hidden=wavelet_mlp_hidden,
        )


class SpeechOnlyBaseline(nn.Module):

    def __init__(
        self,
        sem_dim: int = 2048,
        hidden_dim: int = 512,
        num_classes: int = 2,
    ):
        super().__init__()
        self.sem_projector = Projector(sem_dim, hidden_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        sem: torch.Tensor,
        dyn: torch.Tensor = None,
        mask: torch.Tensor = None,
        return_projections: bool = False,
        **_kwargs,
    ):
        sem_proj = self.sem_projector(sem)
        if mask is not None:
            m = mask.unsqueeze(-1).float()
            pooled = (sem_proj * m).sum(dim=1) / (m.sum(dim=1) + 1e-8)
        else:
            pooled = sem_proj.mean(dim=1)

        logits = self.classifier(pooled)
        output = torch.zeros_like(logits)
        output[:] = logits
        output.logits = logits

        if return_projections:
            return output, {"sem_proj": sem_proj}
        return output


class DynamicOnlyBaseline(nn.Module):

    def __init__(
        self,
        dyn_dim: int = 768,
        hidden_dim: int = 512,
        num_classes: int = 2,
    ):
        super().__init__()
        self.dyn_projector = Projector(dyn_dim, hidden_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        sem: torch.Tensor = None,
        dyn: torch.Tensor = None,
        mask: torch.Tensor = None,
        return_projections: bool = False,
        **_kwargs,
    ):
        dyn_proj = self.dyn_projector(dyn)
        if mask is not None:
            m = mask.unsqueeze(-1).float()
            pooled = (dyn_proj * m).sum(dim=1) / (m.sum(dim=1) + 1e-8)
        else:
            pooled = dyn_proj.mean(dim=1)

        logits = self.classifier(pooled)
        output = torch.zeros_like(logits)
        output[:] = logits
        output.logits = logits

        if return_projections:
            return output, {"dyn_proj": dyn_proj}
        return output


TriCon = ImprovedTriCon
