"""Multimodal PTB-XL model components.

This is the first model step after preprocessing and data loading.
It implements three encoders:
- ECG waveform encoder
- tabular metadata encoder
- SCP-code encoder

The branch embeddings are fused with a gated residual head and a final
multi-label classifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Tuple

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    ecg_channels: int = 12
    ecg_embedding_dim: int = 128
    tabular_embedding_dim: int = 128
    scp_embedding_dim: int = 128
    metadata_embedding_dim: int = 128
    fusion_dim: int = 128
    num_heads: int = 4
    dropout: float = 0.3
    num_classes: int = 5


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.block(x) + x)


class ECGEncoder(nn.Module):
    def __init__(self, in_channels: int = 12, embedding_dim: int = 128, num_heads: int = 4) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.res1 = ResidualConvBlock(64)
        self.proj = nn.Sequential(
            nn.Conv1d(64, embedding_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 2,
            dropout=0.1,
            batch_first=True,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.res1(x)
        x = self.proj(x)
        x = x.transpose(1, 2)
        x = self.transformer(x)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        return x

    def forward_debug(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        stem = self.stem(x)
        res = self.res1(stem)
        proj = self.proj(res)
        tokens = proj.transpose(1, 2)
        trans = self.transformer(tokens)
        pooled = self.pool(trans.transpose(1, 2)).squeeze(-1)
        return {
            "input": x,
            "stem": stem,
            "residual": res,
            "projected": proj,
            "tokens": tokens,
            "transformer": trans,
            "embedding": pooled,
        }


class TabularEncoder(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        hidden_dim = max(64, embedding_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SCPEncoder(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        hidden_dim = max(64, embedding_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GatedFusion(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Sigmoid(),
        )
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.residual = nn.Linear(output_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.proj(x)
        gated = self.gate(x)
        return gated * projected + self.residual(projected)


class BidirectionalCrossAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.ecg_to_meta = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.meta_to_ecg = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, ecg_tokens: torch.Tensor, meta_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ecg_ctx, _ = self.ecg_to_meta(query=ecg_tokens, key=meta_tokens, value=meta_tokens)
        meta_ctx, _ = self.meta_to_ecg(query=meta_tokens, key=ecg_tokens, value=ecg_tokens)
        return ecg_ctx, meta_ctx


class MultimodalPTBXLNet(nn.Module):
    def __init__(self, tabular_dim: int, scp_dim: int, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig(num_classes=5)

        self.ecg_encoder = ECGEncoder(
            in_channels=self.config.ecg_channels,
            embedding_dim=self.config.ecg_embedding_dim,
            num_heads=self.config.num_heads,
        )
        self.tab_encoder = TabularEncoder(
            input_dim=tabular_dim,
            embedding_dim=self.config.tabular_embedding_dim,
            dropout=0.2,
        )
        self.scp_encoder = SCPEncoder(
            input_dim=scp_dim,
            embedding_dim=self.config.scp_embedding_dim,
            dropout=0.2,
        )

        self.metadata_projector = nn.Sequential(
            nn.Linear(self.config.tabular_embedding_dim + self.config.scp_embedding_dim, self.config.metadata_embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.dropout),
        )

        self.cross_attention = BidirectionalCrossAttention(
            embed_dim=self.config.ecg_embedding_dim,
            num_heads=self.config.num_heads,
        )

        fusion_input_dim = (
            self.config.ecg_embedding_dim
            + self.config.metadata_embedding_dim
        )
        self.fusion = GatedFusion(fusion_input_dim, self.config.fusion_dim, dropout=self.config.dropout)
        self.projection = nn.Sequential(
            nn.Linear(self.config.fusion_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.dropout),
        )
        self.head = nn.Linear(64, self.config.num_classes)

    def forward(
        self,
        ecg: torch.Tensor,
        tab: torch.Tensor,
        scp: torch.Tensor,
        return_embeddings: bool = False,
        return_probabilities: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        ecg_emb = self.ecg_encoder(ecg)
        tab_emb = self.tab_encoder(tab)
        scp_emb = self.scp_encoder(scp)

        metadata_emb = self.metadata_projector(torch.cat([tab_emb, scp_emb], dim=1))

        ecg_tokens = ecg_emb.unsqueeze(1)
        meta_tokens = metadata_emb.unsqueeze(1)
        ecg_ctx, meta_ctx = self.cross_attention(ecg_tokens, meta_tokens)

        ecg_ctx = ecg_ctx.squeeze(1)
        meta_ctx = meta_ctx.squeeze(1)

        fused = torch.cat([ecg_ctx, meta_ctx], dim=1)
        fused = self.fusion(fused)
        fused = fused + ecg_emb + metadata_emb
        fused = self.projection(fused)
        logits = self.head(fused)

        if return_probabilities:
            probs = torch.sigmoid(logits)
            if return_embeddings:
                return probs, {
                    "ecg": ecg_emb,
                    "tab": tab_emb,
                    "scp": scp_emb,
                    "metadata": metadata_emb,
                    "ecg_ctx": ecg_ctx,
                    "meta_ctx": meta_ctx,
                    "fusion": fused,
                }
            return probs

        if return_embeddings:
            return logits, {
                "ecg": ecg_emb,
                "tab": tab_emb,
                "scp": scp_emb,
                "metadata": metadata_emb,
                "ecg_ctx": ecg_ctx,
                "meta_ctx": meta_ctx,
                "fusion": fused,
            }
        return logits

    def predict_proba(self, ecg: torch.Tensor, tab: torch.Tensor, scp: torch.Tensor) -> torch.Tensor:
        """Return sigmoid probabilities for multilabel inference."""

        return self.forward(ecg, tab, scp, return_probabilities=True)  # type: ignore[return-value]

    def forward_debug(self, ecg: torch.Tensor, tab: torch.Tensor, scp: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return named outputs from the main layers for inspection."""

        ecg_layers = self.ecg_encoder.forward_debug(ecg)
        tab_emb = self.tab_encoder(tab)
        scp_emb = self.scp_encoder(scp)
        metadata_emb = self.metadata_projector(torch.cat([tab_emb, scp_emb], dim=1))

        ecg_tokens = ecg_layers["embedding"].unsqueeze(1)
        meta_tokens = metadata_emb.unsqueeze(1)
        ecg_ctx, meta_ctx = self.cross_attention(ecg_tokens, meta_tokens)
        ecg_ctx = ecg_ctx.squeeze(1)
        meta_ctx = meta_ctx.squeeze(1)

        fusion_input = torch.cat([ecg_ctx, meta_ctx], dim=1)
        gated = self.fusion(fusion_input)
        residual_added = gated + ecg_layers["embedding"] + metadata_emb
        projection = self.projection(residual_added)
        logits = self.head(projection)
        probs = torch.sigmoid(logits)

        return {
            "ecg_input": ecg_layers["input"],
            "ecg_stem": ecg_layers["stem"],
            "ecg_residual": ecg_layers["residual"],
            "ecg_projected": ecg_layers["projected"],
            "ecg_tokens": ecg_layers["tokens"],
            "ecg_transformer": ecg_layers["transformer"],
            "ecg_embedding": ecg_layers["embedding"],
            "tab_embedding": tab_emb,
            "scp_embedding": scp_emb,
            "metadata_embedding": metadata_emb,
            "ecg_context": ecg_ctx,
            "metadata_context": meta_ctx,
            "fusion_input": fusion_input,
            "gated_fusion": gated,
            "residual_added": residual_added,
            "projection": projection,
            "logits": logits,
            "probabilities": probs,
        }


def build_model_from_batches(batch: Dict[str, torch.Tensor], num_classes: int) -> MultimodalPTBXLNet:
    """Helper to infer encoder input sizes from a batch."""

    tabular_dim = batch["tab"].shape[-1]
    scp_dim = batch["scp"].shape[-1]
    config = ModelConfig(num_classes=num_classes)
    return MultimodalPTBXLNet(tabular_dim=tabular_dim, scp_dim=scp_dim, config=config)


if __name__ == "__main__":
    # Prefer a real processed PTB-XL sample so the output reflects actual cleaned data.
    base_dir = Path(__file__).resolve().parent.parent / "processed"
    if base_dir.exists():
        project_root = Path(__file__).resolve().parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from src.multimodal_data import load_processed_datasets

        datasets = load_processed_datasets(base_dir)
        sample = datasets["train"][0]
        model = MultimodalPTBXLNet(
            tabular_dim=sample["tab"].shape[-1],
            scp_dim=sample["scp"].shape[-1],
            config=ModelConfig(num_classes=sample["label"].shape[-1]),
        )
        ecg = sample["ecg"].unsqueeze(0)
        tab = sample["tab"].unsqueeze(0)
        scp = sample["scp"].unsqueeze(0)
    else:
        # Fallback only if processed arrays are not present yet.
        ecg = torch.randn(2, 12, 5000)
        tab = torch.randn(2, 25)
        scp = torch.randn(2, 71)
        model = MultimodalPTBXLNet(tabular_dim=25, scp_dim=71, config=ModelConfig(num_classes=5))

    debug = model.forward_debug(ecg, tab, scp)
    for name, tensor in debug.items():
        print(f"{name}: {tuple(tensor.shape)}")