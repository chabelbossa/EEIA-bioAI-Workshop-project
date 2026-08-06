# EEIA — bioAI Workshop
# Author: Généreux Akotenou — PhD student, BioinformaticLabs College of Computing / UM6P
# https://github.com/Genereux-akotenou/EEIA-bioAI-Workshop
# Licensed under the MIT License. See LICENSE at the repo root.

"""
probe.py — sonder un SAE le long d'une région génomique annotée.

Le SAE est entraîné **sans étiquette**. Pour savoir si une caractéristique apprise
correspond à quelque chose de biologique, on la regarde s'activer le long d'un morceau
de génome dont le GFF donne les positions codantes.

    probe  = GenomeProbe.load("../2-data/probe", "probe-genome")
    feats  = SAEInterpreter.extract_features(sae, probe.activations)
    scores = SignalScore.fisher_score(feats, probe.coding)
    top    = SignalScore.rank(scores, top=12)
    SAEInterpreter.plot_features(feats, top, annotations=probe.annotations())

Ce fichier est complet (outillage d'inspection) — utilisez-le tel quel. La partie
pédagogique, c'est la construction du SAE, pas ce code.
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

CODING_COLOR = "#FF9800"      # orange : région codante (CDS)
NONCODING_COLOR = "#9C27B0"   # violet : région non codante


# ============================================================================
def mask_to_annotations(mask) -> List[Tuple[int, int, str, str]]:
    """Masque booléen -> [(début, fin, étiquette, couleur), ...] pour l'affichage."""
    mask = np.asarray(mask).astype(bool)
    annotations, start, current = [], 0, mask[0]
    for i in range(1, len(mask)):
        if mask[i] != current:
            annotations.append((start, i, "codant" if current else "non codant",
                                CODING_COLOR if current else NONCODING_COLOR))
            start, current = i, mask[i]
    annotations.append((start, len(mask), "codant" if current else "non codant",
                        CODING_COLOR if current else NONCODING_COLOR))
    return annotations


# ============================================================================
class GenomeProbe:
    """Région continue de génome : activations Evo2 par position + masque codant."""

    def __init__(self, activations: np.ndarray, coding: np.ndarray, meta: dict):
        self.activations = activations      # (L, d_model)
        self.coding = coding.astype(bool)   # (L,) True = dans une CDS
        self.meta = meta

    @classmethod
    def load(cls, probe_dir, name: str) -> "GenomeProbe":
        d = Path(probe_dir) / name
        meta = json.loads((d / "meta.json").read_text())
        acts = np.fromfile(d / "acts.dat", dtype=np.float32).reshape(
            meta["total_tokens"], meta["d_model"])
        return cls(acts, np.load(d / "labels.npy"), meta)

    def annotations(self):
        return mask_to_annotations(self.coding)

    def slice(self, start: int, end: int) -> "GenomeProbe":
        """Sous-région, pour zoomer sans tout recharger."""
        m = dict(self.meta)
        m["start"] = self.meta.get("start", 0) + start
        m["total_tokens"] = end - start
        return GenomeProbe(self.activations[start:end], self.coding[start:end], m)

    def __repr__(self):
        m = self.meta
        return (f"<GenomeProbe {m.get('contig')}:{m.get('start')} "
                f"| {len(self.coding)} positions | {self.coding.mean():.0%} codantes "
                f"| {m.get('n_cds_in_span', '?')} gènes>")


# ============================================================================
class SAEInterpreter:
    """Boîte à outils d'inspection d'un SAE génomique."""

    @staticmethod
    @torch.no_grad()
    def extract_features(sae, activations, batch_size: int = 2048,
                         device: str = "cpu", scale_factor: Optional[float] = None):
        """Activations Evo2 (L, d_model) -> caractéristiques du SAE (L, d_hidden)."""
        sae = sae.to(device).eval()
        out = []
        for s in range(0, len(activations), batch_size):
            x = torch.tensor(np.asarray(activations[s:s + batch_size]),
                             dtype=torch.float32, device=device)
            if scale_factor is not None:
                x = x / scale_factor
            out.append(sae.encode(x).cpu().numpy())
        return np.concatenate(out, axis=0)

    @staticmethod
    def get_all_unique_active_features(feature_ts: np.ndarray) -> List[int]:
        """Toutes les caractéristiques allumées au moins une fois sur la région."""
        return np.flatnonzero(feature_ts.sum(axis=0) > 0).tolist()

    @staticmethod
    def get_active_features_at_position(feature_ts: np.ndarray, pos: int) -> List[int]:
        """Les caractéristiques actives à une position, triées par intensité."""
        col = feature_ts[pos]
        idx = np.flatnonzero(col > 0)
        return idx[np.argsort(col[idx])[::-1]].tolist()

    @staticmethod
    def check_periodicity(feature_ts: np.ndarray, feat_idx: int, period: int = 3) -> float:
        """Rapport de puissance à la période demandée (3 = cadre de lecture des codons).

        Puissance à la fréquence 1/period divisée par la puissance moyenne : c'est un
        rapport signal/bruit. Plus il est grand, plus l'oscillation est régulière.
        """
        signal = feature_ts[:, feat_idx]
        signal = signal - signal.mean()
        n = len(signal)
        spectrum = np.abs(np.fft.fft(signal))
        freqs = np.fft.fftfreq(n)
        idx = int(np.argmin(np.abs(freqs - 1.0 / period)))
        avg = spectrum[1:n // 2].mean()
        return float(spectrum[idx] / (avg + 1e-9))

    @staticmethod
    def rank_by_periodicity(feature_ts: np.ndarray, candidates=None, period: int = 3):
        """[(caractéristique, rapport de puissance), ...] trié par ordre décroissant."""
        if candidates is None:
            candidates = SAEInterpreter.get_all_unique_active_features(feature_ts)
        scores = [(int(j), SAEInterpreter.check_periodicity(feature_ts, j, period))
                  for j in candidates]
        scores.sort(key=lambda t: t[1], reverse=True)
        return scores

    @staticmethod
    def plot_features(feature_ts: np.ndarray, selected_indices,
                      tokens: Optional[List[str]] = None,
                      annotations: Optional[List[Tuple]] = None,
                      title: str = "Activations des caractéristiques du SAE",
                      same_scale: bool = False,
                      color: str = "#2196F3",
                      line_width: float = 1.5,
                      fill_alpha: float = 0.2,
                      y_label_size: int = 12,
                      legend_size: int = 12,
                      fig_width: Optional[float] = None,
                      show: bool = True):
        """Une caractéristique par ligne, régions annotées en fond.

        annotations : (début, fin, étiquette, couleur) — voir mask_to_annotations.
        same_scale=False : chaque ligne a sa propre échelle, ce qui rend visibles les
        caractéristiques de faible amplitude.
        """
        import matplotlib.pyplot as plt

        selected_indices = list(selected_indices)
        n, T = len(selected_indices), feature_ts.shape[0]
        width = fig_width if fig_width else min(T * 0.2 + 4, 30)
        fig, axes = plt.subplots(n, 1, figsize=(width, 2.2 * n), sharex=True,
                                 squeeze=False)
        axes = axes[:, 0]
        x = np.arange(T)
        global_ymax = feature_ts[:, selected_indices].max() * 1.1 + 1e-5

        for ax, feat_idx in zip(axes, selected_indices):
            y = feature_ts[:, feat_idx]
            ax.plot(x, y, lw=line_width, color=color, label=f"Feat {feat_idx}")
            ax.fill_between(x, 0, y, alpha=fill_alpha, color=color)

            if annotations:
                for start, end, label, colour in annotations:
                    ax.axvspan(start, end, alpha=0.2, color=colour, label=label)

            ax.set_xlim(0, T - 1)
            ax.set_ylim(0, global_ymax if same_scale else y.max() * 1.1 + 1e-5)
            ax.set_ylabel(f"Feat {feat_idx}", fontsize=y_label_size, fontweight="bold")

            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            if by_label:
                ax.legend(by_label.values(), by_label.keys(), loc="upper right",
                          fontsize=legend_size, framealpha=0.6)
            ax.grid(True, alpha=0.15)

        if tokens and len(tokens) == T:
            axes[-1].set_xticks(x)
            axes[-1].set_xticklabels(tokens, rotation=90, fontsize=7, family="monospace")
        else:
            axes[-1].set_xlabel("Position (paires de bases)", fontsize=y_label_size)

        fig.suptitle(title, fontsize=y_label_size + 2, fontweight="bold", y=1.01)
        plt.tight_layout()
        if show:
            plt.show()
        return fig


# ============================================================================
class SignalScore:
    """Scores de séparation codant / non-codant, caractéristique par caractéristique."""

    @staticmethod
    def fisher_score(feature_ts: np.ndarray, coding_mask: np.ndarray,
                     normalize: bool = False):
        """(moy_codant - moy_non_codant)² / (variances sommées).

        Le score de référence : il récompense un écart de moyenne net ET stable.
        """
        coding, noncoding = feature_ts[coding_mask], feature_ts[~coding_mask]
        score = ((coding.mean(axis=0) - noncoding.mean(axis=0)) ** 2 /
                 (coding.var(axis=0) + noncoding.var(axis=0) + 1e-8))
        return SignalScore.normalize(score) if normalize else score

    @staticmethod
    def selectivity_score(feature_ts: np.ndarray, coding_mask: np.ndarray,
                          thresh: float = 0, normalize: bool = False):
        """|P(active | codant) - P(active | non-codant)| : différence de fréquence
        d'allumage entre les deux types de régions."""
        active = feature_ts > thresh
        score = np.abs(active[coding_mask].mean(axis=0) -
                       active[~coding_mask].mean(axis=0))
        return SignalScore.normalize(score) if normalize else score

    @staticmethod
    def mutual_info(feature_ts: np.ndarray, coding_mask: np.ndarray,
                    normalize: bool = False):
        """Information mutuelle avec l'étiquette (capte aussi le non-linéaire).
        Lent sur des dizaines de milliers de caractéristiques."""
        from sklearn.feature_selection import mutual_info_classif
        mi = mutual_info_classif(feature_ts, coding_mask.astype(int),
                                 discrete_features=False)
        return SignalScore.normalize(mi) if normalize else mi

    @staticmethod
    def normalize(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    @staticmethod
    def rank(scores: np.ndarray, top: Optional[int] = None) -> np.ndarray:
        """Indices triés par score décroissant, scores nuls écartés."""
        nz = np.flatnonzero(scores > 0)
        order = nz[np.argsort(scores[nz])[::-1]]
        return order[:top] if top else order
