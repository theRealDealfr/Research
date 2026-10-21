"""
explain.py  -  Explainability Module for Pediatric SSD Hierarchical Classifier
================================================================================
Implements four interpretability methods:

1. Integrated Gradients (IG)
   Attributes the model's prediction back to the pooled speech embedding and
   normalized age using a zero baseline and 50 interpolation steps.

2. Temporal Frame Saliency
   Gradient magnitude w.r.t. every WavLM frame (T x 1024) from the last
   hidden state, giving a per-frame importance score that maps to audio time.

3. Layer Contribution Analysis  (stats embed_mode only)
   Leverages the LayerWeightedPool softmax weights to report the learned
   contribution % of each selected WavLM transformer layer.

4. Age Sensitivity Analysis
   Sweeps normalized age from 0.0 to 1.0 (5.67 to 12.83 years) and records
   the model's SSD probability, revealing how age influences the decision.

Usage (standalone):
    python explain.py --audio sample_child_speech.wav --age 7.5 --plot

Usage (from app / infer):
    from explain import explain_prediction
    result = explain_prediction(audio_path, age)
"""

from __future__ import annotations

import os
import math
import argparse
import numpy as np
import torch
import librosa

from infer import (
    AAFFClassifier,
    LayerWeightedPool,
    _DEFAULT_STATS_LAYERS,
    _per_layer_stats,
    get_loaded_hierarchical_models,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_AGE_MIN = 5.67
_AGE_MAX = 12.83
_TARGET_SR = 16_000
_MAX_DURATION = 12.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_audio(audio_path: str):
    arr, _ = librosa.load(audio_path, sr=_TARGET_SR, mono=True)
    max_len = int(_TARGET_SR * _MAX_DURATION)
    arr = arr[:max_len]
    return arr, len(arr) / _TARGET_SR


def _norm_age(age: float) -> float:
    return float(np.clip((age - _AGE_MIN) / (_AGE_MAX - _AGE_MIN), 0.0, 1.0))


def _age_tensor(norm_age: float, device: torch.device) -> torch.Tensor:
    return torch.tensor([[norm_age]], dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# ExplainabilityEngine
# ---------------------------------------------------------------------------

class ExplainabilityEngine:
    """
    Wraps loaded models and implements all four interpretability methods.

    Parameters
    ----------
    stage1_model  : AAFFClassifier (eval mode, on device)
    wavlm         : AutoModel (eval mode, frozen, on device)
    extractor     : AutoFeatureExtractor
    device        : torch.device
    pool          : LayerWeightedPool | None  (required for stats mode)
    stats_layers  : list[int]  1-based layer indices
    embed_mode    : 'last' | 'stats'
    """

    def __init__(
        self,
        stage1_model,
        wavlm,
        extractor,
        device: torch.device,
        pool=None,
        stats_layers=None,
        embed_mode: str = "last",
    ):
        self.stage1_model = stage1_model
        self.wavlm = wavlm
        self.extractor = extractor
        self.device = device
        self.pool = pool
        self.stats_layers = stats_layers or _DEFAULT_STATS_LAYERS
        self.embed_mode = embed_mode

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def explain(
        self,
        audio_path: str,
        age: float,
        ig_steps: int = 50,
        age_sweep_steps: int = 30,
    ) -> dict:
        """
        Run all four explainability analyses and return a unified dict.
        """
        audio_arr, duration = _load_audio(audio_path)
        norm_age_val = _norm_age(age)

        emb, frame_hidden, feature_mask = self._embed_audio(audio_arr)

        ig_result = self._integrated_gradients(emb, norm_age_val, ig_steps)
        ts_result = self._temporal_saliency(frame_hidden, feature_mask, norm_age_val, duration)
        layer_result = self._layer_contributions() if self.embed_mode == "stats" else None
        age_result = self._age_sensitivity(emb, age, age_sweep_steps)

        return {
            "audio_file": os.path.basename(audio_path),
            "duration": duration,
            "age": age,
            "normalized_age": norm_age_val,
            "embed_mode": self.embed_mode,
            "integrated_gradients": ig_result,
            "temporal_saliency": ts_result,
            "layer_contributions": layer_result,
            "age_sensitivity": age_result,
        }

    # ------------------------------------------------------------------
    # 1. Integrated Gradients
    # ------------------------------------------------------------------

    def _integrated_gradients(self, emb: torch.Tensor, norm_age: float, steps: int = 50) -> dict:
        D = emb.shape[-1]
        baseline_emb = torch.zeros_like(emb)
        baseline_age = torch.zeros(1, 1, dtype=torch.float32, device=self.device)
        target_age = _age_tensor(norm_age, self.device)

        alphas = torch.linspace(0.0, 1.0, steps, device=self.device)
        interp_emb = baseline_emb + alphas[:, None, None] * (emb - baseline_emb)
        interp_emb = interp_emb.view(steps, D)
        interp_age = baseline_age + alphas[:, None, None] * (target_age - baseline_age)
        interp_age = interp_age.view(steps, 1)

        interp_emb = interp_emb.detach().requires_grad_(True)
        interp_age = interp_age.detach().requires_grad_(True)

        self.stage1_model.eval()
        logits = self.stage1_model(interp_emb, interp_age)
        ssd_logits = logits[:, 0]

        grad_emb, grad_age = torch.autograd.grad(ssd_logits.sum(), [interp_emb, interp_age])

        ig_emb = (emb.view(D) - baseline_emb.view(D)) * grad_emb.mean(dim=0)
        ig_age = (target_age.view(1) - baseline_age.view(1)) * grad_age.mean(dim=0)

        ig_emb_np = ig_emb.detach().cpu().numpy()
        ig_age_val = float(ig_age.detach().cpu().numpy()[0])

        with torch.no_grad():
            logit_x = float(self.stage1_model(emb, target_age)[0, 0].item())
            logit_base = float(self.stage1_model(baseline_emb, baseline_age)[0, 0].item())

        ig_sum = ig_emb_np.sum() + ig_age_val
        completeness_err_pct = float(
            abs(ig_sum - (logit_x - logit_base)) / max(abs(logit_x - logit_base), 1e-6) * 100.0
        )

        abs_attrs = np.abs(ig_emb_np)
        top10_idx = abs_attrs.argsort()[::-1][:10].tolist()
        top10 = [{"dim": int(i), "attribution": float(ig_emb_np[i])} for i in top10_idx]

        with torch.no_grad():
            ssd_prob = float(
                torch.softmax(self.stage1_model(emb, target_age), dim=-1)[0, 0].item() * 100.0
            )

        return {
            "emb_attributions": ig_emb_np.tolist(),
            "age_attribution": ig_age_val,
            "emb_abs_top10": top10,
            "emb_polarity": {
                "positive_pct": float((ig_emb_np > 0).mean() * 100.0),
                "negative_pct": float((ig_emb_np < 0).mean() * 100.0),
            },
            "ig_approx_error_pct": completeness_err_pct,
            "ssd_logit": float(logit_x),
            "ssd_prob": ssd_prob,
        }

    # ------------------------------------------------------------------
    # 2. Temporal Frame Saliency
    # ------------------------------------------------------------------

    def _temporal_saliency(
        self,
        frame_hidden: torch.Tensor,
        feature_mask: torch.Tensor,
        norm_age: float,
        duration: float,
    ) -> dict:
        T = frame_hidden.shape[1]
        age_t = _age_tensor(norm_age, self.device)
        mask = feature_mask.float()
        n_valid = mask.sum().item()

        frame_h = frame_hidden.detach().requires_grad_(True)
        m = mask.unsqueeze(-1)
        pooled = (frame_h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)

        self.stage1_model.eval()
        logit_ssd = self.stage1_model(pooled, age_t)[0, 0]
        logit_ssd.backward()

        frame_grad = frame_h.grad
        saliency = frame_grad.norm(dim=-1).squeeze(0)
        saliency = saliency * mask.squeeze(0)
        saliency_np = saliency.detach().cpu().numpy()

        # WavLM frame stride ~ 20 ms (320 samples @ 16 kHz)
        frame_stride_sec = 320.0 / _TARGET_SR
        frame_times = [i * frame_stride_sec + frame_stride_sec / 2 for i in range(T)]

        valid_n = int(n_valid)
        peak_idx = int(np.argmax(saliency_np[:valid_n])) if valid_n > 0 else 0
        peak_time = float(frame_times[peak_idx]) if T > 0 else 0.0

        regions = _find_top_regions(saliency_np[:valid_n], frame_times[:valid_n], k=5, window=10)

        return {
            "frame_saliency": saliency_np.tolist(),
            "frame_times_sec": frame_times,
            "peak_frame_idx": peak_idx,
            "peak_time_sec": peak_time,
            "top5_regions": regions,
            "valid_frames": valid_n,
            "total_frames": T,
        }

    # ------------------------------------------------------------------
    # 3. Layer Contribution Analysis
    # ------------------------------------------------------------------

    def _layer_contributions(self):
        if self.pool is None:
            return None

        with torch.no_grad():
            weights = torch.softmax(self.pool.weights, dim=0).cpu().numpy()

        contributions = (weights * 100.0).tolist()
        dominant_idx = int(np.argmax(weights))

        return {
            "layers": self.stats_layers,
            "raw_weights": weights.tolist(),
            "contributions": contributions,
            "dominant_layer": self.stats_layers[dominant_idx],
            "dominant_pct": float(contributions[dominant_idx]),
        }

    # ------------------------------------------------------------------
    # 4. Age Sensitivity Analysis
    # ------------------------------------------------------------------

    def _age_sensitivity(self, emb: torch.Tensor, actual_age: float, steps: int = 30) -> dict:
        self.stage1_model.eval()
        norm_ages = np.linspace(0.0, 1.0, steps)
        ages_years = [_AGE_MIN + a * (_AGE_MAX - _AGE_MIN) for a in norm_ages]
        ssd_probs, td_probs = [], []

        with torch.no_grad():
            for na in norm_ages:
                age_t = _age_tensor(float(na), self.device)
                probs = torch.softmax(self.stage1_model(emb, age_t), dim=-1)[0].cpu().numpy()
                ssd_probs.append(float(probs[0] * 100.0))
                td_probs.append(float(probs[1] * 100.0))

        flips = []
        for i in range(len(ssd_probs) - 1):
            if (ssd_probs[i] - 50.0) * (ssd_probs[i + 1] - 50.0) < 0:
                t = (50.0 - ssd_probs[i]) / (ssd_probs[i + 1] - ssd_probs[i])
                crossing = ages_years[i] + t * (ages_years[i + 1] - ages_years[i])
                flips.append(float(crossing))

        slope = float(np.polyfit(norm_ages, np.array(ssd_probs), 1)[0])

        actual_norm = _norm_age(actual_age)
        with torch.no_grad():
            actual_prob = float(
                torch.softmax(
                    self.stage1_model(emb, _age_tensor(actual_norm, self.device)), dim=-1
                )[0, 0].item() * 100.0
            )

        return {
            "age_years": ages_years,
            "ssd_probs": ssd_probs,
            "td_probs": td_probs,
            "decision_flips": flips,
            "age_sensitivity_slope": slope,
            "actual_age": actual_age,
            "actual_ssd_prob": actual_prob,
        }

    # ------------------------------------------------------------------
    # Private: audio embedding with gradient-capable frame tensors
    # ------------------------------------------------------------------

    def _embed_audio(self, audio_arr):
        max_length = int(_TARGET_SR * _MAX_DURATION)
        inputs = self.extractor(
            audio_arr,
            sampling_rate=_TARGET_SR,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        input_values = inputs["input_values"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        need_hidden = self.embed_mode == "stats"
        with torch.no_grad():
            outputs = self.wavlm(
                input_values=input_values,
                attention_mask=attention_mask,
                output_hidden_states=need_hidden,
            )

        last_hidden = outputs.last_hidden_state  # (1, T, 1024)
        feature_mask = self.wavlm._get_feature_vector_attention_mask(
            last_hidden.shape[1], attention_mask
        )

        if self.embed_mode == "stats":
            hidden_all = outputs.hidden_states
            layers_0based = [l - 1 for l in self.stats_layers]
            stats = _per_layer_stats(hidden_all, feature_mask, layers_0based)
            pooled = self.pool(stats)
        else:
            mask = feature_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        return pooled.detach(), last_hidden.detach(), feature_mask.detach()


# ---------------------------------------------------------------------------
# Helper: top-k high-saliency contiguous temporal regions (sliding window)
# ---------------------------------------------------------------------------

def _find_top_regions(saliency, times, k=5, window=10):
    if len(saliency) == 0:
        return []
    s = saliency / (saliency.max() + 1e-9)
    T = len(s)
    win = min(window, T)
    scores = np.convolve(s, np.ones(win) / win, mode="valid")
    regions = []
    used = np.zeros(len(scores), dtype=bool)
    for _ in range(k):
        if used.all():
            break
        masked = np.where(used, -1.0, scores)
        if masked.max() < 0:
            break
        idx = int(np.argmax(masked))
        end_idx = min(idx + win - 1, T - 1)
        regions.append({
            "start_sec": float(times[idx]),
            "end_sec": float(times[end_idx]),
            "mean_saliency": float(scores[idx]),
            "start_frame": idx,
            "end_frame": int(end_idx),
        })
        lo = max(0, idx - win // 2)
        hi = min(len(used), idx + win)
        used[lo:hi] = True
    return regions


# ---------------------------------------------------------------------------
# Convenience wrapper for use in app.py / infer.py
# ---------------------------------------------------------------------------

def explain_prediction(
    audio_path: str,
    age: float,
    stage1_checkpoint: str = "./results_controlled_baselines/model_c_aaff_best.pt",
    stage2_checkpoint: str = "./results_controlled_baselines/model_stage2_disorder_best.pt",
    model_checkpoint: str = "microsoft/wavlm-large",
    embed_mode: str = "last",
    stats_layers=None,
    ig_steps: int = 50,
    age_sweep_steps: int = 30,
) -> dict:
    """
    High-level convenience wrapper.  Loads models (cached), runs prediction,
    and returns the full explainability report as a JSON-serializable dict.
    """
    if stats_layers is None:
        stats_layers = _DEFAULT_STATS_LAYERS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb_dim = len(stats_layers) * 2 * 1024 if embed_mode == "stats" else 1024
    in_features = emb_dim + 1

    feature_extractor, wavlm, stage1_model, stage2_model, pool, _layers = (
        get_loaded_hierarchical_models(
            stage1_checkpoint=stage1_checkpoint,
            stage2_checkpoint=stage2_checkpoint,
            model_checkpoint=model_checkpoint,
            device=device,
            embed_mode=embed_mode,
            stats_layers=stats_layers,
            stage1_in_features=in_features,
            stage2_in_features=in_features,
        )
    )

    engine = ExplainabilityEngine(
        stage1_model=stage1_model,
        wavlm=wavlm,
        extractor=feature_extractor,
        device=device,
        pool=pool,
        stats_layers=_layers,
        embed_mode=embed_mode,
    )

    return engine.explain(audio_path=audio_path, age=age, ig_steps=ig_steps, age_sweep_steps=age_sweep_steps)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _print_report(r: dict):
    print("\n" + "=" * 65)
    print("       EXPLAINABILITY REPORT - PEDIATRIC SSD CLASSIFIER")
    print("=" * 65)
    print(f"  Audio     : {r['audio_file']}  ({r['duration']:.2f}s)")
    print(f"  Age       : {r['age']:.2f} yrs  ->  norm {r['normalized_age']:.4f}")
    print(f"  Embed mode: {r['embed_mode']}")

    ig = r["integrated_gradients"]
    print(f"\n-- Integrated Gradients --")
    print(f"  SSD Probability     : {ig['ssd_prob']:.2f}%")
    print(f"  Age Attribution     : {ig['age_attribution']:+.4f}")
    print(f"  IG Approx Error     : {ig['ig_approx_error_pct']:.2f}%  (completeness check)")
    print(f"  Emb Polarity        : {ig['emb_polarity']['positive_pct']:.1f}% pos / {ig['emb_polarity']['negative_pct']:.1f}% neg dims")
    print(f"  Top-3 Attribution Dims:")
    for entry in ig["emb_abs_top10"][:3]:
        print(f"    dim[{entry['dim']:4d}] = {entry['attribution']:+.4f}")

    ts = r["temporal_saliency"]
    print(f"\n-- Temporal Frame Saliency --")
    print(f"  Valid Frames        : {ts['valid_frames']} / {ts['total_frames']}")
    print(f"  Peak Importance     : frame {ts['peak_frame_idx']}  @ {ts['peak_time_sec']:.3f}s")
    print(f"  Top-3 Regions:")
    for reg in ts["top5_regions"][:3]:
        print(f"    [{reg['start_sec']:.2f}s - {reg['end_sec']:.2f}s]  mean saliency = {reg['mean_saliency']:.4f}")

    if r.get("layer_contributions"):
        lc = r["layer_contributions"]
        print(f"\n-- Layer Contribution (Stats Mode) --")
        for lay, pct in zip(lc["layers"], lc["contributions"]):
            bar = "X" * int(pct / 2)
            print(f"  Layer {lay:2d} : {pct:6.2f}%  {bar}")
        print(f"  Dominant Layer : {lc['dominant_layer']}  ({lc['dominant_pct']:.2f}%)")

    ag = r["age_sensitivity"]
    print(f"\n-- Age Sensitivity Analysis --")
    print(f"  Actual SSD Prob     : {ag['actual_ssd_prob']:.2f}%  @ age {ag['actual_age']:.2f}")
    print(f"  Sensitivity Slope   : {ag['age_sensitivity_slope']:+.2f}  delta%SSD / delta age_norm")
    if ag["decision_flips"]:
        for flip in ag["decision_flips"]:
            print(f"  Decision Boundary   : {flip:.2f} yrs  (SSD <-> TD crossing)")
    else:
        print("  Decision Boundary   : No crossing in [5.67, 12.83] yr range")
    print("=" * 65 + "\n")


def _plot_report(r: dict, audio_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("[warn] matplotlib not available -- skipping plots.")
        return

    stem = os.path.splitext(os.path.basename(audio_path))[0]
    out_path = f"{stem}_explainability.png"

    fig = plt.figure(figsize=(16, 10), facecolor="#0b0f19")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    text_col = "#e5e7eb"
    grid_col = "#1f2937"
    cyan = "#06b6d4"
    indigo = "#6366f1"
    emerald = "#10b981"
    red = "#ef4444"

    def _ax_style(ax, title):
        ax.set_facecolor("#111827")
        ax.tick_params(colors=text_col, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(grid_col)
        ax.set_title(title, color=cyan, fontsize=10, fontweight="bold", pad=8)
        ax.yaxis.label.set_color(text_col)
        ax.xaxis.label.set_color(text_col)

    # Panel 1: Temporal Saliency (full width)
    ax1 = fig.add_subplot(gs[0, :])
    ts = r["temporal_saliency"]
    valid_n = ts["valid_frames"]
    times = ts["frame_times_sec"][:valid_n]
    sal = np.array(ts["frame_saliency"][:valid_n])
    sal_norm = sal / (sal.max() + 1e-9)
    ax1.fill_between(times, sal_norm, alpha=0.35, color=cyan)
    ax1.plot(times, sal_norm, color=cyan, linewidth=1.0)
    for reg in ts["top5_regions"]:
        ax1.axvspan(reg["start_sec"], reg["end_sec"], alpha=0.18, color=indigo)
    ax1.axvline(ts["peak_time_sec"], color=red, linewidth=1.5, linestyle="--",
                label=f"Peak @ {ts['peak_time_sec']:.2f}s")
    ax1.set_xlabel("Time (seconds)", fontsize=9)
    ax1.set_ylabel("Normalised Saliency", fontsize=9)
    ax1.legend(fontsize=8, facecolor=grid_col, labelcolor=text_col)
    _ax_style(ax1, "Temporal Frame Saliency  (d SSD logit / d frame embedding)")

    # Panel 2: Top-10 IG dims
    ax2 = fig.add_subplot(gs[1, 0])
    ig = r["integrated_gradients"]
    top10 = ig["emb_abs_top10"]
    dims = [e["dim"] for e in top10]
    attrs = [e["attribution"] for e in top10]
    colors = [emerald if a > 0 else red for a in attrs]
    ax2.barh([f"dim {d}" for d in dims], attrs, color=colors, height=0.6)
    ax2.axvline(0, color=text_col, linewidth=0.7)
    ax2.set_xlabel("Attribution", fontsize=9)
    ax2.text(0.98, 0.04, f"Age IG: {ig['age_attribution']:+.4f}",
             transform=ax2.transAxes, ha="right", fontsize=8, color="#f59e0b")
    _ax_style(ax2, "Top-10 Integrated Gradient Dims")

    # Panel 3: Age Sensitivity
    ax3 = fig.add_subplot(gs[1, 1])
    ag = r["age_sensitivity"]
    ax3.plot(ag["age_years"], ag["ssd_probs"], color=red, linewidth=1.8, label="P(SSD)")
    ax3.plot(ag["age_years"], ag["td_probs"], color=emerald, linewidth=1.8, linestyle="--", label="P(TD)")
    ax3.axhline(50, color=text_col, linewidth=0.8, linestyle=":")
    ax3.axvline(ag["actual_age"], color="#f59e0b", linewidth=1.5, linestyle="--",
                label=f"Child age {ag['actual_age']:.1f}yr")
    for flip in ag["decision_flips"]:
        ax3.axvline(flip, color=indigo, linewidth=1.2, linestyle=":",
                    label=f"Boundary @ {flip:.1f}yr")
    ax3.set_xlabel("Age (years)", fontsize=9)
    ax3.set_ylabel("Probability (%)", fontsize=9)
    ax3.set_ylim(0, 100)
    ax3.legend(fontsize=8, facecolor=grid_col, labelcolor=text_col)
    _ax_style(ax3, "Age Sensitivity  (Prob vs Chronological Age)")

    fig.suptitle(
        f"Explainability -- {r['audio_file']}  |  Age {r['age']:.1f} yr  |  SSD {ig['ssd_prob']:.1f}%",
        color=text_col, fontsize=12, fontweight="bold", y=0.98,
    )

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[explain] Plot saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Explainability Analysis for Pediatric SSD Classifier")
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--age", type=float, required=True)
    parser.add_argument("--stage1_checkpoint", default="./results_controlled_baselines/model_c_aaff_best.pt")
    parser.add_argument("--stage2_checkpoint", default="./results_controlled_baselines/model_stage2_disorder_best.pt")
    parser.add_argument("--model_checkpoint", default="microsoft/wavlm-large")
    parser.add_argument("--embed_mode", choices=["last", "stats"], default="last")
    parser.add_argument("--stats_layers", default="6,9,12,15,18,24")
    parser.add_argument("--ig_steps", type=int, default=50)
    parser.add_argument("--age_sweep_steps", type=int, default=30)
    parser.add_argument("--plot", action="store_true", help="Generate and save matplotlib explainability figure")
    args = parser.parse_args()

    layers = [int(x.strip()) for x in args.stats_layers.split(",")]

    print("\nLoading models and computing explainability...")
    result = explain_prediction(
        audio_path=args.audio,
        age=args.age,
        stage1_checkpoint=args.stage1_checkpoint,
        stage2_checkpoint=args.stage2_checkpoint,
        model_checkpoint=args.model_checkpoint,
        embed_mode=args.embed_mode,
        stats_layers=layers,
        ig_steps=args.ig_steps,
        age_sweep_steps=args.age_sweep_steps,
    )

    _print_report(result)

    if args.plot:
        _plot_report(result, args.audio)


if __name__ == "__main__":
    main()
