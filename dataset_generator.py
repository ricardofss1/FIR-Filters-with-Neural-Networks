import argparse
import os

import numpy as np
from scipy import signal

from fir_utils import build_response_masks, design_reference_filter


def normalize_profile_name(profile):
    if profile == "hard-focused":
        return "hard"
    return profile


def sample_log_uniform(rng, low, high):
    return float(10 ** rng.uniform(np.log10(low), np.log10(high)))


def sample_edge_biased_cutoff(rng, min_fc, max_fc):
    mode = rng.choice(["low_edge", "high_edge", "mid"], p=[0.36, 0.36, 0.28])
    if mode == "low_edge":
        upper = min(max_fc, max(min_fc * 12.0, 0.08))
        return sample_log_uniform(rng, min_fc, max(upper, min_fc * 1.01))
    if mode == "high_edge":
        lower = max(min_fc, min(max_fc - 0.02, 0.28))
        return float(rng.uniform(lower, max_fc))
    lower = max(min_fc, 0.06)
    upper = min(max_fc, 0.34)
    if upper <= lower:
        return float(rng.uniform(min_fc, max_fc))
    return float(rng.uniform(lower, upper))


def sample_order(rng, min_order, max_order, profile):
    if profile == "hard":
        fraction = float(rng.beta(1.2, 3.0))
    else:
        fraction = float(rng.random())
    order = min_order + int(round((max_order - min_order) * fraction))
    return int(np.clip(order, min_order, max_order))


def sample_spec(
    rng,
    min_order=8,
    max_order=128,
    min_fc=0.005,
    max_fc=0.45,
    min_trans=0.002,
    max_trans=0.12,
    profile="broad",
):
    """
    Generate valid lowpass or highpass specifications in normalized frequency.

    broad:
        Cobertura geral do espaco de projeto.
    hard:
        Favorece transicoes estreitas, alta atenuacao, ordem apertada e
        regioes proximas das bordas de frequencia.
    """
    profile = normalize_profile_name(profile)
    ftype = rng.choice(["lowpass", "highpass"])

    if profile == "hard":
        fcut = sample_edge_biased_cutoff(rng, min_fc, max_fc)
        trans_upper = min(max_trans, 0.05)
        trans = sample_log_uniform(rng, min_trans, max(trans_upper, min_trans * 1.01))
        rp = float(rng.uniform(0.01, 0.35))
        attenuation = sample_log_uniform(rng, 50.0, max(100.0, 50.0))
        order = sample_order(rng, min_order, max_order, profile)
    else:
        fcut = float(rng.uniform(min_fc, max_fc))
        trans = sample_log_uniform(rng, min_trans, max_trans)
        rp = float(rng.uniform(0.01, 1.0))
        attenuation = sample_log_uniform(rng, 30.0, 100.0)
        order = sample_order(rng, min_order, max_order, profile)

    return {
        "type": ftype,
        "fc": float(fcut),
        "trans": float(trans),
        "Rp": float(rp),
        "As": float(attenuation),
        "order": int(order),
    }


def is_valid_spec(spec, nmax=256, min_trans=0.002):
    """Apply lightweight constraints to keep the generated specs feasible."""
    fc = spec["fc"]
    trans = spec["trans"]
    rp = spec["Rp"]
    attenuation = spec["As"]
    order = spec["order"]

    if fc + trans >= 0.5:
        return False
    if fc - trans <= 0.0 and spec["type"] == "highpass":
        return False
    if trans < min_trans:
        return False
    if order > nmax:
        return False
    if attenuation < 20:
        return False
    if rp > 2.0:
        return False
    return True


def design_fir(spec, method="remez"):
    return design_reference_filter(
        fc=spec["fc"],
        trans=spec["trans"],
        rp=spec["Rp"],
        attenuation_db=spec["As"],
        order=spec["order"],
        ftype=spec["type"],
        method=method,
        fs=1.0,
    )


def evaluate_candidate_against_request(taps, request_spec, n_fft=1024):
    freqs, response = signal.freqz(taps, worN=n_fft, fs=1.0)
    response_db = 20 * np.log10(np.maximum(np.abs(response), 1e-10))
    passband_mask, stopband_mask = build_response_masks(
        freqs, request_spec["fc"], request_spec["trans"], request_spec["type"]
    )

    if not np.any(passband_mask) or not np.any(stopband_mask):
        return float("inf"), float("inf"), float("-inf")

    passband = response_db[passband_mask]
    stopband = response_db[stopband_mask]
    ripple = float(np.max(passband) - np.min(passband))
    attenuation = float(-np.max(stopband))
    gain_bias = float(abs(np.mean(passband)))

    ripple_violation = max(ripple - request_spec["Rp"], 0.0)
    attenuation_violation = max(request_spec["As"] - attenuation, 0.0)
    order_penalty = max(len(taps) - request_spec["order"], 0) * 0.02
    score = 2.0 * ripple_violation + attenuation_violation + 0.1 * gain_bias + order_penalty
    return score, ripple, attenuation


def evaluate_search_candidate(candidate_spec, request_spec, method):
    try:
        taps = design_fir(candidate_spec, method=method)
    except Exception:
        try:
            taps = design_fir(candidate_spec, method="firwin")
        except Exception:
            return None

    candidate = dict(candidate_spec)
    candidate["order"] = int(len(taps))
    score, ripple, attenuation = evaluate_candidate_against_request(taps, request_spec)
    return {
        "spec": candidate,
        "taps": taps,
        "score": float(score),
        "ripple": float(ripple),
        "attenuation": float(attenuation),
    }


def candidate_sort_key(candidate):
    return (
        candidate["score"],
        len(candidate["taps"]),
        abs(candidate["spec"]["fc"]),
        abs(candidate["spec"]["trans"]),
    )


def candidate_signature(spec):
    return (
        round(float(spec["fc"]), 6),
        round(float(spec["trans"]), 6),
        round(float(spec["Rp"]), 6),
        round(float(spec["As"]), 4),
        int(spec["order"]),
        str(spec["type"]),
    )


def propose_adjusted_spec(base_spec, rng, nmax, min_fc=0.005, min_trans=0.002, phase="explore"):
    candidate = dict(base_spec)

    if phase == "refine":
        fc_frac = 0.06
        trans_scale = (0.88, 1.18)
        rp_scale = (0.90, 1.12)
        attenuation_delta = 4.0
        order_delta = 12
    else:
        fc_frac = 0.22
        trans_scale = (0.55, 1.75)
        rp_scale = (0.70, 1.35)
        attenuation_delta = 12.0
        order_delta = 40

    candidate["fc"] = float(
        np.clip(
            base_spec["fc"] * (1.0 + rng.uniform(-fc_frac, fc_frac)),
            min_fc,
            0.45,
        )
    )
    candidate["trans"] = float(
        np.clip(
            base_spec["trans"] * (10 ** rng.uniform(np.log10(trans_scale[0]), np.log10(trans_scale[1]))),
            min_trans,
            0.12,
        )
    )
    candidate["Rp"] = float(
        np.clip(
            base_spec["Rp"] * (10 ** rng.uniform(np.log10(rp_scale[0]), np.log10(rp_scale[1]))),
            0.01,
            1.0,
        )
    )
    candidate["As"] = float(
        np.clip(base_spec["As"] + rng.uniform(-attenuation_delta, attenuation_delta), 30.0, 100.0)
    )
    candidate["order"] = int(
        np.clip(base_spec["order"] + rng.integers(-order_delta, order_delta + 1), 8, nmax)
    )
    return candidate


def compute_difficulty_weight(request_spec, direct_candidate, target_candidate):
    direct_score = max(float(direct_candidate["score"]), 0.0)
    target_score = max(float(target_candidate["score"]), 0.0)
    improvement = max(direct_score - target_score, 0.0)

    transition_term = max(0.04 - request_spec["trans"], 0.0) / 0.04
    attenuation_term = max(request_spec["As"] - 55.0, 0.0) / 45.0
    order_term = max(80.0 - request_spec["order"], 0.0) / 72.0
    edge_distance = min(request_spec["fc"], 0.5 - request_spec["fc"])
    edge_term = max(0.06 - edge_distance, 0.0) / 0.06

    difficulty = 1.0
    difficulty += 0.40 * np.log1p(direct_score)
    difficulty += 0.20 * np.log1p(target_score)
    difficulty += 0.20 * np.log1p(improvement)
    difficulty += 0.10 * transition_term
    difficulty += 0.05 * attenuation_term
    difficulty += 0.05 * order_term
    difficulty += 0.05 * edge_term
    return float(np.clip(difficulty, 1.0, 6.0))


def find_adjusted_target_spec(
    request_spec,
    method,
    search_candidates,
    rng,
    nmax,
    min_fc=0.005,
    min_trans=0.002,
):
    direct_candidate = evaluate_search_candidate(request_spec, request_spec, method)
    if direct_candidate is None:
        raise RuntimeError("Unable to synthesize the direct candidate for the request spec.")

    candidate_pool = [direct_candidate]
    seen_signatures = {candidate_signature(direct_candidate["spec"])}

    if search_candidates > 0:
        phase1_budget = max(1, int(np.ceil(search_candidates * 0.6)))
        phase2_budget = max(1, search_candidates - phase1_budget)
        beam_width = min(5, max(2, int(np.sqrt(search_candidates)) + 1))

        for _ in range(phase1_budget):
            candidate_spec = propose_adjusted_spec(
                request_spec,
                rng,
                nmax=nmax,
                min_fc=min_fc,
                min_trans=min_trans,
                phase="explore",
            )
            if not is_valid_spec(candidate_spec, nmax=nmax, min_trans=min_trans):
                continue

            evaluated = evaluate_search_candidate(candidate_spec, request_spec, method)
            if evaluated is None:
                continue

            signature = candidate_signature(evaluated["spec"])
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            candidate_pool.append(evaluated)

        beam = sorted(candidate_pool, key=candidate_sort_key)[:beam_width]
        if beam:
            local_budgets = np.full(len(beam), phase2_budget // len(beam), dtype=np.int32)
            local_budgets[: phase2_budget % len(beam)] += 1

            for seed_candidate, local_budget in zip(beam, local_budgets):
                current_anchor = seed_candidate["spec"]
                current_best = seed_candidate
                for _ in range(int(local_budget)):
                    candidate_spec = propose_adjusted_spec(
                        current_anchor,
                        rng,
                        nmax=nmax,
                        min_fc=min_fc,
                        min_trans=min_trans,
                        phase="refine",
                    )
                    if not is_valid_spec(candidate_spec, nmax=nmax, min_trans=min_trans):
                        continue

                    evaluated = evaluate_search_candidate(candidate_spec, request_spec, method)
                    if evaluated is None:
                        continue

                    signature = candidate_signature(evaluated["spec"])
                    if signature not in seen_signatures:
                        seen_signatures.add(signature)
                        candidate_pool.append(evaluated)

                    if candidate_sort_key(evaluated) < candidate_sort_key(current_best):
                        current_best = evaluated
                        current_anchor = evaluated["spec"]

    best_candidate = min(candidate_pool, key=candidate_sort_key)
    difficulty = compute_difficulty_weight(request_spec, direct_candidate, best_candidate)
    metadata = {
        "direct_score": float(direct_candidate["score"]),
        "target_score": float(best_candidate["score"]),
        "score_improvement": float(direct_candidate["score"] - best_candidate["score"]),
        "difficulty": difficulty,
    }
    return best_candidate["spec"], best_candidate["taps"], metadata


def generate_dataset(
    n_samples=50000,
    nmax=256,
    out_fname="fir_dataset.npz",
    min_order=8,
    max_order=128,
    min_fc=0.005,
    max_fc=0.45,
    min_trans=0.002,
    max_trans=0.12,
    method="remez",
    seed=None,
    search_candidates=0,
    profile="broad",
):
    rng = np.random.default_rng(seed)
    profile = normalize_profile_name(profile)

    specs = []
    targets = []
    coefs = np.zeros((n_samples, nmax), dtype=np.float32)
    orders = np.zeros(n_samples, dtype=np.int32)
    difficulty = np.zeros(n_samples, dtype=np.float32)
    direct_scores = np.zeros(n_samples, dtype=np.float32)
    target_scores = np.zeros(n_samples, dtype=np.float32)
    score_improvements = np.zeros(n_samples, dtype=np.float32)

    index = 0
    attempts = 0
    while index < n_samples:
        attempts += 1
        request_spec = sample_spec(
            rng,
            min_order=min_order,
            max_order=max_order,
            min_fc=min_fc,
            max_fc=max_fc,
            min_trans=min_trans,
            max_trans=max_trans,
            profile=profile,
        )
        if not is_valid_spec(request_spec, nmax=nmax, min_trans=min_trans):
            continue

        try:
            if search_candidates > 0:
                target_spec, taps, search_info = find_adjusted_target_spec(
                    request_spec=request_spec,
                    method=method,
                    search_candidates=search_candidates,
                    rng=rng,
                    nmax=nmax,
                    min_fc=min_fc,
                    min_trans=min_trans,
                )
            else:
                taps = design_fir(request_spec, method=method)
                target_spec = dict(request_spec)
                target_spec["order"] = int(len(taps))
                direct_score, _, _ = evaluate_candidate_against_request(taps, request_spec)
                direct_candidate = {
                    "spec": dict(request_spec),
                    "taps": taps,
                    "score": float(direct_score),
                }
                target_candidate = {
                    "spec": dict(target_spec),
                    "taps": taps,
                    "score": float(direct_score),
                }
                search_info = {
                    "direct_score": float(direct_score),
                    "target_score": float(direct_score),
                    "score_improvement": 0.0,
                    "difficulty": compute_difficulty_weight(
                        request_spec,
                        direct_candidate,
                        target_candidate,
                    ),
                }
        except Exception:
            try:
                taps = design_fir(request_spec, method="firwin")
                target_spec = dict(request_spec)
                target_spec["order"] = int(len(taps))
                direct_score, _, _ = evaluate_candidate_against_request(taps, request_spec)
                direct_candidate = {
                    "spec": dict(request_spec),
                    "taps": taps,
                    "score": float(direct_score),
                }
                target_candidate = {
                    "spec": dict(target_spec),
                    "taps": taps,
                    "score": float(direct_score),
                }
                search_info = {
                    "direct_score": float(direct_score),
                    "target_score": float(direct_score),
                    "score_improvement": 0.0,
                    "difficulty": compute_difficulty_weight(
                        request_spec,
                        direct_candidate,
                        target_candidate,
                    ),
                }
            except Exception:
                continue

        length = len(taps)
        if length > nmax:
            continue

        request_type_flag = 0 if request_spec["type"] == "lowpass" else 1
        target_type_flag = 0 if target_spec["type"] == "lowpass" else 1

        coefs[index, :length] = taps
        specs.append(
            [
                request_spec["fc"],
                request_spec["trans"],
                request_spec["Rp"],
                request_spec["As"],
                request_spec["order"],
                request_type_flag,
            ]
        )
        targets.append(
            [
                target_spec["fc"],
                target_spec["trans"],
                target_spec["Rp"],
                target_spec["As"],
                length,
                target_type_flag,
            ]
        )
        orders[index] = length
        difficulty[index] = search_info["difficulty"]
        direct_scores[index] = search_info["direct_score"]
        target_scores[index] = search_info["target_score"]
        score_improvements[index] = search_info["score_improvement"]
        index += 1

    specs = np.array(specs, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)
    full_path = os.path.abspath(out_fname)
    np.savez_compressed(
        out_fname,
        specs=specs,
        targets=targets,
        coefs=coefs,
        orders=orders,
        difficulty=difficulty,
        direct_scores=direct_scores,
        target_scores=target_scores,
        score_improvements=score_improvements,
        profile=np.array([profile]),
    )
    print(f"Saved dataset to: {full_path}")
    print(f"Profile: {profile}")
    print(f"Discard rate: {(attempts - n_samples) / attempts:.2%}")
    print(f"Adjusted targets enabled: {search_candidates > 0} (candidates={search_candidates})")
    print(
        "Difficulty stats: "
        f"mean={difficulty.mean():.3f}, median={np.median(difficulty):.3f}, p90={np.quantile(difficulty, 0.9):.3f}"
    )
    print(
        "Direct/target score stats: "
        f"direct_mean={direct_scores.mean():.3f}, target_mean={target_scores.mean():.3f}, "
        f"improvement_mean={score_improvements.mean():.3f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Generate a synthetic FIR dataset.")
    parser.add_argument("--n-samples", type=int, default=50000, help="Number of filters to generate")
    parser.add_argument("--nmax", type=int, default=256, help="Maximum number of stored taps per filter")
    parser.add_argument("--out", type=str, default="fir_dataset.npz", help="Output .npz file")
    parser.add_argument("--min-order", type=int, default=8, help="Minimum number of taps to sample")
    parser.add_argument("--max-order", type=int, default=128, help="Maximum number of taps to sample")
    parser.add_argument("--min-fc", type=float, default=0.005, help="Minimum normalized cutoff frequency")
    parser.add_argument("--max-fc", type=float, default=0.45, help="Maximum normalized cutoff frequency")
    parser.add_argument("--min-trans", type=float, default=0.002, help="Minimum normalized transition width")
    parser.add_argument("--max-trans", type=float, default=0.12, help="Maximum normalized transition width")
    parser.add_argument("--method", type=str, choices=["remez", "firwin"], default="remez")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed")
    parser.add_argument(
        "--search-candidates",
        type=int,
        default=0,
        help="Number of local candidate adjustments evaluated for each request.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        choices=["broad", "hard", "hard-focused"],
        default="broad",
        help="Sampling profile. Use 'hard' for a hard-focused dataset.",
    )
    args = parser.parse_args()

    generate_dataset(
        n_samples=args.n_samples,
        nmax=args.nmax,
        out_fname=args.out,
        min_order=args.min_order,
        max_order=args.max_order,
        min_fc=args.min_fc,
        max_fc=args.max_fc,
        min_trans=args.min_trans,
        max_trans=args.max_trans,
        method=args.method,
        seed=args.seed,
        search_candidates=args.search_candidates,
        profile=args.profile,
    )


if __name__ == "__main__":
    main()
