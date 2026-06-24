import argparse
import bz2
import json
import pickle
import zipfile
from pathlib import PurePosixPath

import numpy as np


def hashable(grid):
    return tuple(tuple(int(x) for x in row) for row in np.asarray(grid).tolist())


def as_array(grid):
    return np.asarray(grid, dtype=int)


def candidate_features(samples):
    features = {}
    for sample in samples:
        solution = as_array(sample["solution"])
        h = hashable(solution)
        if h not in features:
            features[h] = {
                "solution": solution,
                "votes": 0,
                "beam_scores": [],
                "score_aug": [],
            }
        features[h]["votes"] += 1
        features[h]["beam_scores"].append(float(sample["beam_score"]))
        features[h]["score_aug"].append([float(x) for x in sample["score_aug"]])
    return features


def score_full_probmul_3(features, baseline=3):
    scored = []
    for cand in features.values():
        inf_score = np.sum([baseline - x for x in cand["beam_scores"]])
        aug_score = np.mean([np.sum([baseline - s for s in scores]) for scores in cand["score_aug"]])
        scored.append((inf_score + aug_score, cand["solution"]))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [solution for _, solution in scored]


def score_kgmon(features):
    scored = []
    for cand in features.values():
        inf_score = cand["votes"]
        aug_score = np.mean([np.mean(scores) for scores in cand["score_aug"]])
        scored.append((inf_score - aug_score, cand["solution"]))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [solution for _, solution in scored]


# ADD NEW SELECTION FUNCTIONS HERE


SCORERS = {
    "score_kgmon": score_kgmon,
    "score_full_probmul_3": score_full_probmul_3,
}


def read_dump(dump_path, solutions_path=None):
    decoded = {}
    timings = []
    timeouts = 0
    solutions = None

    with zipfile.ZipFile(dump_path, "r") as zf:
        for name in zf.namelist():
            path = PurePosixPath(name)
            if path.name == "arc-agi_evaluation_solutions.json":
                solutions = json.loads(zf.read(name).decode("utf-8"))
                continue
            if "inference_outputs" not in path.parts or path.name == "":
                continue
            outputs = pickle.loads(bz2.decompress(zf.read(name)))
            basekey = path.name.split(".")[0]
            decoded.setdefault(basekey, [])
            decoded[basekey].extend(outputs)
            for sample in outputs:
                for time_key in ["elapsed", "elapsed_time", "spend_time", "time_sec", "seconds"]:
                    if time_key in sample:
                        timings.append(float(sample[time_key]))
                        break
                if sample.get("timeout") or sample.get("timed_out"):
                    timeouts += 1

    if solutions is None:
        if solutions_path is None:
            raise FileNotFoundError("arc-agi_evaluation_solutions.json not found in dump")
        with open(solutions_path, "r") as f:
            solutions = json.load(f)

    return decoded, solutions, timings, timeouts


def label_for(basekey, solutions):
    task_id, output_nr = basekey.rsplit("_", 1)
    return as_array(solutions[task_id][int(output_nr)])


def rank_of_correct(ordered, correct):
    for i, guess in enumerate(ordered, start=1):
        if np.array_equal(guess, correct):
            return i
    return None


def evaluate(decoded, solutions, scorer_names):
    num_outputs = {}
    oracle_hits = {}
    selected_hits = {name: {} for name in scorer_names}
    ranks = {name: {} for name in scorer_names}
    n_cands = {}

    for basekey, samples in sorted(decoded.items()):
        task_id, output_nr = basekey.rsplit("_", 1)
        num_outputs[task_id] = max(num_outputs.get(task_id, 0), int(output_nr) + 1)
        correct = label_for(basekey, solutions)
        features = candidate_features(samples)
        n_cands[basekey] = len(features)
        oracle_hits[basekey] = any(np.array_equal(correct, cand["solution"]) for cand in features.values())

        for name in scorer_names:
            ordered = SCORERS[name](features)
            rank = rank_of_correct(ordered, correct)
            ranks[name][basekey] = rank
            selected_hits[name][basekey] = rank is not None and rank <= 2

    task_count = len(num_outputs)

    def pass_at_2(hits):
        if task_count == 0:
            return 0.0
        score = 0.0
        for basekey, hit in hits.items():
            if hit:
                task_id = basekey.rsplit("_", 1)[0]
                score += 1.0 / num_outputs[task_id]
        return score / task_count

    oracle_score = pass_at_2(oracle_hits)
    selected_scores = {name: pass_at_2(hits) for name, hits in selected_hits.items()}
    best_name = max(selected_scores, key=selected_scores.get)
    best_selected = selected_scores[best_name]

    worst = []
    for basekey, hit in oracle_hits.items():
        if hit and not selected_hits[best_name][basekey]:
            rank_kgmon = ranks.get("score_kgmon", {}).get(basekey)
            rank_probmul = ranks.get("score_full_probmul_3", {}).get(basekey)
            present_ranks = [r for r in [rank_kgmon, rank_probmul] if r is not None]
            best_rank = min(present_ranks) if present_ranks else 10**9
            worst.append((best_rank, basekey, n_cands[basekey], rank_kgmon, rank_probmul))
    worst = sorted(worst, key=lambda x: (-x[0], x[1]))[:10]

    return oracle_score, selected_scores, best_selected, worst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", default="val_dump.zip")
    parser.add_argument("--solutions", default=None)
    parser.add_argument("--scorer", choices=sorted(SCORERS), default=None)
    args = parser.parse_args()

    scorer_names = [args.scorer] if args.scorer else ["score_kgmon", "score_full_probmul_3"]
    decoded, solutions, timings, timeouts = read_dump(args.dump, args.solutions)
    oracle_score, selected_scores, best_selected, worst = evaluate(decoded, solutions, scorer_names)

    print(f"ORACLE_PASS@2: {oracle_score:.3f}")
    print(
        "SELECTED_PASS@2: "
        + ", ".join(f"{name}={selected_scores[name]:.3f}" for name in scorer_names)
    )
    print(f"GAP: {oracle_score - best_selected:.3f}")
    if timings:
        timing_array = np.asarray(timings, dtype=float)
        print(
            "TIMING: "
            f"median={np.median(timing_array):.1f}s "
            f"p90={np.percentile(timing_array, 90):.1f}s "
            f"max={np.max(timing_array):.1f}s "
            f"timeouts={timeouts}/{len(timings)}"
        )
    print("WORST10 (in pool, not in top-2): task_id n_cand rank_kgmon rank_probmul")
    for _, basekey, n_cand, rank_kgmon, rank_probmul in worst:
        print(f"{basekey} {n_cand} {rank_kgmon} {rank_probmul}")


if __name__ == "__main__":
    main()
