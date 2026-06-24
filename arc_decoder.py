import os
import bz2
import pickle
import numpy as np

def hashable(guess):
    return tuple(map(tuple, guess))

def score_sum(guesses, getter):
    guess_list = list(guesses.values())
    scores = {}
    for g in guess_list:
        h = hashable(g["solution"])
        x = scores[h] = scores.get(h, [[], g["solution"]])
        x[0].append(g)
    scores = [(getter(sc), o) for sc, o in scores.values()]
    scores = sorted(scores, key=(lambda x: x[0]), reverse=True)
    ordered_outputs = [x[-1] for x in scores]
    return ordered_outputs

def getter_full_probmul_3(guesses, baseline=3):
    inf_score = np.sum([baseline-g["beam_score"] for g in guesses])
    aug_score = np.mean([np.sum([baseline-s for s in g["score_aug"]]) for g in guesses])
    return inf_score + aug_score

def score_full_probmul_3(guesses):
    return score_sum(guesses, getter_full_probmul_3)

def getter_kgmon(guesses):
    inf_score = len(guesses)
    aug_score = np.mean([np.mean(g["score_aug"]) for g in guesses])
    return inf_score - aug_score

def score_kgmon(guesses):
    return score_sum(guesses, getter_kgmon)


selection_algorithms = [
    score_full_probmul_3,
    score_kgmon,
]


class ArcDecoder:
    
    def __init__(self, dataset, n_guesses):
        self.dataset = dataset
        self.n_guesses = n_guesses
        self.decoded_results = {}

    def load_decoded_results(self, store, run_name=""):
        for key in os.listdir(store):
            with bz2.BZ2File(os.path.join(store, key)) as f:
                outputs = pickle.load(f)
            base_key = key.split(".")[0]
            self.decoded_results[base_key] = self.decoded_results.get(base_key, {})
            for i, sample in enumerate(outputs):
                self.decoded_results[base_key][f"{key}{run_name}.out{i}"] = sample

    def run_selection_algo(self, selection_algorithm=score_kgmon):
        return {bk: selection_algorithm({k: g for k, g in v.items()}) for bk, v in self.decoded_results.items()}

    def benchmark_selection_algos(self):
        scorers = [
            ("score_kgmon", score_kgmon),
            ("score_full_probmul_3", score_full_probmul_3),
        ]
        top_n = 2
        labels = {}
        num_outputs = {}
        n_cands = {}
        oracle_hits = {}
        selected_hits = {name: {} for name, _ in scorers}
        ranks = {name: {} for name, _ in scorers}
        size_differs = {}
        timings = []
        timeouts = 0

        for basekey, basevalues in self.decoded_results.items():
            task_id, output_nr = basekey.rsplit("_", 1)
            num_outputs[task_id] = max(num_outputs.get(task_id, 0), int(output_nr) + 1)
            correct_solution = np.asarray(self.dataset.replies[basekey][0])
            labels[basekey] = correct_solution

            input_grid = np.asarray(self.dataset.queries[basekey]["test"][0]["input"])
            size_differs[basekey] = np.shape(input_grid) != np.shape(correct_solution)

            pool = {}
            for sample in basevalues.values():
                solution = np.asarray(sample["solution"])
                pool.setdefault(hashable(solution), solution)
                for time_key in ["elapsed", "elapsed_time", "spend_time", "time_sec", "seconds"]:
                    if time_key in sample:
                        timings.append(float(sample[time_key]))
                        break
                if sample.get("timeout") or sample.get("timed_out"):
                    timeouts += 1

            n_cands[basekey] = len(pool)
            oracle_hits[basekey] = any(np.array_equal(correct_solution, solution) for solution in pool.values())

            for name, scorer in scorers:
                ordered = scorer({k: g for k, g in basevalues.items()})
                rank = None
                for i, guess in enumerate(ordered, start=1):
                    if np.array_equal(correct_solution, guess):
                        rank = i
                        break
                ranks[name][basekey] = rank
                selected_hits[name][basekey] = rank is not None and rank <= top_n

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

        def group_acc(keys):
            if not keys:
                return 0.0
            hits = selected_hits[best_name]
            return sum(1 for key in keys if hits[key]) / len(keys)

        diff_keys = [key for key, differs in size_differs.items() if differs]
        same_keys = [key for key, differs in size_differs.items() if not differs]

        print(f"ORACLE_PASS@2: {oracle_score:.3f}")
        print(
            "SELECTED_PASS@2: "
            f"score_kgmon={selected_scores['score_kgmon']:.3f}, "
            f"score_full_probmul_3={selected_scores['score_full_probmul_3']:.3f}"
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
        print(
            "SIZE_DIFFERS: "
            f"{len(diff_keys)}/{len(size_differs)} tasks have output size != input size; "
            f"acc_diff={group_acc(diff_keys):.3f} acc_same={group_acc(same_keys):.3f}"
        )
        print("WORST10 (in pool, not in top-2): task_id n_cand rank_kgmon rank_probmul")

        worst = []
        for basekey, hit in oracle_hits.items():
            if hit and not selected_hits[best_name][basekey]:
                rank_kgmon = ranks["score_kgmon"][basekey]
                rank_probmul = ranks["score_full_probmul_3"][basekey]
                best_rank = min(rank_kgmon or 10**9, rank_probmul or 10**9)
                worst.append((best_rank, basekey, n_cands[basekey], rank_kgmon, rank_probmul))
        worst = sorted(worst, key=lambda x: (-x[0], x[1]))[:10]
        for _, basekey, n_cand, rank_kgmon, rank_probmul in worst:
            print(f"{basekey} {n_cand} {rank_kgmon} {rank_probmul}")
