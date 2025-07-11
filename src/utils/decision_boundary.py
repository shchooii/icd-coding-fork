import torch


def f1_score_db_tuning(logits, targets, groups, average="macro", type="per_group"):
    if average not in ["micro", "macro"]:
        raise ValueError("Average must be either 'micro' or 'macro'")
    dbs = torch.linspace(0, 1, 100)
    tp = torch.zeros((len(dbs), targets.shape[1]))
    fp = torch.zeros((len(dbs), targets.shape[1]))
    fn = torch.zeros((len(dbs), targets.shape[1]))
    for idx, db in enumerate(dbs):
        predictions = (logits > db).long()
        tp[idx] = torch.sum((predictions) * (targets), dim=0)
        fp[idx] = torch.sum(predictions * (1 - targets), dim=0)
        fn[idx] = torch.sum((1 - predictions) * targets, dim=0)
    per_cls_f1 = tp / (tp + 0.5 * (fp + fn) + 1e-10)
    if average == "micro":
        f1_scores = tp.sum(1) / (tp.sum(1) + 0.5 * (fp.sum(1) + fn.sum(1)) + 1e-10)
    else:
        f1_scores = torch.mean(tp / (tp + 0.5 * (fp + fn) + 1e-10), dim=1)
    if type == "single":
        best_f1 = f1_scores.max()
        best_db = dbs[f1_scores.argmax()]
        print(f"Best F1: {best_f1:.4f} at DB: {best_db:.4f}")
        return best_f1, best_db
    if type == "per_class":
        best_f1, best_idx_each = per_cls_f1.max(0)
        best_db_each = dbs[best_idx_each]
        print(f"Best F1: {best_f1} at DB: {best_db_each}")
        return best_f1, best_db_each
    if type == "per_group":
        thr_vec = torch.full((targets.shape[1],), 0.5)
        cls_f1 = tp / (tp + 0.5 * (fp + fn) + 1e-10)
        best_f1_g, best_db_g = {}, {}
        for g, idxs in groups.items():
            idxs = torch.as_tensor(idxs, device=logits.device)
            if average == "micro":
                g_tp = tp[:, idxs].sum(1)
                g_fp = fp[:, idxs].sum(1)
                g_fn = fn[:, idxs].sum(1)
                g_f1 = g_tp / (g_tp + 0.5 * (g_fp + g_fn) + 1e-10)
            else:
                g_f1 = cls_f1[:, idxs].mean(1)
            best = g_f1.argmax()
            best_f1_g[g] = g_f1[best].item()
            best_db_g[g] = dbs[best].item()
            thr_vec[torch.as_tensor(idxs, dtype=torch.long)] = best_db_g[g]
        return best_f1_g, thr_vec