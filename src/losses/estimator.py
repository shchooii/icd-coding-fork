import torch
import torch.nn as nn
import numpy as np

class EstimatorCV():
    def __init__(self, feature_num, class_num):
        super(EstimatorCV, self).__init__()
        self.class_num = class_num
        self.CoVariance = torch.zeros(class_num, feature_num, feature_num)
        self.Ave        = torch.zeros(class_num, feature_num)
        self.Amount     = torch.zeros(class_num)
        self.Prop       = torch.zeros(class_num)
        self.Cov_pos    = torch.zeros(class_num)
        self.Cov_neg    = torch.zeros(class_num)
        self.Sigma_cj   = torch.zeros(class_num, class_num)  # CPU
        self.Ro_cj      = torch.zeros(class_num, class_num)  # CPU
        self.Tao_cj     = torch.zeros(class_num, class_num)  # CPU

    # 기존 numpy -> torch 기반으로 교체
    @torch.no_grad()
    def update_CV(self, labels: torch.Tensor, logits: torch.Tensor):
        """
        labels: [N, C] (0/1)
        logits: [N, C]
        returns: (Prop, Cov_pos, Cov_neg, Sigma_cj, Ro_cj, Tao_cj)
        """
        dev  = logits.device
        eps  = 1e-8

        N, C = labels.shape
        assert C == self.class_num
        onehot = labels.to(logits.dtype)                # [N, C] on GPU

        # ---- 활성 클래스만 선택 ----
        sum_pos  = onehot.sum(0)                        # [C]
        act_mask = sum_pos > 0
        if act_mask.sum() == 0:
            # 이 배치에 양성 라벨이 없으면 갱신 없이 현 상태 반환
            return (self.Prop.to(dev), self.Cov_pos.to(dev), self.Cov_neg.to(dev),
                    self.Sigma_cj, self.Ro_cj, self.Tao_cj)

        idx     = act_mask.nonzero(as_tuple=False).squeeze(1)   # [Ca] GPU indices
        idx_cpu = idx.cpu()                                     # [Ca] CPU indices
        Ca      = idx.numel()

        M = onehot[:, idx]                                      # [N, Ca] (GPU)

        # ---- prior p(c) ----
        pr_C_act = sum_pos[idx] / float(max(N, 1))              # [Ca] (GPU)

        # ---- 행 합/제곱합 (전체 로그릿 기준) ----
        row_sum   = logits.sum(dim=1)                           # [N]
        row_sumsq = (logits * logits).sum(dim=1)                # [N]

        # ---- 공발생 카운트 Co = M^T M ----
        Co = M.t().mm(M)                                        # [Ca, Ca]

        # ---- σ_cj: S_c∩S_j 에서의 전체 로그릿 분산 ----
        # 공발생 가중 합/제곱합
        S_sum   = M.t().mm(M * row_sum.view(N, 1))              # [Ca, Ca]
        S_sumsq = M.t().mm(M * row_sumsq.view(N, 1))            # [Ca, Ca]
        # 총 원소 수(샘플수×C)
        Mtot = (Co * C).clamp_min(eps)                          # [Ca, Ca]
        mean = S_sum / Mtot
        var  = S_sumsq / Mtot - mean * mean                     # [Ca, Ca]

        # ---- ρ_cj, τ_cj ----
        sum_pos_act = sum_pos[idx].clamp_min(eps)               # [Ca]
        # ρ_cj = |c∩j| / |j|
        Ro = Co / sum_pos_act.view(1, Ca)                       # [Ca, Ca]
        # τ_cj = (N - |c|) / (|j| - |c∩j|)
        denom = (sum_pos_act.view(1, Ca) - Co).clamp_min(eps)   # [Ca, Ca]
        Tao   = (float(N) - sum_pos_act.view(Ca, 1)) / denom    # [Ca, Ca]

        # ---- min–max 정규화 (활성 블록 기준) ----
        def minmax_norm(x):
            xmin = x.amin()
            xmax = x.amax()
            return (x - xmin) / (xmax - xmin + eps)

        Sigma_n = minmax_norm(var)                               # [Ca, Ca]
        Ro_n    = minmax_norm(Ro)                                # [Ca, Ca]
        Tao_n   = minmax_norm(Tao)                               # [Ca, Ca]

        # ---- per-class pos/neg 분산 (전체 로그릿 매트릭스에 대한 분산) ----
        # pos: S_c, neg: ¬S_c
        S_pos   = (row_sum.view(N, 1)   * M).sum(dim=0)          # [Ca]
        S2_pos  = (row_sumsq.view(N, 1) * M).sum(dim=0)          # [Ca]
        M_pos   = (sum_pos_act * C).clamp_min(eps)               # [Ca]
        mean_p  = S_pos / M_pos
        var_pos = S2_pos / M_pos - mean_p * mean_p               # [Ca]

        Mneg        = 1.0 - M
        cnt_neg     = (float(N) - sum_pos_act).clamp_min(eps)    # [Ca]
        S_neg       = (row_sum.view(N, 1)   * Mneg).sum(dim=0)   # [Ca]
        S2_neg      = (row_sumsq.view(N, 1) * Mneg).sum(dim=0)   # [Ca]
        M_neg_total = (cnt_neg * C).clamp_min(eps)               # [Ca]
        mean_n      = S_neg / M_neg_total
        var_neg     = S2_neg / M_neg_total - mean_n * mean_n     # [Ca]

        # ---- EMA 가중치 (상태는 CPU, 계산은 GPU) ----
        amount_act_cpu = self.Amount[idx_cpu]                    # [Ca] CPU
        amount_act     = amount_act_cpu.to(dev)                  # [Ca] GPU

        w_pr     = sum_pos_act / (sum_pos_act + amount_act + eps)   # [Ca] GPU
        w_pr_neg = cnt_neg    / (cnt_neg    + amount_act + eps)     # [Ca] GPU
        w_cj     = Co / (Co + amount_act.view(Ca, 1) + eps)         # [Ca, Ca] GPU

        # ---- 스칼라 상태 갱신 (CPU 인덱싱) ----
        w_pr_cpu     = w_pr.detach().to(self.Prop.dtype).cpu()
        w_pr_neg_cpu = w_pr_neg.detach().to(self.Prop.dtype).cpu()

        self.Prop[idx_cpu]    = (self.Prop[idx_cpu]    * (1 - w_pr_cpu)     + pr_C_act.detach().cpu() * w_pr_cpu)
        self.Cov_pos[idx_cpu] = (self.Cov_pos[idx_cpu] * (1 - w_pr_cpu)     + var_pos.detach().cpu()  * w_pr_cpu)
        self.Cov_neg[idx_cpu] = (self.Cov_neg[idx_cpu] * (1 - w_pr_neg_cpu) + var_neg.detach().cpu()  * w_pr_neg_cpu)
        self.Amount[idx_cpu]  = self.Amount[idx_cpu] + sum_pos_act.detach().cpu()

        # ---- 행렬 상태 갱신 (CPU 인덱싱) ----
        ii_cpu, jj_cpu = torch.meshgrid(idx_cpu, idx_cpu, indexing='ij')

        Sigma_n_cpu = Sigma_n.detach().to(self.Sigma_cj.dtype).cpu()
        Ro_n_cpu    = Ro_n.detach().to(self.Ro_cj.dtype).cpu()
        Tao_n_cpu   = Tao_n.detach().to(self.Tao_cj.dtype).cpu()
        w_cj_cpu    = w_cj.detach().to(Sigma_n_cpu.dtype).cpu()

        self.Sigma_cj[ii_cpu, jj_cpu] = self.Sigma_cj[ii_cpu, jj_cpu] * (1 - w_cj_cpu) + Sigma_n_cpu * w_cj_cpu
        self.Ro_cj[ii_cpu, jj_cpu]    = self.Ro_cj[ii_cpu, jj_cpu]    * (1 - w_cj_cpu) + Ro_n_cpu    * w_cj_cpu
        self.Tao_cj[ii_cpu, jj_cpu]   = self.Tao_cj[ii_cpu, jj_cpu]   * (1 - w_cj_cpu) + Tao_n_cpu   * w_cj_cpu

        # ---- 반환 (2D 통계는 GPU, C×C 행렬은 CPU 유지) ----
        return (self.Prop.detach().to(dev),
                self.Cov_pos.detach().to(dev),
                self.Cov_neg.detach().to(dev),
                self.Sigma_cj.detach().cpu(),
                self.Ro_cj.detach().cpu(),
                self.Tao_cj.detach().cpu())

