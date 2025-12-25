import torch
import torch.nn as nn

from torch.nn.init import xavier_uniform_
import torch.nn.functional as F

class LabelAttention(nn.Module):
    def __init__(self, input_size: int, projection_size: int, num_classes: int):
        super().__init__()
        self.first_linear = nn.Linear(input_size, projection_size, bias=False)
        self.second_linear = nn.Linear(projection_size, num_classes, bias=False)
        self.third_linear = nn.Linear(input_size, num_classes)
        self._init_weights(mean=0.0, std=0.03)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """LAAT attention mechanism

        Args:
            x (torch.Tensor): [batch_size, seq_len, input_size]

        Returns:
            torch.Tensor: [batch_size, num_classes]
        """
        weights = torch.tanh(self.first_linear(x))
        att_weights = self.second_linear(weights)
        att_weights = torch.nn.functional.softmax(att_weights, dim=1).transpose(1, 2)
        weighted_output = att_weights @ x
        return (
            self.third_linear.weight.mul(weighted_output)
            .sum(dim=2)
            .add(self.third_linear.bias)
        )

    def _init_weights(self, mean: float = 0.0, std: float = 0.03) -> None:
        """
        Initialise the weights

        Args:
            mean (float, optional): Mean of the normal distribution. Defaults to 0.0.
            std (float, optional): Standard deviation of the normal distribution. Defaults to 0.03.
        """

        torch.nn.init.normal_(self.first_linear.weight, mean, std)
        torch.nn.init.normal_(self.second_linear.weight, mean, std)
        torch.nn.init.normal_(self.third_linear.weight, mean, std)


class LabelAttention2(nn.Module):
    def __init__(self, input_size: int, projection_size: int, num_classes: int):
        super().__init__()
        self.first_linear = nn.Linear(input_size, projection_size, bias=False)
        self.second_linear = nn.Linear(projection_size, num_classes, bias=False)
        self.third_linear = nn.Linear(input_size, num_classes)
        self._init_weights(mean=0.0, std=0.03)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask_1d: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        """
        x: [B, L, H]
        attention_mask_1d: [B, L] (0/1 or bool)
        return:
          logits: [B, C]
          aux(optional): {
              "att_weights": [B, C, L],
              "weighted_output": [B, C, H],
              "embed_mean": [B, H],
          }
        """
        # ---- 원래 LAAT 로직 그대로 ----
        weights = torch.tanh(self.first_linear(x))            # [B,L,P]
        att_logits = self.second_linear(weights)              # [B,L,C]
        att_weights = F.softmax(att_logits, dim=1).transpose(1, 2)  # [B,C,L]
        weighted_output = att_weights @ x                     # [B,C,H]
        logits = (
            self.third_linear.weight.mul(weighted_output)
            .sum(dim=2)
            .add(self.third_linear.bias)
        )  # [B,C]

        if not return_aux:
            return logits

        # ---- embed_mean: "마스크 반영 mean" (COMIC embed로 쓰기 가장 안전한 최소 버전) ----
        if attention_mask_1d is None:
            embed_mean = x.mean(dim=1)  # [B,H]
        else:
            m = attention_mask_1d
            if m.dtype != torch.float32:
                m = m.float()
            m = m.unsqueeze(-1)                         # [B,L,1]
            denom = m.sum(dim=1).clamp_min(1.0)         # [B,1]
            embed_mean = (x * m).sum(dim=1) / denom     # [B,H]

        aux = {
            "att_weights": att_weights,
            "weighted_output": weighted_output,
            "embed_mean": embed_mean,
        }
        return logits, aux

    def _init_weights(self, mean: float = 0.0, std: float = 0.03) -> None:
        torch.nn.init.normal_(self.first_linear.weight, mean, std)
        torch.nn.init.normal_(self.second_linear.weight, mean, std)
        torch.nn.init.normal_(self.third_linear.weight, mean, std)


class CAMLAttention(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.first_linear = nn.Linear(input_size, num_classes)
        xavier_uniform_(self.first_linear.weight)
        self.second_linear = nn.Linear(input_size, num_classes)
        xavier_uniform_(self.second_linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """CAML attention mechanism

        Args:
            x (torch.Tensor): [batch_size, input_size, seq_len]

        Returns:
            torch.Tensor: [batch_size, num_classes]
        """
        x = torch.tanh(x)
        weights = torch.softmax(self.first_linear.weight.matmul(x), dim=2)
        weighted_output = weights @ x.transpose(1, 2)
        return (
            self.second_linear.weight.mul(weighted_output)
            .sum(2)
            .add(self.second_linear.bias)
        )
