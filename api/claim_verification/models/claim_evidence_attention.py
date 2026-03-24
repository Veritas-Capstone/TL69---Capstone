import torch
import torch.nn as nn


class ClaimEvidenceAttentionModel(nn.Module):
    def __init__(self, encoder, hidden_dim=1024, num_labels=3, num_heads=4):
        super().__init__()
        self.encoder = encoder  # e.g., RoBERTa
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Multi-head claim-aware attention:
        # For each head h: α_i^h = softmax(v_hᵀ tanh(Wc_h * c + We_h * eᵢ))
        self.W_c = nn.Parameter(torch.empty(num_heads, hidden_dim, hidden_dim))
        self.W_e = nn.Parameter(torch.empty(num_heads, hidden_dim, hidden_dim))
        self.v = nn.Parameter(torch.empty(num_heads, hidden_dim))

        # Adaptive fusion: learn instance-specific head weights
        self.head_gate = nn.Linear(2 * hidden_dim, 1)

        # Classifier on [c, e_att, |c−e_att|, c*e_att]
        self.classifier = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_labels),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize attention parameters with Xavier for stability
        nn.init.xavier_uniform_(self.W_c)
        nn.init.xavier_uniform_(self.W_e)
        nn.init.xavier_uniform_(self.v)
        nn.init.xavier_uniform_(self.head_gate.weight)
        if self.head_gate.bias is not None:
            nn.init.zeros_(self.head_gate.bias)

    def forward(self, claim_inputs, evidence_inputs, evidence_mask, labels=None):
        B, K, L = evidence_inputs["input_ids"].shape
        device = next(self.parameters()).device

        # Encode claims → [B, D]
        c_out = self.encoder(**claim_inputs).last_hidden_state[:, 0, :]  # [CLS] token

        # Encode evidence sentences → flatten to [B*K, L]
        flat_ids = evidence_inputs["input_ids"].reshape(B * K, L).to(device)
        flat_mask = evidence_inputs["attention_mask"].reshape(B * K, L).to(device)
        e_out = self.encoder(input_ids=flat_ids, attention_mask=flat_mask)
        flat_e_emb = e_out.last_hidden_state[:, 0, :]  # [CLS] → [B*K, D]
        e_emb = flat_e_emb.view(B, K, -1)  # → [B, K, D]

        # Multi-head attention: compute α over K for each head
        # c_proj: [B, H, D], e_proj: [B, H, K, D]
        c_proj = torch.einsum("bd,hde->bhd", c_out, self.W_c)
        e_proj = torch.einsum("bkd,hde->bhke", e_emb, self.W_e)
        u = torch.einsum("bhke,he->bhk", torch.tanh(c_proj.unsqueeze(2) + e_proj), self.v)
        u = u.masked_fill(evidence_mask.unsqueeze(1) == 0, float("-inf"))
        alpha = torch.softmax(u, dim=2)  # [B, H, K]

        # Per-head attended evidence: e_att_heads [B, H, D]
        e_att_heads = torch.einsum("bhk,bkd->bhd", alpha, e_emb)

        # Adaptive head fusion
        c_rep = c_out.unsqueeze(1).expand_as(e_att_heads)  # [B, H, D]
        gate_in = torch.cat([c_rep, e_att_heads], dim=-1)  # [B, H, 2D]
        gate_logits = self.head_gate(gate_in).squeeze(-1)  # [B, H]
        head_weights = torch.softmax(gate_logits, dim=1)   # [B, H]

        # Fused evidence vector
        e_att = torch.einsum("bh,bhd->bd", head_weights, e_att_heads)

        # Combine with claim: [c, e_att, |c−e_att|, c*e_att]
        h = torch.cat([c_out, e_att, torch.abs(c_out - e_att), c_out * e_att], dim=1)

        logits = self.classifier(h)

        output = {
            "logits": logits,
            "attn_weights": alpha,
            "head_weights": head_weights,
        }
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            output["loss"] = loss

        return output
