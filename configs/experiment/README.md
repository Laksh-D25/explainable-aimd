# Ablations

Each file overrides `configs/model/mert95_attnpool.yaml`. Together they form the
table that justifies the corrected architecture instead of asserting it — and
they are also where the as-drawn diagram variants get their fair hearing.

| Config | Isolates | Question it answers |
|---|---|---|
| `baseline.yaml` | — | the corrected default |
| `pool_mean.yaml` | frame pooling | does attention pooling actually beat average pooling on *music*? (the central novelty claim) |
| `layers_last.yaml` | layer aggregation | is aggregating 13 states better than last-layer features? |
| `self_attn_on.yaml` | the diagram's self-attention block | does it earn its parameters on top of two pooling stages? |
| `song_mean.yaml` | clip→song aggregation | does attentive song pooling beat mean over clips? |
| `no_augment.yaml` | training augmentation | how much of the robustness result comes from augmentation? |

Run the sweep only after the architecture is settled on the dev subset — the
Kaggle GPU quota (~30 h/week) is the binding constraint.
