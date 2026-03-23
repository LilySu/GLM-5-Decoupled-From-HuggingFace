# GLM-5 Training Infrastructure on Ascend

## Cluster Configuration

| Spec | Value |
|------|-------|
| Chip | Huawei Ascend 910B |
| Scale | ~100,000 chips |
| Framework | MindSpeed (MindSpore-based) |
| Node | Atlas 800T A3 (8 NPUs per node) |
| Nodes | ~12,500 nodes |
| Interconnect | HCCS intra-node, RoCE/IB inter-node |
| Data | 28.5 trillion tokens |
| Model | 744B total, 40B active (MoE) |

## MindSpeed Framework

MindSpeed-LLM is Ascend's distributed training framework providing:
- Distributed pre-training (data parallel + tensor parallel + pipeline parallel)
- Distributed instruction fine-tuning
- Distributed preference alignment (RLHF/DPO)
- Dynamic graph multi-level pipelined deployment

## Parallelism Strategy

GLM-5's MoE architecture requires multi-dimensional parallelism:

```
                ┌─────────────────────────┐
                │   Expert Parallelism     │ 256 experts distributed across nodes
                │   (EP across nodes)      │
                ├─────────────────────────┤
                │   Tensor Parallelism     │ Attention heads split within node
                │   (TP within node)       │ (8-way TP on 8 NPUs)
                ├─────────────────────────┤
                │   Pipeline Parallelism   │ Layers distributed across node groups
                │   (PP across node groups)│
                ├─────────────────────────┤
                │   Data Parallelism       │ Batch distributed across PP groups
                │   (DP across clusters)   │
                └─────────────────────────┘
```

## DSA Training-Specific Optimizations

The Ascend team developed specialized fused operators for DSA TRAINING:

1. **Lightning Indexer Loss memory optimization** — reduces GPU memory during the indexer's KL-divergence training loss computation
2. **Cube-Vector pipeline parallelism** — overlaps matrix multiply (Cube) with activation/norm (Vector) during the forward/backward pass

## Key Challenge: 100K Chip Reliability

Training on 100K chips for months requires:
- Automatic checkpoint saving and recovery
- Fault-tolerant communication (node failure → automatic re-routing)
- Dynamic graph support for variable batch/sequence during training
- Gradient accumulation across failed steps
