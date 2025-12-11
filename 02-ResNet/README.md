# ResNet-18 Implementation & Ablation Study

A PyTorch implementation of ResNet-18 on CIFAR-10, based on the paper ["Deep Residual Learning for Image Recognition"](https://arxiv.org/abs/1512.03385) by He et al. (2015).

## 🧪 Ablation Study: The Power of Skip Connections
I conducted an experiment to verify the hypothesis that residual connections facilitate gradient flow. I trained two models with identical depth (18 layers) and hyperparameters:
1.  **ResNet-18:** Standard architecture with skip connections ($y = F(x) + x$).
2.  **PlainNet-18:** The same architecture with skip connections removed ($y = F(x)$).

### Results (10 Epochs)
| Model | Test Accuracy | Train Accuracy | Convergence Speed |
|-------|---------------|----------------|-------------------|
| **ResNet-18** | **87.98%** | 89.07% | 🚀 Fast (76% at epoch 5) |
| **PlainNet-18** | 84.30% | 84.44% | 🐢 Slower (66% at epoch 5) |

**Conclusion:** Even at a depth of 18 layers, the residual path significantly improves optimization and final accuracy.

## 🛠️ Tech Specs
- **Dataset:** CIFAR-10 (Augmented: RandomCrop, HorizontalFlip)
- **Architecture:** Custom `BasicBlock` implementation handling stride/channel changes.
- **Training:** SGD + Momentum (0.9) + Weight Decay (5e-4) + CosineAnnealingLR.
- **Hardware:** Apple M2 Pro (MPS Acceleration).

## 🚀 Usage

Train ResNet-18 (Default):
```bash
python train.py --model resnet --epochs 20

```‍‍‍

Train PlainNet-18 (Ablation):

python train.py --model plain --epochs 20