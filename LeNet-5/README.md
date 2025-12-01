# LeNet-5 Implementation in PyTorch

A modern reproduction of the classic LeNet-5 architecture proposed by Yann LeCun et al. (1998) in "Gradient-Based Learning Applied to Document Recognition".

## 📊 Results
I achieved **99.0% accuracy** on the MNIST test set, exceeding the original paper's baseline (~98.9% without distortions).

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **99.00%** |
| **Test Loss** | 0.0020 |
| **Epochs** | 20 |
| **Hardware** | Apple M2 Pro (MPS Acceleration) |

## 🛠️ Architecture Details
- **Input:** 32x32 Grayscale images (MNIST resized from 28x28)
- **Activation:** `Tanh` (Historical accuracy)
- **Pooling:** `AvgPool2d`
- **Optimizer:** SGD + Momentum (0.9)

## 🚀 How to Run

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
    ```

Train the Model: You can train the model from scratch using the CLI.


# Default settings (10 epochs)
python train.py

# Custom settings
python train.py --epochs 20 --batch_size 32 --lr 0.01
📂 File Structure
model.py: PyTorch implementation of the LeNet-5 class.

data_setup.py: Data downloading and preprocessing pipeline.

train.py: Training loop, evaluation, and CLI argument parsing.

lenet5_impl.ipynb: Exploratory notebook with visualizations.

🔗 Reference
Gradient-Based Learning Applied to Document Recognition (1998)
