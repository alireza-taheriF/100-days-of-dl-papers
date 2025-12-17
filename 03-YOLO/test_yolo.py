import torch
from loss import YoloLoss
from utils import intersection_over_union

def test_iou():
    print("🧪 Testing IoU Function...")
    box1 = torch.tensor([0.5, 0.5, 0.2, 0.2]).unsqueeze(0)
    box2 = torch.tensor([0.5, 0.5, 0.2, 0.2]).unsqueeze(0)
    iou = intersection_over_union(box1, box2, box_format="midpoint")
    
    if abs(iou - 1.0) < 1e-5:
        print("   ✅ Identical boxes IoU is 1.0")
    else:
        print(f"   ❌ Identical boxes IoU failed! Got {iou}")

    box1 = torch.tensor([0.1, 0.1, 0.1, 0.1]).unsqueeze(0)
    box2 = torch.tensor([0.9, 0.9, 0.1, 0.1]).unsqueeze(0)
    iou = intersection_over_union(box1, box2, box_format="midpoint")
    
    if abs(iou - 0.0) < 1e-5:
        print("   ✅ Disjoint boxes IoU is 0.0")
    else:
        print(f"   ❌ Disjoint boxes IoU failed! Got {iou}")

def test_loss():
    print("\n🧪 Testing YOLO Loss Function...")
    loss_fn = YoloLoss()

    predictions = torch.randn(2, 1470) 
    
    target = torch.rand(2, 7, 7, 30) 

    try:
        loss = loss_fn(predictions, target)
        print(f"   ✅ Loss forward pass successful!")
        print(f"   📊 Calculated Loss: {loss.item():.4f}")
    except Exception as e:
        print(f"   ❌ Loss function crashed!")
        print(e)

if __name__ == "__main__":
    test_iou()
    test_loss()