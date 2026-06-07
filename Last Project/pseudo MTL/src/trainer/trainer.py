# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# import numpy as np
# from src.trainer.loss import MultiTaskLoss
# from src.trainer.metrics import MultiTaskMetrics

# def train_model(model, train_loader=None, val_loader=None, epochs=50, lr=0.001, val_step=5):
#     custom_criterion = MultiTaskLoss(weight_task1=2.0, weight_task2=0.2)
#     optimizer = optim.Adam(model.parameters(), lr=lr)
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
    
#     print(f"Training started on device: {device}")
    
#     for epoch in range(epochs):

#         model.train()

#         train_loss = 0.0
#         train_l1 = 0.0
#         train_l2 = 0.0

#         for features, labels_task1, labels_task2 in train_loader:

#             features = features.to(device)
#             labels_task1 = labels_task1.to(device)
#             labels_task2 = labels_task2.to(device)

#             optimizer.zero_grad()

#             pred1, pred2 = model(features)

#             total_loss, l1, l2 = custom_criterion(
#                 pred1,
#                 labels_task1,
#                 pred2,
#                 labels_task2
#             )

#             total_loss.backward()

#             optimizer.step()

#             train_loss += total_loss.item()
#             train_l1 += l1.item()
#             train_l2 += l2.item()

#         avg_total = train_loss / len(train_loader)
#         avg_l1 = train_l1 / len(train_loader)
#         avg_l2 = train_l2 / len(train_loader)

#         print(
#             f"Epoch [{epoch+1}/{epochs}] "
#             f"- total_loss={avg_total:.4f} "
#             f"- l1={avg_l1:.4f} "
#             f"- l2={avg_l2:.4f}"
#         )

#         # Jalankan validasi setiap val_step
#         if (epoch + 1) % val_step == 0:
#             if val_loader is not None:
#                 validation(model, val_loader, device, epoch, epochs)
#             else:
#                 print(f"skip validation")

# def validation(model, val_loader, device, epoch, epochs):
#     model.eval()
#     mtl_metrics = MultiTaskMetrics()

#     with torch.no_grad():
#         for features, labels1, labels2 in val_loader:
#             features, labels1, labels2 = features.to(device), labels1.to(device), labels2.to(device)
#             p1, p2 = model(features)
            
#             # _, pred1 = torch.max(p1, 1)
#             # val_acc1 += (pred1 == labels1).sum().item()
            
#             # _, pred2 = torch.max(p2, 1)
#             # val_acc2 += (pred2 == labels2).sum().item()
            
#             # total += labels1.size(0)
#             # Update data ke dalam metrics
#             mtl_metrics.update(p1, labels1, p2, labels2)
    
#     # Hitung dan tampilkan hasil
#     res = mtl_metrics.compute()
#     print(f"Digit Acc: {res['task_digit']['acc']:.4f} | F1: {res['task_digit']['f1']:.4f}")
#     print(f"Speaker Acc: {res['task_speaker']['acc']:.4f} | F1: {res['task_speaker']['f1']:.4f}")

# def prepare_dataloader(X, y1, y2, batch_size=32):
#     X_tensor = torch.FloatTensor(X)
#     y1_tensor = torch.LongTensor(y1)
#     y2_tensor = torch.LongTensor(y2)
#     dataset = TensorDataset(X_tensor, y1_tensor, y2_tensor)
#     return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- CONTOH PENYIAPAN DATA ---
# Anggap X_train adalah hasil feature_extraction yang sudah di-stack jadi numpy array
# X_train shape: (n_samples, 45)
# y_digit shape: (n_samples,) -> label 0-9
# y_speaker shape: (n_samples,) -> label 0-n_speaker

# Cara pakai:
# model = ShallowMTLModel(input_size=45, num_classes_task1=10, num_classes_task2=6)
# train_loader = prepare_dataloader(X_train_np, y_digit_np, y_speaker_np)
# val_loader = prepare_dataloader(X_val_np, y_digit_val, y_speaker_val)
# train_model(model, train_loader, val_loader)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from src.trainer.loss import MultiTaskLoss
from src.trainer.metrics import MultiTaskMetrics

def train_model(model, train_loader=None, val_loader=None, 
                epochs=50, lr=0.001, val_step=1, 
                patience=10, save_path="best_model.pt"):
    
    custom_criterion = MultiTaskLoss(weight_task1=1.0, weight_task2=0.2)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, verbose=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Training started on device: {device}")

    # History untuk plotting
    history = {
        "train_loss": [], "train_l1": [], "train_l2": [],
        "val_loss": [],
        "digit_acc": [], "digit_f1": [],
        "speaker_acc": [], "speaker_f1": []
    }

    best_val_digit_acc = 0.0
    no_improve = 0  # early stopping counter

    for epoch in range(epochs):
        # ── Training ──
        model.train()
        train_loss, train_l1, train_l2 = 0.0, 0.0, 0.0

        for features, labels_task1, labels_task2 in train_loader:
            features = features.to(device)
            labels_task1 = labels_task1.to(device)
            labels_task2 = labels_task2.to(device)

            optimizer.zero_grad()
            pred1, pred2 = model(features)
            total_loss, l1, l2 = custom_criterion(pred1, labels_task1, pred2, labels_task2)
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()
            train_l1 += l1.item()
            train_l2 += l2.item()

        avg_total = train_loss / len(train_loader)
        avg_l1    = train_l1   / len(train_loader)
        avg_l2    = train_l2   / len(train_loader)

        history["train_loss"].append(avg_total)
        history["train_l1"].append(avg_l1)
        history["train_l2"].append(avg_l2)

        print(f"Epoch [{epoch+1}/{epochs}]"
              f" - total_loss={avg_total:.4f}"
              f" - l1={avg_l1:.4f}"
              f" - l2={avg_l2:.4f}", end="")

        # ── Validasi setiap val_step ──
        if val_loader is not None and (epoch + 1) % val_step == 0:
            res, val_loss = validation(model, val_loader, custom_criterion, device, epoch, epochs)

            history["val_loss"].append(val_loss)
            history["digit_acc"].append(res['task_digit']['acc'])
            history["digit_f1"].append(res['task_digit']['f1'])
            history["speaker_acc"].append(res['task_speaker']['acc'])
            history["speaker_f1"].append(res['task_speaker']['f1'])

            scheduler.step(val_loss)

            # Simpan model terbaik berdasarkan digit accuracy
            if res['task_digit']['acc'] > best_val_digit_acc:
                best_val_digit_acc = res['task_digit']['acc']
                torch.save(model.state_dict(), save_path)
                print(f" ← best model saved! (digit_acc={best_val_digit_acc:.4f})")
                no_improve = 0
            else:
                no_improve += 1
                print()

            # Early stopping
            if no_improve >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
        else:
            print()

    return history


def validation(model, val_loader, criterion, device, epoch, epochs):
    model.eval()
    mtl_metrics = MultiTaskMetrics()
    val_loss = 0.0

    with torch.no_grad():
        for features, labels1, labels2 in val_loader:
            features = features.to(device)
            labels1  = labels1.to(device)
            labels2  = labels2.to(device)

            p1, p2 = model(features)

            # Hitung val loss juga
            total_loss, _, _ = criterion(p1, labels1, p2, labels2)
            val_loss += total_loss.item()

            mtl_metrics.update(p1, labels1, p2, labels2)

    res = mtl_metrics.compute()
    val_loss /= len(val_loader)

    print(f"\n  → Val Loss: {val_loss:.4f}"
          f" | Digit  Acc: {res['task_digit']['acc']:.4f}"
          f" F1: {res['task_digit']['f1']:.4f}"
          f" | Speaker Acc: {res['task_speaker']['acc']:.4f}"
          f" F1: {res['task_speaker']['f1']:.4f}")

    return res, val_loss


def prepare_dataloader(X, y1, y2, batch_size=32, shuffle=True):
    X_tensor  = torch.FloatTensor(X)
    y1_tensor = torch.LongTensor(y1)
    y2_tensor = torch.LongTensor(y2)
    dataset   = TensorDataset(X_tensor, y1_tensor, y2_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)