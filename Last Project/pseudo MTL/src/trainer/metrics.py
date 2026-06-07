import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

class MultiTaskMetrics:
    def __init__(self):
        # Tempat menyimpan prediksi dan target selama satu fase evaluasi
        self.reset()

    def reset(self):
        self.all_preds1 = []
        self.all_targets1 = []
        self.all_preds2 = []
        self.all_targets2 = []

    def update(self, p1, t1, p2, t2):
        """
        Update hasil prediksi dari satu batch.
        p1, p2: logits dari model (tensor)
        t1, t2: labels asli (tensor)
        """
        # Konversi tensor ke numpy dan ambil kelas tertinggi (argmax)
        pred1 = torch.argmax(p1, dim=1).cpu().numpy()
        pred2 = torch.argmax(p2, dim=1).cpu().numpy()
        
        self.all_preds1.extend(pred1)
        self.all_targets1.extend(t1.cpu().numpy())
        
        self.all_preds2.extend(pred2)
        self.all_targets2.extend(t2.cpu().numpy())

    def compute(self):
        """Menghitung metrik akhir untuk kedua task"""
        
        # Task 1: Digit Recognition
        acc1 = accuracy_score(self.all_targets1, self.all_preds1)
        f1_1 = f1_score(self.all_targets1, self.all_preds1, average='weighted')
        
        # Task 2: Speaker Identification
        acc2 = accuracy_score(self.all_targets2, self.all_preds2)
        f1_2 = f1_score(self.all_targets2, self.all_preds2, average='weighted')
        
        results = {
            'task_digit': {'acc': acc1, 'f1': f1_1},
            'task_speaker': {'acc': acc2, 'f1': f1_2}
        }
        
        return results

    def print_full_report(self):
        """Mencetak laporan klasifikasi lengkap (bagus untuk lampiran tesis)"""
        print("\n=== REPORT TASK 1: DIGIT RECOGNITION ===")
        print(classification_report(self.all_targets1, self.all_preds1))
        
        print("\n=== REPORT TASK 2: SPEAKER IDENTIFICATION ===")
        print(classification_report(self.all_targets2, self.all_preds2))

# Digunakan untuk debugging/visualisasi hasil
def get_confusion_matrix(targets, preds):
    return confusion_matrix(targets, preds)