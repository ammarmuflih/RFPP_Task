import argparse
import torch
from src.utils.config import load_config
from src.data.loader import get_dataloader
from src.features.pipeline import FeaturePipeline
from src.models.mlp import MultiTaskMLP
from src.evaluation.metrics import compute_multitask_metrics
from src.evaluation.visualizer import plot_confusion_matrix

def evaluate(config_path, model_weight):
    config = load_config(config_path)
    pipeline = FeaturePipeline(config['features'])
    
    # 1. Load Test Data
    test_loader = get_dataloader(config['data']['test_json'], pipeline, batch_size=1, shuffle=False)

    # 2. Load Trained Model
    model = MultiTaskMLP(
        input_dim=config['model']['input_dim'],
        num_digits=10,
        num_speakers=config['model']['num_speakers']
    )
    model.load_state_dict(torch.load(model_weight))
    model.eval()

    # 3. Inference
    all_preds = {'digit': [], 'speaker': []}
    all_labels = {'digit': [], 'speaker': []}

    with torch.no_grad():
        for features, labels in test_loader:
            digit_out, speaker_out = model(features)
            
            all_preds['digit'].append(digit_out.argmax(1))
            all_preds['speaker'].append(speaker_out.argmax(1))
            all_labels['digit'].append(labels['digit'])
            all_labels['speaker'].append(labels['speaker'])

    # 4. Metrics & Visualization
    results = compute_multitask_metrics(all_labels, all_preds)
    print(f"Test Accuracy - Digit: {results['digit_acc']:.2%}")
    print(f"Test Accuracy - Speaker: {results['speaker_acc']:.2%}")

    # Simpan plot ke folder output
    plot_confusion_matrix(all_labels['digit'], all_preds['digit'], title="Confusion Matrix Digits")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, help="Path ke .pth file")
    args = parser.parse_args()
    evaluate(args.config, args.model)