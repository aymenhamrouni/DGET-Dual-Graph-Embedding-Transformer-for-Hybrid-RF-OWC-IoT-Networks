# train_DGET.py: DGET training/validation/plotting entry point
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import tikzplotlib

from utils_DGET import (TransformerGNN, compute_class_weights, create_batched_dataloaders,
                        normalize_edge_features, normalize_node_features, normalize_edge_weights,
                        postProcessing, generate_all_combinations_indices)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# --- Configuration ---
size = 5  # Set to N_APs + N_d of your generated dataset
VIZ = True
edge_classes = 8
n_repeats = 10
epochs = 100
models_dir = 'models'
ensure_dir(models_dir)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# --- Data loading ---
import csv
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from tqdm import tqdm

# --- Read and process input files (mostly identical logic, as in original DGET.py) ---

def load_dataset(size):
    time_length = 0
    edge_index = []
    edge_features = []
    allowedTechnologies = []
    numberofnodespersnapshot = size*(size-1)
    MonteCarlo = 0
    # --- Edges features ---
    all_data = []
    with open(f'dataset/inputEdgesFeatures{size}.csv', 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        next(csv_reader)
        for row in csv_reader:
            all_data.append([float(row[4]), float(row[5]), \
                            float(row[6]), float(row[7])])
            time_length = float(row[0]) if float(row[0]) > time_length else time_length
    data_array = np.array(all_data)
    mins = data_array.min(axis=0)
    maxs = data_array.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1
    with open(f'dataset/inputEdgesFeatures{size}.csv', 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        next(csv_reader)
        for row in csv_reader:
            edge_index.append([int(row[1]), int(row[2])])
            allowedTechnologies.append(int(float(row[9])))
            raw_values = np.array([float(row[4]), float(row[5]),
                                   float(row[6]), float(row[7])])
            edge_features.append([float(row[3])] + raw_values.tolist() + [float(row[8])])
    # --- Nodes features ---
    input_node_features = []
    with open(f'dataset/inputNodesFeatures{size}.csv', 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        next(csv_reader)
        for row in csv_reader:
            input_node_features.append([float(row[2])])
            if int(row[0]) == int(time_length):
                MonteCarlo += 1
    MonteCarlo = int(MonteCarlo/size)
    input_node_features = np.stack(input_node_features)
    # --- Labels/recorded ---
    labelsOrignal, labelsAugmented = [], []
    with open(f'dataset/recordedEdgesFeatures{size}.csv', 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        next(csv_reader)
        for row in csv_reader:
            labelsOrignal.append([float(row[6])])
            labelsAugmented.append([float(row[7])])
    labelsOrignal = np.stack(labelsOrignal)
    labelsAugmented = np.stack(labelsAugmented)
    recorded_node_features = []
    with open(f'dataset/recordedNodesFeatures{size}.csv', 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        next(csv_reader)
        for row in csv_reader:
            recorded_node_features.append([float(row[2])])
    # --- Normalizations ---
    edge_features = normalize_edge_features(np.array(edge_features))
    allowedTechnologies = normalize_edge_weights(np.array(allowedTechnologies))
    labelsOrignal = normalize_edge_weights(np.array(labelsOrignal))
    input_node_features = normalize_node_features(np.array(input_node_features))
    edge_index = np.stack(edge_index)
    labelsAugmented = np.stack(labelsAugmented)
    recorded_node_features = np.stack(recorded_node_features)
    # --- Data object construction ---
    def create_graph_data(node_features, edge_index, edge_features, edge_weights, labels):
        i = 0
        j = 0
        data_list = []
        dataset = []
        for m in range(MonteCarlo):
            while i < (int(time_length)+1)*size*(m+1):
                x = torch.tensor(node_features[i:i+size], dtype=torch.float)
                edge_index_t = torch.tensor(edge_index[j:j+(size*(size-1))], dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(edge_features[j:j+(size*(size-1))], dtype=torch.float)
                edge_weight = torch.tensor(edge_weights[j:j+(size*(size-1))], dtype=torch.float)
                y = torch.tensor(labels[j:j+(size*(size-1))], dtype=torch.long)
                data = Data(x=x, edge_index=edge_index_t, edge_attr=edge_attr, edge_weight=edge_weight, y=y)
                data_list.append(data)
                i += size
                j += (size*(size-1))
            dataset.append(data_list)
            data_list = []
        return dataset
    data_list_known = create_graph_data(input_node_features, edge_index, edge_features, allowedTechnologies, labelsAugmented)
    data_list_recorded = create_graph_data(recorded_node_features, edge_index, edge_features, labelsOrignal, labelsAugmented)
    return data_list_known, data_list_recorded, MonteCarlo, time_length

def plot_roc_curve(all_labels, all_probs, edge_classes, title="ROC Curves", filename="roc_curves"):
    plt.figure(figsize=(10, 10))
    colors = plt.cm.rainbow(np.linspace(0, 1, edge_classes))
    for i in range(edge_classes):
        binary_labels = (all_labels == i).astype(int)
        class_probs = all_probs[:, i]
        fpr, tpr, _ = roc_curve(binary_labels, class_probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2,
                 label=f'ROC curve (class {i}) (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    tikzplotlib.save(f"{filename}.tex")
    plt.savefig(f"{filename}.png")
    plt.show()

def plot_confusion(orig_labels, orig_preds, noisy_labels=None, noisy_preds=None, filename="confusion_matrices"):
    fig, axs = (plt.subplots(1, 2, figsize=(15, 6)) if noisy_labels is not None else plt.subplots(figsize=(7,6)))
    axs = np.atleast_1d(axs)
    cm_orig = confusion_matrix(orig_labels, orig_preds)
    import seaborn as sns
    sns.heatmap(cm_orig, annot=True, fmt='d', cmap='Blues', ax=axs[0])
    axs[0].set_title('Confusion Matrix - Original Data')
    axs[0].set_xlabel('Predicted')
    axs[0].set_ylabel('True')
    if noisy_labels is not None:
        cm_noisy = confusion_matrix(noisy_labels, noisy_preds)
        sns.heatmap(cm_noisy, annot=True, fmt='d', cmap='Blues', ax=axs[1])
        axs[1].set_title('Confusion Matrix - Noisy Data')
        axs[1].set_xlabel('Predicted')
        axs[1].set_ylabel('True')
    plt.tight_layout()
    plt.savefig(f"{filename}.png")
    plt.show()

def plot_history_curves(history):
    # Losses
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label="Training Loss", color="blue", marker='o', markevery=25)
    plt.plot(history['val_loss'], label="Validation Loss", color="orange", linestyle='-.', marker='o', markevery=25)
    plt.plot(history['train_classification_loss'], label="Training Classification Loss", color="blue", marker='*', markevery=25)
    plt.plot(history['val_classification_loss'], label="Validation Classification Loss", color="orange", linestyle='-.', marker='*', markevery=25)
    plt.plot(history['train_consistency_loss'], label="Training Consistency Loss", color="blue", marker='^', markevery=25)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("history_losses.png")
    plt.show()
    # Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_acc'], label="Training Accuracy", color='blue')
    plt.plot(history['val_acc'], label="Validation Accuracy", color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("history_accuracy.png")
    plt.show()

def save_best_model(model, out_path):
    torch.save(model, out_path)

# ===================== MAIN TRAINING LOGIC =====================
data_list_known, data_list_recorded, MonteCarlo, time_length = load_dataset(size)
Int = generate_all_combinations_indices(size-1)
class_weights = compute_class_weights(data_list_known)
class_weights = np.array([ i if i!=np.inf else 0 for i in class_weights])
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

def main():
    # Split train/test
    train_known, test_known = train_test_split(data_list_known, test_size=0.2, random_state=42)
    train_recorded, test_recorded = train_test_split(data_list_recorded, test_size=0.2, random_state=42)

    model = TransformerGNN(node_features=32, num_classes=edge_classes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4, weight_decay=5e-4, betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=5e-4, epochs=epochs, steps_per_epoch=len(train_known),
        pct_start=0.1, div_factor=25, final_div_factor=1e2)
    # Training
    history = {
        'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'learning_rates': [],
        'train_consistency_loss': [], 'train_classification_loss': [],
        'val_consistency_loss': [], 'val_classification_loss': [], 
        'accuracies_improved': [], 'avgSwitch': []
    }
    best_val_loss = float('inf')
    best_model_state = None
    patience, patience_counter = 10, 0
    for epoch in range(epochs):
        model.train()
        total_train_loss, total_classification_loss, total_consistency_loss = 0, 0, 0
        correct_train, total_train = 0, 0
        train_loader, val_loader = create_batched_dataloaders(
            train_known, train_recorded, test_known, test_recorded, batch_size=32)
        # Training loop
        for batch_idx, (known_batch, recorded_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            scores, consistency_loss, _, _, _, _ = model(known_batch, recorded_batch)
            known_labels = []
            for instance in known_batch:
                for snapshot in instance:
                    known_labels.append(snapshot.y)
            known_labels = torch.cat(known_labels, dim=0).squeeze(1)
            LossWeighted = torch.nn.NLLLoss(weight=class_weights_tensor)
            classification_loss = LossWeighted(scores, known_labels)
            consistency_weight = min(0.1 * (1 + epoch / epochs), 0.5)
            loss = classification_loss + consistency_weight * consistency_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item() * len(known_batch)
            total_classification_loss += classification_loss * len(known_batch)
            total_consistency_loss += consistency_loss * len(known_batch) / 50
            preds = scores.argmax(dim=1)
            correct_train += (preds == known_labels).sum().item()
            total_train += known_labels.size(0)
        scheduler.step()
        # Metrics reporting
        avg_train_loss = total_train_loss / len(train_known)
        avg_classification_loss = total_classification_loss / len(train_known)
        avg_consistency_loss = total_consistency_loss / len(train_known)
        train_acc = correct_train / total_train
        # Validation
        model.eval()
        total_val_loss, total_val, correct_val = 0, 0, 0
        total_consistency_loss_val, total_classification_loss_val = 0, 0
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for known_batch, recorded_batch in val_loader:
                scores, consistency_loss, _, _, _, _ = model(known_batch, None)
                known_labels = []
                for instance in known_batch:
                    for snapshot in instance:
                        known_labels.append(snapshot.y)
                known_labels = torch.cat(known_labels, dim=0).squeeze(1)
                classification_loss = torch.nn.NLLLoss(weight=class_weights_tensor)(scores, known_labels)
                consistency_weight = min(0.1 * (1 + epoch / epochs), 0.5)
                loss = classification_loss + consistency_weight * consistency_loss
                total_val_loss += loss.item() * len(known_batch)
                preds = scores.argmax(dim=1)
                correct_val += (preds == known_labels).sum().item()
                total_val += known_labels.size(0)
                total_consistency_loss_val += consistency_loss * len(known_batch)/50
                total_classification_loss_val += classification_loss * len(known_batch)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(known_labels.cpu().numpy())
                all_probs.extend(torch.exp(scores).cpu().numpy())
        avg_val_loss = total_val_loss / len(test_known)
        val_acc = correct_val / total_val
        avg_consistency_loss_val = total_consistency_loss_val / len(test_known)
        avg_classification_loss_val = total_classification_loss_val / len(test_known)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_classification_loss'].append(avg_classification_loss)
        history['val_classification_loss'].append(avg_classification_loss_val)
        history['train_consistency_loss'].append(avg_consistency_loss)
        history['val_consistency_loss'].append(avg_consistency_loss_val)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        print(f"Epoch {epoch + 1}/{epochs}: train loss {avg_train_loss:.4f} val loss {avg_val_loss:.4f} train acc {train_acc:.4f} val acc {val_acc:.4f}")
        # Save best model by val loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            save_best_model(model, os.path.join(models_dir, f'best_model_size{size}.pt'))
            print(f"(Epoch {epoch + 1}) Saved best model.")
        # Early stopping
        if avg_val_loss < best_val_loss:
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping after {epoch + 1} epochs.")
            break
    # Restore best model for downstream plots/evals
    if best_model_state:
        model.load_state_dict(best_model_state)
    # Evaluate and plot
    plot_history_curves(history)
    plot_roc_curve(np.array(all_labels).flatten(), np.array(all_probs), edge_classes)
    plot_confusion(np.array(all_labels).flatten(), np.array(all_preds).flatten())
    print(classification_report(np.array(all_labels).flatten(), np.array(all_preds).flatten()))

if __name__ == '__main__':
    main()
