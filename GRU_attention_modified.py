# GRU_attention_modified.py
# Returns avg_loss at the end of training for external comparison.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import pickle
import os
import shutil
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

# --- Shared helpers ---
def set_seed(seed=42):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

class EarlyStopping:
    def __init__(self, patience=15, verbose=False, delta=0, path='checkpoint.pth'):
        self.patience=patience; self.verbose=verbose; self.counter=0; self.best_score=None; self.early_stop=False; self.val_loss_min=np.Inf; self.delta=delta; self.path=path
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None: self.best_score=score; self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter+=1
            if self.verbose and (self.counter % 5 == 0 or self.counter == self.patience) : 
                print(f'   > EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience: self.early_stop = True
        else: self.best_score=score; self.save_checkpoint(val_loss, model); self.counter=0
    def save_checkpoint(self, val_loss, model):
        if self.verbose: print(f'   > Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path); self.val_loss_min = val_loss

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__(); self.Wa=nn.Linear(hidden_dim, hidden_dim); self.Ua=nn.Linear(hidden_dim, hidden_dim); self.Va=nn.Linear(hidden_dim, 1)
    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query.unsqueeze(1)) + self.Ua(keys))).squeeze(2)
        attn_weights = F.softmax(scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), keys).squeeze(1)
        return context, attn_weights

class GRU_Attention_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_prob):
        super(GRU_Attention_Model, self).__init__()
        gru_dropout = dropout_prob if num_layers > 1 else 0
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=gru_dropout, bidirectional=False)
        self.attention = Attention(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
    def forward(self, x):
        gru_out, h_n = self.gru(x)
        last_layer_hidden_state = h_n[-1, :, :]
        context_vector, _ = self.attention(last_layer_hidden_state, gru_out)
        x = self.fc1(context_vector); x = self.relu(x); x = self.fc2(x)
        return x

# --- Main training entry ---
def run_training(
    output_dir,
    data_path='btc_labeled_data.csv',
    time_steps=60,
    n_splits=5,
    batch_size=64,
    epochs=100,
    hidden_dim=16,
    num_layers=1,
    dropout_prob=0.5,
    learning_rate=0.001,
    weight_decay=0.01
):
    os.makedirs(output_dir, exist_ok=True)
    set_seed(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        df = pd.read_csv(data_path, index_col=0, parse_dates=[0])
    except FileNotFoundError:
        print(f"[ERROR] Data file not found at '{data_path}'."); return None
    
    FEATURE_COLUMNS = [col for col in df.columns if col != 'label']; LABEL_COLUMN = 'label'
    raw_features = df[FEATURE_COLUMNS].values; raw_labels = df[LABEL_COLUMN].values
    X, y = [], []
    for i in range(len(raw_features) - time_steps):
        X.append(raw_features[i:(i + time_steps)]); y.append(raw_labels[i + time_steps])
    X, y = np.array(X), np.array(y)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    history_folds, scores_folds = [], {'val_loss': [], 'val_accuracy': []}

    for fold, (train_index, test_index) in enumerate(tscv.split(X), 1):
        print(f"\n--- Cross-Validation Fold {fold}/{n_splits} ---")
        X_train_raw, X_test_raw = X[train_index], X[test_index]; y_train, y_test = y[train_index], y[test_index]
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_train_reshaped = X_train_raw.reshape(-1, X_train_raw.shape[-1]); scaler.fit(X_train_reshaped)
        X_train_tensor = torch.from_numpy(scaler.transform(X_train_reshaped).reshape(X_train_raw.shape)).float().to(device)
        X_test_reshaped = X_test_raw.reshape(-1, X_test_raw.shape[-1])
        X_test_tensor = torch.from_numpy(scaler.transform(X_test_reshaped).reshape(X_test_raw.shape)).float().to(device)
        y_train_tensor = torch.from_numpy(y_train).long().to(device); y_test_tensor = torch.from_numpy(y_test).long().to(device)
        train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)

        INPUT_DIM = X_train_tensor.shape[2]; OUTPUT_DIM = len(np.unique(y_train))
        model = GRU_Attention_Model(INPUT_DIM, hidden_dim, num_layers, OUTPUT_DIM, dropout_prob).to(device)
        
        class_counts = pd.Series(y_train).value_counts()
        class_weights = (len(y_train) / (OUTPUT_DIM * class_counts.sort_index())).values
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float).to(device))
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.2, verbose=False)

        model_path = os.path.join(output_dir, f'best_model_fold_{fold}.pth')
        early_stopping = EarlyStopping(patience=15, verbose=True, path=model_path)
        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

        for epoch in range(epochs):
            model.train()
            for inputs, labels in train_loader:
                optimizer.zero_grad(); outputs = model(inputs); loss = criterion(outputs, labels); loss.backward(); optimizer.step()
            
            model.eval()
            with torch.no_grad():
                train_outputs = model(X_train_tensor); train_loss = criterion(train_outputs, y_train_tensor)
                _, train_pred = torch.max(train_outputs, 1); train_acc = (train_pred == y_train_tensor).sum().item() / len(y_train_tensor)
                val_outputs = model(X_test_tensor); val_loss = criterion(val_outputs, y_test_tensor)
                _, val_pred = torch.max(val_outputs, 1); val_acc = (val_pred == y_test_tensor).sum().item() / len(y_test_tensor)
                
            history['loss'].append(train_loss.item()); history['accuracy'].append(train_acc)
            history['val_loss'].append(val_loss.item()); history['val_accuracy'].append(val_acc)
            
            if (epoch + 1) % 10 == 0:
                 print(f"   Epoch [{epoch+1:03d}/{epochs}] | Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%}")

            scheduler.step(val_loss.item()); early_stopping(val_loss, model)
            if early_stopping.early_stop: print("   > Early stopping triggered."); break
        
        history_folds.append(history)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        with torch.no_grad():
            final_outputs = model(X_test_tensor); final_loss = criterion(final_outputs, y_test_tensor).item()
            _, final_predicted = torch.max(final_outputs.data, 1); final_accuracy = (final_predicted == y_test_tensor).sum().item() / len(y_test_tensor)
        scores_folds['val_loss'].append(final_loss); scores_folds['val_accuracy'].append(final_accuracy)

    print(f"\nSaving model and plotting results to '{output_dir}'...")
    if n_splits > 0:
        final_model_path = os.path.join(output_dir, 'best_model_final.pth')
        shutil.copyfile(os.path.join(output_dir, f'best_model_fold_{n_splits}.pth'), final_model_path)
        
        final_scaler_path = os.path.join(output_dir, 'final_scaler.pkl')
        with open(final_scaler_path, 'wb') as f: pickle.dump(scaler, f)
        
        # Save feature column list for inference pipeline compatibility.
        feature_columns_path = os.path.join(output_dir, 'feature_columns.pkl')
        with open(feature_columns_path, 'wb') as f:
            pickle.dump(FEATURE_COLUMNS, f)
        print(f"   > Feature columns saved to '{feature_columns_path}'")
        # -------------------------------------------------------------
        
    avg_acc = np.mean(scores_folds['val_accuracy']); std_acc = np.std(scores_folds['val_accuracy'])
    avg_loss = np.mean(scores_folds['val_loss'])  # Compute average validation loss.
    print(f"\nAverage Validation Loss: {avg_loss:.4f}, Average Validation Accuracy: {avg_acc:.2%} (Std Dev: {std_acc:.2%})")
    
    last_fold_history = history_folds[-1]
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(last_fold_history['accuracy'], label='Training Accuracy')
    plt.plot(last_fold_history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Model Accuracy (Last Fold)\nAvg Val Acc: {avg_acc:.2%}')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(last_fold_history['loss'], label='Training Loss')
    plt.plot(last_fold_history['val_loss'], label='Validation Loss')
    plt.title(f'Model Loss (Last Fold)\nAvg Val Loss: {avg_loss:.4f}')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"   > Training history chart saved to '{plot_path}'")
    
    # Return average validation loss for brute-force ranking.
    return avg_loss
