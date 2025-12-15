import PureAttention as pa
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class HousingModel(pa.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Arhitectura: Input(8) -> Linear(32) -> ReLU -> Linear(1)
        self.l1 = pa.Linear(input_dim, 128)
        self.relu = pa.ReLU()
        self.l2 = pa.Linear(128, 1)

    def forward(self, x):
        x = self.l1.forward(x)
        x = self.relu.forward(x)
        x = self.l2.forward(x)
        return x

    def parameters(self):
        return self.l1.parameters() + self.l2.parameters()

def main():
    # 1. Pregătirea datelor (Scikit-Learn)
    print("Se încarcă datele California Housing...")
    data = fetch_california_housing()
    X, y = data.data, data.target

    # Foarte important: Scalarea datelor. Rețelele nu învață bine pe date brute.
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Redimensionăm y să fie (N, 1) nu vector (N,)
    y = y.reshape(-1, 1)

    # Convertim la float32 (CUDA preferă float32)
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    # Split train/test (folosim doar un subset mic pentru test rapid dacă vrei)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_scaled = scaler.fit_transform(X_train)
    # Luăm un batch mai mic pentru antrenare (ex: 1000 exemple) ca să vedem viteza
    # Sau poți antrena pe tot setul dacă implementarea ta suportă memorie suficientă.
    BATCH_SIZE = 2000
    X_batch = X_train_scaled[:BATCH_SIZE]
    y_batch = y_train[:BATCH_SIZE]
    
    # Flatten pentru transferul către PureAttention (presupunând că to_device vrea 1D array)
    input_data_flat = X_batch.flatten()
    target_data_flat = y_batch.flatten()

    B = X_batch.shape[0]  # Batch size
    IN = X_batch.shape[1] # 8 features
    OUT = 1               # 1 prediction (price)

    # 2. Setup PureAttention Tensors
    # Atenție: Asigură-te că Tensor-ul tău știe dimensiunile corecte
    input_tensor = pa.Tensor([B, IN], False)
    input_tensor.to_device(input_data_flat)

    target_tensor = pa.Tensor([B, OUT], False)
    target_tensor.to_device(target_data_flat)

    # 3. Model și Optimizer
    model = HousingModel(input_dim=IN)
    
    # Learning rate poate necesita ajustare. 0.001 sau 0.01 sunt standard.
    optimizer = pa.Adam(model.parameters(), lr=0.00001) 
    criterion = pa.MSE()

    print(f"Start training on {B} samples...")

    # 4. Training Loop
    epochs = 1000
    for i in range(epochs):
        optimizer.zero_grad()

        # Forward pass
        pred = model.forward(input_tensor)
        
        # Loss calculation
        loss = criterion.forward(pred, target_tensor)

        # Backward pass
        loss.backward()
        
        # Weight update
        optimizer.step()

        if i % 50 == 0:
            # Loss-ul returnat de MSE ar trebui să scadă
            val_loss = loss.to_host()[0]
            print(f"Epoch {i}, MSE Loss: {val_loss:.6f}")

    print("Antrenament finalizat.")
    
    # 5. (Optional) Verificare vizuală pe primele 5 exemple
    preds = pred.to_host()
    # Deoarece output-ul e flatten, îl remodelăm mental sau afișăm direct
    print("\nVerificare (Primele 5 exemple):")
    print(f"{'Predicție':<15} | {'Real (Target)':<15} | {'Diferență':<15}")
    print("-" * 50)
    for k in range(5):
        p = preds[k]
        t = target_data_flat[k]
        print(f"{p:<15.4f} | {t:<15.4f} | {abs(p-t):<15.4f}")

if __name__ == "__main__":
    main()