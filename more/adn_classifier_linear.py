import torch
import torch.nn as nn

class ADNClassifierLinear(nn.Module):
    def __init__(self, input_size, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)

        #  Remplacement du Conv1d par un Linear sur la dimension de séquence (input_size)
        self.linear_pos = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):  # x: (batch_size, seq_length)
        x = self.embedding(x)      # x: (batch_size, seq_length, embed_size)

        x = x.transpose(1, 2)      # x: (batch_size, embed_size, seq_length)
        x = self.linear_pos(x)     # x: (batch_size, embed_size, hidden_size)
        x = self.relu(x)

        x = x.transpose(1, 2)      # x: (batch_size, hidden_size, embed_size)
        x = self.pool(x)           # x: (batch_size, hidden_size, 1)
        x = x.squeeze(-1)          # x: (batch_size, hidden_size)

        x = self.classifier(x)     # x: (batch_size, 1)
        return x.squeeze(-1)       # x: (batch_size,)

    def fit(self, X, y, epochs=10, batch_size=32, learning_rate=0.001, validation_data=None):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        history = {'loss': [], 'val_accuracy': [], 'train_accuracy': []}

        for epoch in range(epochs):
            loss_total = 0.0
            self.train()
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.forward(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                loss_total += loss.item() * batch_X.size(0)
            loss_mean = loss_total / len(dataloader.dataset)

            train_acc = self.evaluate(X, y)
            val_acc = self.evaluate(validation_data[0], validation_data[1]) if validation_data else 0.0
            history['loss'].append(loss_mean)
            history['val_accuracy'].append(val_acc)
            history['train_accuracy'].append(train_acc)

            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss_mean:.4f}')
            print(f'Training Accuracy: {train_acc * 100:.2f}%')

            if validation_data:
                print(f'Validation Accuracy: {val_acc * 100:.2f}%')
            print('----------------------------------------')

        return history

    def evaluate(self, X, y):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(X)
            predictions = outputs >= 0
            accuracy = (predictions == y).float().mean()
        return accuracy.item()

    def save(self, path, example_input):
        model_traced = torch.jit.trace(self, example_input)
        model_traced.save(path)
