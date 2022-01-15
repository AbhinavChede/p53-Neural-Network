import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torch import nn, optim

# Processing the raw datset from the repository so it is easier to use
def process_data(csv_path, attributes):
    n = [str(i) for i in range(attributes)]
    df = pd.read_csv(csv_path, low_memory=False, names=n)
    df = df.reset_index()
    df = df.fillna(0)
    labels = df[df.columns[attributes - 1]].map(lambda x: 1 if x == 'active' else 0)
    df.drop(df.columns[(attributes - 1):(attributes + 1)], axis=1, inplace=True)
    df = ((df-df.min()) / (df.max() - df.min())) # Normalizing the data
    df = df.replace(0, 0.0000001) # Swapping zeros with small decimals so that no issues arise with logarithms
    data_list = df.values.tolist()
    df.drop(df.columns[0:attributes], axis=1, inplace=True)
    df[0] = data_list
    df[1] = labels
    return df

# A custom dataset class I can use for my processed dataset
class CustomDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, df, transforms=None):
        self.data = df
        self.data_arr = np.asarray(self.data.iloc[:, 0])
        self.label_arr = np.asarray(self.data.iloc[:, 1])
        self.transforms = transforms

    def __getitem__(self, index):
        data_instance = self.data_arr[index]
        data_label = self.label_arr[index]
        if self.transforms is not None:
            data_tensor = self.transforms(np.array(np.reshape(data_instance, (-1, 1))))
        return (data_tensor, data_label)

    def __len__(self):
        return len(self.data.index)

# Testing the accuracy of the model
def test_model(test_loader, model):
    correct_count, all_count = 0, 0
    for datas, labels in test_loader:
        for i in range(len(labels)):
            data = datas[i].view(1, 5408)
            with torch.no_grad():
                logps = model(data.float())

            
            ps = torch.exp(logps)
            probab = list(ps.numpy()[0])
            pred_label = probab.index(max(probab))
            true_label = labels.numpy()[i]
            if(true_label == pred_label):
                correct_count += 1
            all_count += 1
    print("Number Of Mutants Tested =", all_count)
    print("Model Accuracy =", (correct_count/all_count))
    print()

# Training the model
def train_model(epochs, learning_rate, m, loader, criterion):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=m)
    for e in range(epochs):
        running_loss = 0
        for data, labels in loader:
            data = data.view(data.shape[0], -1)        

            optimizer.zero_grad()            
            output = model(data.float())
            loss = criterion(output, labels)            

            loss.backward()            

            optimizer.step()            
            running_loss += loss.item()
        else:   
            print("Epoch {} - Training loss: {}".format(e, running_loss/len(train_loader)))


if __name__ == "__main__":
    attributes = 5409
    fraction = 0.7
    csv_path = 'C:/Users/abhin/Desktop/Post Ap CS/Neural Network/Data Sets/Dataset.csv'
    transformations = transforms.Compose([transforms.ToTensor()])                          

    dataset = process_data(csv_path, attributes)

    dataset_X = dataset[0]
    dataset_y = dataset[1]

    # Splitting the datset randomly
    train_dataset = dataset.sample(frac=fraction) # training set
    test_dataset = dataset.loc[~dataset.index.isin(train_dataset.index)] # test set

    train_custom_dataset = CustomDataset(train_dataset, transforms=transformations)
    test_custom_dataset = CustomDataset(test_dataset, transforms=transformations)

    train_loader = torch.utils.data.DataLoader(dataset=train_custom_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_custom_dataset, batch_size=64, shuffle=True)
    
    input_size = 5408
    hidden_sizes = [256, 16] # hidden layers
    output_size = 2

    model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                        nn.ReLU(),
                        nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                        nn.ReLU(),
                        nn.Linear(hidden_sizes[1], output_size),
                        nn.LogSoftmax(dim=1))

    criterion = nn.NLLLoss()

    learning_rate = 0.0002
    m = 0.9
    epochs = 4

    train_model(epochs, learning_rate, m, train_loader, criterion)

    test_model(test_loader, model)

    torch.save(model, './p53_mutant_model.pt')