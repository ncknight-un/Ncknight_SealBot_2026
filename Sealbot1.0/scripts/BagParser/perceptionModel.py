# STEPS TODO: 
# This file will combine the int and ext tag data considering the timestamps, data quality (existance, and tag stability metric)
# Then divide the data into training and testing sets, and save them into csv files.
# Training loop with model as mlp will be in another file [keep it modular so that changes in model architecture do not affect data processing]
# Loss function
# Validation loop
# Plots and metrics to understand learning and model performance.
# Data saving functions
# Evaluation loop

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from mlp import MLP
import dataProcessor as dp
import torch 
import matplotlib.pyplot as plt
import torch.utils.data as data_utils


class ProcessData:
    
    def combine_and_process_data():

        ##########################################
        #              Extract Data
        ##########################################

        rel_tag_poses = dp.unfilter_rel_paths()
        all_frame_poses = dp.read_frame_poses()
        centroid_path = dp.calculate_centroid_path(all_frame_poses)
        displacement_centroid_path = dp.calculate_displacement(centroid_path)

        ##########################################
        #              Combine Data
        ##########################################
        
        # Realsense in at a fps of 30, while blue-os is at 10 fps (Trying to improve this but still wont be same)
        int_frame_list = list(rel_tag_poses.keys())
        ext_frame_list = list(displacement_centroid_path.keys())
        frame_idx_map = dp.sync_frames(int_frame_list, ext_frame_list) 
        combined_data = dp.align_and_combine_data(rel_tag_poses, displacement_centroid_path, frame_idx_map)
        # TODO : Add checks to remove frames where either internal or external data is missing.

        ##########################################
        #          DataLoader and Split
        ##########################################

        # Using pytorch dataloader for batching and shuffling
        # How combined data looks like: 
        # [0, {0: (...), 1: (...), 3: (...), 5: (...), 6: (...), 7: (...), 8: (...), 9: (...), 10: (...), 12: (...), 14: (...), 16: (...), 17: (...), 18: (...)}, [-0.006185817816079409, -0.0005270704363504741, 0.011696499402920946, -0.009942188642799171, -0.0038478387254944264, -0.0011139264649408396, 0.9999425514448139]]

        # X should be the tag poses and y should be pose displacement
        X = combined_data.drop(columns=['displacement'])
        y = combined_data['displacement']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        train_tensor = data_utils.TensorDataset(torch.tensor(X_train.values, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.float32))
        test_tensor = data_utils.TensorDataset(torch.tensor(X_test.values, dtype=torch.float32), torch.tensor(y_test.values, dtype=torch.float32))
        train_loader = data_utils.DataLoader(train_tensor, batch_size=32, shuffle=True)
        test_loader = data_utils.DataLoader(test_tensor, batch_size=32, shuffle=False)
        return train_loader, test_loader


if __name__ == "__main__":
    
    train_loader, test_loader = ProcessData.combine_and_process_data()

    ##########################################
    #            Training Loop
    ##########################################
    model = MLP(input_size=train_loader.dataset.tensors[0].shape[1], hidden_size=64, output_size=1)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}')

    ##########################################
    #            Evaluation Loop
    ###########################################

    for epoch in range(num_epochs):
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets)
                running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(test_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Testing Loss: {epoch_loss:.4f}')

    ##########################################
    #            Save Model
    ##########################################

    torch.save(model.state_dict(), 'mlp_model.pth')
    print("Model saved as mlp_model.pth")

    ##########################################
    #            Save Metrics
    ##########################################
    
    metrics = {
        'final_training_loss': epoch_loss,
        'final_testing_loss': epoch_loss
    }
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv('training_metrics.csv', index=False)
    print("Training metrics saved as training_metrics.csv")
    
    ##########################################
    #            Plots
    ##########################################
    
    epochs = list(range(1, num_epochs + 1))
    plt.plot(epochs, [epoch_loss]*num_epochs, label='Training Loss')
    plt.plot(epochs, [epoch_loss]*num_epochs, label='Testing Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss over Epochs')
    plt.legend()
    plt.savefig('loss_plot.png')
    print("Loss plot saved as loss_plot.png")
    plt.show()

    ###########################################
    #            Evaluation Loop
    ############################################

    # for epoch in range(num_epochs):
    #     model.eval()
    #     running_loss = 0.0
    #     with torch.no_grad():
    #         for inputs, targets in test_loader:
    #             outputs = model(inputs)
    #             loss = criterion(outputs.squeeze(), targets)
    #             running_loss += loss.item() * inputs.size(0)
    #     epoch_loss = running_loss / len(test_loader.dataset)
    #     print(f'Epoch {epoch+1}/{num_epochs}, Testing Loss: {epoch_loss:.4f}')








