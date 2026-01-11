import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class Conv1DVAE(nn.Module):
    def __init__(self, input_dim=90, latent_dim=10):
        super(Conv1DVAE, self).__init__()
        
        # Encoder: 1D Convolutions
        # Input shape: (Batch, 1, 90)
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Calculate flattened size dynamically or pre-calculated
        # 90 -> 45 -> 23 (approx)
        self.flatten_size = 32 * (input_dim // 4) # integer division approx

        # Layer 1: 45. Layer 2: 23.
        self.flatten_size = 32 * 23 
        
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, self.flatten_size)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # 23 -> 46
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1), # 46 -> 92 (approx)

        )
        self.final_trim = input_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # x shape: (Batch, Features) -> Reshape to (Batch, 1, Features)
        x = x.unsqueeze(1)
        
        h = self.encoder(x)
        h_flat = h.view(h.size(0), -1)
        
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        z = self.reparameterize(mu, logvar)
        
        # Decode
        z_projected = self.decoder_input(z)
        z_reshaped = z_projected.view(z_projected.size(0), 32, 23)
        
        recon = self.decoder(z_reshaped)
        
        # Fix dimensions (ConvTranspose sometimes gives slightly different sizes)
        if recon.shape[2] != self.final_trim:
            recon = nn.functional.interpolate(recon, size=self.final_trim)
            
        # Squeeze back to (Batch, Features)
        return recon.squeeze(1), mu, logvar

def train_conv_vae(features, input_dim=90, latent_dim=10, epochs=50):
    # Standardize data type
    dataset = TensorDataset(torch.FloatTensor(features))
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = Conv1DVAE(input_dim=input_dim, latent_dim=latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    print("Training Convolutional VAE...")
    for epoch in range(epochs):
        for batch in dataloader:
            x = batch[0]
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            
            # Loss
            MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = MSE + KLD
            
            loss.backward()
            optimizer.step()
            
    return model

def get_latent_conv(model, features):
    model.eval()
    with torch.no_grad():
        x = torch.FloatTensor(features).unsqueeze(1) # Add channel dim
        h = model.encoder(x)
        h_flat = h.view(h.size(0), -1)
        return model.fc_mu(h_flat).numpy()

class CVAE(nn.Module):
    def __init__(self, input_dim=90, num_classes=19, hidden_dim=64, latent_dim=10):
        super(CVAE, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Encoder: Takes [x, y]
        # We concat x (90) + y (num_classes)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
        # Decoder: Takes [z, y]
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y):
        # y needs to be one-hot encoded 
        xy = torch.cat([x, y], dim=1)
        h = self.encoder(xy)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        
        zy = torch.cat([z, y], dim=1)
        recon_x = self.decoder(zy)
        return recon_x, mu, logvar

def train_cvae(features, labels, num_classes, input_dim=90, latent_dim=10, epochs=50):
    # Converting to tensors
    x_tensor = torch.FloatTensor(features)
    # One-hot encode labels
    y_tensor = torch.nn.functional.one_hot(torch.LongTensor(labels), num_classes=num_classes).float()
    
    dataset = TensorDataset(x_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = CVAE(input_dim=input_dim, num_classes=num_classes, latent_dim=latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"Training CVAE with {num_classes} classes...")
    for epoch in range(epochs):
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x_batch, y_batch)
            
            MSE = nn.functional.mse_loss(recon_x, x_batch, reduction='sum')
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = MSE + KLD
            
            loss.backward()
            optimizer.step()
            
    return model

def get_cvae_latent(model, features, labels, num_classes):
    model.eval()
    with torch.no_grad():
        x = torch.FloatTensor(features)
        y = torch.nn.functional.one_hot(torch.LongTensor(labels), num_classes=num_classes).float()
        
        # Pass through encoder manually
        xy = torch.cat([x, y], dim=1)
        h = model.encoder(xy)
        mu = model.fc_mu(h)
        return mu.numpy()
