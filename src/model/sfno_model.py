import torch
import torch.nn as nn
import torch_harmonics as th
import torch.nn.functional as F


class SphericalFourierLayer(nn.Module):
    """
    Spherical Fourier Neural Operator Layer
    
    Applies filtering in the spherical harmonic domain
    """
    def __init__(self, hidden_dim, lmax, modes=None, activation='gelu'):
        super(SphericalFourierLayer, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.lmax = lmax
        
        # Number of modes to keep in the Fourier domain
        if modes is None:
            # By default, keep all modes up to lmax
            self.modes = lmax + 1
        else:
            self.modes = min(modes, lmax + 1)
        
        # Number of spherical harmonic coefficients up to degree lmax
        self.num_coeffs = (self.lmax + 1) ** 2
        
        # Complex weights for spherical harmonic domain mixing
        # We use real-valued parameters to represent complex weights
        # For each input-output channel pair, we have weights for each mode
        self.weights_real = nn.Parameter(torch.FloatTensor(hidden_dim, hidden_dim, self.modes))
        self.weights_imag = nn.Parameter(torch.FloatTensor(hidden_dim, hidden_dim, self.modes))
        
        # Bias in the Fourier domain
        self.bias = nn.Parameter(torch.FloatTensor(hidden_dim, self.num_coeffs))
        
        # Linear transformations
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        
        # Set activation function
        if activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'relu':
            self.activation = F.relu
        else:
            self.activation = lambda x: x
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize weights and biases"""
        nn.init.xavier_normal_(self.weights_real)
        nn.init.xavier_normal_(self.weights_imag)
        nn.init.zeros_(self.bias)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input features in spherical harmonic domain
               [batch_size, channels, num_coeffs]
               
        Returns:
            y: Output features in spherical harmonic domain
               [batch_size, channels, num_coeffs]
        """
        batch_size, channels, num_coeffs = x.shape
        
        # Apply spectral filter
        x_modes = x[:, :, :self.modes]  # [batch_size, channels, modes]
        
        # Reshape for matrix multiplication
        x_modes = x_modes.permute(0, 2, 1)  # [batch_size, modes, channels]
        
        # Complex matrix multiplication in real form
        y_real = torch.zeros(batch_size, self.modes, self.hidden_dim, device=x.device)
        y_imag = torch.zeros(batch_size, self.modes, self.hidden_dim, device=x.device)
        
        for b in range(batch_size):
            for m in range(self.modes):
                y_real[b, m] = torch.matmul(x_modes[b, m], self.weights_real[:, :, m])
                y_imag[b, m] = torch.matmul(x_modes[b, m], self.weights_imag[:, :, m])
        
        # Reshape back
        y_modes = torch.complex(y_real, y_imag)
        y_modes = y_modes.permute(0, 2, 1)  # [batch_size, hidden_dim, modes]
        
        # Construct output by keeping high modes unchanged
        y = torch.zeros(batch_size, self.hidden_dim, num_coeffs, device=x.device, dtype=torch.complex64)
        y[:, :, :self.modes] = y_modes
        
        # Add bias
        y = y + self.bias.unsqueeze(0)
        
        # Apply linear transformation and activation
        y = y.view(batch_size, -1)
        y = self.linear(y)
        y = self.activation(y)
        y = y.view(batch_size, self.hidden_dim, num_coeffs)
        
        return y

class SphericalFNO(nn.Module):
    """
    Spherical Fourier Neural Operator for global weather prediction
    """
    def __init__(self, in_channels, hidden_dim, out_channels, 
                lmax=20, nlat=32, nlon=64, num_layers=4):
        super(SphericalFNO, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.lmax = lmax
        self.nlat = nlat
        self.nlon = nlon
        self.num_layers = num_layers
        
        # Setup spherical harmonic transforms
        self.sht = th.RealSHT(nlat, nlon, lmax)
        self.isht = self.sht.inverse
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        
        # Spherical Fourier layers
        self.sfno_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.sfno_layers.append(SphericalFourierLayer(hidden_dim, lmax))
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, out_channels)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input features [batch_size, in_channels, nlat, nlon]
            
        Returns:
            y: Output features [batch_size, out_channels, nlat, nlon]
        """
        batch_size = x.size(0)
        
        # Project input channels
        x = x.permute(0, 2, 3, 1)  # [batch_size, nlat, nlon, in_channels]
        x = self.input_proj(x)  # [batch_size, nlat, nlon, hidden_dim]
        x = x.permute(0, 3, 1, 2)  # [batch_size, hidden_dim, nlat, nlon]
        
        # Transform to spherical harmonic domain
        x_coeffs = []
        for i in range(self.hidden_dim):
            coeffs = self.sht(x[:, i])  # [batch_size, num_coeffs]
            x_coeffs.append(coeffs)
        
        x_coeffs = torch.stack(x_coeffs, dim=1)  # [batch_size, hidden_dim, num_coeffs]
        
        # Apply SFNO layers
        for layer in self.sfno_layers:
            x_coeffs = layer(x_coeffs)
        
        # Transform back to spatial domain
        y = torch.zeros(batch_size, self.hidden_dim, self.nlat, self.nlon, device=x.device)
        for i in range(self.hidden_dim):
            y[:, i] = self.isht(x_coeffs[:, i])
        
        # Project to output channels
        y = y.permute(0, 2, 3, 1)  # [batch_size, nlat, nlon, hidden_dim]
        y = self.output_proj(y)  # [batch_size, nlat, nlon, out_channels]
        y = y.permute(0, 3, 1, 2)  # [batch_size, out_channels, nlat, nlon]
        
        return y

class TemporalSFNO(nn.Module):
    """
    Temporal Spherical Fourier Neural Operator for weather forecasting
    """
    def __init__(self, in_channels, hidden_dim, out_channels, 
                lmax=20, nlat=32, nlon=64, num_layers=4,
                history_steps=3, forecast_steps=1,
                inference_mode=False):  # <-- NEW ARG
        super(TemporalSFNO, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.lmax = lmax
        self.nlat = nlat
        self.nlon = nlon
        self.history_steps = history_steps
        self.forecast_steps = forecast_steps
        self.inference_mode = inference_mode  # <-- STORE FLAG
        
        # Time embedding
        self.time_embedding = nn.Linear(1, hidden_dim)
        
        # SFNO for spatial processing
        self.sfno = SphericalFNO(in_channels + hidden_dim, hidden_dim, hidden_dim, 
                                 lmax, nlat, nlon, num_layers)
        
        # LSTM for temporal processing
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, out_channels * forecast_steps)
        )
    
    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input features [batch_size, history_steps, in_channels, nlat, nlon]

        Returns:
            y: Output features [batch_size, forecast_steps, out_channels, nlat, nlon]
        """
        batch_size, _, _, nlat, nlon = x.shape

        if self.inference_mode:
            # Use only the last timestep for spatial processing
            t = self.history_steps - 1
            time = torch.ones(batch_size, 1, 1, 1, device=x.device) * (t / self.history_steps)
            time_emb = self.time_embedding(time)
            time_emb = time_emb.expand(batch_size, self.hidden_dim, nlat, nlon)

            x_t = x[:, t]  # Last timestep
            x_augmented = torch.cat([x_t, time_emb], dim=1)
            sfno_out = self.sfno(x_augmented)
            sfno_pool = F.adaptive_avg_pool2d(sfno_out, 1).flatten(1)

            lstm_input = sfno_pool.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        else:
            # Process each time step
            sfno_outputs = []
            for t in range(self.history_steps):
                time = torch.ones(batch_size, 1, 1, 1, device=x.device) * (t / self.history_steps)
                time_emb = self.time_embedding(time)
                time_emb = time_emb.expand(batch_size, self.hidden_dim, nlat, nlon)

                x_t = x[:, t]
                x_augmented = torch.cat([x_t, time_emb], dim=1)
                sfno_out = self.sfno(x_augmented)
                sfno_pool = F.adaptive_avg_pool2d(sfno_out, 1).flatten(1)
                sfno_outputs.append(sfno_pool)

            lstm_input = torch.stack(sfno_outputs, dim=1)  # [batch_size, history_steps, hidden_dim]

        # Process with LSTM
        lstm_out, _ = self.lstm(lstm_input)
        final_state = lstm_out[:, -1]  # [batch_size, hidden_dim]

        # Generate forecasts
        forecast_flat = self.output_proj(final_state)  # [batch_size, out_channels * forecast_steps]
        forecast = forecast_flat.view(batch_size, self.forecast_steps, self.out_channels)

        # Expand to spatial dimensions
        forecast = forecast.unsqueeze(3).unsqueeze(4)
        forecast = forecast.expand(batch_size, self.forecast_steps, self.out_channels, nlat, nlon)

        return forecast

if __name__ == "__main__":
    # Test the model with dummy data
    batch_size = 4
    in_channels = 4  # e.g., temperature, pressure, wind_u, wind_v
    hidden_dim = 32
    out_channels = 4
    history_steps = 3
    forecast_steps = 1
    nlat, nlon = 32, 64
    lmax = 10
    
    # Create dummy input data
    x = torch.randn(batch_size, history_steps, in_channels, nlat, nlon)
    
    # Initialize model
    model = TemporalSFNO(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        out_channels=out_channels,
        lmax=lmax,
        nlat=nlat,
        nlon=nlon,
        history_steps=history_steps,
        forecast_steps=forecast_steps
    )
    
    # Forward pass
    y = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")