import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint, odeint
from TorchDiffEqPack.odesolver import odesolve

######### Neural ODE    
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        
        self._initialize_weights()  

    def _initialize_weights(self):
        
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.xavier_uniform_(self.layer3.weight) 
       
        nn.init.zeros_(self.layer1.bias)
        nn.init.zeros_(self.layer2.bias)
        nn.init.zeros_(self.layer3.bias)
        

    def forward(self, x):
        x = self.layer1(x)
        x = nn.Tanh()(x)
        
        x = self.layer2(x)
        x = nn.Tanh()(x)
        
        x = self.layer3(x)
        return x
    
class HybridModel(nn.Module):
    def __init__(self, model_phy, model_nn, input_meas, state_meas, device, N, bacth_number, a, centers):
        super().__init__()
        self.model_phy = model_phy
        self.model_nn = model_nn
        self.input_meas = input_meas
        self.state_meas = state_meas
        self.device = device
        self.N = N 
        self.bacth_number = bacth_number
        self.a = a
        self.centers = centers 
    def forward(self, t, state):
        state = state.to(self.device)
        # t: shape (batch_size,)
        # state: shape (batch_size, state_dim)
        t_batch = create_t_batch(t, self.input_meas, self.N, self.bacth_number)
        u, NEh_meas = self.get_input(t_batch)  # u: shape (batch_size, input_dim)
        u=u.unsqueeze(1)
        ail_sp = u[:,0, 0]  # Shape: (batch_size,)
        elev_sp = -u[:,0, 1]
        flaps_sp = u[:, 0, 3]
        load_meas = u[:, 0, 4]
        length_meas = u[:, 0, 5]
        N_meas = NEh_meas[:, 0]
        E_meas = NEh_meas[:, 1]
        h_meas = NEh_meas[:, 2]

        state_angles_deg = state[:, 3:7] * 180 / torch.pi #quaternions
        state_rates_deg = state[:, 7:10] * 180 / torch.pi

        # Gaussian basis functions controller
        states_controller = torch.cat([
            state[:, 0:3],                     
            q2e(state[:, 3:7])* 180 / torch.pi,               
            state_rates_deg,                                    
            state[:, 10:13]                    
        ], dim=1)  
        states_expanded = states_controller.unsqueeze(1)         
        diff = states_expanded - self.centers.unsqueeze(0)              
        dist_sq = (diff ** 2).sum(dim=-1)                       
        dist_sq_normalized = dist_sq / (100*100)             
        phi_matrix_rudder = torch.exp(-0.5 * dist_sq_normalized) 
        rud_sp = -torch.clamp((phi_matrix_rudder @ self.a).squeeze(1), -30, 30) 
        
        # Compute va, alfa, beta, and v_kite_relative_b in batched fashion
        va, alfa, beta, v_kite_relative_b = self.va_computation(state)
    
        # Physical model prediction
        model_pred = self.model_phy(
            state, ail_sp, elev_sp, rud_sp, flaps_sp,
            load_meas, length_meas, va, alfa, beta, v_kite_relative_b,
            N_meas, E_meas, h_meas
        )

        #min-max scaling neural network inputs
        min = torch.tensor([  24.6239,  -13.5442,  -13.3981,  -0.2774*180/torch.pi,  -0.6295*180/torch.pi, -0.5937*180/torch.pi, -0.7701*180/torch.pi, -26.1794,   -2.9779,  -68.9290, -364.7302,  332.5255,  189.1163,
                                  3.5000,  -20,   10.0000, 2901.0911,  481.8000,
                              34, -14, -5]) 
        max = torch.tensor([  53.4398,   14.7434,   -6.6381,   0.9414*180/torch.pi,   0.4554*180/torch.pi,  0.5605*180/torch.pi, 0.7149*180/torch.pi, 12.4082,   26.4609,   -9.8182, -230.7785,  467.1510,  303.6810,
                             14, 20, 2.0000e+01, 5.1583e+03, 5.5592e+02,
                            55, 20, 12])
    
        nn_input = 2*(torch.cat([
            state[:, 0:3],              # u, v, w
            state_angles_deg,           # quaternions in degrees
            state_rates_deg,            # P, Q, R in degrees
            state[:, 10:13],            # N, E, h
            elev_sp.unsqueeze(1),
            rud_sp.unsqueeze(1),
            flaps_sp.unsqueeze(1),
            load_meas.unsqueeze(1),
            length_meas.unsqueeze(1),
            va.unsqueeze(1),
            (alfa * 180 / torch.pi).unsqueeze(1),
            (beta * 180 / torch.pi).unsqueeze(1)
        ], dim=1)-min)/(max-min)-1

        # Neural network prediction
        nn_pred = self.model_nn(nn_input)  # Shape: (batch_size, nn_output_dim)
        pred = model_pred.clone()  # Create a copy of model_pred that requires gradients
        pred[:, 0:3] = pred[:, 0:3] + nn_pred[:, 0:3]  # Correct  linear velocity derivatives
        pred[:, 7:10] = pred[:, 7:10] + nn_pred[:, 3:6] # Correct  angular velocity derivatives

        return pred # Shape: (batch_size, state_dim)
    
    def get_input(self, t_batch):
        # t_batch: shape (batch_size,)
        # Perform batched searchsorted
        indices = torch.searchsorted(self.input_meas[:, 6], t_batch, right=True) - 1
        indices = indices.clamp(min=0, max=self.input_meas.shape[0] - 1)  # Ensure indices are valid
        return self.input_meas[indices, :], self.state_meas[indices,9:12]  
    
    def va_computation(self, states):
        # states: shape (batch_size, state_dim)
        batch_size = states.shape[0]
        device = states.device

        # Extract state variables
        q0 = states[:, 3]    # Shape: (batch_size,)
        q1 = states[:, 4]
        q2 = states[:, 5]
        q3 = states[:, 6]
        v_kite_b = states[:, 0:3]  # Shape: (batch_size, 3)

        # WIND profile (interpolation based on the height (state 12))
        wind_profile_h = torch.tensor([0, 100, 200, 300, 400, 600], dtype=states.dtype, device=device)  # Shape: (6,)
        wind_profile_v = torch.tensor([13, 14.32, 13.82, 10.63, 9.18, 10], dtype=states.dtype, device=device)  # Shape: (6,)

        h = states[:, 12]  # Shape: (batch_size,)

        # Clamp h to the range of wind_profile_h
        h_clamped = h.clamp(min=wind_profile_h[0], max=wind_profile_h[-1])

        # Perform linear interpolation to compute vw for each sample in the batch
        indices = torch.searchsorted(wind_profile_h, h_clamped, right=True)
        indices = indices.clamp(max=wind_profile_h.size(0) - 1)
        i_left = (indices - 1).clamp(min=0)
        i_right = indices

        x0 = wind_profile_h[i_left]  # Shape: (batch_size,)
        x1 = wind_profile_h[i_right]
        y0 = wind_profile_v[i_left]
        y1 = wind_profile_v[i_right]

        dx = x1 - x0
        dx[dx == 0] = 1e-6  # Prevent division by zero

        frac = (h_clamped - x0) / dx
        vw = y0 + frac * (y1 - y0)  # Shape: (batch_size,)

        # Wind direction
        wind_dir = torch.tensor(-54.5939, dtype=states.dtype, device=device)

        # Compute wind vector in NED frame
        wind_dir_rad = torch.deg2rad(wind_dir)
        cos_wind_dir = torch.cos(wind_dir_rad)
        sin_wind_dir = torch.sin(wind_dir_rad)

        v_w_n = torch.stack([
            -vw * cos_wind_dir,
            -vw * sin_wind_dir,
            torch.zeros_like(vw)
        ], dim=1)  # Shape: (batch_size, 3)

        # Compute Direction Cosine Matrix (DCM) for each sample
        DCM = torch.zeros((batch_size, 3, 3), dtype=states.dtype, device=device)

        DCM[:, 0, 0] = q0**2 + q1**2 - q2**2 - q3**2
        DCM[:, 0, 1] = 2 * (q1 * q2 + q0 * q3)
        DCM[:, 0, 2] = 2 * (q1 * q3 - q0 * q2)
        
        DCM[:, 1, 0] = 2 * (q1 * q2 - q0 * q3)
        DCM[:, 1, 1] = q0**2 - q1**2 + q2**2 - q3**2
        DCM[:, 1, 2] = 2 * (q2 * q3 + q0 * q1)
        
        DCM[:, 2, 0] = 2 * (q1 * q3 + q0 * q2)
        DCM[:, 2, 1] = 2 * (q2 * q3 - q0 * q1)
        DCM[:, 2, 2] = q0**2 - q1**2 - q2**2 + q3**2

        # Compute v_kite_n
        v_kite_b_expanded = v_kite_b.unsqueeze(2)  # Shape: (batch_size, 3, 1)
        DCM_T = DCM.transpose(1, 2)
        v_kite_n = torch.bmm(DCM_T, v_kite_b_expanded).squeeze(2)  # Shape: (batch_size, 3)

        # Compute v_kite_relative_n
        v_kite_relative_n = v_kite_n - v_w_n  # Shape: (batch_size, 3)

        # Compute v_kite_relative_b
        v_kite_relative_n_expanded = v_kite_relative_n.unsqueeze(2)
        v_kite_relative_b = torch.bmm(DCM, v_kite_relative_n_expanded).squeeze(2).to(self.device)  # Shape: (batch_size, 3)

        # Compute va
        va = torch.norm(v_kite_relative_b, dim=1).to(self.device)  # Shape: (batch_size,)
        va = va + (va == 0).float() * torch.finfo(states.dtype).eps  # Prevent division by zero

        # Compute angle of attack (alfa) and sideslip angle (beta)
        alfa = torch.atan2(v_kite_relative_b[:, 2], v_kite_relative_b[:, 0]).to(self.device)  # Shape: (batch_size,)
        beta = torch.asin(v_kite_relative_b[:, 1] / va).to(self.device)

        return va, alfa, beta, v_kite_relative_b

# manual interpolation for the wind profile    
def h_poly(t):
    # t: shape (batch_size,)
    batch_size = t.shape[0]
    device = t.device
    dtype = t.dtype

    # Compute tt: shape (batch_size, 4)
    exponents = torch.arange(4, device=device, dtype=dtype).unsqueeze(0)  # Shape: (1, 4)
    t_expanded = t.unsqueeze(1)  # Shape: (batch_size, 1)
    tt = t_expanded ** exponents  # Shape: (batch_size, 4)
    tt = tt.transpose(0, 1)  # Shape: (4, batch_size)
    A = torch.tensor([
        [1, 0, -3, 2],
        [0, 1, -2, 1],
        [0, 0, 3, -2],
        [0, 0, -1, 1]
    ], dtype=dtype, device=device)  # Shape: (4, 4)

    # Compute hh: Shape: (4, batch_size)
    hh = torch.matmul(A, tt)  # Matrix multiplication

    # Transpose hh to get shape (batch_size, 4)
    hh = hh.transpose(0, 1)  # Shape: (batch_size, 4)

    return hh

# manual interpolation for the wind profile    
def interp(x, y, xs):
    # x: shape (n_points,)
    # y: shape (n_points,)
    # xs: shape (batch_size,)

    device = xs.device
    dtype = xs.dtype
    n_points = x.size(0)
    batch_size = xs.size(0)

    # Compute slopes m between points
    m = (y[1:] - y[:-1]) / (x[1:] - x[:-1])  # Shape: (n_points - 1,)
    m = torch.cat([
        m[[0]],                        # First slope
        (m[1:] + m[:-1]) / 2,          # Average of adjacent slopes
        m[[-1]]                        # Last slope
    ])  # Shape: (n_points,)

    # Search for indices
    idxs = torch.searchsorted(x[1:], xs)
    idxs = torch.clamp(idxs, 0, n_points - 2)  # Ensure indices are valid

    # Compute dx
    x_idx = x[idxs]          # Shape: (batch_size,)
    x_idx1 = x[idxs + 1]     # Shape: (batch_size,)
    dx = x_idx1 - x_idx      # Shape: (batch_size,)

    # Avoid division by zero
    dx_nonzero = dx.clone()
    dx_nonzero[dx_nonzero == 0] = 1e-6

    # Compute normalized t for h_poly
    t = (xs - x_idx) / dx_nonzero  # Shape: (batch_size,)

    # Compute hh using h_poly
    hh = h_poly(t)  # Shape: (batch_size, 4)

    # Get y and m at idxs and idxs + 1
    y_idx = y[idxs]          # Shape: (batch_size,)
    y_idx1 = y[idxs + 1]     # Shape: (batch_size,)
    m_idx = m[idxs]
    m_idx1 = m[idxs + 1]

    # Reshape for broadcasting
    y_idx = y_idx.unsqueeze(1)        # Shape: (batch_size, 1)
    y_idx1 = y_idx1.unsqueeze(1)
    m_idx = m_idx.unsqueeze(1)
    m_idx1 = m_idx1.unsqueeze(1)
    dx = dx.unsqueeze(1)              # Shape: (batch_size, 1)

    # Compute interpolated values
    interpolated = (
        hh[:, 0].unsqueeze(1) * y_idx +
        hh[:, 1].unsqueeze(1) * m_idx * dx +
        hh[:, 2].unsqueeze(1) * y_idx1 +
        hh[:, 3].unsqueeze(1) * m_idx1 * dx
    )  # Shape: (batch_size, 1)

    return interpolated.squeeze(1)  # Shape: (batch_size,)

def create_t_batch(t, input_meas, N, num_batches):
    t_list = []
    for i in range(num_batches):
        index = i * N
        offset = input_meas[index, 6]  
        t_shifted = t + offset
        t_list.append(t_shifted)
    
    # Stack the list of time vectors along a new dimension
    t_batch = torch.stack(t_list, dim=0)
    return t_batch

def quaterInit(phi, theta, psi):
    # quaternion initialization
    phi_half = phi / 2
    theta_half = theta / 2
    psi_half = psi / 2
    
    cos_phi_half = torch.cos(phi_half)
    sin_phi_half = torch.sin(phi_half)
    cos_theta_half = torch.cos(theta_half)
    sin_theta_half = torch.sin(theta_half)
    cos_psi_half = torch.cos(psi_half)
    sin_psi_half = torch.sin(psi_half)
    
    q0 = (cos_phi_half * cos_theta_half * cos_psi_half +
          sin_phi_half * sin_theta_half * sin_psi_half)
    q1 = (sin_phi_half * cos_theta_half * cos_psi_half -
          cos_phi_half * sin_theta_half * sin_psi_half)
    q2 = (cos_phi_half * sin_theta_half * cos_psi_half +
          sin_phi_half * cos_theta_half * sin_psi_half)
    q3 = (cos_phi_half * cos_theta_half * sin_psi_half -
          sin_phi_half * sin_theta_half * cos_psi_half)
    
    q = torch.tensor([q0, q1, q2, q3], dtype=torch.float32)
    return q

def mse_loss(y_sim, state_meas, w):
    eul_angles = q2e(y_sim[:,:,3:7]) # from quaternions to euler angles
    y_sim2 = torch.cat([y_sim[:, :, 0:3], eul_angles, y_sim[:, :, 7:13]], dim=-1) 
    loss = (((y_sim2 - state_meas) * w) **2).mean()
    return loss

def q2e(q):
    # Convert quaternions to Euler angles
    q_norm = torch.linalg.norm(q, dim=-1, keepdim=True)
    q = q / q_norm

    # Extract quaternion components
    q0 = q[..., 0]
    q1 = q[..., 1]
    q2 = q[..., 2]
    q3 = q[..., 3]

    DCM_11 = q0**2 + q1**2 - q2**2 - q3**2
    DCM_12 = 2 * (q1 * q2 + q0 * q3)
    DCM_13 = 2 * (q1 * q3 - q0 * q2)
    DCM_23 = 2 * (q2 * q3 + q0 * q1)
    DCM_33 = q0**2 - q1**2 - q2**2 + q3**2

    # Compute Euler angles
    phi = torch.atan2(DCM_23, DCM_33)  # Roll (phi)
    theta = -torch.asin(DCM_13)        # Pitch (theta)
    psi = torch.atan2(DCM_12, DCM_11)  # Yaw (psi)
    
    eul = torch.stack([phi, theta, psi], dim=-1)  # Shape: (number of samples, number of batches, 3)
    return eul
