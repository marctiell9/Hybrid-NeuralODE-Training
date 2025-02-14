import torch.utils
from HybridModel import* 
from scipy.io import loadmat, savemat
from KM1 import KM1kite
import time
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

device='cpu'

# Load the DataSet
mat_data = loadmat('NODEdataset_traction1600.mat')
controller_CVX = loadmat('rud_controller.mat') # Gaussian controller Identified from data

ail_sp = torch.tensor(mat_data["ail_filtered"],dtype=torch.float32).to(device)
elev_sp = torch.tensor(mat_data["elev_filtered"],dtype=torch.float32).to(device)
rud_sp = torch.tensor(mat_data["rud_filtered"],dtype=torch.float32).to(device)
flaps_sp = torch.tensor(mat_data["flaps_filtered"],dtype=torch.float32).to(device)
load_meas = torch.tensor(mat_data["load_meas"],dtype=torch.float32).to(device)
length_meas = torch.tensor(mat_data["length_meas"],dtype=torch.float32).to(device)
time_vector = torch.tensor(mat_data["time_vector"],dtype=torch.float32).to(device) 

N_meas = torch.tensor(mat_data["N_meas"],dtype=torch.float32).to(device)
E_meas = torch.tensor(mat_data["E_meas"],dtype=torch.float32).to(device)
h_meas = torch.tensor(mat_data["h_meas"],dtype=torch.float32).to(device)
u_meas = torch.tensor(mat_data["u_meas"],dtype=torch.float32).to(device)
v_meas = torch.tensor(mat_data["v_meas"],dtype=torch.float32).to(device)
w_meas = torch.tensor(mat_data["w_meas"],dtype=torch.float32).to(device)
phi_meas = torch.tensor(mat_data["phi_meas"],dtype=torch.float32).to(device) 
theta_meas = torch.tensor(mat_data["theta_meas"],dtype=torch.float32).to(device)
psi_meas = torch.tensor(mat_data["psi_meas"],dtype=torch.float32).to(device)
p_meas = torch.tensor(mat_data["p_meas"],dtype=torch.float32).to(device)
q_meas = torch.tensor(mat_data["q_meas"],dtype=torch.float32).to(device)
r_meas = torch.tensor(mat_data["r_meas"],dtype=torch.float32).to(device)

starting_index = 0 
batch_number = 120 #number of batches
N = 400 #prediction horizon
identification_index=starting_index + N*batch_number

identification_index = N*batch_number
ail_sp = ail_sp[starting_index:identification_index].to(device)
elev_sp = elev_sp[starting_index:identification_index].to(device)
rud_sp = rud_sp[starting_index:identification_index].to(device)
flaps_sp = flaps_sp[starting_index:identification_index].to(device)
load_meas = load_meas[starting_index:identification_index].to(device)
length_meas = length_meas[starting_index:identification_index].to(device)
time_id = time_vector[starting_index:identification_index].to(device)

N_meas = N_meas[starting_index:identification_index].to(device)
E_meas = E_meas[starting_index:identification_index].to(device)
h_meas = h_meas[starting_index:identification_index].to(device)
u_meas = u_meas[starting_index:identification_index].to(device)
v_meas = v_meas[starting_index:identification_index].to(device)
w_meas = w_meas[starting_index:identification_index].to(device)
phi_meas = phi_meas[starting_index:identification_index].to(device)
theta_meas = theta_meas[starting_index:identification_index].to(device)
psi_meas = psi_meas[starting_index:identification_index].to(device)
p_meas = p_meas[starting_index:identification_index].to(device)
q_meas = q_meas[starting_index:identification_index].to(device)
r_meas = r_meas[starting_index:identification_index].to(device)

input_meas=torch.stack([ail_sp, elev_sp, rud_sp, flaps_sp, load_meas, length_meas, time_id]).squeeze(-1).T.to(device)
state_meas=torch.stack([ u_meas, v_meas, w_meas ,
                         phi_meas* torch.pi / 180, theta_meas* torch.pi / 180, psi_meas* torch.pi / 180,
                           p_meas* torch.pi / 180, q_meas* torch.pi / 180, r_meas* torch.pi / 180,
                             N_meas, E_meas, h_meas,]).squeeze(-1).T.to(device)

matlab_param = loadmat('th_hat_traction.mat')  # Load physical parameters

physical_params = nn.ParameterDict({
    'CYp': nn.Parameter(torch.tensor(matlab_param["th_hat"][0], dtype=torch.float32).squeeze(0)),
    'Clp': nn.Parameter(torch.tensor(matlab_param["th_hat"][1], dtype=torch.float32).squeeze(0)),
    'Cnp': nn.Parameter(torch.tensor(matlab_param["th_hat"][2], dtype=torch.float32).squeeze(0)),
    'CLq': nn.Parameter(torch.tensor(matlab_param["th_hat"][3], dtype=torch.float32).squeeze(0)),
    'Cmq': nn.Parameter(torch.tensor(matlab_param["th_hat"][4], dtype=torch.float32).squeeze(0)),
    'CYr': nn.Parameter(torch.tensor(matlab_param["th_hat"][5], dtype=torch.float32).squeeze(0)),
    'Clr': nn.Parameter(torch.tensor(matlab_param["th_hat"][6], dtype=torch.float32).squeeze(0)),
    'Cnr': nn.Parameter(torch.tensor(matlab_param["th_hat"][7], dtype=torch.float32).squeeze(0)),
    'T_cg_x': nn.Parameter(torch.tensor(matlab_param["th_hat"][8], dtype=torch.float32).squeeze(0)),
    'T_cg_z': nn.Parameter(torch.tensor(matlab_param["th_hat"][9], dtype=torch.float32).squeeze(0)),
    'Ixx': nn.Parameter(torch.tensor(matlab_param["th_hat"][10], dtype=torch.float32).squeeze(0)),
    'Iyy': nn.Parameter(torch.tensor(matlab_param["th_hat"][11], dtype=torch.float32).squeeze(0)),
    'Izz': nn.Parameter(torch.tensor(matlab_param["th_hat"][12], dtype=torch.float32).squeeze(0)),
    'Ixz': nn.Parameter(torch.tensor(matlab_param["th_hat"][13], dtype=torch.float32).squeeze(0)),
    'Cdt': nn.Parameter(torch.tensor(matlab_param["th_hat"][14], dtype=torch.float32).squeeze(0)),
    'CD_0': nn.Parameter(torch.tensor(matlab_param["th_hat"][15], dtype=torch.float32).squeeze(0)),
    'CD_alfa': nn.Parameter(torch.tensor(matlab_param["th_hat"][16], dtype=torch.float32).squeeze(0)),
    'CD_flap': nn.Parameter(torch.tensor(matlab_param["th_hat"][17], dtype=torch.float32).squeeze(0)),
    'CL_0': nn.Parameter(torch.tensor(matlab_param["th_hat"][18], dtype=torch.float32).squeeze(0)),
    'CL_alfa': nn.Parameter(torch.tensor(matlab_param["th_hat"][19], dtype=torch.float32).squeeze(0)),
    'CL_flap': nn.Parameter(torch.tensor(matlab_param["th_hat"][20], dtype=torch.float32).squeeze(0)),
    'Cm_0': nn.Parameter(torch.tensor(matlab_param["th_hat"][21], dtype=torch.float32).squeeze(0)),
    'Cm_alfa': nn.Parameter(torch.tensor(matlab_param["th_hat"][22], dtype=torch.float32).squeeze(0)),
    'Cm_elev': nn.Parameter(torch.tensor(matlab_param["th_hat"][23], dtype=torch.float32).squeeze(0)),
    'Cn_rud': nn.Parameter(torch.tensor(matlab_param["th_hat"][24], dtype=torch.float32).squeeze(0)),
}).to(device)

#hybrid model initialization
input_size=21 #12 states + 4 control inputs + alpha, beta, va, tether load and tether length
hidden_size= 8
output_size=6 # 3 linear velocities, 3 angular velocities
NN=MLP(input_size, hidden_size, output_size).to(device) #Neural ODE initialization
kite=KM1kite(physical_params, device, input_meas).to(device) # physical model 
hybrid_model=HybridModel(kite, NN, input_meas, state_meas, device, N, batch_number, torch.tensor(controller_CVX["a"], dtype=torch.float32).to(device),
                          torch.tensor(controller_CVX["centers"], dtype=torch.float32).squeeze(0).to(device)
                        )
optimizer = torch.optim.Adam(NN.parameters(), lr=0.0005)

#create initial state bacthes for integration function initialization
initial_state_list = []
for i in range(batch_number):
    index = i * N
    initial_state_sample = torch.cat([
        torch.tensor([u_meas[index], v_meas[index], w_meas[index]], device=device, dtype=torch.float32),
        quaterInit(phi_meas[index]*torch.pi/180 ,theta_meas[index]*torch.pi/180 ,psi_meas[index]*torch.pi/180).to(device), 
        torch.tensor([p_meas[index] * torch.pi / 180,
                      q_meas[index] * torch.pi / 180,
                      r_meas[index] * torch.pi / 180,
                      N_meas[index], E_meas[index], h_meas[index]], device=device, dtype=torch.float32)
          ]).to(device)
    initial_state_list.append(initial_state_sample)
    initial_state_batch = torch.stack(initial_state_list, dim=0) #shape (Batch,13)
    initial_state_batch = torch.tensor(initial_state_batch, requires_grad=True)

# weight coefficients for the loss function
w = 1 / torch.abs(state_meas).max(dim=0).values

state_meas = state_meas.view(batch_number, N, 12).permute(1, 0, 2)  
time_id=time_id.view(-1).to(device) - time_id[0] 

epochs=200 
loss_history = []

for epoch in range(epochs):
    start_time = time.time() 
    
    y_sim = odeint(hybrid_model, initial_state_batch, time_id[0:N], method='rk4', options={'step_size': 0.045})   
    
    loss = mse_loss(y_sim, state_meas, w)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds", "LOSS", loss, "EPOCH", epoch)
torch.save(hybrid_model.state_dict(), 'hybrid_model.pth') #save the model

