import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
# ==============================================================================
# 1. PHYSICAL MODEL
# ==============================================================================
BETA = 0.0065
L_GEN = 0.0005
LAMBDA_PRE = 0.1
ALPHA = -0.002
K_HEAT = 0.05
GAMMA_NOMINAL = 0.05
MATH_LIMIT = 1e4 
P_NOMINAL_REF = 50.0
T_EQ = (K_HEAT / GAMMA_NOMINAL) * P_NOMINAL_REF 

def model_derivatives(state, rho_ext, gamma, dt):
    P, C, T = state
    if P > MATH_LIMIT: P = MATH_LIMIT
    if P < 0: P = 0  
    if T > MATH_LIMIT: T = MATH_LIMIT
    if T < 0: T = 0 
    
    rho_doppler = ALPHA * (T - T_EQ)
    rho = rho_ext + rho_doppler
    
    dP = ((rho - BETA) / L_GEN) * P + (LAMBDA_PRE * C)
    dC = (BETA / L_GEN) * P - (LAMBDA_PRE * C)
    dT = (K_HEAT * P) - (gamma * T)
    
    return np.array([dP, dC, dT])

def rk4_step(state, rho_ext, gamma, dt):
    if np.isnan(state).any() or np.isinf(state).any():
        return np.array([0.0, 0.0, 0.0])

    k1 = model_derivatives(state, rho_ext, gamma, dt)
    k2 = model_derivatives(state + 0.5*dt*k1, rho_ext, gamma, dt)
    k3 = model_derivatives(state + 0.5*dt*k2, rho_ext, gamma, dt)
    k4 = model_derivatives(state + dt*k3, rho_ext, gamma, dt)
    
    new_state = state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    new_state[new_state < 0] = 0.0
    new_state[new_state > MATH_LIMIT] = MATH_LIMIT
    return new_state
# ==============================================================================
# 2. The Digital Twin
# ==============================================================================
class DigitalTwinHybrid:
    def __init__(self):
        self.state_estimate = None 
        # nu=0.0005: Tolerates extreme outliers (very robust)
        # gamma=0.01: Creates a smooth boundary, ignoring rapid noise spikes
        self.svm_model = svm.OneClassSVM(nu=0.0005, kernel='rbf', gamma=0.01) 
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def initialize(self, p0):
        t0 = (K_HEAT / GAMMA_NOMINAL) * p0
        c0 = (BETA / (L_GEN * LAMBDA_PRE)) * p0
        self.state_estimate = np.array([p0, c0, t0])

    def predict_physics(self, dt):
        self.state_estimate = rk4_step(self.state_estimate, 0.0, GAMMA_NOMINAL, dt)
        return self.state_estimate

    def check_safety(self, measurement):
        p_meas, _ = measurement
        trip = False
        if p_meas > 65.0: 
            trip = True
        return trip, ""

    def monitor(self, measurement, dt):
        p_meas, t_meas = measurement
        
        state_pred = self.predict_physics(dt)
        p_pred = state_pred[0]
        t_pred = state_pred[2]
        
        # Synchronization Filter
        self.state_estimate[0] = 0.9 * p_pred + 0.1 * p_meas
        self.state_estimate[2] = 0.9 * t_pred + 0.1 * t_meas
        
        res_p = p_meas - p_pred
        res_t = t_meas - t_pred
        
        if np.isnan(res_p): res_p = 0.0
        if np.isnan(res_t): res_t = 0.0

        is_anom = False
        if self.is_trained:
            feats = np.array([[res_p, res_t]])
            try:
                feats_sc = self.scaler.transform(feats)
                pred = self.svm_model.predict(feats_sc)
                if pred[0] == -1: is_anom = True
            except: is_anom = True
                
        return [res_p, res_t], is_anom, 0

# ==============================================================================
# 3. DATA GENERATION
# ==============================================================================
def generate_plant_data_long_run():
    t_sim = np.arange(0, 3600, 0.1)
    data = []
    
    state_real = np.array([50.0, (BETA/(L_GEN*LAMBDA_PRE))*50, (K_HEAT/GAMMA_NOMINAL)*50])
    
    # IMPORTANT: Set the noise level here to maintain consistency.
    NOISE_P_SIGMA = 0.5
    NOISE_T_SIGMA = 0.2

    for t in tqdm(t_sim, desc="Simulating Plant  ", unit="steps", ncols=100, colour='green'):
        current_gamma = GAMMA_NOMINAL
        rho_in = 0.0
        
        noise_p = np.random.normal(0, NOISE_P_SIGMA) 
        noise_t = np.random.normal(0, NOISE_T_SIGMA)
        
        # Event Schedule
        if 2000 <= t < 2800:
            current_gamma = 0.040 # Thermal Drift
        elif t >= 2800:
            rho_in = 0.0035 # RIA Accident
            
        state_real = rk4_step(state_real, rho_in, current_gamma, 0.1)
        meas = [state_real[0] + noise_p, state_real[2] + noise_t]
        data.append(meas)
        
    return np.array(data), t_sim

# ==============================================================================
# 4. EXECUTION
# ==============================================================================
print("\n=== NUCLEAR MONITORING SYSTEM ===\n")

twin_trainer = DigitalTwinHybrid()
twin_trainer.initialize(50.0)

X_train_raw = []
st_tr = np.array([50.0, (BETA/(L_GEN*LAMBDA_PRE))*50, (K_HEAT/GAMMA_NOMINAL)*50])

print("1. Training (Same noise as the plant)...")
NOISE_TRAIN_P = 0.55
NOISE_TRAIN_T = 0.22

for i in tqdm(range(3000), desc="Calibrating SVM    ", unit="samples", ncols=100, colour='cyan'): 
    noise_p = np.random.normal(0, NOISE_TRAIN_P) 
    noise_t = np.random.normal(0, NOISE_TRAIN_T)
    st_tr = rk4_step(st_tr, 0.0, GAMMA_NOMINAL, 0.1)
    meas = [st_tr[0] + noise_p, st_tr[2] + noise_t]
    
    if i == 0: twin_trainer.initialize(50.0)
    
    res, _, _ = twin_trainer.monitor(meas, 0.1)
    if not np.isnan(res).any():
        X_train_raw.append(res)

X_train_raw = np.array(X_train_raw)
twin_trainer.scaler.fit(X_train_raw)
twin_trainer.svm_model.fit(twin_trainer.scaler.transform(X_train_raw))
twin_trainer.is_trained = True
print("   -> Calibrated AI.")
print("\n2. Running Scenario...")
plant_data, time_axis = generate_plant_data_long_run()

print("\n3. Processing Diagnostics with Persistence Filter...")
monitor = twin_trainer
monitor.initialize(plant_data[0][0])
monitor.state_estimate[2] = plant_data[0][1]

res_p_hist, res_t_hist, warnings, hard_trips = [], [], [], []
reactor_tripped = False

# --- FILTER CONFIGURATION ---
# We need X consecutive error readings to confirm the alarm
# If dt=0.1s and we want to wait 0.5s, we need 5 persistences
PERSISTENCE_LIMIT = 5 
persistence_counter = 0

for meas in tqdm(plant_data, desc="Analyzing Signals ", unit="pts", ncols=100, colour='yellow'):
    #1. Security (Physical)
    trip, _ = monitor.check_safety(meas)
    if trip: reactor_tripped = True
    hard_trips.append(1 if reactor_tripped else 0)
    
    #2. AI Monitoring
    residuals, is_anom_raw, _ = monitor.monitor(meas, 0.1)
    
    # --- FILTER LOGIC (NEW) ---
    warning_final = 0
    
    if is_anom_raw:
        persistence_counter += 1
    else:
        persistence_counter = 0 # Reset if the signal returns to normal.
        
    # The warning is only triggered if the error persists for 5 consecutive cycles.
    if persistence_counter >= PERSISTENCE_LIMIT:
        warning_final = 1
    warnings.append(warning_final)
    res_p_hist.append(residuals[0])
    res_t_hist.append(residuals[1])

# ==============================================================================
# 5. PLOT
# ==============================================================================
fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# 1. Reality
ax[0].plot(time_axis, plant_data[:, 0], 'k', lw=0.6, label='Measured Power')
ax[0].axvspan(2000, 2800, color='yellow', alpha=0.3, label='Drift (Loss of Cooling)')
ax[0].axvspan(2800, 3600, color='red', alpha=0.3, label='Accident (RIA)')
ax[0].set_ylabel('MW')
ax[0].legend(loc='upper right')
ax[0].set_title('Real Plant (3600s)')
ax[0].grid(True, alpha=0.5)

# 2. Residuals (Note the scale is close to zero at the beginning)
ax[1].plot(time_axis, res_p_hist, 'b', lw=0.3, alpha=0.9, label='P Waste')
ax[1].plot(time_axis, res_t_hist, 'g', lw=0.3, alpha=0.9, label='T Waste')
ax[1].set_ylabel('Estimation Error')
ax[1].legend()
ax[1].set_title('Digital Twin (Physics-Informed Residuals)')
ax[1].grid(True, alpha=0.5)

# 3. Decisions
ax[2].fill_between(time_axis, 0, warnings, step='mid', color='orange', alpha=1.0, label='AI: Early Detection')
ax[2].plot(time_axis, hard_trips, 'r-', lw=2.5, label='SECURITY SYSTEM: Trip')
ax[2].set_xlabel('Time (seconds)')
ax[2].set_yticks([0, 1])
ax[2].set_yticklabels(['NORMAL', 'ANOMALY/TRIP'])
ax[2].legend(loc='center left')
ax[2].set_title('Hybrid System Operation')

plt.tight_layout()
plt.show()
