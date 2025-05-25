import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

a = 0.05 # Yield stiffness ratio
Vjy = 125 # Yield shear force (kN)
lam = [0, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6] # Load factor (multiple force steps)
pref = [125/15, 125/15*2, 125/15*3, 125/15*4, 125/15*5] # Load distribution per floor 
heights = [0, 3, 6, 9, 12, 15] # Total structure height (m)
k_tipikal = 100 #  typical column stiffness (kN/cm)
dlimit = Vjy / k_tipikal # Yield limit of typical column

# Convert pref to column array
pref = np.atleast_2d(np.array(pref)).T

# Initialize lists to store results of each iteration
u_fix=[]
u_fix.append(np.zeros((5,1))) # Initial displacement (zero)

fs_in_fix=[]
fs_in_fix.append(np.zeros((5,1))) # Initial internal force (zero)

# Initial stiffness of each floor
k=np.array([k_tipikal, k_tipikal, k_tipikal, k_tipikal, k_tipikal])
k=np.array(k)

# Initialize global stiffness matrix (for 5 floors)
kHat_fix = []
kHat_fix.append(np.array([[k[0]+k[1], -k[1], 0, 0, 0],
       [-k[1], k[1]+k[2], -k[3], 0,0],
       [0, -k[2], k[2]+k[3],-k[3],0],
       [0, 0, -k[3], k[3]+k[4], -k[4]],
        [0,0,0,-k[4], k[4]]]))

# Initialize variables for iteration
u = []
u.append(u_fix[0])
fs_in = []
fs_in.append(fs_in_fix[0])
kHat = []
kHat.append(kHat_fix[0])
p =[np.zeros((5,1))]
R =[np.zeros((5,1))]
v = 100
St =[np.zeros((5,1))]

# Iteration for each lambda (load factor)
for i in range(0,len(lam)-1):
    j = 0  
    
    # Store previous iteration results
    u.append(u_fix[i])
    fs_in.append(fs_in_fix[i])
    kHat.append(kHat_fix[i])
    p.append(pref*lam[i+1]) # Calculate load for next lambda

    # Newton-Raphson iteration for convergence
    while v > 0:
        j=j+1
        R = (p[i+1]-fs_in[1+i]) # Calculate residual
        v = np.linalg.norm(R) # Norm of residual
        v = v.tolist()
        if v < float(10**(-3)):
            break
        
        # Calculate displacement correction
        dU = np.dot(np.linalg.inv(kHat[i+1]), R)
        u[i+1] = u[i+1] + dU
        
        # Calculate story drift (relative displacement between floors)
        St = np.array([u[i+1][0],u[i+1][1]-u[i+1][0], u[i+1][2]-u[i+1][1], u[i+1][3]-u[i+1][2], u[i+1][4]-u[i+1][3]])
        
        temp = []
        # Check if story drift exceeds yield limit
        for count, value in enumerate(St,0):
            if value < dlimit:
                # If not yielded, stiffness remains
                k[count] = k[count]
                kHat[i+1] = [[k[0]+k[1], -k[1], 0, 0, 0],
                [-k[1], k[1]+k[2], -k[2], 0,0],
                [0, -k[2], k[2]+k[3],-k[3],0],
                [0, 0, -k[3], k[3]+k[4], -k[4]],
                [0,0,0,-k[4], k[4]]]
                
                temp.append(k[count]*value)
                
            else:
                # If yielded, reduce stiffness
                k[count] = a * k_tipikal
                kHat[i+1] = [[k[0]+k[1], -k[1], 0, 0, 0],
                [-k[1], k[1]+k[2], -k[2], 0,0],
                [0, -k[2], k[2]+k[3],-k[3],0],
                [0, 0, -k[3], k[3]+k[4], -k[4]],
                [0,0,0,-k[4], k[4]]]
                
                temp.append(Vjy+k[count]*(value-dlimit))

        # Calculate internal force per floor
        temp[0] = temp[0]-temp[1]
        temp[1] = temp[1]-temp[2]
        temp[2] = temp[2]-temp[3]
        temp[3] = temp[3]-temp[4]   
       
        fs_in[i+1] = np.array(temp)
        
    # Store iteration results
    u_fix.append(u[i+1])
    fs_in_fix.append(fs_in[i+1])
    kHat_fix.append(kHat[i+1])

# --- Result plotting section --- #

# Create empty DataFrame for results
df = pd.DataFrame()

# Store floor displacement results for each lambda
for count, i in enumerate(u):     
    _l = [0]
    col = f'λ= {lam[count]}'
    for j in range(0,len(pref)):
        _l.append(i[j][0])      
        
    df[col] = _l

# Add floor names
df['Floor'] = ['Base', 'Floor 1', 'Floor 2', 'Floor 3', 'Floor 4', 'Floor 5']

# Set 'Floor' column as index
df.set_index('Floor', inplace=True)

# Compute disp/height for top floor
floor_5 = df.loc['Floor 5']
floor_5_values = df.loc['Floor 5'].tolist()
disp_per_height_top = [i / heights[-1] for i in floor_5_values]

# Compute base shear
base_shear = []
for i in lam:
    base_shear.append(float(i*sum(pref)[0]))

# Store story shear per floor for each lambda
story_shear_by_lambda = []

for shear_arr in fs_in_fix:
    shear_per_lambda = []
    for i in range(5):  # for V1 to V5
        selected = shear_arr[i:]  # take from i-th floor to base
        total = np.sum(selected, axis=0)[0]
        shear_per_lambda.append(float(total))
    story_shear_by_lambda.append(shear_per_lambda)

story_shear_by_floor = list(map(list, zip(*story_shear_by_lambda)))

# Convert to list of lists (lambda, floor)
disp_per_lambda = [arr.flatten().tolist() for arr in u_fix]

# Transpose to format: [floor][lambda]
disp_per_floor = list(map(list, zip(*disp_per_lambda)))

# Compute story displacement between floors
relative_disp_per_floor = []
for i in range(len(disp_per_floor)):
    if i == 0:
        relative_disp_per_floor.append(disp_per_floor[i])  # Floor 1 stays the same
    else:
        delta = [disp_per_floor[i][j] - disp_per_floor[i-1][j] for j in range(len(disp_per_floor[i]))]
        relative_disp_per_floor.append(delta)

# Create plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

# --- LEFT: Floor vs Displacement ---#

for column in df.columns:
    axes[0].plot(df[column], df.index, marker='o', label=column)
axes[0].set_xlabel('Displacement (cm)')
axes[0].set_ylabel('Floor')
axes[0].set_title('Floor vs Displacement\nfor Different λ Values')
axes[0].legend(title='λ Values')
axes[0].grid(True)
axes[0].set_yticks(df.index)

# --- RIGHT: Base Shear vs Drift Ratio ---#
axes[1].plot(disp_per_height_top, base_shear, marker='o', color='tab:red')
axes[1].set_xlabel('Displacement/Height')
axes[1].set_ylabel('Base Shear (kN)')
axes[1].set_title('Base Shear vs Displacement/Height')
axes[1].grid(True)

plt.tight_layout()

# Create plot
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 10))
axes = axes.flatten()

max_y = 250  # Y-axis
max_x = 20 # optional, for uniform X-axis

for i in range(5):
    axes[i].plot(relative_disp_per_floor[i], story_shear_by_floor[i], marker='o', color='black')
    axes[i].set_title(f'Relative Displacement vs Base Shear - Floor {i+1}')
    axes[i].set_xlabel('Relative Displacement (cm)')
    axes[i].set_ylabel('Base Shear (kN)')
    axes[i].set_ylim(0, max_y)
    axes[i].set_xlim(0, max_x)
    axes[i].grid(True)
axes[5].axis('off') # Leave last subplot empty

plt.tight_layout()
plt.show()