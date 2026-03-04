import numpy as np
import matplotlib.pyplot as plt

# Data diamond (Caratage, Price)
data = [
    (0.12, 223), (0.15, 322), (0.15, 323), (0.15, 315), (0.15, 298),
    (0.15, 287), (0.15, 316), (0.16, 328), (0.16, 342), (0.16, 336),
    (0.16, 345), (0.16, 332), (0.16, 339), (0.16, 338), (0.17, 355),
    (0.17, 350), (0.17, 352), (0.17, 353), (0.17, 318), (0.17, 346),
    (0.17, 350), (0.17, 345), (0.18, 325), (0.18, 462), (0.18, 468),
    (0.18, 438), (0.18, 419), (0.18, 443), (0.19, 485), (0.20, 498),
    (0.21, 483), (0.23, 595), (0.23, 595), (0.23, 553), (0.25, 642),
    (0.25, 750), (0.25, 655), (0.25, 678), (0.25, 675), (0.26, 663),
    (0.26, 693), (0.27, 720), (0.28, 823), (0.29, 860), (0.32, 918),
    (0.32, 919), (0.33, 945), (0.35, 1086)
]

data_array = np.array(data)
x = data_array[:, 0]  # Caratage
y = data_array[:, 1]  # Price

theta0 = 0
theta1 = 0
alpha = 0.1
m = len(x) #panjangnya
J_values = []
iterations = 0
convergence_threshold = 0.01

print("="*70)
print("LINEAR REGRESSION - DIAMOND PRICE PREDICTION")
print("="*70)
print(f"Jumlah data (m): {m}")
print(f"Learning rate (α): {alpha}")
print(f"Convergence threshold: {convergence_threshold}")
print("="*70)
print(f"{'Iter':<6} {'θ₀':<12} {'θ₁':<12} {'J(θ₀,θ₁)':<15} {'Δ J':<15}")
print("="*70)

previous_J = float('inf')


while True:
    # Prediksi: h(x) = θ₀ + θ₁ * x
    h_theta = theta0 + theta1 * x
    
    # Error: h(x) - y
    error = h_theta - y
    
    # Cost function: J(θ₀, θ₁) = (1/2m) * Σ(error²)
    J = (1 / (2 * m)) * np.sum(error ** 2)
    J_values.append(J)
    
    # Dapat nilai delta J (selisih dengan iterasi sebelumnya)
    delta_J = abs(previous_J - J)
    
    # Print perkembangan setiap iterasi
    print(f"iterasi: {iterations:<6} | theta0: {theta0:<12.6f} | theta1: {theta1:<12.6f} | Cost J: {J:<15.6f} |  ΔJ: {delta_J:<15.6f}")
    
    # cek convergent pada delta J 
    if delta_J < convergence_threshold:
        print("="*70)
        print(f"KONVERGEN! (Δ J = {delta_J:.6f} < {convergence_threshold})")
        print("="*70)
        break
    
    # Update theta0 dan theta1
    temp_theta0 = theta0 - alpha * (1 / m) * np.sum(error)
    temp_theta1 = theta1 - alpha * (1 / m) * np.sum(error * x)
    
    theta0 = temp_theta0
    theta1 = temp_theta1
    
    previous_J = J
    iterations += 1
    
    # Cek biar gak infinite loop
    if iterations > 100000:
        print("Mencapai batas iterasi maksimal!")
        break

print("\n" + "="*70)
print("HASIL FINAL")
print("="*70)
print(f"Total iterasi: {iterations}")
print(f"Theta 0 (θ₀): {theta0:.6f}")
print(f"Theta 1 (θ₁): {theta1:.6f}")
print(f"Cost Function (J): {J:.6f}")
print(f"Persamaan Linear: Price = {theta0:.2f} + {theta1:.2f} * Caratage (X)")
print("="*70)

# Visualisasi - Grafik 
plt.figure(figsize=(12, 7))

# Plot data asli (cluster)
plt.scatter(x, y, color='blue', s=80, alpha=0.6, label='Data Actual', edgecolors='black', linewidth=0.5)

# Plot garis 
x_line = np.linspace(x.min(), x.max(), 100)
y_line = theta0 + theta1 * x_line
plt.plot(x_line, y_line, color='red', linewidth=2.5, label=f'Linear Regression')

plt.xlabel('Caratage (x)', fontsize=12, fontweight='bold')
plt.ylabel('Price (y)', fontsize=12, fontweight='bold')
plt.title('Linear Regression - Diamond Price Prediction', fontsize=14, fontweight='bold')
plt.legend(fontsize=11, loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()