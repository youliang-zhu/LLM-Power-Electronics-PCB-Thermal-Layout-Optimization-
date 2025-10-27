import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ========== 1. 定义 PCB 尺寸 ==========
pcb_width = 800  # mm
pcb_height = 600  # mm
resolution = 10  # 网格精度 mm

# 计算网格数量
nx = pcb_width // resolution + 1
ny = pcb_height // resolution + 1

# ========== 2. 定义元件信息 ==========
# components = [
#     {"name": "C1", "x": 638, "y": 299, "power": 0.1},  # 电容功耗很小
#     {"name": "C2", "x": 316, "y": 301, "power": 0.1},
#     {"name": "Y1", "x": 95, "y": 263, "power": 0.05},  # 晶振功耗很低
#     {"name": "R1", "x": 197, "y": 108, "resistance": 100e3},  # 100KΩ
#     {"name": "R2", "x": 700, "y": 106, "resistance": 100e3},  
#     {"name": "R3", "x": 524, "y": 104, "resistance": 1e3},    # 1KΩ
#     {"name": "R4", "x": 346, "y": 104, "resistance": 1e3},
#     {"name": "Q1", "x": 276, "y": 479, "power": 0.5},  # 晶体管功耗
#     {"name": "Q2", "x": 657, "y": 473, "power": 0.5}
# ]

components = [{'name': 'C1', 'x': 90, 'y': 326, 'power': 0.1}, 
              {'name': 'C2', 'x': 556, 'y': 155, 'power': 0.1}, 
              {'name': 'Y1', 'x': 234, 'y': 140, 'power': 0.05},
                {'name': 'R1', 'x': 505, 'y': 153, 'resistance': 100000.0}, 
                {'name': 'R2', 'x': 25, 'y': 81, 'resistance': 100000.0}, 
                {'name': 'R3', 'x': 442, 'y': 122, 'resistance': 1000.0}, 
                {'name': 'R4', 'x': 59, 'y': 5, 'resistance': 1000.0}, 
                {'name': 'Q1', 'x': 263, 'y': 90, 'power': 0.5}, 
                {'name': 'Q2', 'x': 528, 'y': 40, 'power': 0.5}]

Vcc = 12  # 供电电压 12V

# ========== 3. 计算功耗 ==========
for comp in components:
    if "resistance" in comp:
        comp["power"] = (Vcc ** 2) / comp["resistance"]  # 计算电阻功耗
        print(f"{comp['name']} 功耗: {comp['power']:.4f} W")
    else:
        print(f"{comp['name']} 功耗: {comp['power']:.4f} W")

# ========== 4. 定义热导率 ==========
# 假设PCB基板的热导率为FR-4材料
k_pcb = 0.3  # W/(m·K) FR-4材料的热导率
k = np.full((ny, nx), k_pcb)  # 初始化整个PCB的热导率

# ========== 5. 初始化温度场 ==========
T = np.zeros((ny, nx))  # 初始温度设为0度
ambient_temp = 25.0  # 环境温度，单位°C
T.fill(ambient_temp)  # 初始化为环境温度

# 设置元件为热源，并赋予功率密度
power_density = np.zeros((ny, nx))
component_positions = []

for comp in components:
    # 将mm坐标转为网格索引
    x_idx = min(int(comp["x"] // resolution), nx-1)
    y_idx = min(int(comp["y"] // resolution), ny-1)
    
    # 存储元件位置供后面使用
    component_positions.append((x_idx, y_idx))
    
    # 设置功率密度 - 单位面积功率，但保持适当的数值范围
    # 1W/mm² = 1,000,000 W/m²，太大导致溢出，缩小到合理范围
    area_mm2 = resolution * resolution
    power_density[y_idx, x_idx] = comp["power"] / area_mm2 * 100  # 单位修改为W/(100mm²)

# ========== 6. 使用有限差分法迭代求解稳态温度分布 ==========
# 定义求解参数
tolerance = 1e-3  # 收敛容差
max_iter = 5000  # 最大迭代次数
alpha = 0.8  # 松弛因子，减小以提高稳定性

# 热传导系数
heat_transfer_coef = 0.1  # 热传导系数，单位为合适的尺度

# 边界条件: 假设PCB边缘固定为环境温度
# 迭代求解
for it in range(max_iter):
    T_old = T.copy()
    
    # 对非边界点进行迭代计算
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            # 跳过组件位置，保持为热源
            if (i, j) in component_positions:
                continue
            
            # 使用有限差分法计算温度分布（简化版本以避免溢出）
            T[j, i] = (1-alpha) * T_old[j, i] + alpha * (
                0.25 * (T_old[j, i+1] + T_old[j, i-1] + T_old[j+1, i] + T_old[j-1, i]) + 
                heat_transfer_coef * power_density[j, i]
            )
    
    # 对元件位置应用热源方程
    for comp_x, comp_y in component_positions:
        idx = component_positions.index((comp_x, comp_y))
        comp = components[idx]
        
        # 基于周围温度和热源功率计算热点温度
        surrounding_temp = 0.0
        count = 0
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx_pos = comp_x + dx
            ny_pos = comp_y + dy
            if 0 <= nx_pos < nx and 0 <= ny_pos < ny:
                surrounding_temp += T_old[ny_pos, nx_pos]
                count += 1
        
        if count > 0:
            surrounding_temp /= count
            
        # 使用更稳定的热源模型
        thermal_resistance = 5.0  # °C/W，降低热阻值
        T[comp_y, comp_x] = surrounding_temp + comp["power"] * thermal_resistance
    
    # 应用边界条件：边缘保持在环境温度
    T[0, :] = ambient_temp
    T[-1, :] = ambient_temp
    T[:, 0] = ambient_temp
    T[:, -1] = ambient_temp
    
    # 计算温度变化的最大差异
    diff = np.max(np.abs(T - T_old))
    
    # 打印迭代进度
    if it % 500 == 0:
        print(f"迭代 {it}, 温度变化: {diff:.6f}")
    
    # 检查收敛
    if diff < tolerance:
        print(f"稳态已达到，迭代次数: {it+1}")
        break
else:
    print(f"达到最大迭代次数 {max_iter}，尚未完全收敛，最终温度变化: {diff:.6f}")

# 打印最高温度和最低温度值
print(f"最大相对温度值: {T.max():.4f}")
print(f"最小相对温度值: {T.min():.4f}")

# ========== 7. 绘制热分布图 ==========
# np.savetxt('temperature_data.csv', T, delimiter=',')

plt.figure(figsize=(10, 6))
im = plt.imshow(T, cmap='hot', extent=[0, pcb_width, 0, pcb_height], origin='lower')
plt.colorbar(label="Temperature (°C)")
plt.title("PCB Thermal Distribution (Finite Difference Method)")
plt.xlabel("X (mm)")
plt.ylabel("Y (mm)")

# 标记元件位置和温度值
for comp in components:
    x_idx = min(int(comp["x"] // resolution), nx-1)
    y_idx = min(int(comp["y"] // resolution), ny-1)
    temp_value = T[y_idx, x_idx]
    plt.text(comp["x"], comp["y"]-15, comp["name"], fontsize=10, color='white', ha='center')
    plt.text(comp["x"], comp["y"]+15, f"{temp_value:.1f}°C", fontsize=8, color='cyan', ha='center')

# 添加标注 - 使用英文避免中文乱码问题
plt.figtext(0.5, 0.01, 
            "Note: Temperature values are based on finite difference solution of 2D heat equation.\n"
            "The simulation considers component power and PCB thermal conductivity.", 
            ha="center", fontsize=9, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})

plt.show()

# 可选：导出数据
# df = pd.DataFrame([(i*resolution, j*resolution, T[j, i]) 
#                  for j in range(ny) for i in range(nx)], 
#                 columns=["X (mm)", "Y (mm)", "Temperature (°C)"])
# df.to_csv("thermal_distribution_fdm.csv", index=False)
# print("✅ 热分布数据已导出！")