import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class ThermalCalculator:
    """PCB热分析计算类，用于计算和可视化PCB布局的温度分布"""
    
    def __init__(self, board_width, board_height, resolution=10, ambient_temp=25.0):
        """
        初始化热计算器
        
        参数:
            board_width (float): PCB板宽度(mm)
            board_height (float): PCB板高度(mm)
            resolution (float): 网格分辨率(mm)
            ambient_temp (float): 环境温度(°C)
        """
        self.board_width = board_width
        self.board_height = board_height
        self.resolution = resolution
        self.ambient_temp = ambient_temp
        
        # 计算网格数量
        self.nx = int(board_width // resolution + 1)
        self.ny = int(board_height // resolution + 1)
        
        # 组件功率数据库 (名称: 功率W)
        self.component_power_db = {
            "C1": 0.1,  # 电容功耗很小
            "C2": 0.1,
            "Y1": 0.05,  # 晶振功耗很低
            "R1": (12**2)/100e3,  # 电阻功耗根据阻值计算
            "R2": (12**2)/100e3,
            "R3": (12**2)/1e3,
            "R4": (12**2)/1e3,
            "Q1": 0.5,  # 晶体管功耗
            "Q2": 0.5
        }
    
    def calculate_thermal_distribution(self, solution):
        """
        Calculate PCB thermal distribution with improved component heat distribution
        """
        print("***********************Starting thermal calculation for solution...\n")
        # Initialize temperature to ambient
        T = np.ones((self.ny, self.nx)) * self.ambient_temp
        
        # Initialize power density map
        power_density = np.zeros((self.ny, self.nx))
        
        # Map components to the grid - distribute power across component area
        # print(solution.modules)
        for module in solution.modules:
            # Get component dimensions and convert to grid cells
            width, height = module.get_dimensions()
            
            # Calculate grid indices that cover the component
            x_start = max(0, int(module.x // self.resolution))
            y_start = max(0, int(module.y // self.resolution))
            x_end = min(self.nx-1, int((module.x + width) // self.resolution))
            y_end = min(self.ny-1, int((module.y + height) // self.resolution))
            
            # Get component power from database
            power = self.component_power_db.get(module.name, 0.1)  # Default 0.1W
            
            # Calculate number of cells this component covers
            cell_count = max(1, (x_end - x_start + 1) * (y_end - y_start + 1))
            
            # Distribute power across all cells covered by the component
            for i in range(x_start, x_end + 1):
                for j in range(y_start, y_end + 1):
                    power_density[j, i] = power / cell_count
        
        # Solve thermal distribution using finite difference
        tolerance = 1e-3
        max_iter = 5000
        alpha = 0.8  # Relaxation factor
        k = 0.1  # Thermal conductivity coefficient
        
        for it in range(max_iter):
            T_old = T.copy()
            
            # Update interior points
            for i in range(1, self.nx-1):
                for j in range(1, self.ny-1):
                    # Heat equation with source term
                    T[j, i] = (1-alpha) * T_old[j, i] + alpha * (
                        0.25 * (T_old[j, i+1] + T_old[j, i-1] + T_old[j+1, i] + T_old[j-1, i]) + 
                        k * power_density[j, i]
                    )
            
            # Apply boundary conditions (could use more realistic ones)
            # Consider using a Robin boundary condition instead of fixed temperature
            T[0, :] = 0.9 * T[1, :] + 0.1 * self.ambient_temp  # Bottom edge
            T[-1, :] = 0.9 * T[-2, :] + 0.1 * self.ambient_temp  # Top edge
            T[:, 0] = 0.9 * T[:, 1] + 0.1 * self.ambient_temp  # Left edge
            T[:, -1] = 0.9 * T[:, -2] + 0.1 * self.ambient_temp  # Right edge
            
            # Check convergence
            diff = np.max(np.abs(T - T_old))
            if diff < tolerance:
                print(f"Thermal analysis: Steady state reached in {it+1} iterations")
                break
        else:
            print(f"Thermal analysis: Maximum iterations reached, final diff: {diff:.6f}")
        
        # Save data to solution
        # solution.temperature = T
        # solution.max_temp = float(T.max())
        # solution.min_temp = float(T.min())
        # solution.resolution = self.resolution
        # solution.nx = self.nx
        # solution.ny = self.ny

        solution.max_temp = float(T.max())
        solution.min_temp = float(T.min())
        solution.temperature = T.copy()  
        solution.resolution = self.resolution
        solution.nx = self.nx
        solution.ny = self.ny
        
        print(f"Maximum temperature: {solution.max_temp:.2f}°C")
        print(f"Minimum temperature: {solution.min_temp:.2f}°C")
        print(f"Temperature difference: {(solution.max_temp - solution.min_temp):.2f}°C")
        
        return solution
    
    def visualize_thermal_distribution(self, solution):
        """
        可视化PCB温度分布
        
        参数:
            solution: 包含温度分布数据的解决方案对象
        """
        if not hasattr(solution, 'temperature'):
            print("错误：解决方案没有温度分布数据。请先运行calculate_thermal_distribution方法。")
            return
        
        plt.figure(figsize=(10, 6))
        
        # 热分布图 - 使用更好的对比度范围
        T = solution.temperature
        np.savetxt('temperature_data_project.csv', T, delimiter=',')
        im = plt.imshow(T, cmap='hot', 
                       extent=[0, self.board_width, 0, self.board_height], 
                       origin='lower')
        plt.colorbar(im, label="Temperature (°C)")
        plt.title("PCB Thermal Distribution")
        plt.xlabel("X (mm)")
        plt.ylabel("Y (mm)")
        
        # # 标记最高温度位置
        # max_pos = np.unravel_index(solution.temperature.argmax(), solution.temperature.shape)
        # max_y, max_x = max_pos
        # max_real_x = max_x * solution.resolution
        # max_real_y = max_y * solution.resolution
        # max_temp = solution.temperature[max_y, max_x]
        
        # # 在最高温度点绘制标记
        # plt.plot(max_real_x, max_real_y, 'o', color='cyan', markersize=8)
        # plt.text(max_real_x, max_real_y + 20, f"Max: {max_temp:.1f}°C", 
        #         color='white', fontsize=10, ha='center', 
        #         bbox=dict(facecolor='red', alpha=0.5))
        
        # # 绘制模块轮廓和标注温度
        # for module in solution.modules:
        #     width, height = module.get_dimensions()
        #     rect = patches.Rectangle((module.x, module.y), width, height, 
        #                             linewidth=1, edgecolor='cyan', facecolor='none')
        #     plt.gca().add_patch(rect)
            
        #     # 获取元件在温度场中的区域
        #     x_start = max(0, int(module.x // solution.resolution))
        #     y_start = max(0, int(module.y // solution.resolution))
        #     x_end = min(solution.nx-1, int((module.x + width) // solution.resolution))
        #     y_end = min(solution.ny-1, int((module.y + height) // solution.resolution))
            
        #     # 计算元件区域内的温度
        #     temps = []
        #     for x in range(x_start, x_end+1):
        #         for y in range(y_start, y_end+1):
        #             if 0 <= y < solution.temperature.shape[0] and 0 <= x < solution.temperature.shape[1]:
        #                 temps.append(solution.temperature[y, x])
            
        #     # 如果没有有效温度，使用中心点温度作为备选
        #     center_x = module.x + width/2
        #     center_y = module.y + height/2
        #     x_idx = min(int(center_x // solution.resolution), solution.nx-1)
        #     y_idx = min(int(center_y // solution.resolution), solution.ny-1)
        #     center_temp = solution.temperature[y_idx, x_idx]
            
        #     # 计算区域最高温度
        #     component_max_temp = max(temps) if temps else center_temp
            
        #     # 标记元件名称和温度
        #     plt.text(center_x, center_y-15, module.name, fontsize=10, color='white', ha='center')
        #     plt.text(center_x, center_y+15, f"{component_max_temp:.1f}°C", fontsize=8, color='cyan', ha='center')
        
        # 添加说明注释
        plt.figtext(0.5, 0.01, 
                   "Note: Temperature values are based on finite difference solution of 2D heat equation.\n"
                   "The simulation considers component power and PCB thermal conductivity.", 
                   ha="center", fontsize=9, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
        
        plt.tight_layout()
        plt.show()
    
    def get_component_temperatures(self, solution):
        """
        获取所有组件的温度
        
        参数:
            solution: 包含温度分布数据的解决方案对象
            
        返回:
            dict: 包含组件ID和对应温度的字典
        """
        if not hasattr(solution, 'temperature'):
            print("错误：解决方案没有温度分布数据")
            return {}
        
        temperatures = {}
        
        for module in solution.modules:
            # 获取模块中心点对应的网格索引
            center_x = module.x + module.get_dimensions()[0]/2
            center_y = module.y + module.get_dimensions()[1]/2
            x_idx = min(int(center_x // solution.resolution), solution.nx-1)
            y_idx = min(int(center_y // solution.resolution), solution.ny-1)
            
            # 记录温度
            temperatures[module.id] = float(solution.temperature[y_idx, x_idx])
            
        return temperatures
    
    def get_thermal_metrics(self, solution):
        """
        获取热分析指标
        
        参数:
            solution: 包含温度分布数据的解决方案对象
            
        返回:
            dict: 热分析指标字典
        """
        if not hasattr(solution, 'temperature'):
            print("错误：解决方案没有温度分布数据")
            return {}
        
        metrics = {
            "max_temp": float(solution.max_temp),
            "min_temp": float(solution.min_temp),
            "avg_temp": float(np.mean(solution.temperature)),
            "temp_variance": float(np.var(solution.temperature)),
            "temp_range": float(solution.max_temp - solution.min_temp),
            "component_temps": self.get_component_temperatures(solution)
        }
        
        return metrics
    
    def export_temperature_data(self, solution, filename="temperature_data.csv"):
        """
        导出温度场数据到CSV文件
        
        参数:
            solution: 包含温度分布数据的解决方案对象
            filename: 输出CSV文件名
        """
        if not hasattr(solution, 'temperature'):
            print("错误：解决方案没有温度分布数据")
            return
        
        np.savetxt(filename, solution.temperature, delimiter=',')
        print(f"温度数据已导出到 {filename}")

# def convert_solution_to_component_data(solution):
#     """
#     将Solution对象转换为标准组件数据格式
    
#     Parameters:
#     solution (Solution): 包含PCB布局信息的Solution对象
    
#     Returns:
#     list: 包含组件数据的列表，每个组件是一个带有name、x、y、width和height的字典
#     """
#     component_data = []
    
#     for module in solution.modules:
#         # 获取模块的名称和位置
#         name = module.name
#         x = int(module.x)  # 转为整数，确保与期望输出格式一致
#         y = int(module.y)

#         width, height = module.get_dimensions()
#         width = int(width)
#         height = int(height)
        
#         # 创建组件数据字典
#         component = {
#             "name": name,
#             "x": x,
#             "y": y,
#             "width": width,
#             "height": height
#         }
        
#         component_data.append(component)
    
#     return component_data


def convert_solution_to_component_data(solution):

    component_data = []
    
    for module in solution.modules:
        # 获取模块的名称和位置
        name = module.name
        x = int(module.x)  # 转为整数，确保与期望输出格式一致
        y = int(module.y)
        
        # 创建基本组件数据字典
        component = {
            "name": name,
            "x": x,
            "y": y
        }
        
        # 根据组件类型添加特定属性
        if name.startswith('C'):  # 电容
            component["power"] = 0.1
        elif name.startswith('Y'):  # 晶振
            component["power"] = 0.05
        elif name.startswith('R'):  # 电阻
            # 判断是100KΩ还是1KΩ (基于示例数据的模式)
            if name in ['R1', 'R2']:
                component["resistance"] = 100e3
            else:
                component["resistance"] = 1e3
        elif name.startswith('Q'):  # 晶体管
            component["power"] = 0.5
        
        component_data.append(component)
    
    return component_data