import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from library.thermal_calculation import ThermalCalculator

class MockModule:
    """Mock Module class to mimic the PCB components"""
    def __init__(self, id, x, y, width, height, name=None):
        self.id = id
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.name = name
        self.rotation = 0  # Default rotation
        
    def get_dimensions(self):
        """Return width and height based on rotation"""
        if self.rotation in [0, 180]:
            return self.width, self.height
        else:  # 90 or 270 degrees
            return self.height, self.width

class MockSolution:
    """Mock Solution class to simulate a PCB layout solution"""
    def __init__(self, modules, board_width, board_height):
        self.modules = modules
        self.board_width = board_width
        self.board_height = board_height
        self.objectives = [0, 0, 0, 0]  # Placeholder for objectives

def test_thermal_calculator():
    # PCB board dimensions
    board_width = 800
    board_height = 600
    
    # component_data = [
    #     {"name": "C1", "x": 638, "y": 299, "width": 20, "height": 20},
    #     {"name": "C2", "x": 316, "y": 301, "width": 20, "height": 20},
    #     {"name": "Y1", "x": 95, "y": 263, "width": 20, "height": 40},
    #     {"name": "R1", "x": 197, "y": 108, "width": 10, "height": 30},
    #     {"name": "R2", "x": 700, "y": 106, "width": 10, "height": 30},
    #     {"name": "R3", "x": 524, "y": 104, "width": 10, "height": 30},
    #     {"name": "R4", "x": 346, "y": 104, "width": 10, "height": 30},
    #     {"name": "Q1", "x": 276, "y": 479, "width": 30, "height": 30},
    #     {"name": "Q2", "x": 657, "y": 473, "width": 30, "height": 30}
    # ]

    component_data = [{'name': 'C1', 'x': 40, 'y': 175, 'width': 130, 'height': 230}, 
                      {'name': 'C2', 'x': 422, 'y': 70, 'width': 130, 'height': 230}, 
                      {'name': 'Y1', 'x': 319, 'y': 91, 'width': 154, 'height': 54}, 
                      {'name': 'R1', 'x': 654, 'y': 305, 'width': 60, 'height': 232}, 
                      {'name': 'R2', 'x': 154, 'y': 296, 'width': 60, 'height': 232}, 
                      {'name': 'R3', 'x': 483, 'y': 483, 'width': 232, 'height': 60}, 
                      {'name': 'R4', 'x': 644, 'y': 248, 'width': 60, 'height': 232}, 
                      {'name': 'Q1', 'x': 516, 'y': 494, 'width': 98, 'height': 79}, 
                      {'name': 'Q2', 'x': 501, 'y': 235, 'width': 79, 'height': 98}]
    
    # Create module objects
    modules = []
    for i, comp in enumerate(component_data):
        module = MockModule(
            id=i, 
            x=comp["x"], 
            y=comp["y"], 
            width=comp["width"], 
            height=comp["height"], 
            name=comp["name"]
        )
        modules.append(module)
    
    # Create a mock solution
    solution = MockSolution(modules, board_width, board_height)
    
    # Create thermal calculator
    thermal_calculator = ThermalCalculator(board_width, board_height, resolution=10, ambient_temp=25.0)
    
    # Print component power database for reference
    print("Component Power Database:")
    for comp_name, power in thermal_calculator.component_power_db.items():
        print(f"{comp_name}: {power} W")
    
    # Debug: Print the grid dimensions
    print(f"\nGrid dimensions: {thermal_calculator.nx} x {thermal_calculator.ny}")
    
    # Calculate thermal distribution
    print("\nCalculating thermal distribution...")
    solution = thermal_calculator.calculate_thermal_distribution(solution)
    
    # Print thermal metrics
    print("\nThermal Metrics:")
    metrics = thermal_calculator.get_thermal_metrics(solution)
    for key, value in metrics.items():
        if key != "component_temps":
            print(f"{key}: {value}")
    
    print("\nComponent Temperatures:")
    for module_id, temp in metrics["component_temps"].items():
        module = solution.modules[module_id]
        print(f"{module.name} (ID {module_id}): {temp:.2f}°C")
    
    # Visualize the thermal distribution
    print("\nVisualizing thermal distribution...")
    thermal_calculator.visualize_thermal_distribution(solution)
    
    # Add a second visualization with component outlines
    plt.figure(figsize=(10, 6))
    
    # Plot the temperature distribution
    im = plt.imshow(solution.temperature, cmap='hot', 
                   extent=[0, board_width, 0, board_height], 
                   origin='lower')
    plt.colorbar(im, label="Temperature (°C)")
    plt.title("PCB Thermal Distribution with Component Outlines")
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    
    # Draw component outlines
    for module in solution.modules:
        width, height = module.get_dimensions()
        rect = patches.Rectangle((module.x, module.y), width, height, 
                                linewidth=1, edgecolor='cyan', facecolor='none')
        plt.gca().add_patch(rect)
        
        # Add component name label
        plt.text(module.x + width/2, module.y + height/2, 
                module.name, color='white', ha='center', va='center')
    
    # Mark the hottest point
    max_pos = np.unravel_index(solution.temperature.argmax(), solution.temperature.shape)
    max_y, max_x = max_pos
    max_real_x = max_x * solution.resolution
    max_real_y = max_y * solution.resolution
    plt.plot(max_real_x, max_real_y, 'o', color='cyan', markersize=8)
    plt.text(max_real_x, max_real_y + 20, f"Max: {solution.max_temp:.1f}°C", 
             color='white', fontsize=10, ha='center',
             bbox=dict(facecolor='red', alpha=0.5))
    
    plt.tight_layout()
    plt.show()
    
    # Save the temperature data to CSV for inspection
    np.savetxt('debug_temperature_data.csv', solution.temperature, delimiter=',')
    print("Temperature data saved to 'debug_temperature_data.csv'")

if __name__ == "__main__":
    test_thermal_calculator()