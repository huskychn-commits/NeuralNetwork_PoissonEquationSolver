import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import time

def charge_density(x, y, z):
    """定义电荷密度函数 rho = 100*x*y*z^2"""
    return 100 * x * y * z**2

def random_walk_solver(args):
    """
    蒙特卡洛随机游走求解单个网格点的电势
    """
    i, j, k, px, max_steps, dx, bounds = args
    x = -1.0 + i * dx
    y = -1.0 + j * dx
    z = -1.0 + k * dx
    
    # 如果是边界点，电势为0
    if (i == 0 or i == px-1 or j == 0 or j == px-1 or k == 0 or k == px-1):
        return 0.0
    
    # 蒙特卡洛模拟参数
    n_walks = 1000  # 每个点的随机游走次数
    total_potential = 0.0
    
    for _ in range(n_walks):
        # 初始化随机游走
        x_pos, y_pos, z_pos = x, y, z
        steps = 0
        path_integral = 0.0
        
        while steps < max_steps:
            # 检查是否到达边界
            if (abs(x_pos) >= 1.0 or abs(y_pos) >= 1.0 or abs(z_pos) >= 1.0):
                # 边界条件 phi=0
                break
            
            # 随机选择移动方向 (6个可能方向)
            direction = np.random.randint(0, 6)
            if direction == 0:
                x_pos += dx  # 右
            elif direction == 1:
                x_pos -= dx  # 左
            elif direction == 2:
                y_pos += dx  # 上
            elif direction == 3:
                y_pos -= dx  # 下
            elif direction == 4:
                z_pos += dx  # 前
            else:
                z_pos -= dx  # 后
            
            # 累积路径积分（电荷密度贡献）
            rho = charge_density(x_pos, y_pos, z_pos)
            path_integral += rho * (dx**2) / 6.0  # 三维情况下的格林函数系数
            
            steps += 1
        
        total_potential += path_integral
    
    # 返回平均电势
    return total_potential / n_walks

def solve_poisson_monte_carlo(px=128, max_steps=10000):
    """
    使用蒙特卡洛方法求解泊松方程
    
    参数:
    px: 网格点数 (每个维度)
    max_steps: 最大随机游走步数
    """
    print(f"开始求解泊松方程，网格大小: {px}×{px}×{px}")
    print(f"计算区域: [-1, 1]^3")
    print(f"电荷密度: ρ = 100*x*y*z^2")
    print(f"边界条件: φ = 0")
    
    # 网格参数
    dx = 2.0 / (px - 1)  # 网格间距
    
    # 初始化电势数组
    phi = np.zeros((px, px, px))
    
    # 准备参数列表用于并行计算
    params = []
    for i in range(px):
        for j in range(px):
            for k in range(px):
                params.append((i, j, k, px, max_steps, dx, [-1, 1]))
    
    print(f"总网格点数: {px**3}")
    print(f"使用 {mp.cpu_count()} 个CPU核心进行并行计算")
    
    # 使用多进程并行计算
    start_time = time.time()
    
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = list(tqdm(pool.imap(random_walk_solver, params), 
                          total=len(params), 
                          desc="蒙特卡洛求解进度"))
    
    # 将结果整理回3D数组
    idx = 0
    for i in range(px):
        for j in range(px):
            for k in range(px):
                phi[i, j, k] = results[idx]
                idx += 1
    
    end_time = time.time()
    print(f"计算完成，耗时: {end_time - start_time:.2f} 秒")
    
    return phi, dx

def analyze_solution(phi, dx):
    """分析求解结果"""
    print("\n=== 结果分析 ===")
    print(f"电势范围: [{np.min(phi):.6f}, {np.max(phi):.6f}]")
    print(f"电势平均值: {np.mean(phi):.6f}")
    print(f"电势标准差: {np.std(phi):.6f}")
    
    # 检查边界条件
    boundary_vals = np.concatenate([
        phi[0, :, :].flatten(),   # x=-1 边界
        phi[-1, :, :].flatten(),  # x=1 边界
        phi[:, 0, :].flatten(),   # y=-1 边界
        phi[:, -1, :].flatten(),   # y=1 边界
        phi[:, :, 0].flatten(),    # z=-1 边界
        phi[:, :, -1].flatten()    # z=1 边界
    ])
    print(f"边界条件检查 - 边界点电势均值: {np.mean(boundary_vals):.6f}")
    print(f"边界条件检查 - 边界点电势最大绝对值: {np.max(np.abs(boundary_vals)):.6f}")

def save_results(phi, dx, filename="poisson_solution.npz"):
    """保存结果到.npz文件"""
    # 创建坐标网格
    x = np.linspace(-1, 1, phi.shape[0])
    y = np.linspace(-1, 1, phi.shape[1])
    z = np.linspace(-1, 1, phi.shape[2])
    
    # 保存到.npz文件
    np.savez(filename,
             potential=phi,       # 电势场
             x_coords=x,          # x坐标
             y_coords=y,          # y坐标  
             z_coords=z,          # z坐标
             dx=dx,               # 网格间距
             charge_density="100*x*y*z^2",  # 电荷密度函数
             boundary_condition="phi=0",   # 边界条件
             method="Monte Carlo Random Walk")
    
    print(f"\n结果已保存到: {filename}")
    print("文件包含以下数组:")
    print("  - potential: 电势场 (3D数组)")
    print("  - x_coords, y_coords, z_coords: 坐标数组")
    print("  - dx: 网格间距")
    print("  - 其他元数据")

def verify_solution(phi, dx, test_points=10):
    """验证解的正确性（通过有限差分法检查）"""
    print("\n=== 解的正确性验证 ===")
    
    # 随机选择一些内部点进行验证
    for _ in range(test_points):
        # 选择内部点（避免边界）
        i = np.random.randint(1, phi.shape[0]-1)
        j = np.random.randint(1, phi.shape[1]-1)
        k = np.random.randint(1, phi.shape[2]-1)
        
        # 计算拉普拉斯算子的有限差分近似
        laplacian = (phi[i+1, j, k] - 2*phi[i, j, k] + phi[i-1, j, k] + 
                     phi[i, j+1, k] - 2*phi[i, j, k] + phi[i, j-1, k] + 
                     phi[i, j, k+1] - 2*phi[i, j, k] + phi[i, j, k-1]) / (dx**2)
        
        # 计算该点的电荷密度
        x = -1 + i * dx
        y = -1 + j * dx
        z = -1 + k * dx
        rho = charge_density(x, y, z)
        
        # 检查泊松方程 ∇²φ = -ρ
        residual = laplacian + rho
        if _ < 3:  # 只显示前3个点的详细信息
            print(f"点({x:.2f}, {y:.2f}, {z:.2f}): ∇²φ = {laplacian:.4f}, -ρ = {-rho:.4f}, 残差 = {residual:.4f}")

def main():
    """主函数"""
    # 设置参数
    px = 128  # 网格点数
    max_steps = 5000  # 最大随机游走步数
    
    # 求解泊松方程
    phi, dx = solve_poisson_monte_carlo(px, max_steps)
    
    # 分析结果
    analyze_solution(phi, dx)
    
    # 验证解的正确性
    verify_solution(phi, dx)
    
    # 保存结果
    save_results(phi, dx, f"poisson_solution_px{px}.npz")
    
    print("\n=== 计算完成 ===")

if __name__ == "__main__":
    # 在Windows上运行多进程时需要这个条件
    main()