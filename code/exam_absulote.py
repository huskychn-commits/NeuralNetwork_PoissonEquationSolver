import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os
import matplotlib

# 解决中文显示问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = True

# 解决OpenMP库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class PINN(nn.Module):
    """物理信息神经网络 - 用于求解三维泊松方程"""
    
    def __init__(self, hidden_layers=5, hidden_neurons=256):
        super(PINN, self).__init__()
        
        # 输入层：3个坐标 (x, y, z)
        self.input_layer = nn.Linear(3, hidden_neurons)
        
        # 隐藏层：5层，每层256个神经元
        self.hidden_layers = nn.ModuleList()
        for _ in range(hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_neurons, hidden_neurons))
        
        # 输出层：1个输出 (电势φ)
        self.output_layer = nn.Linear(hidden_neurons, 1)
        
        # 激活函数：全部使用tanh
        self.activation = nn.Tanh()
    
    def forward(self, x):
        # 输入层
        x = self.activation(self.input_layer(x))
        
        # 隐藏层
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        
        # 输出层
        x = self.output_layer(x)
        return x

def load_model(model_path='pinn.pth'):
    """加载训练好的模型"""
    device = torch.device('cpu')
    model = PINN(hidden_layers=5, hidden_neurons=256).to(device)
    
    # 尝试多个可能的路径
    possible_paths = [
        model_path,  # 当前目录
        f'../{model_path}',  # 上级目录
        f'../../{model_path}'  # 上上级目录
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
            print(f"模型已从 {path} 加载")
            return model
    
    raise FileNotFoundError(f"模型文件 {model_path} 不存在，请先运行训练程序")

def compute_potential_grid(model, px=128):
    """
    在[-1,1]^3网格上计算电势
    
    参数:
    model: 训练好的神经网络模型
    px: 每个维度的网格点数
    
    返回:
    x, y, z: 1D坐标数组
    phi: 3D电势数组 (px, px, px)
    """
    # 创建网格坐标
    x = np.linspace(-1.0, 1.0, px)
    y = np.linspace(-1.0, 1.0, px)
    z = np.linspace(-1.0, 1.0, px)
    
    # 创建3D网格
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # 将网格点转换为神经网络输入格式
    grid_points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32)
    
    # 使用神经网络计算电势
    print(f"计算 {px}x{px}x{px} 网格上的电势...")
    with torch.no_grad():
        phi_flat = model(grid_tensor).numpy().flatten()
    
    # 重塑为3D数组
    phi = phi_flat.reshape((px, px, px))
    
    return x, y, z, phi

def visualize_potential(x, y, z, phi):
    """可视化电势分布"""
    
    # 初始切面坐标
    sx, sy, sz = 0.1, 0.1, 0.1
    
    # 找到最接近指定坐标的索引
    ix = int(np.argmin(np.abs(x - sx)))
    iy = int(np.argmin(np.abs(y - sy)))
    iz = int(np.argmin(np.abs(z - sz)))
    
    # 选择用于显示的子数组
    slice_x = phi[ix, :, :].T    # y (x-axis), z (y-axis) -> transpose 以匹配 extent
    slice_y = phi[:, iy, :].T    # x, z
    slice_z = phi[:, :, iz].T    # x, y
    
    # 统一色标范围（对称）
    vmin = float(np.min(phi))
    vmax = float(np.max(phi))
    vmax_abs = max(abs(vmin), abs(vmax))
    vmin, vmax = -vmax_abs, vmax_abs
    
    # 创建 figure 与子图，并为滑条与 colorbar 留出空间
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.subplots_adjust(left=0.06, right=0.86, bottom=0.18, top=0.95, wspace=0.3)
    cmap = 'RdBu_r'
    
    # 全区域的 colorbar 范围（最小/最大电势）
    vmin_global = float(np.min(phi))
    vmax_global = float(np.max(phi))
    
    im0 = axs[0].imshow(slice_x, extent=[y[0], y[-1], z[0], z[-1]], origin='lower', 
                        cmap=cmap, vmin=vmin_global, vmax=vmax_global, aspect='auto')
    axs[0].set_xlabel('y')
    axs[0].set_ylabel('z')
    axs[0].set_title(f'x = {sx} (ix={ix})')
    
    im1 = axs[1].imshow(slice_y, extent=[x[0], x[-1], z[0], z[-1]], origin='lower', 
                        cmap=cmap, vmin=vmin_global, vmax=vmax_global, aspect='auto')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('z')
    axs[1].set_title(f'y = {sy} (iy={iy})')
    
    im2 = axs[2].imshow(slice_z, extent=[x[0], x[-1], y[0], y[-1]], origin='lower', 
                        cmap=cmap, vmin=vmin_global, vmax=vmax_global, aspect='auto')
    axs[2].set_xlabel('x')
    axs[2].set_ylabel('y')
    axs[2].set_title(f'z = {sz} (iz={iz})')
    
    # colorbar 放在最右侧外部
    cbar_ax = fig.add_axes([0.88, 0.18, 0.02, 0.7])
    cbar = fig.colorbar(im2, cax=cbar_ax)
    cbar.set_label('电势 φ')
    
    # 添加滑条 (Slider)，范围 [-1,1]
    axcolor = 'lightgoldenrodyellow'
    ax_sx = fig.add_axes([0.06, 0.06, 0.24, 0.03], facecolor=axcolor)
    ax_sy = fig.add_axes([0.36, 0.06, 0.24, 0.03], facecolor=axcolor)
    ax_sz = fig.add_axes([0.66, 0.06, 0.18, 0.03], facecolor=axcolor)
    
    slider_x = Slider(ax_sx, 'x', -1.0, 1.0, valinit=sx)
    slider_y = Slider(ax_sy, 'y', -1.0, 1.0, valinit=sy)
    slider_z = Slider(ax_sz, 'z', -1.0, 1.0, valinit=sz)
    
    def update(val):
        sxv = float(slider_x.val)
        syv = float(slider_y.val)
        szv = float(slider_z.val)
        ix_new = int(np.argmin(np.abs(x - sxv)))
        iy_new = int(np.argmin(np.abs(y - syv)))
        iz_new = int(np.argmin(np.abs(z - szv)))
        
        new_x = phi[ix_new, :, :].T
        new_y = phi[:, iy_new, :].T
        new_z = phi[:, :, iz_new].T
        
        im0.set_data(new_x)
        im1.set_data(new_y)
        im2.set_data(new_z)
        
        axs[0].set_title(f'x = {sxv:.3f} (ix={ix_new})')
        axs[1].set_title(f'y = {syv:.3f} (iy={iy_new})')
        axs[2].set_title(f'z = {szv:.3f} (iz={iz_new})')
        
        # 保持 colorbar 范围为全局最小/最大
        for im in (im0, im1, im2):
            im.set_clim(vmin_global, vmax_global)
        cbar.mappable.set_clim(vmin_global, vmax_global)
        fig.canvas.draw_idle()
    
    slider_x.on_changed(update)
    slider_y.on_changed(update)
    slider_z.on_changed(update)
    
    plt.show()

def analyze_solution(phi):
    """分析求解结果"""
    print("\n=== PINN求解结果分析 ===")
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

def find_benchmark_data():
    """智能查找基准数据文件"""
    import os
    
    # 可能的基准数据文件路径
    possible_paths = [
        # 从当前目录开始向上查找
        '../Benchmark_Solver/网格法直接求解微分方程/phi.npz',
        '../../Benchmark_Solver/网格法直接求解微分方程/phi.npz',
        '../../../Benchmark_Solver/网格法直接求解微分方程/phi.npz',
        # 从根目录开始查找
        'Benchmark_Solver/网格法直接求解微分方程/phi.npz',
        # 尝试查找任何包含Benchmark_Solver的路径
        '**/Benchmark_Solver/**/phi.npz',
        '../**/Benchmark_Solver/**/phi.npz',
        '../../**/Benchmark_Solver/**/phi.npz'
    ]
    
    # 首先尝试直接路径
    for path in possible_paths[:6]:  # 前6个是具体路径
        if os.path.exists(path):
            return path
    
    # 如果具体路径没找到，使用glob查找
    import glob
    for pattern in possible_paths[6:]:  # 最后2个是glob模式
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0]
    
    # 如果还没找到，尝试在当前目录及其父目录中搜索
    current_dir = os.getcwd()
    for root, dirs, files in os.walk(current_dir):
        if 'phi.npz' in files and 'Benchmark_Solver' in root:
            return os.path.join(root, 'phi.npz')
    
    # 向上搜索父目录
    parent_dir = os.path.dirname(current_dir)
    for root, dirs, files in os.walk(parent_dir):
        if 'phi.npz' in files and 'Benchmark_Solver' in root:
            return os.path.join(root, 'phi.npz')
    
    return None

def load_benchmark_data():
    """加载基准求解器的电势数据"""
    benchmark_path = find_benchmark_data()
    
    if benchmark_path is None:
        print("未找到基准数据文件 phi.npz")
        print("请确保基准求解器已经运行并生成了phi.npz文件")
        return None, None, None, None
    
    try:
        data = np.load(benchmark_path)
        x_bench = data['x']
        y_bench = data['y']
        z_bench = data['z']
        phi_bench = data['phi']
        print(f"基准数据已从 {benchmark_path} 加载")
        print(f"基准数据形状: {phi_bench.shape}")
        return x_bench, y_bench, z_bench, phi_bench
    except Exception as e:
        print(f"加载基准数据失败: {e}")
        return None, None, None, None

def compute_potential_comparison(model, px=128):
    """
    计算PINN和基准的电势，并计算差值
    
    返回:
    x, y, z: 坐标数组
    phi_bench: 基准电势
    phi_pinn: PINN电势  
    phi_diff: 差值 (PINN - 基准)
    """
    # 加载基准数据
    x_bench, y_bench, z_bench, phi_bench = load_benchmark_data()
    
    if phi_bench is None:
        # 如果没有基准数据，使用PINN计算
        print("使用PINN计算基准数据...")
        x, y, z, phi_pinn = compute_potential_grid(model, px=px)
        phi_bench = phi_pinn.copy()  # 临时使用PINN作为基准
        phi_diff = np.zeros_like(phi_pinn)
    else:
        # 使用基准数据的网格
        x, y, z = x_bench, y_bench, z_bench
        px = len(x)
        
        # 在基准网格上计算PINN电势
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        grid_points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        grid_tensor = torch.tensor(grid_points, dtype=torch.float32)
        
        print(f"在基准网格上计算PINN电势 ({px}x{px}x{px})...")
        with torch.no_grad():
            phi_flat = model(grid_tensor).numpy().flatten()
        phi_pinn = phi_flat.reshape((px, px, px))
        
        # 计算差值
        phi_diff = phi_pinn - phi_bench
    
    return x, y, z, phi_bench, phi_pinn, phi_diff

def visualize_comparison(x, y, z, phi_bench, phi_pinn, phi_diff):
    """3×3对比可视化：基准、PINN、差值"""
    
    # 初始切面坐标
    sx, sy, sz = 0.1, 0.1, 0.1
    
    # 找到最接近指定坐标的索引
    ix = int(np.argmin(np.abs(x - sx)))
    iy = int(np.argmin(np.abs(y - sy)))
    iz = int(np.argmin(np.abs(z - sz)))
    
    # 创建3×3子图
    fig, axs = plt.subplots(3, 3, figsize=(15, 12))
    fig.subplots_adjust(left=0.06, right=0.86, bottom=0.18, top=0.95, wspace=0.3, hspace=0.3)
    cmap = 'RdBu_r'
    
    # 使用基准数据的全局范围
    vmin_global = float(np.min(phi_bench))
    vmax_global = float(np.max(phi_bench))
    
    # 准备切面数据
    def get_slices(phi_data, ix, iy, iz):
        return {
            'x': phi_data[ix, :, :].T,
            'y': phi_data[:, iy, :].T,
            'z': phi_data[:, :, iz].T
        }
    
    bench_slices = get_slices(phi_bench, ix, iy, iz)
    pinn_slices = get_slices(phi_pinn, ix, iy, iz)
    diff_slices = get_slices(phi_diff, ix, iy, iz)
    
    # 差值使用对称色标
    diff_max = max(abs(np.min(phi_diff)), abs(np.max(phi_diff)))
    vmin_diff, vmax_diff = -diff_max, diff_max
    
    # 行标题
    row_titles = ['基准求解器', 'PINN求解', '差值 (PINN - 基准)']
    
    # 列标题
    col_titles = [f'x = {sx:.3f}', f'y = {sy:.3f}', f'z = {sz:.3f}']
    
    # 绘制所有子图
    images = []
    for row in range(3):
        for col in range(3):
            ax = axs[row, col]
            
            if row == 0:  # 基准
                if col == 0:
                    im = ax.imshow(bench_slices['x'], extent=[y[0], y[-1], z[0], z[-1]], 
                                  origin='lower', cmap=cmap, vmin=vmin_global, vmax=vmax_global, aspect='auto')
                elif col == 1:
                    im = ax.imshow(bench_slices['y'], extent=[x[0], x[-1], z[0], z[-1]], 
                                  origin='lower', cmap=cmap, vmin=vmin_global, vmax=vmax_global, aspect='auto')
                else:  # col == 2
                    im = ax.imshow(bench_slices['z'], extent=[x[0], x[-1], y[0], y[-1]], 
                                  origin='lower', cmap=cmap, vmin=vmin_global, vmax=vmax_global, aspect='auto')
            
            elif row == 1:  # PINN
                if col == 0:
                    im = ax.imshow(pinn_slices['x'], extent=[y[0], y[-1], z[0], z[-1]], 
                                  origin='lower', cmap=cmap, vmin=vmin_global, vmax=vmax_global, aspect='auto')
                elif col == 1:
                    im = ax.imshow(pinn_slices['y'], extent=[x[0], x[-1], z[0], z[-1]], 
                                  origin='lower', cmap=cmap, vmin=vmin_global, vmax=vmax_global, aspect='auto')
                else:  # col == 2
                    im = ax.imshow(pinn_slices['z'], extent=[x[0], x[-1], y[0], y[-1]], 
                                  origin='lower', cmap=cmap, vmin=vmin_global, vmax=vmax_global, aspect='auto')
            
            else:  # row == 2, 差值
                if col == 0:
                    im = ax.imshow(diff_slices['x'], extent=[y[0], y[-1], z[0], z[-1]], 
                                  origin='lower', cmap=cmap, vmin=vmin_global, vmax=vmax_global, aspect='auto')
                elif col == 1:
                    im = ax.imshow(diff_slices['y'], extent=[x[0], x[-1], z[0], z[-1]], 
                                  origin='lower', cmap=cmap, vmin=vmin_global, vmax=vmax_global, aspect='auto')
                else:  # col == 2
                    im = ax.imshow(diff_slices['z'], extent=[x[0], x[-1], y[0], y[-1]], 
                                  origin='lower', cmap=cmap, vmin=vmin_global, vmax=vmax_global, aspect='auto')
            
            images.append(im)
            
            # 设置坐标轴标签
            if row == 2:  # 最后一行显示x轴标签
                if col == 0:
                    ax.set_xlabel('y')
                    ax.set_ylabel('z')
                elif col == 1:
                    ax.set_xlabel('x')
                    ax.set_ylabel('z')
                else:
                    ax.set_xlabel('x')
                    ax.set_ylabel('y')
            else:
                ax.set_xticks([])
                ax.set_yticks([])
            
            # 设置标题
            if row == 0:  # 第一行显示列标题
                ax.set_title(col_titles[col], fontsize=12)
            if col == 0:  # 第一列显示行标题
                ax.text(-0.3, 0.5, row_titles[row], transform=ax.transAxes, 
                       fontsize=12, rotation=90, va='center', ha='right')
    
    # 为每一行添加colorbar，与图像对齐，使用相同的映射方式
    cbar_axes = []
    for row in range(3):
        # 计算每行colorbar的位置，与图像高度对齐
        cbar_height = 0.2  # colorbar高度
        cbar_bottom = 0.75 - row*0.25  # 与图像底部对齐
        cbar_ax = fig.add_axes([0.88, cbar_bottom, 0.02, cbar_height])
        
        # 所有colorbar使用基准数据的全局范围
        cbar = fig.colorbar(images[row*3+2], cax=cbar_ax)
        if row == 2:  # 差值行
            cbar.set_label('差值', fontsize=10)
        else:  # 基准和PINN行
            cbar.set_label('电势 φ', fontsize=10)
        cbar_axes.append(cbar_ax)
    
    # 添加共享滑条
    axcolor = 'lightgoldenrodyellow'
    ax_sx = fig.add_axes([0.06, 0.06, 0.24, 0.03], facecolor=axcolor)
    ax_sy = fig.add_axes([0.36, 0.06, 0.24, 0.03], facecolor=axcolor)
    ax_sz = fig.add_axes([0.66, 0.06, 0.18, 0.03], facecolor=axcolor)
    
    slider_x = Slider(ax_sx, 'x', -1.0, 1.0, valinit=sx)
    slider_y = Slider(ax_sy, 'y', -1.0, 1.0, valinit=sy)
    slider_z = Slider(ax_sz, 'z', -1.0, 1.0, valinit=sz)
    
    def update(val):
        sxv = float(slider_x.val)
        syv = float(slider_y.val)
        szv = float(slider_z.val)
        ix_new = int(np.argmin(np.abs(x - sxv)))
        iy_new = int(np.argmin(np.abs(y - syv)))
        iz_new = int(np.argmin(np.abs(z - szv)))
        
        # 更新所有切面数据
        bench_slices_new = get_slices(phi_bench, ix_new, iy_new, iz_new)
        pinn_slices_new = get_slices(phi_pinn, ix_new, iy_new, iz_new)
        diff_slices_new = get_slices(phi_diff, ix_new, iy_new, iz_new)
        
        # 更新图像数据
        for row in range(3):
            for col in range(3):
                idx = row * 3 + col
                if row == 0:  # 基准
                    if col == 0:
                        images[idx].set_data(bench_slices_new['x'])
                    elif col == 1:
                        images[idx].set_data(bench_slices_new['y'])
                    else:
                        images[idx].set_data(bench_slices_new['z'])
                elif row == 1:  # PINN
                    if col == 0:
                        images[idx].set_data(pinn_slices_new['x'])
                    elif col == 1:
                        images[idx].set_data(pinn_slices_new['y'])
                    else:
                        images[idx].set_data(pinn_slices_new['z'])
                else:  # 差值
                    if col == 0:
                        images[idx].set_data(diff_slices_new['x'])
                    elif col == 1:
                        images[idx].set_data(diff_slices_new['y'])
                    else:
                        images[idx].set_data(diff_slices_new['z'])
        
        # 更新列标题
        col_titles_new = [f'x = {sxv:.3f}', f'y = {syv:.3f}', f'z = {szv:.3f}']
        for col in range(3):
            axs[0, col].set_title(col_titles_new[col], fontsize=12)
        
        fig.canvas.draw_idle()
    
    slider_x.on_changed(update)
    slider_y.on_changed(update)
    slider_z.on_changed(update)
    
    plt.show()

def main():
    """主函数"""
    print("PINN与基准求解器对比可视化")
    print("=" * 50)
    
    try:
        # 加载训练好的模型
        model = load_model('pinn.pth')
        
        # 计算对比数据
        x, y, z, phi_bench, phi_pinn, phi_diff = compute_potential_comparison(model)
        
        # 分析结果
        print("\n=== 基准求解器结果分析 ===")
        analyze_solution(phi_bench)
        print("\n=== PINN求解结果分析 ===")
        analyze_solution(phi_pinn)
        print("\n=== 差值分析 ===")
        print(f"差值范围: [{np.min(phi_diff):.6f}, {np.max(phi_diff):.6f}]")
        print(f"差值均方根: {np.sqrt(np.mean(phi_diff**2)):.6f}")
        
        # 可视化对比
        print("\n开始对比可视化...")
        visualize_comparison(x, y, z, phi_bench, phi_pinn, phi_diff)
        
    except Exception as e:
        print(f"错误: {e}")
        print("请确保已经运行过训练程序并生成了pinn.pth文件")

if __name__ == "__main__":
    main()
