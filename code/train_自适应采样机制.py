import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import time
import os
import matplotlib

# 解决中文显示问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
# 指定数学符号使用 STIX 字体（专门用于数学符号）
matplotlib.rcParams['mathtext.fontset'] = 'stix'  # 使用 STIX 数学字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 可以保持 False

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

def charge_density(x, y, z):
    """电荷密度函数：ρ = 100 * x * y * z^2"""
    return 100 * x * y * z**2

def generate_samples(n_total=10000, ratio_boundary=0.01):
    """
    生成训练样本点
    
    参数:
    n_total: 总采样点数
    ratio_boundary: 每个边界面的采样比例
    """
    # 计算各类采样点数量
    n_boundary_per_face = int(min(n_total * ratio_boundary , n_total/7))  # 7是顺手写的超参数
    n_interior = n_total - 6 * n_boundary_per_face
    
    # 生成体内采样点 (均匀分布在[-1,1]^3内)
    interior_points = torch.rand(n_interior, 3) * 2 - 1
    
    # 生成边界采样点 (6个面)
    boundary_points = []
    
    # x = -1 和 x = 1 边界
    for x_val in [-1.0, 1.0]:
        yz_points = torch.rand(n_boundary_per_face, 2) * 2 - 1
        x_points = torch.full((n_boundary_per_face, 1), x_val)
        face_points = torch.cat([x_points, yz_points], dim=1)
        boundary_points.append(face_points)
    
    # y = -1 和 y = 1 边界
    for y_val in [-1.0, 1.0]:
        xz_points = torch.rand(n_boundary_per_face, 2) * 2 - 1
        y_points = torch.full((n_boundary_per_face, 1), y_val)
        face_points = torch.cat([xz_points[:, 0:1], y_points, xz_points[:, 1:2]], dim=1)
        boundary_points.append(face_points)
    
    # z = -1 和 z = 1 边界
    for z_val in [-1.0, 1.0]:
        xy_points = torch.rand(n_boundary_per_face, 2) * 2 - 1
        z_points = torch.full((n_boundary_per_face, 1), z_val)
        face_points = torch.cat([xy_points, z_points], dim=1)
        boundary_points.append(face_points)
    
    # 合并所有边界点
    boundary_points = torch.cat(boundary_points, dim=0)
    
    # 合并所有点
    all_points = torch.cat([interior_points, boundary_points], dim=0)
    
    # 打乱顺序
    indices = torch.randperm(all_points.shape[0])
    all_points = all_points[indices]
    
    # 分离体内点和边界点
    interior_mask = torch.ones(n_total, dtype=torch.bool)
    interior_mask[n_interior:] = False
    
    interior_points = all_points[interior_mask]
    boundary_points = all_points[~interior_mask]
    
    return interior_points, boundary_points

def compute_pde_loss(model, interior_points):
    """计算PDE损失：Mean((∇²φ + ρ)²)"""
    
    # 需要计算二阶导数，设置requires_grad=True
    interior_points.requires_grad_(True)
    
    # 计算神经网络输出
    phi = model(interior_points)
    
    # 计算一阶导数 ∇φ
    grad_phi = torch.autograd.grad(
        phi, interior_points, 
        grad_outputs=torch.ones_like(phi),
        create_graph=True,
        retain_graph=True
    )[0]
    
    # 计算二阶导数 ∇²φ
    laplacian = 0.0
    for i in range(3):  # 对x,y,z三个方向分别求二阶导
        grad_component = grad_phi[:, i:i+1]
        grad2_component = torch.autograd.grad(
            grad_component, interior_points,
            grad_outputs=torch.ones_like(grad_component),
            create_graph=True,
            retain_graph=True
        )[0][:, i:i+1]
        laplacian += grad2_component
    
    # 计算电荷密度
    x, y, z = interior_points[:, 0:1], interior_points[:, 1:2], interior_points[:, 2:3]
    rho = charge_density(x, y, z)
    
    # PDE残差：∇²φ + ρ
    pde_residual = laplacian + rho
    
    # PDE损失：Mean((∇²φ + ρ)²)
    pde_loss = torch.mean(pde_residual**2)
    
    return pde_loss

def compute_boundary_loss(model, boundary_points):
    """计算边界损失：Mean(φ²)"""
    phi_boundary = model(boundary_points)
    boundary_loss = torch.mean(phi_boundary**2)
    return boundary_loss

def exponential_fit(x_data, y_data):
    """对数据进行指数函数拟合：y = a * exp(b*x)"""
    try:
        # 取对数进行线性拟合
        log_y = np.log(y_data + 1e-10)  # 避免log(0)
        coeffs = np.polyfit(x_data, log_y, 1)
        a = np.exp(coeffs[1])
        b = coeffs[0]
        return a, b
    except:
        return None, None

def boundary_ratio_combined(expected_loss_boundary, expected_loss_pde, boundary_improvement_contribution, boundary_improvement_threshold, base_ratio, sr_adapt_factor, boundary_ratio_upper_limit):
    """计算边缘占比：结合期望损失比例和边界改进贡献度"""
    # 基于期望损失比例计算基础边缘占比
    if expected_loss_pde > 0:
        ratio_from_expected = max(base_ratio, min(boundary_ratio_upper_limit, expected_loss_boundary / expected_loss_pde * base_ratio))
    else:
        ratio_from_expected = base_ratio
    
    # 如果边界项的改进贡献大于阈值，则进一步调整采样比例
    if boundary_improvement_contribution > boundary_improvement_threshold:
        ratio = ratio_from_expected * sr_adapt_factor
    else:
        ratio = ratio_from_expected
    return ratio

def train_model():
    """训练神经网络模型"""
    
    # 设置设备 - 强制使用CPU以避免CUDA兼容性问题
    device = torch.device('cpu')#('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device} ")
    
    # 初始化模型
    model = PINN(hidden_layers=5, hidden_neurons=256).to(device)
    
    # 初始学习率
    initial_lr = 1e-3
    current_lr = initial_lr
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    
    # 训练参数
    n_sample_learn = 10000
    beta = 1  # 边缘权重调控参数
    error_epoch_lim = 1e-3
    stable_count_required = 3
    
    # 学习率衰减参数
    lr_decay_start_epoch = 400    # 400个epoch后开始学习率衰减
    lookback_window = 100         # 观察窗口大小
    improvement_threshold = 0.05  # 5%改进阈值
    lr_decay_factor = 0.997       # 学习率衰减因子（每次乘以0.997）

    # 采样比例自适应参数
    base_ratio = 0.05  # 基础边缘占比
    boundary_ratio_upper_limit = 1/7  # 边缘占比的上限
    improvement_threshold_factor = 1.1  # 改进阈值因子
    sr_adapt_factor = 1.05  # 采样比例自适应因子
    
    # 训练记录
    losses_total = []
    losses_pde = []
    losses_boundary = []
    epochs = []
    learning_rates = []
    boundary_ratios = []  # 记录边缘占比
    
    # 手动终止标志
    manual_stop = False
    
    # 设置绘图
    plt.ion()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    # 损失曲线
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('PINN Training Progress')
    ax1.set_yscale('log')
    line_total, = ax1.plot([], [], 'b-', label='Total Loss')
    line_pde, = ax1.plot([], [], 'r-', label='PDE Loss')
    line_boundary, = ax1.plot([], [], 'g-', label='Boundary Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 学习率曲线
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.set_yscale('log')
    line_lr, = ax2.plot([], [], 'm-', label='Learning Rate')
    ax2.legend()
    ax2.grid(True)
    
    # 边缘占比曲线
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Boundary Ratio')
    ax3.set_title('Adaptive Boundary Sampling Ratio')
    line_boundary_ratio, = ax3.plot([], [], 'c-', label='Boundary Ratio')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout(pad=3.0, h_pad=2.0, w_pad=2.0)
    
    # 添加手动终止按钮
    ax_button = plt.axes([0.81, 0.02, 0.15, 0.05])  # [left, bottom, width, height]
    stop_button = Button(ax_button, '停止训练')
    
    def stop_training(event):
        nonlocal manual_stop
        manual_stop = True
        print("\n用户手动终止训练...")
    
    stop_button.on_clicked(stop_training)
    
    # 训练循环
    epoch = 0
    stable_count = 0
    prev_loss = None
    
    print("开始训练...")
    print("提示：点击图形上的'停止训练'按钮可以手动终止训练")
    start_time = time.time()
    
    while True:
        # 检查手动终止
        if manual_stop:
            print(f"\n训练被手动终止于第 {epoch} 个epoch")
            break
        # 自适应采样机制
        if epoch >= lr_decay_start_epoch and len(losses_total) >= 2 * lookback_window:
            # 获取两个时间段的数据
            # 时间段1: (i-200) ~ (i-100)
            epoch_range1 = np.array(epochs[-2*lookback_window:-lookback_window])
            loss_range1_total = np.array(losses_total[-2*lookback_window:-lookback_window])
            loss_range1_boundary = np.array(losses_boundary[-lookback_window:])
            
            # 时间段2: (i-100) ~ i
            epoch_range2 = np.array(epochs[-lookback_window:])
            loss_range2_total = np.array(losses_total[-lookback_window:])
            loss_range2_boundary = np.array(losses_boundary[-lookback_window:])
            
            # 对两个时间段的边界损失分别进行指数拟合
            a1_boundary, b1_boundary = exponential_fit(epoch_range1, loss_range1_boundary)
            a2_boundary, b2_boundary = exponential_fit(epoch_range2, loss_range2_boundary)
            
            # 对两个时间段的总损失分别进行指数拟合
            a1_total, b1_total = exponential_fit(epoch_range1, loss_range1_total)
            a2_total, b2_total = exponential_fit(epoch_range2, loss_range2_total)
            
            if (a1_boundary is not None and b1_boundary is not None and 
                a2_boundary is not None and b2_boundary is not None and
                a1_total is not None and b1_total is not None and
                a2_total is not None and b2_total is not None):
                
                # 计算边界损失的改进率
                start_boundary1 = a1_boundary * np.exp(b1_boundary * epoch_range1[0])
                end_boundary1 = a1_boundary * np.exp(b1_boundary * epoch_range1[-1])
                improvement_boundary1 = (start_boundary1 - end_boundary1) / start_boundary1
                
                start_boundary2 = a2_boundary * np.exp(b2_boundary * epoch_range2[0])
                end_boundary2 = a2_boundary * np.exp(b2_boundary * epoch_range2[-1])
                improvement_boundary2 = (start_boundary2 - end_boundary2) / start_boundary2
                
                # 计算总损失的改进率
                start_total1 = a1_total * np.exp(b1_total * epoch_range1[0])
                end_total1 = a1_total * np.exp(b1_total * epoch_range1[-1])
                improvement_total1 = (start_total1 - end_total1) / start_total1
                
                start_total2 = a2_total * np.exp(b2_total * epoch_range2[0])
                end_total2 = a2_total * np.exp(b2_total * epoch_range2[-1])
                improvement_total2 = (start_total2 - end_total2) / start_total2
                
                # 计算边界改进对总改进的贡献度
                boundary_improvement_contribution = (improvement_boundary2 - improvement_boundary1) / (improvement_total2 - improvement_total1 + 1e-10)
                
                # 计算当前epoch的期望损失（使用前100个数据进行指数拟合）
                recent_epochs = np.array(epochs[-lookback_window:])
                recent_losses_pde = np.array(losses_pde[-lookback_window:])
                recent_losses_boundary = np.array(losses_boundary[-lookback_window:])
                
                a_pde, b_pde = exponential_fit(recent_epochs, recent_losses_pde)
                a_boundary, b_boundary = exponential_fit(recent_epochs, recent_losses_boundary)
                
                if a_pde is not None and b_pde is not None and a_boundary is not None and b_boundary is not None:
                    expected_loss_pde = a_pde * np.exp(b_pde * epoch)
                    expected_loss_boundary = a_boundary * np.exp(b_boundary * epoch)
                    
                    # 使用结合机制计算边缘占比
                    # 实时计算改进阈值：min(1, 1.5 * 6 * base_ratio)
                    boundary_improvement_threshold = min(1, improvement_threshold_factor * 6 * base_ratio)
                    current_boundary_ratio = boundary_ratio_combined(expected_loss_boundary, expected_loss_pde, boundary_improvement_contribution, boundary_improvement_threshold, base_ratio, sr_adapt_factor, boundary_ratio_upper_limit)
                else:
                    # 拟合失败时使用固定比例
                    current_boundary_ratio = base_ratio
            else:
                # 拟合失败时使用固定比例
                current_boundary_ratio = base_ratio
        else:
            # 使用固定边缘占比
            current_boundary_ratio = base_ratio
        
        # 生成新的采样点
        interior_points, boundary_points = generate_samples(n_total=n_sample_learn, ratio_boundary=current_boundary_ratio)
        interior_points = interior_points.to(device)
        boundary_points = boundary_points.to(device)
        
        # 前向传播
        loss_pde = compute_pde_loss(model, interior_points)
        loss_boundary = compute_boundary_loss(model, boundary_points)
        
        # 计算采样概率
        P_interior = 1 - current_boundary_ratio * 6  # P(采样点在体)
        P_boundary = current_boundary_ratio * 6      # P(采样点在边)
        
        # 总损失公式: (P(采样点在体)*loss_pde + beta*P(采样点在边)*loss_boundary) / (P(采样点在体) + beta*P(采样点在边))
        loss_total = (P_interior * loss_pde + beta * P_boundary * loss_boundary) / (P_interior + beta * P_boundary)
        
        # 反向传播
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        
        # 记录损失、学习率和边缘占比
        losses_total.append(loss_total.item())
        losses_pde.append(loss_pde.item())
        losses_boundary.append(loss_boundary.item())
        epochs.append(epoch)
        learning_rates.append(current_lr)
        boundary_ratios.append(current_boundary_ratio)
        
        # 学习率调整策略（epoch >= 400时启用）
        if epoch >= lr_decay_start_epoch and len(losses_total) >= 2 * lookback_window:
            # 获取两个时间段的数据
            # 时间段1: (i-200) ~ (i-100)
            epoch_range1 = np.array(epochs[-2*lookback_window:-lookback_window])
            loss_range1 = np.array(losses_total[-2*lookback_window:-lookback_window])
            
            # 时间段2: (i-100) ~ i
            epoch_range2 = np.array(epochs[-lookback_window:])
            loss_range2 = np.array(losses_total[-lookback_window:])
            
            # 对两个时间段分别进行指数拟合
            a1, b1 = exponential_fit(epoch_range1, loss_range1)
            a2, b2 = exponential_fit(epoch_range2, loss_range2)
            
            if a1 is not None and b1 is not None and a2 is not None and b2 is not None:
                # 计算两个时间段的改进率
                # 时间段1的改进率：从开始到结束的损失改进
                start_loss1 = a1 * np.exp(b1 * epoch_range1[0])
                end_loss1 = a1 * np.exp(b1 * epoch_range1[-1])
                improvement1 = (start_loss1 - end_loss1) / start_loss1
                
                # 时间段2的改进率：从开始到结束的损失改进
                start_loss2 = a2 * np.exp(b2 * epoch_range2[0])
                end_loss2 = a2 * np.exp(b2 * epoch_range2[-1])
                improvement2 = (start_loss2 - end_loss2) / start_loss2
                
                # 如果时间段2的改进率小于时间段1改进率的95%，说明学习变慢，需要降低学习率
                if improvement2 < improvement1 * (1 - improvement_threshold):
                    current_lr *= lr_decay_factor  # 学习率乘以衰减因子
                    
                    # 更新优化器的学习率
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = current_lr
        
        # 更新绘图
        if epoch % 10 == 0:  # 每10个epoch更新一次绘图
            line_total.set_data(epochs, losses_total)
            line_pde.set_data(epochs, losses_pde)
            line_boundary.set_data(epochs, losses_boundary)
            line_lr.set_data(epochs, learning_rates)
            line_boundary_ratio.set_data(epochs, boundary_ratios)
            ax1.relim()
            ax1.autoscale_view()
            ax2.relim()
            ax2.autoscale_view()
            ax3.relim()
            ax3.autoscale_view()
            plt.draw()
            plt.pause(0.01)
        
        # 检查停止条件
        if prev_loss is not None:
            relative_error = abs(loss_total.item() - prev_loss) / abs(prev_loss)
            if relative_error < error_epoch_lim:
                stable_count += 1
                if epoch % 50 == 0:  # 每50个epoch打印一次详细进度
                    print(f"Epoch {epoch}: Loss = {loss_total.item():.6e}, LR = {current_lr:.2e}, 相对误差 = {relative_error:.6e}, 稳定计数 = {stable_count}")
                
                if stable_count >= stable_count_required:
                    print(f"\n训练完成！达到停止条件：连续{stable_count_required}次相对误差 < {error_epoch_lim}")
                    break
            else:
                stable_count = 0
                if epoch % 100 == 0:  # 每100个epoch打印一次进度
                    print(f"Epoch {epoch}: Loss = {loss_total.item():.6e}, LR = {current_lr:.2e}, 相对误差 = {relative_error:.6e}")
        else:
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss_total.item():.6e}, LR = {current_lr:.2e}")
        
        prev_loss = loss_total.item()
        epoch += 1
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"训练耗时: {training_time:.2f} 秒")
    print(f"总训练轮数: {epoch}")
    
    # 保存模型到当前文件夹
    model_path = 'pinn.pth'
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到 {os.path.abspath(model_path)}")
    
    # 训练完成后保持图像显示但不阻塞程序
    if manual_stop:
        print("\n训练已手动终止！图像窗口保持显示，您可以查看训练曲线。")
    else:
        print("\n训练已完成！图像窗口保持显示，您可以查看训练曲线。")
    print("关闭图像窗口后程序将继续执行。")
    plt.ioff()
    plt.show(block=True)  # block=True 确保图像窗口保持显示直到用户手动关闭
    
    # 弹出训练完成提示，同时在控制台打印
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        if manual_stop:
            message_text = f"PINN训练已手动终止！\n总轮数: {epoch}\n训练时间: {training_time:.2f}秒"
            messagebox.showinfo("训练终止", message_text)
        else:
            message_text = f"PINN训练已完成！\n总轮数: {epoch}\n训练时间: {training_time:.2f}秒"
            messagebox.showinfo("训练完成", message_text)
        root.destroy()
    except:
        if manual_stop:
            print("训练已手动终止！")
        else:
            print("训练完成！")
    
    return model, losses_total, losses_pde, losses_boundary, epochs

if __name__ == "__main__":
    # 训练模型
    model, losses_total, losses_pde, losses_boundary, epochs = train_model()
