import torch
import sys
import platform

def check_cuda_info():
    """检查CUDA可用性和版本信息"""
    
    print("=" * 50)
    print("CUDA和PyTorch环境检测")
    print("=" * 50)
    
    # 系统信息
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    
    # CUDA可用性
    cuda_available = torch.cuda.is_available()
    print(f"CUDA可用: {cuda_available}")
    
    if cuda_available:
        # CUDA版本信息
        print(f"CUDA版本: {torch.version.cuda}")
        
        # GPU设备信息
        device_count = torch.cuda.device_count()
        print(f"GPU设备数量: {device_count}")
        
        for i in range(device_count):
            print(f"\n--- GPU {i} ---")
            print(f"设备名称: {torch.cuda.get_device_name(i)}")
            print(f"设备能力: {torch.cuda.get_device_capability(i)}")
            print(f"设备内存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            
            # 当前设备状态
            if i == torch.cuda.current_device():
                print(f"当前设备: 是")
                print(f"已分配内存: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
                print(f"缓存内存: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
            else:
                print(f"当前设备: 否")
    
    else:
        print("\nCUDA不可用的可能原因:")
        print("1. 没有NVIDIA显卡")
        print("2. 没有安装CUDA驱动")
        print("3. PyTorch版本与CUDA版本不匹配")
        print("4. 显卡计算能力不被当前PyTorch版本支持")
        
        # 检查是否有NVIDIA显卡但CUDA不可用
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                print("\n检测到NVIDIA显卡，但PyTorch无法使用CUDA")
                print("建议更新PyTorch到支持您显卡的版本")
        except:
            pass
    
    print("\n" + "=" * 50)
    
    # 测试简单的CUDA操作
    if cuda_available:
        print("\n测试CUDA操作...")
        try:
            # 创建测试张量
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            
            # 执行矩阵乘法
            z = torch.matmul(x, y)
            
            print("CUDA操作测试通过！")
            print(f"测试张量形状: {z.shape}")
            print(f"测试张量设备: {z.device}")
            
        except Exception as e:
            print(f"CUDA操作测试失败: {e}")
    
    return cuda_available

def check_compatibility():
    """检查显卡计算能力与PyTorch兼容性"""
    print("\n" + "=" * 50)
    print("兼容性检查")
    print("=" * 50)
    
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(current_device)
        device_name = torch.cuda.get_device_name(current_device)
        
        print(f"当前显卡: {device_name}")
        print(f"计算能力: sm_{capability[0]}{capability[1]}")
        
        # 已知支持的CUDA计算能力
        supported_capabilities = [
            "sm_50", "sm_60", "sm_61", "sm_70", "sm_75", 
            "sm_80", "sm_86", "sm_89", "sm_90"
        ]
        
        current_capability = f"sm_{capability[0]}{capability[1]}"
        
        if current_capability in supported_capabilities:
            print(f"✅ 兼容性: 支持")
        else:
            print(f"❌ 兼容性: 不支持")
            print(f"当前PyTorch版本支持的计算能力: {', '.join(supported_capabilities)}")
            print("建议更新PyTorch到支持您显卡的版本")
    
    print("=" * 50)

if __name__ == "__main__":
    check_cuda_info()
    check_compatibility()
    
    # 使用建议
    print("\n使用建议:")
    if torch.cuda.is_available():
        print("✅ 可以使用GPU加速训练")
        print("运行 train.py 时将自动使用GPU")
    else:
        print("⚠️  只能使用CPU进行训练")
        print("运行 train.py 时将使用CPU模式")
        print("如需使用GPU，请更新PyTorch到支持您显卡的版本")
