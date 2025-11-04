Poisson 方程求解器

描述
---
在立方体域 [-1,1]^3 上，求解 Poisson 方程 ∇^2 φ = -ρ，边界条件 φ = 0（Dirichlet）。电荷分布为 ρ = 100 * x * y * z^2，使用自然单位制（ε0 = 1）。

方法
---
使用离散正弦变换（DST-I）将离散拉普拉斯算子对角化（适用于零 Dirichlet 边界），在频域求解后逆变换得到内部解。该方法复杂度约为 O(n^3 log n)。

运行
---
安装依赖：

```powershell
python -m pip install -r requirements.txt
```

运行示例（在 Windows PowerShell）：

```powershell
python poisson_solver.py --px 128 --out phi_128.npz
```

参数
---
- --px: 每一维的网格点数量（包含边界）。默认 128。
- --out: 输出 npz 文件名，保存 x,y,z,phi。
- --save-slice: 同时保存中央 z 切片为文本文件，便于快速查看。

注意
---
- px 应该至少为 3（包含边界）。
- 该实现依赖于 SciPy 的 fftpack 中的 DST/I DST。若 SciPy 版本不同可能需要小改动。