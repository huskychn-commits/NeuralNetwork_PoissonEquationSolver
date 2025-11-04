#!/usr/bin/env python3
"""
Poisson 方程求解器（Dirichlet 边界 phi=0）
域: [-1,1]^3
电荷分布: rho = 100 * x * y * z^2
单位: 自然单位制（ε0 = 1）
数值方法: 使用离散正弦变换（DST-I）在内部格点上求解离散 Poisson 方程

用法:
    python poisson_solver.py --px 128 --out phi_128.npz

输出: 保存 npz 文件，包含网格坐标 (x,y,z) 和解 phi（包含边界点，shape=(px,px,px)）
"""

import argparse
import numpy as np
from scipy.fftpack import dst, idst
import time
import matplotlib.pyplot as plt


def dst3(a):
    # 3D DST-I along each axis (apply 1D DST type=1 along axes 0,1,2)
    a = dst(a, type=1, axis=0)
    a = dst(a, type=1, axis=1)
    a = dst(a, type=1, axis=2)
    return a


def idst3(a):
    # inverse DST-I (apply idst type=1 along each axis)
    a = idst(a, type=1, axis=0)
    a = idst(a, type=1, axis=1)
    a = idst(a, type=1, axis=2)
    return a


def solve_poisson_dirichlet(rho_func, px=128):
    """
    使用 DST-I 在立方体 [-1,1]^3, Dirichlet phi=0 上求解 ∇^2 phi = -rho。
    假设网格点数量为 px 在每个维度（包含边界）。

    返回: x,y,z (1D arrays length px), phi (3D array shape (px,px,px))
    """
    # grid including boundaries
    x = np.linspace(-1.0, 1.0, px)
    y = x.copy()
    z = x.copy()
    h = x[1] - x[0]

    # interior grid size
    n = px - 2
    if n <= 0:
        raise ValueError("px must be >= 3")

    # interior coordinates
    xi = x[1:-1]
    yi = y[1:-1]
    zi = z[1:-1]

    # build rho on interior points
    X = xi[:, None, None]
    Y = yi[None, :, None]
    Z = zi[None, None, :]
    rho_int = rho_func(X, Y, Z)

    # RHS for discrete Poisson: f = -rho
    f = -rho_int

    # Apply DST-I to f (3D)
    t0 = time.time()
    f_hat = dst3(f)

    # eigenvalues for 1D second difference with Dirichlet BC
    k = np.arange(1, n + 1)
    lam = 2.0 * (1.0 - np.cos(np.pi * k / (n + 1.0)))
    # build 3D sum of eigenvalues (shape (n,n,n))
    lam3 = (lam[:, None, None] + lam[None, :, None] + lam[None, None, :])

    # divide by eigenvalues (note Laplacian discretized as (1/h^2)*...), so include h^2
    phi_hat = f_hat / (lam3 / (h * h))

    # inverse transform
    phi_int = idst3(phi_hat)
    # normalization: DST-I forward+inverse yields factor 2*(n+1) per axis when using scipy.fftpack.dst/idst
    # After three inverse transforms we need to divide by (2*(n+1))**3
    norm = (2.0 * (n + 1)) ** 3
    phi_int = phi_int / norm
    t1 = time.time()

    # assemble full phi with boundary zeros
    phi = np.zeros((px, px, px), dtype=phi_int.dtype)
    phi[1:-1, 1:-1, 1:-1] = phi_int

    return x, y, z, phi, (t1 - t0)


def rho_func(X, Y, Z):
    return 100.0 * X * Y * (Z ** 2)


def main():
    parser = argparse.ArgumentParser(description="Solve Poisson in [-1,1]^3 with Dirichlet phi=0")
    parser.add_argument("--px", type=int, default=128, help="number of grid points per axis (including boundaries)")
    parser.add_argument("--out", type=str, default="phi.npz", help="output npz filename")
    parser.add_argument("--save-slice", action="store_true", help="also save one central z-slice as txt for quick inspection")
    args = parser.parse_args()

    px = args.px
    print(f"Solving Poisson on [-1,1]^3 with px={px} (total points {px**3})")

    start = time.time()
    x, y, z, phi, t_dst = solve_poisson_dirichlet(rho_func, px=px)
    total = time.time() - start

    print(f"DST-based solve time (transforms + ops): {t_dst:.3f} s, total elapsed: {total:.3f} s")
    print(f"phi stats: min={phi.min():.6e}, max={phi.max():.6e}, mean={phi.mean():.6e}")

    # save
    np.savez_compressed(args.out, x=x, y=y, z=z, phi=phi)
    print(f"Saved solution to {args.out}")

    if args.save_slice:
        # central z index
        iz = px // 2
        slice_fname = args.out.replace('.npz', f'_z{iz}.txt')
        np.savetxt(slice_fname, phi[:, :, iz])
        print(f"Saved central z-slice to {slice_fname}")

    # 可视化: 在指定三个平面上显示电势剖面（同一色标）
    try:
        # 切面坐标（可按需修改）
        sx, sy, sz = 0.1, 0.1, 0.1

        # 找到最接近指定坐标的索引（包括边界点）
        ix = int(np.argmin(np.abs(x - sx)))
        iy = int(np.argmin(np.abs(y - sy)))
        iz = int(np.argmin(np.abs(z - sz)))

        # 选择用于显示的子数组（包含边界，以方便显示整个域）
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
        # 为滑条和 colorbar 留下空间
        fig.subplots_adjust(left=0.06, right=0.86, bottom=0.18, top=0.95, wspace=0.3)
        cmap = 'RdBu_r'

        # 全区域的 colorbar 范围（最小/最大电势）
        vmin_global = float(np.min(phi))
        vmax_global = float(np.max(phi))

        im0 = axs[0].imshow(slice_x, extent=[y[0], y[-1], z[0], z[-1]], origin='lower', cmap=cmap, vmin=vmin_global, vmax=vmax_global, aspect='auto')
        axs[0].set_xlabel('y')
        axs[0].set_ylabel('z')
        axs[0].set_title(f'x = {sx} (ix={ix})')

        im1 = axs[1].imshow(slice_y, extent=[x[0], x[-1], z[0], z[-1]], origin='lower', cmap=cmap, vmin=vmin_global, vmax=vmax_global, aspect='auto')
        axs[1].set_xlabel('x')
        axs[1].set_ylabel('z')
        axs[1].set_title(f'y = {sy} (iy={iy})')

        im2 = axs[2].imshow(slice_z, extent=[x[0], x[-1], y[0], y[-1]], origin='lower', cmap=cmap, vmin=vmin_global, vmax=vmax_global, aspect='auto')
        axs[2].set_xlabel('x')
        axs[2].set_ylabel('y')
        axs[2].set_title(f'z = {sz} (iz={iz})')

        # colorbar 放在最右侧外部
        cbar_ax = fig.add_axes([0.88, 0.18, 0.02, 0.7])
        cbar = fig.colorbar(im2, cax=cbar_ax)

        # 添加滑条 (Slider)，范围 [-1,1]
        from matplotlib.widgets import Slider
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
    except Exception as e:
        print('Plotting failed:', e)


if __name__ == '__main__':
    main()
