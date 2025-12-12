import torch
import numpy as np
import logging
import sys
import os
from beartype import beartype
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from algorithm.gcastle.trustworthyAI.gcastle.castle.algorithms.gradient.notears.torch.models import MLPModel
# class Constraints_Selection:

#     def __init__(self, use_dir = False,use_active = True,use_inactive = False,use_plus_minus):
from typing import Optional,Literal

from torch import nn

from .partial_order_utils import get_transitive_reduction, get_maximal_paths, adjacency_to_digraph

from torch.func import vmap
from  torch.func import jacrev
# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from torch import nn,functional as F







class BaseConstraints:
    def penalty(self,model,w) -> float:
        raise NotImplementedError





class ActiveConstraints(BaseConstraints):
    def __init__(self,lamb = 0.01,method="max",threshold=0.6, use_transitive_closure = False,sigma=0.1,alpha = 0.5,beta = 0.5,threshold1 = 0.2,threshold2=0.4,**kwargs):
        self.lamb = lamb
        self.method = method
        self.active_threhold = threshold
        self.name = "active"
        self.use_transitive_closure = use_transitive_closure
        self.alpha = alpha
        self.sigma = sigma
        self.beta = beta  
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        
        # 记录约束初始化信息
        logging.info(f"[Constraint Init] ActiveConstraints: lamb={self.lamb}, method={self.method}, threshold={self.active_threhold}")

   
    def penalty(self,model:MLPModel,w,**kwargs) -> float:
        # 获取模型所在设备
        device = next(model.parameters(), None)
        device = device.device if device is not None else torch.device('cpu')
        
        # 确保 w 在正确的设备上
        if isinstance(w, np.ndarray):
            w = torch.tensor(w, device=device, dtype=torch.float32)
        else:
            w = w.to(device)
        
        tmp = 0
        arr = None
        
        if self.use_transitive_closure:
            W = model.get_w()
            assert W.dim() == 2 and W.size(0) == W.size(1), "W 必须是方阵"
            d = W.size(0)
            device = W.device
            dtype = W.dtype
            m = d
            B = W 
            I = torch.eye(d, device=device, dtype=dtype)
            M = I + self.alpha * B
            def matrix_power_fast(M, m: int):
                result = torch.eye(d, device=device, dtype=dtype)
                base = M
                exp = m
                while exp > 0:
                    if exp & 1:
                        result = result @ base
                    base = base @ base
                    exp >>= 1
                return result
            C = matrix_power_fast(M, m)   # C = (I + alpha B)^m

            # 去掉 0 阶项 I，只保留长度 >=1 的“可达性”
            closure_soft = C - I  
            arr =  closure_soft         

   

        else:
            arr = model.get_w()


       
        if self.method == "max":
            # print(model.get_w().shape , self.active_constraints.shape)
            # print(model.get_w() * self.active_constraints)
            tmp = torch.sum(torch.clamp(self.active_threhold * w - arr *  w,0))
        elif self.method == "softplus":
            diff = self.active_threhold - torch.abs(arr) # 假设 arr 可能有负值，建议取 abs
            tmp = torch.sum(w * F.softplus(diff))
        elif self.method == "potential":
            margin = self.threshold1 - arr            # m = tau - a
            amp = F.softplus(margin)                    # s(m) >= 0
            scaled = 1.0 - torch.exp(-  amp / (self.sigma + 1e-12))  # u(m)
            tmp = torch.sum(w * scaled)
        elif self.method == "repulsion":
            tmp = torch.sum(w * (self.beta/(arr + 1e-6)))
        elif self.method == "swish":
            margin = self.threshold2 - arr              # >0：低于阈值，需要惩罚
            gate   = torch.sigmoid(self.beta * margin)  # 平滑 gate，控制过渡锐度
            amp    = F.softplus(margin)                 # >=0，m>0 时 ~ m，m<0 时 ~ 0

            penalty = w * amp * gate
            tmp = torch.sum(penalty)

        else:
            raise ValueError("active_method must be 'max' ")
        


        return self.lamb * tmp


import logging
import numpy as np
import torch
import torch.nn.functional as F

import logging
import math
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F


class InactiveConstraints(BaseConstraints):
    """
    无边先验约束（inactive / no-edge constraints）

    约定:
      - model.get_w(): 返回 [d, d] 非负矩阵 W_pred >= 0，表示边强度
      - penalty(...) 里传入的 w: inactive mask，同形状 [d, d]，
        w_ij = 1 表示我们先验认为 i->j 不应该有边（无边约束）
        w_ij = 0 表示该位置不加无边先验

    惩罚目标:
      - 在 w_ij = 1 的位置上：
          W_pred_ij <= tau 时，轻罚或不罚；
          W_pred_ij > tau 时，惩罚随 (W_pred_ij - tau) 增长，且可以设计成
          比线性更快（如二次）、或势能型（饱和）的形式。

    支持的 method:
      - "baseline" : 直接对禁止边位置做 L1 惩罚，penalty = W_pred
      - "max"      : ReLU 风格 hinge，penalty = max(W_pred - tau, 0)
      - "quad"     : 平滑二次 hinge，penalty ≈ max(W_pred - tau, 0)^2
      - "potential": 势能型，penalty ≈ 1 - exp(-max(W_pred - tau, 0)^2 / sigma)

    同时支持 τ / β / λ 的自适应更新（可通过 adapt_* 开关控制）。
    """

    def __init__(self,
                 lamb: float = 0.01,
                 method: str = "quad",
                 # ---- 初始超参（静态起点） ----
                 tau: float = 0.05,       # 初始阈值: W <= tau 近似不罚
                 beta: float = 20.0,      # 初始 softplus/sigmoid 锐度
                 sigma: float = 1e-3,     # potential 势能 scale

                 # ---- 自适应开关 ----
                 adapt_tau: bool = True,      # 是否让 tau 随 W 自适应
                 adapt_beta: bool = True,     # 是否让 beta 随 margin 标准差自适应
                 adapt_lamb: bool = False,    # 是否让 lamb 随违约率自适应

                 # ---- τ 自适应相关 ----
                 tau_quantile: float = 0.7,   # 使用禁止边权重的 q 分位数来估计 tau
                 tau_min: float = 0.0,
                 tau_max: float = 0.5,
                 tau_ema: float = 0.9,        # τ 的指数滑动平均系数

                 # ---- β 自适应相关 ----
                 beta_target_c: float = 3.0,  # 希望 beta * margin 的典型尺度 ~ [-c, c]
                 beta_min: float = 1.0,
                 beta_max: float = 100.0,
                 beta_ema: float = 0.9,       # β 的指数滑动平均系数

                 # ---- λ 自适应相关 ----
                 r_target: float = 0.1,       # 目标违约率 (margin>0 的比例)
                 lamb_min: float = 1e-4,
                 lamb_max: float = 1.0,
                 lamb_eta: float = 0.5,       # λ 更新步长系数

                 name: str = "inactive",
                 **kwargs):
        super().__init__()
        self.lamb = float(lamb)
        self.method = method

        # 当前（可变）参数
        self.tau = float(tau)
        self.beta = float(beta)
        self.sigma = float(sigma)

        # 开关
        self.adapt_tau = adapt_tau
        self.adapt_beta = adapt_beta
        self.adapt_lamb = adapt_lamb

        # τ 相关
        self.tau_quantile = float(tau_quantile)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.tau_ema = float(tau_ema)

        # β 相关
        self.beta_target_c = float(beta_target_c)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.beta_ema = float(beta_ema)

        # λ 相关
        self.r_target = float(r_target)
        self.lamb_min = float(lamb_min)
        self.lamb_max = float(lamb_max)
        self.lamb_eta = float(lamb_eta)

        self.name = name

        logging.info(
            "[Constraint Init] InactiveConstraints: "
            f"lamb={self.lamb}, method={self.method}, "
            f"tau={self.tau}, beta={self.beta}, sigma={self.sigma}, "
            f"adapt_tau={self.adapt_tau}, adapt_beta={self.adapt_beta}, adapt_lamb={self.adapt_lamb}"
        )

    def _get_device(self, model) -> torch.device:
        p = next(model.parameters(), None)
        if p is not None:
            return p.device
        return torch.device("cpu")

    def _to_tensor(self, arr: Union[np.ndarray, torch.Tensor],
                   device: torch.device,
                   dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if isinstance(arr, np.ndarray):
            return torch.tensor(arr, device=device, dtype=dtype)
        return arr.to(device=device, dtype=dtype)

    def _update_tau_beta_lamb(self,
                              W_forbid: torch.Tensor,
                              margin: torch.Tensor,
                              mask: torch.Tensor) -> None:
        """
        根据当前 W / margin / mask 自适应更新 tau / beta / lamb（原地修改）。
        所有更新都放在 no_grad 块中，不进入 autograd 图。
        """
        with torch.no_grad():
            # 只看有 inactive 先验（mask>0）的条目
            mask_pos = (mask > 0)
            if not mask_pos.any():
                return

            vals = W_forbid[mask_pos]      # 当前禁止边上的 W
            mvals = margin[mask_pos]      # 当前禁止边上的 margin

            # 1) 自适应 τ：用禁止边权重的分位数来估计
            if self.adapt_tau and vals.numel() > 0:
                try:
                    q = torch.quantile(vals, self.tau_quantile).item()
                except AttributeError:
                    # 如果 PyTorch 版本没有 quantile，可 fallback 到 np.percentile
                    q = float(np.percentile(
                        vals.detach().cpu().numpy(),
                        self.tau_quantile * 100.0
                    ))
                tau_new = max(self.tau_min, min(self.tau_max, q))
                # EMA 更新，避免剧烈抖动
                self.tau = self.tau_ema * self.tau + (1.0 - self.tau_ema) * tau_new

            # 重新计算 margin（因为 tau 已更新）
            margin_new = W_forbid - self.tau
            mvals = margin_new[mask_pos]

            # 2) 自适应 β：让 beta * margin 的尺度处在大致 [-c, c]
            if self.adapt_beta and mvals.numel() > 0:
                std = mvals.std().item()
                if std > 1e-8:
                    beta_raw = self.beta_target_c / (std + 1e-8)
                    beta_raw = max(self.beta_min, min(self.beta_max, beta_raw))
                    self.beta = self.beta_ema * self.beta + (1.0 - self.beta_ema) * beta_raw

            # 3) 自适应 λ：根据违约率 (margin>0) 调节
            if self.adapt_lamb and mvals.numel() > 0:
                total = float(mvals.numel())
                viol = float((mvals > 0).sum().item())
                r_t = viol / (total + 1e-8)      # 当前违约率
                # log 形式的负反馈控制
                factor = math.exp(self.lamb_eta * (r_t - self.r_target))
                lamb_new = self.lamb * factor
                lamb_new = max(self.lamb_min, min(self.lamb_max, lamb_new))
                self.lamb = lamb_new

    def penalty(self,
                model,
                w: Union[np.ndarray, torch.Tensor],
                **kwargs) -> torch.Tensor:
        """
        计算无边约束惩罚项。

        参数:
          model : 带 get_w() 方法的模型
          w     : inactive mask，形状与 get_w() 相同，
                  1 表示该位置施加无边约束，0 表示不加。
        返回:
          标量张量 (torch.Tensor)，等于 lamb * Σ penalty_ij
        """
        device = self._get_device(model)

        # inactive mask
        mask = self._to_tensor(w, device=device, dtype=torch.float32)

        # 预测的边权重矩阵（已经是非负）
        W_pred = model.get_w().to(device=device)  # [d, d]

        # 只关心有无边先验的位置
        W_forbid = W_pred * mask

        # 初步计算 margin（后面自适应 tau 时会更新）
        margin = W_forbid - self.tau

        # ---- 自适应更新 tau / beta / lamb ----
        self._update_tau_beta_lamb(W_forbid=W_forbid,
                                   margin=margin,
                                   mask=mask)

        # tau / beta / lamb 已经可能被更新，重新计算 margin
        margin = W_forbid - self.tau

        # ---- 根据 method 计算具体的 penalty_mat ----
        if self.method == "baseline":
            # 等价于原版: mask * W_pred
            penalty_mat = W_forbid

        elif self.method == "max":
            # ReLU 风格: max(W - tau, 0)
            penalty_mat = torch.clamp(margin, min=0.0)

        elif self.method == "quad":
            # 二次 hinge + softplus 平滑:
            # s ≈ max(margin, 0)，penalty ≈ s^2
            s = F.softplus(self.beta * margin) / self.beta
            penalty_mat = s * s

        elif self.method == "potential":
            # 势能型: 1 - exp(-s^2 / sigma), s≈max(margin,0)
            s = F.softplus(self.beta * margin) / self.beta
            penalty_mat = 1.0 - torch.exp(-(s * s) / (self.sigma + 1e-8))

        else:
            raise ValueError(f"[InactiveConstraints] Unknown method: {self.method}")

        loss = self.lamb * penalty_mat.sum()
        return loss



class OrientationConstraints(BaseConstraints):
    def __init__(self, l2_lambda=0.01,l1_lambda = 0.01, alpha=3.0, use_cumulative=True, **kwargs):
        """
        方向约束：通过极大路径施加顺序约束
        
        Parameters
        ----------
        l2_lambda : float
            惩罚系数
        alpha : float or str
            路径边的权重值
            - 如果是数值：直接使用该值（推荐 3.0~5.0）
            - 如果是字符串 'max'/'mean'：动态计算（不推荐，训练初期太小）
        use_cumulative : bool
            是否使用累积路径模式（模仿 partial_notears）
            - True: 路径逐步累积到矩阵上（推荐）
            - False: 每条路径独立计算
        """
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.name = "orient"
        self.alpha = alpha
        self.use_cumulative = use_cumulative
        
        # 记录约束初始化信息
        logging.info(f"[Constraint Init] OrientationConstraints: l1_lambda={self.l1_lambda}, l2_lambda={self.l2_lambda}, "
                    f"alpha={self.alpha}, use_cumulative={self.use_cumulative}")
        
        # 用于记录首次调用信息
        self._first_call = True
        
    def penalty(self, model, w,**kwargs) -> float:
        # 获取模型所在设备
        device = next(model.parameters(), None)
        device = device.device if device is not None else torch.device('cpu')
        
        # 图算法需要在 CPU 上执行（NetworkX 不支持 GPU）
        if isinstance(w, torch.Tensor):
            w_np = w.detach().cpu().numpy()
        else:
            w_np = w
        
        # 在 CPU 上执行图算法
        w_g = adjacency_to_digraph(w_np)
        w_tr = get_transitive_reduction(w_g)
        paths = get_maximal_paths(w_tr)
        # 如果没有路径，返回0
        
        if not paths:
            if self._first_call:
                logging.warning(f"[Constraint Runtime] {self.name}: No paths found in candidate graph!")
                self._first_call = False
            return torch.tensor(0.0, device=device)
        
        # 首次调用时记录路径信息
        if self._first_call:
            logging.info(f"[Constraint Runtime] {self.name}: Found {len(paths)} maximal paths, "
                        f"candidate edges: {int(w_np.sum())}, graph shape: {w_np.shape}")
            self._first_call = False
        
        # 获取模型权重（保持在原设备上，带梯度）
        true_w = model.get_w()
        
        # 处理 alpha 参数
        if isinstance(self.alpha, str):
            if self.alpha.lower() == "max":
                alpha = torch.max(torch.abs(true_w))
            elif self.alpha.lower() == "min":
                alpha = torch.min(torch.abs(true_w[true_w != 0]))  # 排除0
            elif self.alpha.lower() == "mean":
                alpha = torch.mean(torch.abs(true_w[true_w != 0]))
            elif self.alpha.lower() == "sum":
                alpha = torch.sum(torch.abs(true_w))
            else:
                raise ValueError("alpha must be 'max', 'min', 'mean', 'sum', or a number")
            # 确保 alpha 不会太小
            alpha = torch.clamp(alpha, min=1.0)
        elif isinstance(self.alpha, (int, float)):
            alpha = float(self.alpha)
        else:
            raise ValueError("alpha must be a string ('max', 'min', 'mean', 'sum') or a number")
        
        if self.use_cumulative:
            # 累积模式：模仿 partial_notears 的实现
            # 路径逐步累积到矩阵上，后面的路径包含前面所有路径的效果
            h = 0
            A_cumulative = true_w.clone()  # 从当前权重开始累积
            
            for path in paths:
                # 将当前路径的边直接赋值到累积矩阵上（模仿 add_path_to_matrix）
                for u, v in zip(path[:-1], path[1:]):
                    A_cumulative[u, v] = alpha
                
                # 用累积矩阵计算 DAG 约束
                h_val = model.h_func(A_cumulative)
                h += self.l1_lambda *h_val  + self.l2_lambda * h_val * h_val # 线性累加
            
            return  h
        
        else:
            # 独立模式：每条路径单独计算（原始实现）
            sm = 0
            for path in paths:
                # 为这条路径构建掩码
                mask = torch.zeros_like(true_w)
                for u, v in zip(path[:-1], path[1:]):
                    mask[u, v] = alpha
                
                # 路径边直接设为 alpha（赋值而非加法）
                A_with_path = true_w.clone()
                for u, v in zip(path[:-1], path[1:]):
                    A_with_path[u, v] = alpha
                
                h_val = model.h_func(A_with_path)
                sm += self.l1_lambda *h_val  + self.l2_lambda * h_val * h_val  # 线性累加，不用平方
            
            return  sm 

import torch
import torch.nn.functional as F
from torch.func import vmap, jacrev
import numpy as np

class MonoConstraints(BaseConstraints):
    def __init__(self, l1_lambda=0.01, **kwargs):
 
        self.l1_lambda = l1_lambda
        self.name = "mono"

    def _jacobian_batched(self, func, X):
        # func: [B,d] -> [B,d]，返回 J: [B,d,d] with J[b,j,i] = ∂f_j/∂x_i
        def single_f(x1d):  # [d] -> [d]
            return func(x1d.unsqueeze(0)).squeeze(0)
        J_single = jacrev(single_f)     # [d] -> [d,d] (行=输出j, 列=输入i)
        J_batch  = vmap(J_single)       # [B,d] -> [B,d,d]
        return J_batch(X)

    def penalty(self, model, w_adj, **kwargs):
        was_training = getattr(model, "training", False)
        if was_training:
            model.eval()

        def f_pure(X_in):
            return model(X_in)          # [B,d]

        # 1) 雅可比 J: [B,d,d]  (j,i) 顺序
        J = self._jacobian_batched(f_pure, X)

        # 2) 先验 w_adj: [d,d] (i->j)  ——> 对齐到 (j,i)
        w_adj = torch.as_tensor(w_adj, device=J.device, dtype=J.dtype)  # [d,d]
        assert w_adj.dim() == 2 and w_adj.shape[0] == w_adj.shape[1] == J.shape[1]
        wJ = w_adj.T  # [d,d]  (行=输出j, 列=输入i)，与 J 的后两维对齐

        # 3) 掩码（二维即可，会广播到 [B,d,d]）
        inc_mask = (wJ > 0).to(J.dtype)     # 单增：罚负导
        dec_mask = (wJ < 0).to(J.dtype)     # 单减：罚正导

        # 4) 惩罚（逐元素，与 J 广播对齐）
        pen_inc = F.relu(-J) * inc_mask     # max(0, -J_{j,i}) for +1
        pen_dec = F.relu( J) * dec_mask     # max(0,  J_{j,i}) for -1

        denom = (inc_mask + dec_mask).sum().clamp_min(1.0)  # 仅统计被约束的条目数
        penalty = (pen_inc.sum() + pen_dec.sum()) / denom   # 标量

        if was_training:
            model.train()
        return self.l1_lambda * penalty



   








config = {

    "active":{
        "use":True,
        "method":"max",
        "threshold":0.6,
        "lamb":0.01,
        "model":ActiveConstraints
    },
    "inactive":{
        "use":False,
        "lamb":0.01,
        "model":InactiveConstraints
    },
    "plus_minus":{
        "use":False,
        "lamb":0.01,
       
    },
    "orient":{
        "use":False,
        "l2_lambda":0.05,        # 惩罚系数（根据实验调整）
        "alpha":3.0,             # 路径边权重（推荐 3.0~5.0）
        "use_cumulative":True,   # 使用累积模式（推荐）
        "model":OrientationConstraints
    }
}
