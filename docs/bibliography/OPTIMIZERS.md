# Advanced Optimization Algorithms in Transformer Architectures: A Comprehensive Analysis

## The Evolution of Optimization in Neural Networks

The optimization of deep neural networks, particularly highly parameterized Transformer architectures such as Bidirectional Encoder Representations from Transformers (BERT), represents one of the most mathematically complex challenges in modern computational learning theory. The loss landscapes of large language models are characterized by severe non-convexity, saddle points, and block heterogeneity. In the context of block heterogeneity, the Hessian spectrum across different parameter blocks varies dramatically, creating optimization pathways where certain dimensions are exponentially sharper or flatter than others.

Historically, the transition from Convolutional Neural Networks (CNNs) to Transformers necessitated a fundamental shift in optimization paradigms. While standard Stochastic Gradient Descent with momentum dominated vision tasks, it consistently failed to navigate the heterogeneous curvature of attention mechanisms efficiently. Adaptive algorithms, primarily Adam and its decoupled variant AdamW, emerged as the dominant baseline by dynamically scaling the learning rate element-wise based on the first and second moments of the gradients. As the parameter count of models scaled, the memory overhead of maintaining dense momentum and variance tensors drove the field toward novel optimization paradigms.

Recent advancements have fractured the optimization landscape into several distinct theoretical trajectories. These include memory-efficient low-rank approximations (Adafactor, GaLore), second-order Hessian and Matrix approximations (Sophia, Shampoo, SOAP), learning-rate-free distance estimators (Prodigy, Schedule-Free), variance-reduction techniques (MARS, Adan), and matrix-oriented geometric orthogonalizers (Muon, Turbo-Muon). This exhaustive report provides a theoretical and practical analysis of these algorithms, detailing their mathematical Formulations, structural properties, pros and cons, official DOIs, and Pure PyTorch Implementations strictly utilizing native tensor operations.

## 1. Standard Baseline and Adaptive Optimizers

### SGD with Momentum

Features: The original workhorse. Updates parameters using the gradient scaled by a fixed learning rate, plus a momentum term (exponential moving average of past gradients) to accelerate convergence and dampen oscillations.

Pros,Cons
Minimal memory (only 1 buffer for momentum),Requires very careful LR schedule tuning
Excellent generalization in many settings,Slow convergence on NLP/transformer tasks
Well-understood theory,Not adaptive — same LR for all parameters

Paper / DOI: Polyak (1964), Sutskever et al. (2013) / Classical method; no single modern reference.

Mathematical Formulation:Given objective function $f_t(\theta)$ and gradient $g_t = \nabla_\theta f_t(\theta_t)$:$$m_t = \beta m_{t-1} + g_t$$$$\theta_{t+1} = \theta_t - \eta_t m_t$$

Pure PyTorch Implementation:
```Python
import torch
from torch.optim import Optimizer

class SGDMomentum(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.9, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                if group['weight_decay']!= 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                    
                state = self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p)
                    
                buf = state['momentum_buffer']
                buf.mul_(group['momentum']).add_(grad)
                p.add_(buf, alpha=-group['lr'])
```
### Adam and AdamW

Features: Combines first-moment (mean) and second-moment (uncentered variance) exponential moving averages of gradients. Uses bias correction to offset the zero-initialization of these estimates. AdamW fundamentally decouples weight decay from the gradient update mechanism, ensuring that regularization penalizes the weights directly.

Pros,Cons
"Fast convergence, robust to hyperparameters",Known theoretical non-convergence issues with certain β2​
Adaptive per-parameter LR,Can converge to sharp minima leading to worse generalization
De facto standard for fine-tuning,Requires 2× parameter memory for optimizer states

Paper / DOI: Kingma & Ba (2014) [10.48550/arXiv.1412.6980] / Loshchilov & Hutter (2017) [10.48550/arXiv.1711.05101]

Mathematical Formulation:$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$$$\theta_{t+1} = \theta_t - \eta_t \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)$$

Pure PyTorch Implementation (AdamW):
```Python
import torch
from torch.optim import Optimizer

class AdamW_BERT(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                
                p.mul_(1.0 - group['lr'] * group['weight_decay'])
                
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                
                exp_avg.lerp_(p.grad, 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
                
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                step_size = group['lr'] / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)) + group['eps']
                p.addcdiv_(exp_avg, denom, value=-step_size)
```
### RAdam (Rectified Adam)

Features: Identifies that Adam's adaptive learning rate has problematically large variance in early training steps (explaining why warmup is needed). RAdam introduces an automatic variance rectification term, dynamically switching between SGD and Adam behavior depending on the estimated variance.

Pros,Cons
Removes the need for manual warmup tuning,Marginal gains over well-tuned Adam+warmup
More robust out-of-the-box,Same memory footprint as Adam
Drop-in replacement for Adam,Not widely adopted in large-scale pre-training

Paper / DOI: Liu et al. (2020) [10.48550/arXiv.1908.03265]

Mathematical Formulation:Calculate the maximum length of the approximated SMA: $\rho_{\infty} = \frac{2}{1-\beta_2} - 1$.At step $t$, compute the degree of freedom: $\rho_t = \rho_{\infty} - \frac{2t\beta_2^t}{1-\beta_2^t}$.If $\rho_t > 4$, compute the variance rectification term $r_t$ and update adaptively:$$r_t = \sqrt{\frac{(\rho_t - 4)(\rho_t - 2)\rho_{\infty}}{(\rho_{\infty} - 4)(\rho_{\infty} - 2)\rho_t}}$$$$\theta_{t+1} = \theta_t - \eta_t r_t \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$If $\rho_t \leq 4$, behave like SGD: $\theta_{t+1} = \theta_t - \eta_t \hat{m}_t$.

Pure PyTorch Implementation:
```Python
import torch, math
from torch.optim import Optimizer

class RAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            rho_inf = 2.0 / (1.0 - beta2) - 1.0
            
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                if group['weight_decay']!= 0: p.mul_(1 - group['lr'] * group['weight_decay'])
                
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    
                state['step'] += 1
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                
                exp_avg.lerp_(grad, 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                m_hat = exp_avg / bias_correction1
                
                rho_t = rho_inf - 2.0 * state['step'] * (beta2 ** state['step']) / bias_correction2
                
                if rho_t > 4.0:
                    rect = math.sqrt(((rho_t - 4) * (rho_t - 2) * rho_inf) / ((rho_inf - 4) * (rho_inf - 2) * rho_t))
                    v_hat = (exp_avg_sq / bias_correction2).sqrt().add_(group['eps'])
                    p.add_(m_hat / v_hat, alpha=-group['lr'] * rect)
                else:
                    p.add_(m_hat, alpha=-group['lr'])
```
## 2. Advanced Momentum and Variance Reduction (2024-2025)

### Adan (Adaptive Nesterov Momentum)

Features: Reformulates Nesterov acceleration to avoid the extra gradient computation at the extrapolation point. Uses this Nesterov Momentum Estimation (NME) for both first- and second-order moment estimates. Works well across diverse architectures (CNNs, Transformers, GANs).

Pros,Cons
Consistently faster convergence across architectures,3 momentum buffers → higher memory than Adam
Avoids extra forward/backward pass of Nesterov,Relatively new; not yet standard
Strong on both vision and language tasks,Can be unstable near convergence (mitigated by AdanB)

Paper / DOI: Xie et al. (2022) [10.48550/arXiv.2208.06677]

Mathematical Formulation:$$m_k = (1 - \beta_1) m_{k-1} + \beta_1 g_k$$$$v_k = (1 - \beta_2) v_{k-1} + \beta_2 (g_k - g_{k-1})$$$$n_k = (1 - \beta_3) n_{k-1} + \beta_3 [g_k + (1 - \beta_1)(g_k - g_{k-1})]^2$$$$\theta_{k+1} = \theta_k - \eta \frac{m_k + (1 - \beta_1) v_k}{\sqrt{n_k} + \epsilon}$$

Pure PyTorch Implementation:

```Python
import torch
from torch.optim import Optimizer

class Adan(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.98, 0.92, 0.99), eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta1, beta2, beta3 = group['betas']
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                if group['weight_decay'] > 0: p.mul_(1 - group['lr'] * group['weight_decay'])
                
                state = self.state[p]
                if len(state) == 0:
                    state['m'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)
                    state['n'] = torch.zeros_like(p)
                    state['prev_g'] = grad.clone()
                    
                m, v, n, prev_g = state['m'], state['v'], state['n'], state['prev_g']
                
                grad_diff = grad - prev_g
                m.mul_(1 - beta1).add_(grad, alpha=beta1)
                v.mul_(1 - beta2).add_(grad_diff, alpha=beta2)
                
                g_nesterov = grad + grad_diff.mul(1 - beta1)
                n.mul_(1 - beta3).addcmul_(g_nesterov, g_nesterov, value=beta3)
                
                update = m + v.mul(1 - beta1)
                denom = n.sqrt().add_(group['eps'])
                
                p.addcdiv_(update, denom, value=-group['lr'])
                prev_g.copy_(grad)
```
### ADOPT (Modified Adam with Optimal Convergence)

Features: Fixes Adam's theoretical non-convergence by (1) removing the current gradient from the second-moment estimate and (2) reordering the momentum update and normalization. Achieves the optimal $O(1/\sqrt{T})$ rate with any choice of $\beta_2$, without bounded noise assumptions.

Pros,Cons
Provably converges with any β2​ (unlike Adam),Marginal practical gains in some tasks
Drop-in replacement for Adam,Very recent (NeurIPS 2024)
"Superior across vision, NLP, RL, generative tasks",Same memory as Adam

Paper / DOI: Taniguchi et al. (2024) [10.48550/arXiv.2411.02853]

Mathematical Formulation:$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) \frac{g_t}{\sqrt{v_{t-1}} + \epsilon}$$$$\theta_{t+1} = \theta_t - \eta_t m_t$$

Pure PyTorch Implementation:

```Python
import torch
from torch.optim import Optimizer

class ADOPT(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                if group['weight_decay'] > 0: p.mul_(1 - group['lr'] * group['weight_decay'])
                
                state = self.state[p]
                if len(state) == 0:
                    state['m'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)
                    
                m, v = state['m'], state['v']
                
                # ADOPT uses v_{t-1} for the denominator (clamped/maxed for t=1 stability)
                v_prev = torch.max(v, torch.tensor(group['eps'], device=p.device))
                denom = v_prev.sqrt().add_(group['eps'])
                
                # Update first moment using v_{t-1}
                m.mul_(beta1).addcdiv_(grad, denom, value=1 - beta1)
                
                # Update second moment with current gradient
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                p.add_(m, alpha=-group['lr'])
```
### AdEMAMix (Mixture of Two EMAs)

Features: Replaces Adam's single EMA of gradients with a mixture of two EMAs — one fast-decaying (recent gradients) and one slow-decaying (long memory). Demonstrates that gradients remain relevant for tens of thousands of steps.

Pros,Cons
1.3B model on 101B tokens ≈ AdamW on 197B tokens (+95%),Introduces extra hyperparameters (two decay rates + mixing)
Significantly slows model forgetting during training,Needs a scheduler for the slow EMA weight
Simple modification to Adam,Relatively new; less tested beyond LMs


Paper / DOI: Pagliardini et al. (2024) [10.48550/arXiv.2409.03137]

Mathematical Formulation:$$m_{1,t} = \beta_1 m_{1,t-1} + (1 - \beta_1) g_t$$$$m_{2,t} = \beta_3 m_{2,t-1} + (1 - \beta_3) g_t$$$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$$$m_t = m_{1,t} + \alpha m_{2,t}$$$$\theta_{t+1} = \theta_t - \eta_t \frac{m_t}{\sqrt{v_t} + \epsilon}$$

Pure PyTorch Implementation:

```Python
import torch
from torch.optim import Optimizer

class AdEMAMix(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999, 0.9999), alpha=8.0, eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, alpha=alpha, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta1, beta2, beta3 = group['betas']
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                if group['weight_decay'] > 0: p.mul_(1 - group['lr'] * group['weight_decay'])
                
                state = self.state[p]
                if len(state) == 0:
                    state['m1'] = torch.zeros_like(p)
                    state['m2'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)
                    
                m1, m2, v = state['m1'], state['m2'], state['v']
                
                m1.mul_(beta1).add_(grad, alpha=1 - beta1)
                m2.mul_(beta3).add_(grad, alpha=1 - beta3)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                m = m1 + group['alpha'] * m2
                denom = v.sqrt().add_(group['eps'])
                
                p.addcdiv_(m, denom, value=-group['lr'])
```
### MARS (Make Variance Reduction Shine)

Features: A unified framework that reconciles preconditioned gradient methods with variance reduction via a scaled stochastic recursive momentum. Offers instances based on AdamW, Lion, and Shampoo.

Pros,Cons
Brings variance reduction to large-model training (a first),Extra memory for variance-reduced gradient estimates
Consistently outperforms AdamW by a large margin on GPT-2,More complex implementation
Unified framework with multiple instantiations,Scaling behavior still being studied

Paper / DOI: Yuan et al. (2024) [10.48550/arXiv.2411.10438]

Mathematical Formulation (MARS-AdamW variant):$$c_t = g_t + \gamma_t (c_{t-1} - g_{t-1})$$$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) c_t$$$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) c_t^2$$$$\theta_{t+1} = \theta_t - \eta_t \frac{m_t}{\sqrt{v_t} + \epsilon}$$

Pure PyTorch Implementation:

```Python
import torch
from torch.optim import Optimizer

class MARS_AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), gamma=0.025, eps=1e-8, weight_decay=1e-2):
        defaults = dict(lr=lr, betas=betas, gamma=gamma, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            gamma = group['gamma']
            
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                if group['weight_decay'] > 0: p.mul_(1 - group['lr'] * group['weight_decay'])
                
                state = self.state[p]
                if len(state) == 0:
                    state['m'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)
                    state['prev_g'] = grad.clone()
                    state['c'] = grad.clone()
                    
                m, v, prev_g, c = state['m'], state['v'], state['prev_g'], state['c']
                
                # Variance reduction step
                c.copy_(grad + gamma * (c - prev_g))
                
                m.mul_(beta1).add_(c, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(c, c, value=1 - beta2)
                
                denom = v.sqrt().add_(group['eps'])
                p.addcdiv_(m, denom, value=-group['lr'])
                
                prev_g.copy_(grad)
```
### Cautious Optimizers (C-AdamW, C-Lion)

Features: A one-line modification to any momentum-based optimizer: the update is masked so that only components where the momentum and gradient agree in sign are applied. Preserves convergence guarantees while significantly speeding up training.

Pros,Cons
Up to 1.47× speed-up on Llama/MAE pretraining,Marginal effect in some tasks
Trivial to implement (one line of code),Masking can slow down early training slightly
"Works on top of AdamW, Lion, or any momentum method",Very recent; more validation needed

Paper / DOI: Liang et al. (2024) [10.48550/arXiv.2411.16085]

Mathematical Formulation:$$\text{mask} = \mathbb{I}(m_t \odot g_t > 0)$$$$\theta_{t+1} = \theta_t - \eta_t (\text{BaseUpdate}_t \odot \text{mask})$$

Pure PyTorch Implementation (C-AdamW):

```Python
import torch
from torch.optim import Optimizer

class Cautious_AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                if group['weight_decay'] > 0: p.mul_(1 - group['lr'] * group['weight_decay'])
                
                state = self.state[p]
                if len(state) == 0:
                    state['m'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)
                    
                m, v = state['m'], state['v']
                
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Cautious Masking
                mask = (m * grad > 0).to(grad.dtype)
                
                denom = v.sqrt().add_(group['eps'])
                update = (m / denom) * mask
                
                p.add_(update, alpha=-group['lr'])
```
## 3. Large-Batch, Memory-Efficient, and Parameter-Free Optimizers

### LAMB

Features:
The necessity for training LLMs on massive datasets has driven the adoption of extreme batch sizes. LAMB resolves generalization degradation by combining an Adam-style momentum tracker with a layer-wise trust ratio derived from Layer-wise Adaptive Rate Scaling (LARS).

Mathematical Formulation:$$\phi(||x^{(i)}_t||) = \frac{||x^{(i)}_t||}{||r^{(i)}_t||}$$$$x^{(i)}_{t+1} = x^{(i)}_t - \eta_t \phi(||x^{(i)}_t||) \frac{r^{(i)}_t}{||r^{(i)}_t||}$$

Pure PyTorch Implementation:

```Python
import torch
from torch.optim import Optimizer

class LAMB_BERT(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                state = self.state[p]
                
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                adam_step = exp_avg / (exp_avg_sq.sqrt() + group['eps'])
                if group['weight_decay']!= 0: adam_step.add_(p, alpha=group['weight_decay'])
                    
                weight_norm = torch.norm(p).clamp(0, 10.0)
                adam_norm = torch.norm(adam_step)
                trust_ratio = 1.0 if weight_norm == 0 or adam_norm == 0 else weight_norm / adam_norm
                p.add_(adam_step, alpha=-group['lr'] * trust_ratio)
```
### Schedule-Free (AdamW)

Features: Completely removes the need for a learning rate schedule by unifying scheduling and iterate averaging into a single theoretical framework. Introduces no extra hyperparameters.

Pros,Cons
No need to specify total training steps T,"Requires switching between ""train"" and ""eval"" parameter views"
State-of-the-art across convex to large-scale DL,Slightly different API than standard optimizers
No extra hyperparameters,Best results still need good base LR

Paper / DOI: Defazio et al. (2024) [10.48550/arXiv.2405.15682]

Mathematical Formulation:$$z_{t+1} = z_t - \eta \nabla f(y_t)$$$$x_{t+1} = \left(1 - \frac{1}{t+1}\right) x_t + \frac{1}{t+1} z_{t+1}$$$$y_{t+1} = (1 - \beta) z_{t+1} + \beta x_{t+1}$$

Pure PyTorch Implementation:

```Python
import torch
from torch.optim import Optimizer

class ScheduleFree_AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['z'] = p.clone()
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    
                state['step'] += 1
                t = state['step']
                z, v = state['z'], state['exp_avg_sq']
                
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = v.sqrt().add_(group['eps'])
                
                if group['weight_decay'] > 0: z.mul_(1 - group['lr'] * group['weight_decay'])
                z.addcdiv_(grad, denom, value=-group['lr'])
                
                # Simplified iterate averaging approximation for parameter p (y_t)
                c = 1.0 / (t + 1)
                p.data.mul_(1 - c).add_(z, alpha=c)
```
(Note: Adafactor, GaLore, and Prodigy maintain their respective implementations detailed previously within this ecosystem, optimizing memory via matrix factorization and eliminating LR sweeps via distance estimation).

## 4. Second-Order, Geometric, and Orthogonality Optimizers
### Shampoo

Features: A structured second-order optimizer that maintains Kronecker-factored preconditioner matrices (one per dimension of each parameter tensor). Approximates full-matrix AdaGrad with manageable cost.

Pros,Cons
Strong theoretical convergence guarantees,High memory for preconditioner matrices
Captures cross-parameter correlations,Requires periodic expensive eigendecomposition
Proven at Google scale,Complex distributed implementation

Paper / DOI: Gupta et al. (2018) [10.48550/arXiv.1802.09568]

Mathematical Formulation:For a matrix gradient $G_t \in \mathbb{R}^{n \times m}$:$$L_t = L_{t-1} + G_t G_t^T, \quad R_t = R_{t-1} + G_t^T G_t$$$$W_{t+1} = W_t - \eta L_t^{-1/4} G_t R_t^{-1/4}$$

Pure PyTorch Implementation:

```Python
import torch
from torch.optim import Optimizer

class Shampoo(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.0, weight_decay=0.0, update_freq=10):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, update_freq=update_freq)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or p.ndim!= 2: continue
                grad = p.grad
                if group['weight_decay'] > 0: grad = grad + group['weight_decay'] * p
                
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['L'] = torch.eye(p.size(0), device=p.device) * 1e-4
                    state = torch.eye(p.size(1), device=p.device) * 1e-4
                    state['L_inv'] = torch.eye(p.size(0), device=p.device)
                    state = torch.eye(p.size(1), device=p.device)
                    
                state['step'] += 1
                L, R = state['L'], state
                L.addmm_(grad, grad.T)
                R.addmm_(grad.T, grad)
                
                if state['step'] % group['update_freq'] == 1:
                    def compute_inv_root(mat, power=-0.25):
                        evals, evecs = torch.linalg.eigh(mat)
                        return evecs @ torch.diag(evals.clamp(min=1e-6)**power) @ evecs.T
                    state['L_inv'] = compute_inv_root(L)
                    state = compute_inv_root(R)
                    
                precond_grad = state['L_inv'] @ grad @ state
                p.add_(precond_grad, alpha=-group['lr'])
```
### SOAP (Shampoo with Adam in Preconditioner's Eigenbasis)

Features: A simplified and improved version of Shampoo. Key insight: running Adam/Adafactor in the eigenbasis of Shampoo's preconditioner yields a simpler, faster algorithm.

Pros,Cons
Computationally simpler than Shampoo,Still requires eigendecomposition (less frequently)
Combines benefits of Adam and Shampoo,Higher memory than first-order methods
Strong LLM training results,Relatively new; less battle-tested

Paper / DOI: Vyas et al. (2024) [10.48550/arXiv.2409.11321]

Mathematical Formulation: Eigen-decompose the preconditioners: $L = Q_L \Lambda_L Q_L^T$, $R = Q_R \Lambda_R Q_R^T$.Project the gradient: $G_{rot} = Q_L^T G Q_R$.Run standard Adam tracking on $G_{rot}$ to get $M_{rot}, V_{rot}$.Project the update back: $\Delta = Q_L \left(\frac{M_{rot}}{\sqrt{V_{rot}}}\right) Q_R^T$.

Pure PyTorch Implementation:

```Python
import torch
from torch.optim import Optimizer

class SOAP(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01, precond_freq=10):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, precond_freq=precond_freq)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None or p.ndim!= 2: continue
                grad = p.grad
                if group['weight_decay'] > 0: p.mul_(1 - group['lr'] * group['weight_decay'])
                
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['L'] = torch.eye(p.size(0), device=p.device) * group['eps']
                    state = torch.eye(p.size(1), device=p.device) * group['eps']
                    state['Q_L'] = torch.eye(p.size(0), device=p.device)
                    state = torch.eye(p.size(1), device=p.device)
                    state['m_rot'] = torch.zeros_like(p)
                    state['v_rot'] = torch.zeros_like(p)
                    
                state['step'] += 1
                
                state['L'].addmm_(grad, grad.T)
                state.addmm_(grad.T, grad)
                
                if state['step'] % group['precond_freq'] == 1:
                    _, state['Q_L'] = torch.linalg.eigh(state['L'])
                    _, state = torch.linalg.eigh(state)
                
                Q_L, Q_R = state['Q_L'], state
                grad_rot = Q_L.T @ grad @ Q_R
                
                m, v = state['m_rot'], state['v_rot']
                m.lerp_(grad_rot, 1 - beta1)
                v.mul_(beta2).addcmul_(grad_rot, grad_rot, value=1 - beta2)
                
                update_rot = m / (v.sqrt() + group['eps'])
                update = Q_L @ update_rot @ Q_R.T
                
                p.add_(update, alpha=-group['lr'])
```
(Note: Lion, Sophia, Muon, and Turbo-Muon preserve their strictly sign-based momentum and spectral orthogonalization routines configured in the prior foundational matrix algorithms).

Optimizer,Core Mathematical Mechanism,Math Order,Big O (Time),Space (Memory),Target DOI
SGD w/ Mom.,"Fixed scaling, momentum damping",First-order,O(N),1×,Classical
AdamW,"Decoupled Weight Decay, Scale-Freeness",First-order,O(N),2×,10.48550/arXiv.1711.05101
RAdam,Variance rectification heuristic bounds,First-order,O(N),2×,10.48550/arXiv.1908.03265
Adan,Adaptive Nesterov Momentum (NME),First-order,O(N),3×,10.48550/arXiv.2208.06677
ADOPT,Disconnected gradient from 2nd moment,First-order,O(N),2×,10.48550/arXiv.2411.02853
AdEMAMix,Slow + Fast exponential memory mixing,First-order,O(N),3×,10.48550/arXiv.2409.03137
Schedule-Free,Iterate averaging & online integration,First-order,O(N),2×,10.48550/arXiv.2405.15682
MARS,Preconditioned variance reduction,First-order,O(N),3×,10.48550/arXiv.2411.10438
Cautious,Boolean momentum/gradient sign mask,First-order,O(N),2×,10.48550/arXiv.2411.16085
LAMB,Trust Ratio layer adaptation scaling,First-order,O(N),2×,10.48550/arXiv.1904.00962
Lion,Sign-operator update restriction,First-order,O(N),1×,10.48550/arXiv.2302.06675
Adafactor,Rank-1 I-Divergence Variance Approx.,First-order,O(N),Sublinear,10.48550/arXiv.1804.04235
GaLore,SVD Top-k Gradient Projection,First-order,O(Nr+mr2),O(Nr),10.48550/arXiv.2403.03507
Prodigy,D-Adaptation Golden ratio steps,First-order,O(N),2×,10.48550/arXiv.2306.06101
Sophia,Diagonal Hessian estimation & clipping,Second-order,O(N),2×,10.48550/arXiv.2305.14342
Shampoo,Kronecker-factored preconditioners,Second-order,O(N+d3),High,10.48550/arXiv.1802.09568
SOAP,Eigenbasis projection inside Adam,Second-order,O(N+d3),High,10.48550/arXiv.2409.11321
Muon,5-step Newton-Schulz polynomial,Orthogonal,O(n3) per layer,1×,10.48550/arXiv.2505.23737
Turbo-Muon,"AOL Pre-conditioning, 4-step NS",Orthogonal,O(n3) per layer,1×,10.48550/arXiv.2512.04632

(Note: N represents total parameter count; d, n, m denote matrix dimensions; r denotes matrix rank. High mathematical order optimizers incur steeper Big O time complexities due to polynomial or SVD bottlenecks).

Conclusion
The mathematical landscape of Transformer optimization is fractured between scaling parameter dimensions independently (AdamW, Schedule-Free, ADOPT), extracting curvature representations explicitly (Shampoo, SOAP, Sophia), leveraging gradient trajectory variance bounds (MARS, Cautious, AdEMAMix), and treating weight projections as constrained geometric manifolds (Muon, Turbo-Muon). Navigating this complex hierarchy enables deep learning engineers to pinpoint the exact intersection of memory overhead, FLOP throughput, and convergence bounds tailored to their specific hardware limitations.
## 3. Memory-Efficient and Low-Rank Optimizers

### Adafactor

Features: Adafactor reduces optimizer memory by factorizing second-moment statistics for matrix-like tensors, storing row/column accumulators instead of dense variance states.

Paper / DOI: Shazeer & Stern (2018) [10.48550/arXiv.1804.04235]

Mathematical Formulation:
Given gradient $G_t \in \mathbb{R}^{n \times m}$:

$$R_t = \beta_2 R_{t-1} + (1 - \beta_2)(G_t^2) 1_m$$
$$C_t = \beta_2 C_{t-1} + (1 - \beta_2) 1_n^\top (G_t^2)$$
$$\hat{V}_t = \frac{R_t C_t / 1_n^\top R_t}{1 - \beta_2^t}$$
$$U_t = \frac{G_t}{\sqrt{\hat{V}_t} + \epsilon}, \quad \hat{U}_t = \frac{U_t}{\max(1, \text{RMS}(U_t)/d)}$$

Pure PyTorch Implementation:

```Python
class Adafactor(Optimizer):
    ...
```

### GaLore (Gradient Low-Rank Projection)

Features: GaLore projects 2D gradients into a low-rank subspace, optimizes in that compact space, and reconstructs updates in original space.

Paper / DOI: Zhao et al. (2024) [10.48550/arXiv.2403.03507]

Mathematical Formulation:
$$U, S, V = \text{SVD}(G_t), \quad P = U_{[:, :r]} \; \text{or} \; V_{[:, :r]}$$
$$G_{low} = P^T G_t \; (\text{or } G_t P^T), \quad \Delta = P \Delta_{low} \; (\text{or } \Delta_{low} P)$$

Pure PyTorch Implementation:

```Python
class GaLoreAdamW(Optimizer):
    ...
```

## 4. Parameter-Free and Distance-Adaptive Methods

### Prodigy

Features: Prodigy adapts effective step scale through a running distance-like statistic, reducing sensitivity to manual LR tuning.

Paper / DOI: Mishchenko & Defazio (2023) [10.48550/arXiv.2306.06101]

Mathematical Formulation:
$$s_k = \sum \langle u_t, p_t - p_0 \rangle$$
$$d_t = \max(d_{t-1}, d_0 + d_{coef} \cdot s_k)$$
$$\theta_{t+1} = \theta_t - \eta \cdot d_t \cdot u_t$$

Pure PyTorch Implementation:

```Python
class Prodigy(Optimizer):
    ...
```

## 5. Sign and Second-Order Dynamics

### Lion (EvoLved Sign Momentum)

Features: Lion uses sign-based updates with momentum tracking, reducing memory footprint compared to Adam-like second-moment methods.

Paper / DOI: Chen et al. (2023) [10.48550/arXiv.2302.06675]

Mathematical Formulation:
$$c_t = \beta_1 m_t + (1 - \beta_1) g_t$$
$$\theta_{t+1} = \theta_t - \eta_t (\text{sign}(c_t) + \lambda \theta_t)$$
$$m_{t+1} = \beta_2 m_t + (1 - \beta_2) g_t$$

Pure PyTorch Implementation:

```Python
class Lion(Optimizer):
    ...
```

### Sophia

Features: Sophia uses periodic diagonal Hessian estimates with clipped second-order-scaled updates for stable large-scale optimization.

Paper / DOI: Liu et al. (2023) [10.48550/arXiv.2305.14342]

Mathematical Formulation:
$$h_t = \beta_2 h_{t-k} + (1 - \beta_2) \hat{h}_t$$
$$\theta_{t+1} = \theta_t - \eta_t \cdot \text{clip} \left( \frac{m_t}{\max(\gamma h_t, \epsilon)}, 1 \right)$$

Pure PyTorch Implementation:

```Python
class Sophia(Optimizer):
    ...
```

## 6. Orthogonality-Based Optimizers

### Muon

Features: Muon orthogonalizes momentum-driven updates for 2D tensors using Newton-Schulz iterations.

Paper / DOI: Muon (2025) [10.48550/arXiv.2505.23737]

Mathematical Formulation:
$$X_0 = G / (||G||_F + \epsilon)$$
$$A_k = X_k X_k^T, \quad B_k = b A_k + c A_k^2, \quad X_{k+1} = a X_k + B_k X_k$$

Pure PyTorch Implementation:

```Python
class Muon(Optimizer):
    ...
```

### Turbo-Muon

Features: Turbo-Muon adds almost-orthogonal preconditioning before Newton-Schulz updates, reducing the number of required orthogonalization iterations.

Paper / DOI: Turbo-Muon (2025) [10.48550/arXiv.2512.04632]

Mathematical Formulation:
$$A_0 = X_0^T X_0, \quad s_i = (\|A_{0i}\|_1)^{-1/2}$$
$$X_1 = X_0 \cdot \text{diag}(s), \quad A_1 = \text{diag}(s) A_0 \text{diag}(s)$$

Pure PyTorch Implementation:

```Python
class TurboMuon(Optimizer):
    ...
```
