# stump\_to\_gpt

误差引入3 $\sigma$界

画 6 张误差图：

- 位置：$x,y,z$ 误差； 

- 姿态：绕 $x,y,z$的小角度误差，近似对应 roll、pitch、yaw。 

每张图同时包含：

- 实际误差曲线； 

- $+3\sigma$上界； 

- $-3\sigma$下界。 

### 位置误差的 $3\sigma$ 界

假设真实相对位置是 $\mathbf p_{\mathrm{gt}}$，ESKF估计为 $\hat{\mathbf p}$，则：

$\mathbf e_p = \mathbf p_{\mathrm{gt}} - \hat{\mathbf p} = \begin{bmatrix} e_x&e_y&e_z \end{bmatrix}^{T}$

如果你的误差状态排列为：

$\delta\mathbf x= \begin{bmatrix} \delta\boldsymbol\theta& \delta\mathbf v& \delta\mathbf p \end{bmatrix}^{T}$

那么位置协方差是：

$\mathbf P_p=\mathbf P_{6:8,6:8}$

对应三个标准差为：

$\sigma_x=\sqrt{P_{6,6}},\qquad \sigma_y=\sqrt{P_{7,7}},\qquad \sigma_z=\sqrt{P_{8,8}}$

所以要画：

$e_x,\qquad \pm3\sqrt{P_{6,6}}$

$e_y,\qquad \pm3\sqrt{P_{7,7}}$

$e_z,\qquad \pm3\sqrt{P_{8,8}}$

注意这里的 $\sigma$不是实验误差的标准差，也不是RMSE，而是**ESKF协方差矩阵中对应方差的平方根**，并且随时间变化。

### 姿态误差

你的ESKF采用左乘姿态误差时，通常定义：

$\mathbf R_{\mathrm{true}} = \operatorname{Exp}(\delta\boldsymbol\theta) \hat{\mathbf R}$

因此与ESKF误差定义一致的真实姿态误差应当计算为：

$\mathbf e_\theta = \operatorname{Log} \left( \mathbf R_{\mathrm{gt}}\hat{\mathbf R}^{T} \right)$

其中：

$\mathbf e_\theta= \begin{bmatrix} e_{\theta x}&e_{\theta y}&e_{\theta z} \end{bmatrix}^{T}$

它们是旋转向量的小角度分量，误差较小时可以近似理解为：

$e_{\theta x}\approx e_{\mathrm{roll}},\quad e_{\theta y}\approx e_{\mathrm{pitch}},\quad e_{\theta z}\approx e_{\mathrm{yaw}}$

如果姿态误差位于状态的前3维，那么：

$\sigma_{\theta x}=\sqrt{P_{0,0}},\qquad \sigma_{\theta y}=\sqrt{P_{1,1}},\qquad \sigma_{\theta z}=\sqrt{P_{2,2}}$

于是画：

$e_{\theta x},\qquad \pm3\sqrt{P_{0,0}}$

$e_{\theta y},\qquad \pm3\sqrt{P_{1,1}}$

$e_{\theta z},\qquad \pm3\sqrt{P_{2,2}}$

纵轴使用度，把误差和$3\sigma$都转换：

$e_{\theta,\deg}=e_{\theta,\mathrm{rad}}\frac{180}{\pi}$

$3\sigma_{\theta,\deg} = 3\sqrt{P_{\theta\theta}}\frac{180}{\pi}$

