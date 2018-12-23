## 数据

数据使用MNIST的[数据集](http://yann.lecun.com/exdb/mnist/)

## 想法

MNIST提供的是28*28的图片，因此输入层是754维的向量。
隐层
输出层为10维向量

公式定义：
1. 输出层：O1...Ol
2. 隐层：y1...yn
3. 输入层：x1...xm
4. 输入层与隐层间的权值 $V_{ij}$
5. 隐层与输出层之间的权值 $W_{jk}$
6. 使用函数$f(x)=\frac{1}{1+e^{-x}}$ 
7. 准确值d1...dl
8. 学习率eta
9. 隐层与输出层间误差$\delta^o_k = (d_k - O_k)O_k(1-O_k)$
10. 输入层与隐层间误差$\delta^y_j = (\sum^{l}_{k=1}\delta^o_kW_{jk})y_j(1-y_j)$
11. 误差反传时$\Delta W_{jk} = \eta (d_k-O_k)O_k(1-O_k)*y_j$
12. $\Delta V_{ij} = \eta (\sum^{l}_{k=1}\delta^o_kW_{jk})y_j(1-y_j)X_i$

每次计算时先从输入层计算到输出层，然后算出三层间的两个误差，然后更新网络间的权值