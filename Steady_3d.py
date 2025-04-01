import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import time

# 确保使用TensorFlow 2.x和eager execution模式
tf.compat.v1.disable_eager_execution()

# 设置随机种子以确保结果可复现
np.random.seed(1234)
tf.random.set_seed(1234)

# 定义问题参数
Re = 100.0  # 雷诺数
r_cylinder = 0.5  # 圆柱半径
u_inf = 1.0  # 来流速度

# 定义计算域
x_min, x_max = -5.0, 15.0
y_min, y_max = -5.0, 5.0
z_min, z_max = -5.0, 5.0


# 为圆柱绕流生成数据点
def generate_cylinder_points():
    # 内部点（流体域内）
    N_domain = 10000
    x_domain = np.random.uniform(x_min, x_max, N_domain)
    y_domain = np.random.uniform(y_min, y_max, N_domain)
    z_domain = np.random.uniform(z_min, z_max, N_domain)

    # 排除圆柱内部的点
    r_domain = np.sqrt((x_domain ** 2 + y_domain ** 2))
    indices = r_domain > r_cylinder
    x_domain = x_domain[indices]
    y_domain = y_domain[indices]
    z_domain = z_domain[indices]

    # 获取内部点的实际数量（排除圆柱内部后）
    n_domain_actual = len(x_domain)

    # 圆柱表面的点
    N_surface = 2000
    theta = np.random.uniform(0, 2 * np.pi, N_surface)
    x_surface = r_cylinder * np.cos(theta)
    y_surface = r_cylinder * np.sin(theta)
    z_surface = np.random.uniform(z_min, z_max, N_surface)

    # 边界点数量
    N_boundary = 1000

    # 入口点
    x_inlet = np.ones(N_boundary) * x_min
    y_inlet = np.random.uniform(y_min, y_max, N_boundary)
    z_inlet = np.random.uniform(z_min, z_max, N_boundary)

    # 出口点
    x_outlet = np.ones(N_boundary) * x_max
    y_outlet = np.random.uniform(y_min, y_max, N_boundary)
    z_outlet = np.random.uniform(z_min, z_max, N_boundary)

    # 上下边界点
    x_top = np.random.uniform(x_min, x_max, N_boundary)
    y_top = np.ones(N_boundary) * y_max
    z_top = np.random.uniform(z_min, z_max, N_boundary)

    x_bottom = np.random.uniform(x_min, x_max, N_boundary)
    y_bottom = np.ones(N_boundary) * y_min
    z_bottom = np.random.uniform(z_min, z_max, N_boundary)

    # 前后边界点
    x_front = np.random.uniform(x_min, x_max, N_boundary)
    y_front = np.random.uniform(y_min, y_max, N_boundary)
    z_front = np.ones(N_boundary) * z_max

    x_back = np.random.uniform(x_min, x_max, N_boundary)
    y_back = np.random.uniform(y_min, y_max, N_boundary)
    z_back = np.ones(N_boundary) * z_min

    # 将所有数据点合并 - 确保每个数组长度匹配
    x = np.concatenate([x_domain, x_surface, x_inlet, x_outlet, x_top, x_bottom, x_front, x_back])
    y = np.concatenate([y_domain, y_surface, y_inlet, y_outlet, y_top, y_bottom, y_front, y_back])
    z = np.concatenate([z_domain, z_surface, z_inlet, z_outlet, z_top, z_bottom, z_front, z_back])

    print(f"生成数据点: 内部点 {n_domain_actual}, 总点数 {len(x)}")

    return x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)


# PINN网络架构
class PhysicsInformedNN:
    def __init__(self, layers, domain_bounds):
        # 网络结构: 输入 (x,y,z) -> 隐藏层 -> 输出 (u,v,w,p)
        self.layers = layers

        # 域的边界
        self.domain_bounds = domain_bounds

        # 生成训练数据
        self.x, self.y, self.z = generate_cylinder_points()

        # 初始化网络权重和偏置
        self.weights, self.biases = self.initialize_nn(layers)

        # TensorFlow 占位符和变量
        self.x_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.y_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.z_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

        # 输出变量
        self.u_pred, self.v_pred, self.w_pred, self.p_pred, self.f_u_pred, self.f_v_pred, self.f_w_pred, self.f_c_pred = \
            self.net_ns(self.x_tf, self.y_tf, self.z_tf)

        # 定义损失函数
        self.loss = tf.reduce_mean(tf.square(self.f_u_pred) +
                                   tf.square(self.f_v_pred) +
                                   tf.square(self.f_w_pred) +
                                   tf.square(self.f_c_pred))

        # 定义优化器
        self.optimizer = tf.compat.v1.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer.minimize(self.loss)

        # TensorFlow会话
        self.sess = tf.compat.v1.Session()

        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)

    def initialize_nn(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(layers[l], layers[l + 1])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size_in, size_out):
        xavier_stddev = np.sqrt(2 / (size_in + size_out))
        return tf.Variable(tf.random.truncated_normal([size_in, size_out], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = X
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_ns(self, x, y, z):
        # 自动微分变量
        X = tf.concat([x, y, z], 1)

        # 神经网络输出
        uvwp = self.neural_net(X, self.weights, self.biases)
        u = uvwp[:, 0:1]
        v = uvwp[:, 1:2]
        w = uvwp[:, 2:3]
        p = uvwp[:, 3:4]

        # 自动微分获取速度和压力梯度
        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]
        u_z = tf.gradients(u, z)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_yy = tf.gradients(u_y, y)[0]
        u_zz = tf.gradients(u_z, z)[0]

        v_x = tf.gradients(v, x)[0]
        v_y = tf.gradients(v, y)[0]
        v_z = tf.gradients(v, z)[0]
        v_xx = tf.gradients(v_x, x)[0]
        v_yy = tf.gradients(v_y, y)[0]
        v_zz = tf.gradients(v_z, z)[0]

        w_x = tf.gradients(w, x)[0]
        w_y = tf.gradients(w, y)[0]
        w_z = tf.gradients(w, z)[0]
        w_xx = tf.gradients(w_x, x)[0]
        w_yy = tf.gradients(w_y, y)[0]
        w_zz = tf.gradients(w_z, z)[0]

        p_x = tf.gradients(p, x)[0]
        p_y = tf.gradients(p, y)[0]
        p_z = tf.gradients(p, z)[0]

        # NS方程 - 动量方程
        f_u = (u * u_x + v * u_y + w * u_z) + p_x - (1.0 / Re) * (u_xx + u_yy + u_zz)
        f_v = (u * v_x + v * v_y + w * v_z) + p_y - (1.0 / Re) * (v_xx + v_yy + v_zz)
        f_w = (u * w_x + v * w_y + w * w_z) + p_z - (1.0 / Re) * (w_xx + w_yy + w_zz)

        # 连续性方程
        f_c = u_x + v_y + w_z

        return u, v, w, p, f_u, f_v, f_w, f_c

    def callback(self, loss):
        print('Loss:', loss)

    def train(self, nIter=10000):
        tf_dict = {self.x_tf: self.x, self.y_tf: self.y, self.z_tf: self.z}

        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)

            # 打印训练过程
            if it % 100 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print('Iteration: %d, Loss: %.3e, Time: %.2f' %
                      (it, loss_value, elapsed))
                start_time = time.time()

        # 替代L-BFGS的额外Adam训练
        print("Training with Adam optimizer for additional iterations...")
        for it in range(200):
            self.sess.run(self.train_op_Adam, tf_dict)
            if it % 100 == 0:
                loss_value = self.sess.run(self.loss, tf_dict)
                print('Additional Iteration: %d, Loss: %.3e' % (it, loss_value))

    def predict(self, x_star, y_star, z_star):
        tf_dict = {self.x_tf: x_star, self.y_tf: y_star, self.z_tf: z_star}
        u_star = self.sess.run(self.u_pred, tf_dict)
        v_star = self.sess.run(self.v_pred, tf_dict)
        w_star = self.sess.run(self.w_pred, tf_dict)
        p_star = self.sess.run(self.p_pred, tf_dict)
        return u_star, v_star, w_star, p_star


if __name__ == "__main__":
    # 定义神经网络层结构
    layers = [3, 20, 20, 20, 20, 20, 20, 4]

    # 定义计算域边界
    domain_bounds = [x_min, x_max, y_min, y_max, z_min, z_max]

    # 创建PINN模型
    model = PhysicsInformedNN(layers, domain_bounds)

    # 训练模型 - 减少迭代次数以加快演示
    model.train(200)  # 可以根据需要增加迭代次数

    # 生成预测数据点 - 中心平面 (z=0)
    N_test = 100
    x_test = np.linspace(x_min, x_max, N_test)
    y_test = np.linspace(y_min, y_max, N_test)
    X_test, Y_test = np.meshgrid(x_test, y_test)

    x_star = X_test.flatten().reshape(-1, 1)
    y_star = Y_test.flatten().reshape(-1, 1)
    z_star = np.zeros(X_test.size).reshape(-1, 1)

    # 排除圆柱内部的点
    r_test = np.sqrt(x_star ** 2 + y_star ** 2)
    indices = r_test.flatten() > r_cylinder

    x_valid = x_star[indices].reshape(-1, 1)
    y_valid = y_star[indices].reshape(-1, 1)
    z_valid = z_star[indices].reshape(-1, 1)

    print(f"预测点形状: x={x_valid.shape}, y={y_valid.shape}, z={z_valid.shape}")

    # 预测流场
    u_pred, v_pred, w_pred, p_pred = model.predict(x_valid, y_valid, z_valid)

    # 可视化速度场
    U_star = griddata((x_valid.flatten(), y_valid.flatten()), u_pred.flatten(), (X_test, Y_test), method='cubic')
    V_star = griddata((x_valid.flatten(), y_valid.flatten()), v_pred.flatten(), (X_test, Y_test), method='cubic')
    P_star = griddata((x_valid.flatten(), y_valid.flatten()), p_pred.flatten(), (X_test, Y_test), method='cubic')

    # 绘制速度场
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.pcolor(X_test, Y_test, U_star, cmap='jet')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('U-velocity')

    plt.subplot(2, 2, 2)
    plt.pcolor(X_test, Y_test, V_star, cmap='jet')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('V-velocity')

    plt.subplot(2, 2, 3)
    plt.pcolor(X_test, Y_test, P_star, cmap='jet')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Pressure')

    # 绘制流线
    plt.subplot(2, 2, 4)
    plt.streamplot(X_test, Y_test, U_star, V_star, density=1.0, color='k')
    circle = plt.Circle((0, 0), r_cylinder, color='r')
    plt.gca().add_patch(circle)
    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Streamlines')

    plt.tight_layout()
    plt.savefig('cylinder_flow_pinn.png', dpi=300)
    plt.show()
