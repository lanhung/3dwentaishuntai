import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import time
import matplotlib.animation as animation

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
t_min, t_max = 0.0, 10.0  # 时间域


# 为三维瞬态圆柱绕流生成数据点
def generate_cylinder_points():
    # 内部点（流体域内）- 添加时间维度
    N_domain = 20000
    x_domain = np.random.uniform(x_min, x_max, N_domain)
    y_domain = np.random.uniform(y_min, y_max, N_domain)
    z_domain = np.random.uniform(z_min, z_max, N_domain)
    t_domain = np.random.uniform(t_min, t_max, N_domain)

    # 排除圆柱内部的点
    r_domain = np.sqrt((x_domain ** 2 + y_domain ** 2))
    indices = r_domain > r_cylinder
    x_domain = x_domain[indices]
    y_domain = y_domain[indices]
    z_domain = z_domain[indices]
    t_domain = t_domain[indices]

    # 获取内部点的实际数量（排除圆柱内部后）
    n_domain_actual = len(x_domain)

    # 圆柱表面的点（随时间变化）
    N_surface = 3000
    theta = np.random.uniform(0, 2 * np.pi, N_surface)
    x_surface = r_cylinder * np.cos(theta)
    y_surface = r_cylinder * np.sin(theta)
    z_surface = np.random.uniform(z_min, z_max, N_surface)
    t_surface = np.random.uniform(t_min, t_max, N_surface)

    # 边界点数量
    N_boundary = 1500

    # 入口点
    x_inlet = np.ones(N_boundary) * x_min
    y_inlet = np.random.uniform(y_min, y_max, N_boundary)
    z_inlet = np.random.uniform(z_min, z_max, N_boundary)
    t_inlet = np.random.uniform(t_min, t_max, N_boundary)

    # 出口点
    x_outlet = np.ones(N_boundary) * x_max
    y_outlet = np.random.uniform(y_min, y_max, N_boundary)
    z_outlet = np.random.uniform(z_min, z_max, N_boundary)
    t_outlet = np.random.uniform(t_min, t_max, N_boundary)

    # 上下边界点
    x_top = np.random.uniform(x_min, x_max, N_boundary)
    y_top = np.ones(N_boundary) * y_max
    z_top = np.random.uniform(z_min, z_max, N_boundary)
    t_top = np.random.uniform(t_min, t_max, N_boundary)

    x_bottom = np.random.uniform(x_min, x_max, N_boundary)
    y_bottom = np.ones(N_boundary) * y_min
    z_bottom = np.random.uniform(z_min, z_max, N_boundary)
    t_bottom = np.random.uniform(t_min, t_max, N_boundary)

    # 前后边界点
    x_front = np.random.uniform(x_min, x_max, N_boundary)
    y_front = np.random.uniform(y_min, y_max, N_boundary)
    z_front = np.ones(N_boundary) * z_max
    t_front = np.random.uniform(t_min, t_max, N_boundary)

    x_back = np.random.uniform(x_min, x_max, N_boundary)
    y_back = np.random.uniform(y_min, y_max, N_boundary)
    z_back = np.ones(N_boundary) * z_min
    t_back = np.random.uniform(t_min, t_max, N_boundary)

    # 初始时刻点 (t = 0)
    N_initial = 3000
    x_initial = np.random.uniform(x_min, x_max, N_initial)
    y_initial = np.random.uniform(y_min, y_max, N_initial)
    z_initial = np.random.uniform(z_min, z_max, N_initial)
    t_initial = np.zeros(N_initial)

    # 排除初始时刻点中圆柱内部的点
    r_initial = np.sqrt((x_initial ** 2 + y_initial ** 2))
    indices_initial = r_initial > r_cylinder
    x_initial = x_initial[indices_initial]
    y_initial = y_initial[indices_initial]
    z_initial = z_initial[indices_initial]
    t_initial = t_initial[indices_initial]

    # 将所有数据点合并 - 确保每个数组长度匹配
    x = np.concatenate([x_domain, x_surface, x_inlet, x_outlet, x_top, x_bottom, x_front, x_back, x_initial])
    y = np.concatenate([y_domain, y_surface, y_inlet, y_outlet, y_top, y_bottom, y_front, y_back, y_initial])
    z = np.concatenate([z_domain, z_surface, z_inlet, z_outlet, z_top, z_bottom, z_front, z_back, z_initial])
    t = np.concatenate([t_domain, t_surface, t_inlet, t_outlet, t_top, t_bottom, t_front, t_back, t_initial])

    print(f"生成数据点: 内部点 {n_domain_actual}, 总点数 {len(x)}")

    return x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1), t.reshape(-1, 1)


# PINN网络架构 - 瞬态版本
class PhysicsInformedNN:
    def __init__(self, layers, domain_bounds):
        # 网络结构: 输入 (x,y,z,t) -> 隐藏层 -> 输出 (u,v,w,p)
        self.layers = layers

        # 域的边界
        self.domain_bounds = domain_bounds

        # 生成训练数据
        self.x, self.y, self.z, self.t = generate_cylinder_points()

        # 初始化网络权重和偏置
        self.weights, self.biases = self.initialize_nn(layers)

        # TensorFlow 占位符和变量
        self.x_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.y_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.z_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.t_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

        # 输出变量
        self.u_pred, self.v_pred, self.w_pred, self.p_pred, self.f_u_pred, self.f_v_pred, self.f_w_pred, self.f_c_pred = \
            self.net_ns(self.x_tf, self.y_tf, self.z_tf, self.t_tf)

        # 定义损失函数
        # 1. NS方程损失
        self.loss_physics = tf.reduce_mean(tf.square(self.f_u_pred) +
                                           tf.square(self.f_v_pred) +
                                           tf.square(self.f_w_pred) +
                                           tf.square(self.f_c_pred))

        # 2. 初始条件损失 (根据实际需要添加)
        # 3. 边界条件损失 (根据实际需要添加)

        # 总损失
        self.loss = self.loss_physics

        # 定义优化器
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
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

    def net_ns(self, x, y, z, t):
        # 自动微分变量
        X = tf.concat([x, y, z, t], 1)

        # 神经网络输出
        uvwp = self.neural_net(X, self.weights, self.biases)
        u = uvwp[:, 0:1]
        v = uvwp[:, 1:2]
        w = uvwp[:, 2:3]
        p = uvwp[:, 3:4]

        # 自动微分获取速度和压力梯度
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]
        u_z = tf.gradients(u, z)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_yy = tf.gradients(u_y, y)[0]
        u_zz = tf.gradients(u_z, z)[0]

        v_t = tf.gradients(v, t)[0]
        v_x = tf.gradients(v, x)[0]
        v_y = tf.gradients(v, y)[0]
        v_z = tf.gradients(v, z)[0]
        v_xx = tf.gradients(v_x, x)[0]
        v_yy = tf.gradients(v_y, y)[0]
        v_zz = tf.gradients(v_z, z)[0]

        w_t = tf.gradients(w, t)[0]
        w_x = tf.gradients(w, x)[0]
        w_y = tf.gradients(w, y)[0]
        w_z = tf.gradients(w, z)[0]
        w_xx = tf.gradients(w_x, x)[0]
        w_yy = tf.gradients(w_y, y)[0]
        w_zz = tf.gradients(w_z, z)[0]

        p_x = tf.gradients(p, x)[0]
        p_y = tf.gradients(p, y)[0]
        p_z = tf.gradients(p, z)[0]

        # 非稳态NS方程 - 动量方程 (添加时间导数)
        f_u = u_t + (u * u_x + v * u_y + w * u_z) + p_x - (1.0 / Re) * (u_xx + u_yy + u_zz)
        f_v = v_t + (u * v_x + v * v_y + w * v_z) + p_y - (1.0 / Re) * (v_xx + v_yy + v_zz)
        f_w = w_t + (u * w_x + v * w_y + w * w_z) + p_z - (1.0 / Re) * (w_xx + w_yy + w_zz)

        # 连续性方程
        f_c = u_x + v_y + w_z

        return u, v, w, p, f_u, f_v, f_w, f_c

    def callback(self, loss):
        print('Loss:', loss)

    def train(self, nIter=10000):
        tf_dict = {self.x_tf: self.x, self.y_tf: self.y, self.z_tf: self.z, self.t_tf: self.t}

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

        # 多轮Adam训练
        print("Training with Adam optimizer for additional iterations...")
        for it in range(200):
            self.sess.run(self.train_op_Adam, tf_dict)
            if it % 100 == 0:
                loss_value = self.sess.run(self.loss, tf_dict)
                print('Additional Iteration: %d, Loss: %.3e' % (it, loss_value))

    def predict(self, x_star, y_star, z_star, t_star):
        tf_dict = {self.x_tf: x_star, self.y_tf: y_star, self.z_tf: z_star, self.t_tf: t_star}
        u_star = self.sess.run(self.u_pred, tf_dict)
        v_star = self.sess.run(self.v_pred, tf_dict)
        w_star = self.sess.run(self.w_pred, tf_dict)
        p_star = self.sess.run(self.p_pred, tf_dict)
        return u_star, v_star, w_star, p_star


if __name__ == "__main__":
    # 定义神经网络层结构 - 输入增加时间维度
    layers = [4, 40, 40, 40, 40, 40, 40, 4]

    # 定义计算域边界 - 包括时间域
    domain_bounds = [x_min, x_max, y_min, y_max, z_min, z_max, t_min, t_max]

    # 创建PINN模型
    model = PhysicsInformedNN(layers, domain_bounds)

    # 训练模型 - 瞬态模型需要更多迭代
    model.train(200)

    # 生成时间序列的预测数据
    N_test = 100
    x_test = np.linspace(x_min, x_max, N_test)
    y_test = np.linspace(y_min, y_max, N_test)
    X_test, Y_test = np.meshgrid(x_test, y_test)

    # 创建动画
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 存储每个时间点的预测结果
    frames = []
    time_steps = np.linspace(t_min, t_max, 20)  # 20个时间步

    for t_step in time_steps:
        print(f"处理时间步: t = {t_step:.2f}")

        # 创建当前时间的网格点
        x_star = X_test.flatten().reshape(-1, 1)
        y_star = Y_test.flatten().reshape(-1, 1)
        z_star = np.zeros(X_test.size).reshape(-1, 1)  # z = 0 平面
        t_star = np.ones(X_test.size).reshape(-1, 1) * t_step

        # 排除圆柱内部的点
        r_test = np.sqrt(x_star ** 2 + y_star ** 2)
        indices = r_test.flatten() > r_cylinder

        x_valid = x_star[indices].reshape(-1, 1)
        y_valid = y_star[indices].reshape(-1, 1)
        z_valid = z_star[indices].reshape(-1, 1)
        t_valid = t_star[indices].reshape(-1, 1)

        # 预测流场
        u_pred, v_pred, w_pred, p_pred = model.predict(x_valid, y_valid, z_valid, t_valid)

        # 可视化速度场
        U_star = griddata((x_valid.flatten(), y_valid.flatten()), u_pred.flatten(), (X_test, Y_test), method='cubic')
        V_star = griddata((x_valid.flatten(), y_valid.flatten()), v_pred.flatten(), (X_test, Y_test), method='cubic')
        P_star = griddata((x_valid.flatten(), y_valid.flatten()), p_pred.flatten(), (X_test, Y_test), method='cubic')

        # 计算速度幅值
        UV_mag = np.sqrt(U_star ** 2 + V_star ** 2)

        # 存储该时间步的结果
        frames.append((U_star, V_star, P_star, UV_mag, t_step))

    # 创建动画
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))


    def update_frame(frame_idx):
        U_star, V_star, P_star, UV_mag, t_step = frames[frame_idx]

        ax1.clear()
        ax2.clear()

        # 速度场
        c1 = ax1.pcolor(X_test, Y_test, UV_mag, cmap='jet', vmin=0, vmax=2)
        ax1.streamplot(X_test, Y_test, U_star, V_star, density=1.0, color='k', linewidth=0.5)
        ax1.add_patch(plt.Circle((0, 0), r_cylinder, color='r'))
        ax1.set_xlim([x_min, x_max])
        ax1.set_ylim([y_min, y_max])
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title(f'Velocity Field, t = {t_step:.2f}')

        # 压力场
        c2 = ax2.pcolor(X_test, Y_test, P_star, cmap='jet')
        ax2.add_patch(plt.Circle((0, 0), r_cylinder, color='r'))
        ax2.set_xlim([x_min, x_max])
        ax2.set_ylim([y_min, y_max])
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_title(f'Pressure Field, t = {t_step:.2f}')

        return (ax1, ax2)


    # 创建动画
    ani = animation.FuncAnimation(fig, update_frame, frames=len(frames), interval=200, blit=False)

    # 保存动画
    ani.save('cylinder_flow_transient.mp4', writer='ffmpeg', fps=5, dpi=200)

    # 显示最终结果
    plt.figure(figsize=(15, 10))

    # 选择最后一个时间步
    U_star, V_star, P_star, UV_mag, t_step = frames[-1]

    plt.subplot(2, 2, 1)
    plt.pcolor(X_test, Y_test, U_star, cmap='jet')
    plt.colorbar()
    circle = plt.Circle((0, 0), r_cylinder, color='r')
    plt.gca().add_patch(circle)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'U-velocity, t = {t_step:.2f}')

    plt.subplot(2, 2, 2)
    plt.pcolor(X_test, Y_test, V_star, cmap='jet')
    plt.colorbar()
    circle = plt.Circle((0, 0), r_cylinder, color='r')
    plt.gca().add_patch(circle)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'V-velocity, t = {t_step:.2f}')

    plt.subplot(2, 2, 3)
    plt.pcolor(X_test, Y_test, P_star, cmap='jet')
    plt.colorbar()
    circle = plt.Circle((0, 0), r_cylinder, color='r')
    plt.gca().add_patch(circle)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Pressure, t = {t_step:.2f}')

    plt.subplot(2, 2, 4)
    plt.streamplot(X_test, Y_test, U_star, V_star, density=1.0, color='k')
    plt.pcolor(X_test, Y_test, UV_mag, cmap='jet', alpha=0.7)
    plt.colorbar()
    circle = plt.Circle((0, 0), r_cylinder, color='r')
    plt.gca().add_patch(circle)
    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Streamlines & Velocity Magnitude, t = {t_step:.2f}')

    plt.tight_layout()
    plt.savefig('cylinder_flow_pinn_final.png', dpi=300)
    plt.show()
