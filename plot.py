import matplotlib.pyplot as plt
import numpy as np

x = np.array(np.random.rand(10))
y = np.sin(2 * np.pi * x)

# ノイズ生成
noise = np.random.normal(0, 0.1, 10)

fig, ax = plt.subplots()

ax.scatter(x, y, label='no noise')
ax.scatter(x, y + noise, label='noise')

ax.set_title('test')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_xlim(0, 1)
ax.set_ylim(-2, 2)
ax.grid()

ax.legend()
plt.show()