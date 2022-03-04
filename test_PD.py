import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
"""N = 100
x = 0.0
dx = 0.0
T = 0.001
Kp = 0.9
Kd = 0.0001

ylist = [0.0] * N
for i in range(N):
    y = Kp * (1.0 - x) + Kd * (0.0 - dx)
    ylist[i] = y
    dx = (y - x) / 0.001
    x = y

print(ylist)

t = np.linspace(0.0, N*T, N+1)
plt.figure()
plt.plot(t[1:], ylist)
plt.show()"""


N = 1000
x = 0.0
dx = 0.0
ddx = 0.0
dt = 0.0005
#dt = 0.001
I = 0.0000045
Kp = 5
Kd = 0.1


y = [0.0] * N
for i in range(N):
    ddx = (Kp * (1.0 - x) + Kd * (0.0 - dx) ) / I / 81
    """print("Kp:", Kp * (1.0 - x) / I / 81)
    print("Kd:", Kd * (0.0 - dx) / I / 81)
    print("ddx:", ddx)"""
    x += dx * dt + ddx * dt**2 * 0.5
    dx += ddx * dt
    y[i] = x

t = np.linspace(0.0, N*dt, N+1)
fig = plt.figure()

h, = plt.plot(t[1:], y)
ax = plt.gca()

# Make a horizontal slider to control the time.
axcolor = 'lightgoldenrodyellow'
axtime = plt.axes([0.25, 0.03, 0.65, 0.03], facecolor=axcolor)
Kp_slider = Slider(
    ax=axtime,
    label='Kp',
    valmin=0.0,
    valmax=50.0,
    valinit=0.0,
)
axtime = plt.axes([0.25, 0.00, 0.65, 0.03], facecolor=axcolor)
Kd_slider = Slider(
    ax=axtime,
    label='Kd',
    valmin=0.0,
    valmax=1.0,
    valinit=0.0,
)

def update():
    Kp = Kp_slider.val
    Kd = Kd_slider.val
    x = 0.0
    dx = 0.0
    ddx = 0.0
    for i in range(N):
        ddx = (Kp * (1.0 - x) + Kd * (0.0 - dx) ) / I / 81
        x += dx * dt + ddx * dt**2 * 0.5
        dx += ddx * dt
        y[i] = x

    h.set_ydata(y)
    ax.set_ylim([0.0, 2.0])
    fig.canvas.draw_idle()

def update_Kp(val):
    Kp_slider.val = np.round(val, decimals=3)
    update()

def update_Kd(val):
    Kd_slider.val = np.round(val, decimals=3)
    update()

Kp_slider.on_changed(update_Kp)
Kd_slider.on_changed(update_Kd)

plt.show()
