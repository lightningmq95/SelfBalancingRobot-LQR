import numpy as np
from scipy.linalg import solve_continuous_are
import math

def sysCall_init():
    global body, left_motor, right_motor, prismatic_joint, arm_joint
    global m, M, L, I, g
    global K
    global prev_theta, prev_time

    global vel_offset, manual_rotate, stationary, setpoint_pitch

    sim = require('sim')

    body = sim.getObject('/body')
    left_motor = sim.getObject('/left_joint')
    right_motor = sim.getObject('/right_joint')
    prismatic_joint = sim.getObject('/Prismatic_joint')
    arm_joint = sim.getObject('/arm_joint')

    # Physical parameters
    m = 0.248
    M = 0.018
    L = 0.010
    I = 0.00133
    g = 9.81

    # LQR gain matrix K calculation
    den = I*(m+M) + m*M*L**2

    A = np.array([
        [0, 1, 0, 0],
        [m*g*L*(m+M)/den, 0, 0, 0],
        [0, 0, 0, 1],
        [-m**2 * g * L**2 / den, 0, 0, 0]
    ])

    B = np.array([
        [0],
        [(I + m*L**2) / den],
        [0],
        [m*L / den]
    ])

    Q = np.diag([1, 1, 1, 1])
    R = np.array([[1]])

    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P

    print('LQR gain matrix K:', K)

    # Init state
    prev_theta = 0
    prev_time = sim.getSimulationTime()

    # Manual control vars
    vel_offset = 0.20
    manual_rotate = 0.0
    stationary = True
    setpoint_pitch = 0.0


def sysCall_actuation():
    global prev_theta, prev_time
    global vel_offset, manual_rotate, stationary, setpoint_pitch

    sim = require('sim')

    # Handle keyboard input
    message, data, data2 = sim.getSimulatorMessage()

    if message == sim.message_keypress:
        if data[0] == 2007:  # Up arrow
            manual_rotate = 0
            stationary = False
            setpoint_pitch = -vel_offset
        elif data[0] == 2008:  # Down arrow
            manual_rotate = 0
            stationary = False
            setpoint_pitch = vel_offset
        elif data[0] == 2009:  # Right arrow
            manual_rotate = 2
        elif data[0] == 2010:  # Left arrow
            manual_rotate = -2
        elif data[0] == 122:  # Z key to STOP & BALANCE
            stationary = True
            manual_rotate = 0
            setpoint_pitch = 0
        elif (data[0] == 113): # q
            if sim.getJointTargetVelocity(prismatic_joint) == 0:
                sim.setJointTargetVelocity(prismatic_joint, -0.1)
            else:
                sim.setJointTargetVelocity(prismatic_joint, 0)
        elif (data[0] == 101): # e
            if sim.getJointTargetVelocity(prismatic_joint) == 0:
                sim.setJointTargetVelocity(prismatic_joint, 0.1)
            else:
                sim.setJointTargetVelocity(prismatic_joint, 0)
        elif (data[0] == 119): # w
            if sim.getJointTargetVelocity(arm_joint) == 0:
                sim.setJointTargetVelocity(arm_joint, 3.3)
            else:
                sim.setJointTargetVelocity(arm_joint, 0)
        elif (data[0] == 115): # s
            if sim.getJointTargetVelocity(arm_joint) == 0:
                sim.setJointTargetVelocity(arm_joint, -3.3)
            else:
                sim.setJointTargetVelocity(arm_joint, 0)

    # Time update
    now = sim.getSimulationTime()
    dt = now - prev_time
    prev_time = now

    eulerAngles = sim.getObjectOrientation(body, -1)
    roll, yaw, theta = sim.alphaBetaGammaToYawPitchRoll(eulerAngles[0], eulerAngles[1], eulerAngles[2])

    # Angular velocity theta_dot (derivative)
    theta_dot = (theta - prev_theta) / dt if dt > 0 else 0
    prev_theta = theta

    # Get average wheel velocity as x_dot
    v_left = sim.getJointVelocity(left_motor)
    v_right = sim.getJointVelocity(right_motor)
    x_dot = (v_left + v_right) / 2

    # Position x (wheel avg position)
    p_left = sim.getJointPosition(left_motor)
    p_right = sim.getJointPosition(right_motor)
    x_pos = (p_left + p_right) / 2

    # Adjust theta error with manual pitch offset for forward/backward control
    theta_error = theta - setpoint_pitch

    # State vector
    x = np.array([theta_error, theta_dot, x_pos, x_dot])

    # LQR control input u
    u = float(-K @ x)

    # Apply turning rotation velocity offset (differential speed)
    v_left_cmd = u + manual_rotate
    v_right_cmd = u - manual_rotate

    # Apply to motors
    sim.setJointTargetVelocity(left_motor, v_left_cmd)
    sim.setJointTargetVelocity(right_motor, v_right_cmd)
    print(u)