import time
import numpy as np
from utils import *

import sim
import sys
import modern_robotics as mr

clientID = sim.simxStart('127.0.0.1',19997,True,True,5000,5)
if(clientID!=-1):
    print('Connected Successfully')
else: 
    sys.exit('Failed to connect')

returnCode = sim.simxStartSimulation(clientID, sim.simx_opmode_oneshot_wait)
if returnCode == sim.simx_return_ok:
    print('Simulation started')
else:
    print('Failed to start simulation')



class Env:
    def __init__(self, path) -> None:
        self.model_handles = {}
        self.path = path

    def generate_object(self, model, position, orientation = None, inflation = 0.0):
        print(f'Spawning {model} at {position}, {orientation if orientation is not None else [0, 0, 0]}')
        model_path = self.path + model
        
        res, model_handle = sim.simxLoadModel(clientID, model_path, 0, sim.simx_opmode_oneshot_wait)
        if not res == sim.simx_return_ok:
            print('Model loading Failed')
            return
        res = sim.simxSetObjectPosition(clientID, model_handle, -1, position, sim.simx_opmode_oneshot_wait)
        if not res == sim.simx_return_ok:
            print('Pose set Failed')
            return
        if orientation is not None:
            res = sim.simxSetObjectOrientation(clientID, model_handle, -1, orientation, sim.simx_opmode_oneshot_wait)
            if not res == sim.simx_return_ok:
                print('Orientation set Failed')
                return
            
        res, min_x = sim.simxGetObjectFloatParameter(clientID, model_handle, sim.sim_objfloatparam_modelbbox_min_x, sim.simx_opmode_oneshot_wait)
        res, min_y = sim.simxGetObjectFloatParameter(clientID, model_handle, sim.sim_objfloatparam_modelbbox_min_y, sim.simx_opmode_oneshot_wait)
        res, min_z = sim.simxGetObjectFloatParameter(clientID, model_handle, sim.sim_objfloatparam_modelbbox_min_z, sim.simx_opmode_oneshot_wait)
        res, max_x = sim.simxGetObjectFloatParameter(clientID, model_handle, sim.sim_objfloatparam_modelbbox_max_x, sim.simx_opmode_oneshot_wait)
        res, max_y = sim.simxGetObjectFloatParameter(clientID, model_handle, sim.sim_objfloatparam_modelbbox_max_y, sim.simx_opmode_oneshot_wait)
        res, max_z = sim.simxGetObjectFloatParameter(clientID, model_handle, sim.sim_objfloatparam_modelbbox_max_z, sim.simx_opmode_oneshot_wait)
        
        if res == sim.simx_return_ok:
            self.model_handles[model_handle] = {'position':position, 'orientation': orientation, 'bbox':[min_x - inflation, min_y - inflation, min_z + inflation, max_x + inflation, max_y + inflation, max_z + inflation]}
        else:
            print('BBox fetch failed')
            return

        return model_handle

    def plot_object(self, model_handle, position = None, orientation = None, style = 'b-'):
        x,y,z = self.model_handles[model_handle]['position'] if position is None else position
        _, _, yaw = self.model_handles[model_handle]['orientation'] if orientation is None else orientation
        bbox = self.model_handles[model_handle]['bbox']
        plot_rectangle(x, y, bbox, yaw, style)

    def show_env(self,):
        for model_handle in self.model_handles:
            self.plot_object(model_handle)

    def pause_env(self):
        returnCode = sim.simxPauseSimulation(clientID, sim.simx_opmode_oneshot_wait)
        if returnCode == sim.simx_return_ok:
            print('Simulation Paused')
        else:
            print('Failed to pause simulation')

class Bot:
    def __init__(self) -> None:
        error_code, self.youb = sim.simxGetObjectHandle(clientID, '/youBot',sim.simx_opmode_oneshot_wait)
        error_code, self.dr12 = sim.simxGetObjectHandle(clientID, '/dr12',sim.simx_opmode_oneshot_wait)

        error_code, self.rollingJoint_fl = sim.simxGetObjectHandle(clientID, '/youBot/rollingJoint_fl',sim.simx_opmode_oneshot_wait)
        error_code, self.rollingJoint_rl = sim.simxGetObjectHandle(clientID, '/youBot/rollingJoint_rl',sim.simx_opmode_oneshot_wait)
        error_code, self.rollingJoint_rr = sim.simxGetObjectHandle(clientID, '/youBot/rollingJoint_rr',sim.simx_opmode_oneshot_wait)
        error_code, self.rollingJoint_fr = sim.simxGetObjectHandle(clientID, '/youBot/rollingJoint_fr',sim.simx_opmode_oneshot_wait)

        error_code, self.roll_joint_Arm0 = sim.simxGetObjectHandle(clientID,'/youBot/youBotArmJoint0',sim.simx_opmode_oneshot_wait)
        error_code, self.roll_joint_Arm1 = sim.simxGetObjectHandle(clientID,'/youBot/youBotArmJoint1',sim.simx_opmode_oneshot_wait)
        error_code, self.roll_joint_Arm2 = sim.simxGetObjectHandle(clientID,'/youBot/youBotArmJoint2',sim.simx_opmode_oneshot_wait)
        error_code, self.roll_joint_Arm3 = sim.simxGetObjectHandle(clientID,'/youBot/youBotArmJoint3',sim.simx_opmode_oneshot_wait)
        error_code, self.roll_joint_Arm4 = sim.simxGetObjectHandle(clientID,'/youBot/youBotArmJoint4',sim.simx_opmode_oneshot_wait)

        self.Kp = np.array([
            [2.0, 0, 0], 
            [0, -2.0, 0],
            [0, 0, -5]], dtype=float
        )
        self.Kd = np.array([
            [0, 0.1, 0], 
            [0.1, 0, 0],
            [0, 0, -0.3]], dtype=float
        )

        # End-effector to base relative transform
        self.M_0e = np.array([[1,0,0,0.033],
                        [0,1,0,0],
                        [0,0,1,0.6546],
                        [0,0,0,1]])
        
        # B list for arm on th eyou-bot
        self.Blist =  np.array([[0,0,1,0,0.0330,0],
                        [0,-1,0,-0.5076,0,0],
                        [0,-1,0,-0.3526,0,0],
                        [0,-1,0,-0.2176,0,0],
                        [0,0,1,0,0,0]]).T
        
        # offset from base to arm transform
        self.T_b0 = np.array([[1,0,0,0.1662],
                        [0,1,0,0],
                        [0,0,1,0.0026],
                        [0,0,0,1]])
        
        # Set Kp and Ki
        self.kp = np.eye(6) * 1
        self.ki = np.eye(6)* 0

        self.ec = np.zeros(6)

        # Constants from MR wiki
        radius = 0.0475
        l = 0.235 
        w =  0.15

        # init robot's H matrix
        h = radius/4 * np.array([[-1/(l+w), 1/(l+w), 1/(l+w), -1/(l+w)],
                        [1, 1, 1, 1],
                        [-1, 1, -1, 1]]).T

        h1 = np.array([[-1/(l+w), 1/(l+w), 1/(l+w), -1/(l+w)],
                        [1, 1, 1, 1],
                        [-1, 1, -1, 1]])
        self.h1 = np.concatenate((np.zeros((2,4)),h1*radius/4, np.zeros((1,4))), axis=0)
    
    def getArmState(self, ):
        
        sols =  [
            sim.simxGetJointPosition(clientID,self.roll_joint_Arm0,sim.simx_opmode_oneshot_wait),
            sim.simxGetJointPosition(clientID,self.roll_joint_Arm1,sim.simx_opmode_oneshot_wait),
            sim.simxGetJointPosition(clientID,self.roll_joint_Arm2,sim.simx_opmode_oneshot_wait),
            sim.simxGetJointPosition(clientID,self.roll_joint_Arm3,sim.simx_opmode_oneshot_wait),
            sim.simxGetJointPosition(clientID,self.roll_joint_Arm4,sim.simx_opmode_oneshot_wait),]
        qs_list = []
        for res, qs in sols:
            if not res == sim.simx_return_ok:
                return None
            qs_list.append(qs)

        return qs_list
    
    def getPoseState(self, ):
        sols = [sim.simxGetObjectPosition(clientID, self.youb, -1, sim.simx_opmode_oneshot_wait),
        sim.simxGetObjectOrientation(clientID, self.youb, -1, sim.simx_opmode_oneshot_wait)]
        qs_list = []
        for res, qs in sols:
            if not res == sim.simx_return_ok:
                return None
            qs_list.append(qs)
        position, (alpha, beta, gamma) = qs_list
        R_original = euler_to_rotation_matrix(alpha, beta, gamma)
        T = np.array([[0, 0, -1],
                    [0, -1, 0],
                    [1, 0, 0]])
        R_modified = T.dot(R_original)
        alpha_new, beta_new, gamma_new = rotation_matrix_to_euler(R_modified)
        return [position, [alpha_new, beta_new, gamma_new]]
    
    def setPoseState(self, position, orientation):
        res = [sim.simxSetObjectPosition(clientID, self.youb, -1, position,  sim.simx_opmode_oneshot_wait),
               sim.simxSetObjectOrientation(clientID, self.youb, -1, orientation,  sim.simx_opmode_oneshot_wait)]
        for r in res:
            if not r == sim.simx_return_ok:
                return False
            
        return True
    
    def setWeelState(self, fl, rl, rr, fr):


        sim.simxSetJointTargetVelocity(clientID, self.rollingJoint_fl, fl, sim.simx_opmode_oneshot_wait)
        sim.simxSetJointTargetVelocity(clientID, self.rollingJoint_rl, rl, sim.simx_opmode_oneshot_wait)
        sim.simxSetJointTargetVelocity(clientID, self.rollingJoint_rr, rr, sim.simx_opmode_oneshot_wait)
        sim.simxSetJointTargetVelocity(clientID, self.rollingJoint_fr, fr, sim.simx_opmode_oneshot_wait)

    def setMovement(self, forwBackVel, leftRightVel, rotVel):
        fl = -forwBackVel-leftRightVel-rotVel
        rl = -forwBackVel+leftRightVel-rotVel
        rr = -forwBackVel-leftRightVel+rotVel
        fr = -forwBackVel+leftRightVel+rotVel

        self.setWeelState(fl, rl, rr, fr)
    
    def setArmState(self, qs):
        res = [
        sim.simxSetJointTargetPosition(clientID, self.roll_joint_Arm0, qs[0], sim.simx_opmode_oneshot_wait),
        sim.simxSetJointTargetPosition(clientID, self.roll_joint_Arm1, qs[1], sim.simx_opmode_oneshot_wait),
        sim.simxSetJointTargetPosition(clientID, self.roll_joint_Arm2, qs[2], sim.simx_opmode_oneshot_wait),
        sim.simxSetJointTargetPosition(clientID, self.roll_joint_Arm3, qs[3], sim.simx_opmode_oneshot_wait),
        sim.simxSetJointTargetPosition(clientID, self.roll_joint_Arm4, qs[4], sim.simx_opmode_oneshot_wait),]

        for r in res:
            if not r == sim.simx_return_ok:
                return False
            
        return True
    def getDR12State(self):
        sols = [sim.simxGetObjectPosition(clientID, self.dr12, -1, sim.simx_opmode_oneshot_wait),
        sim.simxGetObjectOrientation(clientID, self.dr12, -1, sim.simx_opmode_oneshot_wait)]
        qs_list = []
        for res, qs in sols:
            if not res == sim.simx_return_ok:
                return None
            qs_list.append(qs)
        return qs_list
    
    # def moveArm(self, pose):
    #     xg, yg, zg = pose
    #     (x,y,z), (_, _, yaw) = self.getPoseState()

    #     R = np.array([
    #         [np.cos(yaw), np.sin(yaw), 0],
    #         [-np.sin(yaw), np.cos(yaw), 0],
    #         [0, 0, 1]
    #     ])

    #     vecR = np.array([xg-x, yg-y, zg])

    #     vecG = np.dot(R, vecR)
    #     vecG = vecG - np.array([0.11354, -0.12149, 0.04892])
    #     vecG[0] = -vecG[0]
    #     vecG[1] = -vecG[1]
    #     solved = False
    #     i = 0
    #     l = []
    #     print(vecG)
    #     print('\n\n')
    #     while not solved and i < 2000:
    #         rand_vecG = vecG + np.random.uniform(-0.05, 0.05, vecG.shape)
    #         l.append(rand_vecG)
    #         qs, solves = self.ks.inverse(rand_vecG.tolist(), [0,0,0])
    #         solved = solves[0] and solves[1]
    #         i += 1
    #     if solved:
    #         print(f'{i} ARM Pose:',rand_vecG)
    #         self.setArmState(qs[0])
        
    #     return solved
    
    def lineFollow(self, line, position, orientaion, coa):
        (x_init,y_init), (x_goal,y_goal) = line
        x, y, z = position
        yaw = orientaion[2]

        vec1 = np.array([x_goal - x_init, y_goal - y_init])
        vec2 = np.array([x_goal - x, y_goal - y])
        vec1_hat = vec1 / np.linalg.norm(vec1)

        cross_track_error = np.cross(vec2, vec1_hat)

        course_angle = np.arctan2(vec1[1], vec1[0])
        course_angle_err = ssa(course_angle - yaw)

        dist = np.linalg.norm(vec2)
        vLR = cross_track_error * 10.0
        rotV = -20.0 * course_angle_err
        if dist < coa:
            self.moveToGoal((x_goal, y_goal, course_angle))

        self.setMovement(10.0, vLR, rotV)
        self.setArmState([0, 0, 0, 0, 0])
        return dist < coa

    def moveToGoal(self, goal):
        [x, y, z],orientation = self.getPoseState()
        gx, gy, gyaw = goal
        yaw = orientation[2]

        R = np.array([
            [np.cos(yaw), np.sin(yaw), 0],
            [-np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        e_global = np.array([
            gx - x, gy - y, ssa(gyaw - yaw)
        ])
        e_local = np.dot(R, e_global)
        V = (self.Kp @ e_local)
        self.setMovement(V[0], V[1], V[2])
    
    def exit(self,):
        sim.simxFinish(clientID)

    # def getTrajectory(self, ):

    #     err, (x, y, z) = sim.simxGetObjectPosition(clientID, self.roll_joint_Arm4, -1, sim.simx_opmode_oneshot_wait)
    #     err, (roll, pitch, yaw) = sim.simxGetObjectPosition(clientID, self.roll_joint_Arm4, -1, sim.simx_opmode_oneshot_wait)
        
    #     T_se_initial = euler_to_rotation_matrix(roll, pitch, yaw).tolist()
    #     T_se_initial.append([0, 0, 0, 1])
    #     T_se_initial[0].append(x)
    #     T_se_initial[1].append(y)
    #     T_se_initial[2].append(z)
    #     T_se_initial = np.array(T_se_initial)
    #     (xf , yf, zf), (alpha, beta, gamma) = [0, 0, 0.25], [0, 0, 0]


        
    #     T_se_standoff_s = np.dot(T_sc_initial, T_ce_standoff)
    #     traj = mr.ScrewTrajectory(T_se_initial, T_se_standoff_s, 3, 500, 5)

    # def moveArm(self, dt):
    #     joint_angles = np.array(self.getArmState())
    #     err, (x, y, z) = sim.simxGetObjectPosition(clientID, self.roll_joint_Arm4, -1, sim.simx_opmode_oneshot_wait)
    #     err, (roll, pitch, yaw) = sim.simxGetObjectPosition(clientID, self.roll_joint_Arm4, -1, sim.simx_opmode_oneshot_wait)
    #     (xf , yf, zf), (alpha, beta, gamma) = [0, 0, 0.25], [0, 0, 0]
    #     X_d = euler_to_rotation_matrix(alpha, beta, gamma).tolist()
    #     X_d.append([0, 0, 0, 1])
    #     X_d[0].append(xf)
    #     X_d[1].append(yf)
    #     X_d[2].append(zf)

    #     T_sb = euler_to_rotation_matrix(roll, pitch, yaw).tolist()
    #     T_sb.append([0, 0, 0, 1])
    #     T_sb[0].append(x)
    #     T_sb[1].append(y)
    #     T_sb[2].append(z)

    #     T_sb = np.array(T_sb)
    #     X_d = X_dn = np.array(X_d)

    #     # Compute Transforms T_0e offset arm to end effector, T_eb end-effector to base
    #     T_0e = mr.FKinBody(self.M_0e, self.Blist, joint_angles)
        
    #     T_eb = np.matmul(np.linalg.inv(T_0e), np.linalg.inv(self.T_b0))

    #     X = np.matmul(T_sb,T_be)


    #     # Compute feed forward ref twise
    #     X_inv = np.linalg.inv(X)
    #     X_d_inv = np.linalg.inv(X_d)
    #     X_dn_inv = np.linalg.inv(X_dn)
    #     inp = np.matmul(X_d_inv, X_dn)/dt
    #     Vd = mr.se3ToVec(mr.MatrixLog6(inp))

    #     # Compute Ad x1 xd Vd
    #     adj_x1 = mr.Adjoint(np.matmul(X_inv, X_d))
    #     adj_vd = np.matmul(adj_x1, Vd)

    #     # add error integral
    #     x_error = mr.se3ToVec(mr.MatrixLog6(np.matmul(X_inv, X_d)))
    #     self.ec = self.ec + x_error*dt
    #     v_twist = adj_vd + np.matmul(self.kp, x_error) + np.matmul(self.ki, self.ec)
        
    #     # Calculate Jacobians
    #     J_arm = mr.JacobianBody(self.Blist, joint_angles)
    #     J_base = np.matmul(mr.Adjoint(T_eb), self.h1)
    #     J_end = np.concatenate((J_base, J_arm), axis=1)
    #     J_inv = np.linalg.pinv(J_end)

    #     # Compute Wheel and arm control outputs
    #     controls = np.matmul(J_inv, v_twist)

    #     # Split control into individual variables
    #     joint_control = controls[0:5]
    #     wheel_control = controls[5:9]

    #     self.setArmState(joint_control)
    #     self.setWeelState(*wheel_control.tolist())