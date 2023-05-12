import csv
import math
import os
import sys
from random import Random
from typing import List
import numpy as np
import sqlalchemy
import torch
from pyrr import Quaternion, Matrix33, Matrix44, Vector3
import pyrr
from brain import PPObrain
from config import ACTION_CONSTRAINT, NUM_ITERATIONS, NUM_PARALLEL_AGENT
from pyrr import Quaternion, Vector3
from revolve2.actor_controller import ActorController
from revolve2.core.database.serializers import DbNdarray1xn
from revolve2.core.modular_robot import Body
from revolve2.core.physics.actor import Actor
from revolve2.core.physics.running import (ActorControl, ActorState, Batch,
                                           Environment, PosedActor, Runner)
from runner_train_mujoco import LocalRunnerTrain
from sqlalchemy.ext.declarative import declarative_base

# For testing whether a number is close to zero
_FLOAT_EPS = np.finfo(np.float64).eps
_EPS4 = _FLOAT_EPS * 4.0

class PPOOptimizer():
    _runner: Runner
    _body: Body
    _actor: Actor
    _dof_ids: List[int]
    _controller: ActorController

    _rng: Random

    _simulation_time: int
    _sampling_frequency: float
    _control_frequency: float
    _visualize: bool
    _num_agents: int
    _task: str
    _file_path: str

    def __init__(
            self,
            rng: Random,
            simulation_time: int,
            sampling_frequency: float,
            control_frequency: float,
            visualize: bool,
            num_agents: int,
            robot_body: Body,
            task: str,
            file_path: str,
    ) -> None:

        self._visualize = visualize
        print("torch" in sys.modules)
        self._init_runner()
        self._rng = rng
        self._simulation_time = simulation_time
        self._sampling_frequency = sampling_frequency
        self._control_frequency = control_frequency
        self._num_agents = num_agents
        self._body = robot_body
        self._task = task
        self._actor, self._dof_ids = self._body.to_actor()
        self._file_path = file_path

    def _init_runner(self) -> None:
        self._runner = LocalRunnerTrain(headless=(not self._visualize))

    def _control(self, environment_index: int, dt: float, control: ActorControl, observations):
        action, value, logp = self._controller.get_dof_targets([torch.tensor(obs) for obs in observations])
        control.set_dof_targets(0, torch.clip(action, -ACTION_CONSTRAINT, ACTION_CONSTRAINT))
        # controller.train() TODO
        # here you could map action to cpg 
        #print("action", action)
        return action.tolist(), value.item(), logp.item()

    async def train(self, from_checkpoint: bool = False):
        """
        Create the agents, insert them in the simulation and run it
        args:
            agents: list of agents to simulate
            from_checkpoint: if True resumes training from the last checkpoint
        """
        print(from_checkpoint)
        # prepare file to log statistics
        if not from_checkpoint:
            with open(self._file_path + '/statistics.csv', 'w', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['mean_rew', 'mean_val'])
            with open(self._file_path + '/fitnesses.csv', 'w', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['individual_id', 'fitness'])

        # all parallel agents share the same brain
        brain = PPObrain(from_checkpoint=from_checkpoint)
        self._controller = brain.make_controller(self._body, self._dof_ids, self._file_path)

        for iteration_num in range(NUM_ITERATIONS):

            batch = Batch(
                simulation_time=self._simulation_time,
                sampling_frequency=self._sampling_frequency,
                control_frequency=self._control_frequency,
                control=self._control,
            )

            # insert agents in the simulation environment
            bounding_box = self._actor.calc_aabb()
            for _ in range(self._num_agents):
                env = Environment()
                env.actors.append(
                    PosedActor(
                        self._actor,
                        Vector3(
                            [
                                0.0,
                                0.0,
                                bounding_box.size.z / 2.0,
                            ]
                        ),
                        Quaternion(),
                        [0.0 for _ in range(len(self._dof_ids))],
                    )
                )
                batch.environments.append(env)

            # run the simulation
            batch_results = await self._runner.run_batch(batch, self._controller, self._num_agents)  ########################

            fitnesses = [
                self._calculate_fitness(
                    environment_result.environment_states[0].actor_states[0],
                    environment_result.environment_states[-1].actor_states[0],
                    environment_result.environment_states,
                    self._task
                )
                for environment_result in batch_results.environment_results
            ]

            with open(self._file_path + '/fitnesses.csv', 'a', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                for idx in range(self._num_agents):
                    id = iteration_num * NUM_PARALLEL_AGENT + idx
                    writer.writerow([id, fitnesses[idx]])
        return

    @staticmethod
    def _calculate_fitness(begin_state: ActorState, end_state: ActorState, states: ActorState, task) -> float:
        
        if task == "gait":
            #print("shouldnt be here")
            # distance traveled on the xy plane
            return math.sqrt(
                (begin_state.position[0] - end_state.position[0]) ** 2
                + ((begin_state.position[1] - end_state.position[1]) ** 2)
            )


        if task == "rotation":
            vertical_angle_limit = math.pi/4.

            total_angle = 0.0
            vertical_limit = math.sin(vertical_angle_limit)
            unit_vectors = []
            normal_vector = Vector3([1.,0.,0.])
            
            for i in range(0, len(states)):
                quat = states[i].actor_states[0].orientation
                #quat = np.array([quat[0], quat[1], quat[2], quat[3]])
                #print("quat", quat)
                quaternion = pyrr.quaternion.create(x=quat[0], y=quat[1], z=quat[2], w=quat[3], dtype=None) # create quaternion in pyrr
                #print("quaternion", quaternion)
                vect = Quaternion(quaternion)*Vector3(normal_vector) # rotate vector with quaternion
                #vect2 = pyrr.quaternion.apply_to_vector(quaternion, normal_vector)
                #print("vect", vect)
                #print("vect2", vect2)
                unit_vectors.append(vect)
                ''' 
                euler = quat2euler(quat)
                x = math.cos(euler[2])*math.cos(euler[1])
                y = math.sin(euler[2])*math.cos(euler[1])
                z = math.sin(euler[1])       
                unit_vectors.append((x,y,z))
                '''
            #print("unit vectors", unit_vectors)
            for i in range(1, len(states)):
                
                u: Vector3 = unit_vectors[i-1]
                v: Vector3 = unit_vectors[i]
                #print("u", u)
                #print("v", v)
                if abs(u.z) > vertical_limit:
                    return total_angle

                dot = u.x*v.x + u.y*v.y       # dot product between [x1, y1] and [x2, y2]
                det = u.x*v.y - u.y*v.x       # determinant
                delta = math.atan2(det, dot)  # atan2(y, x) or atan2(sin, cos)
                #print("delta", delta)
                total_angle += delta
            #print("total_angle", total_angle)
            #print("---------------------")
            
            return total_angle




 
def euler_from_quaternion(quat):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """

        x = quat[0]
        y = quat[1]
        z = quat[2]
        w = quat[3]
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
        return [roll_x, pitch_y, yaw_z] # in radians

def quat2mat(quat):

    """ Convert Quaternion to Euler Angles.  See rotation.py for notes """
    quat = np.asarray(quat, dtype=np.float64)
    assert quat.shape[-1] == 4, "Invalid shape quat {}".format(quat)

    #x = quat[0]
    #y = quat[1]
    #z = quat[2]
    #w = quat[3]
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    Nq = np.sum(quat * quat, axis=-1)
    s = 2.0 / Nq
    X, Y, Z = x * s, y * s, z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ, zZ = y * Y, y * Z, z * Z

    mat = np.empty(quat.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 0, 0] = 1.0 - (yY + zZ)
    mat[..., 0, 1] = xY - wZ
    mat[..., 0, 2] = xZ + wY
    mat[..., 1, 0] = xY + wZ
    mat[..., 1, 1] = 1.0 - (xX + zZ)
    mat[..., 1, 2] = yZ - wX
    mat[..., 2, 0] = xZ - wY
    mat[..., 2, 1] = yZ + wX
    mat[..., 2, 2] = 1.0 - (xX + yY)
    return np.where((Nq > _FLOAT_EPS)[..., np.newaxis, np.newaxis], mat, np.eye(3))

def mat2euler(mat):
    """ Convert Rotation Matrix to Euler Angles.  See rotation.py for notes """
    mat = np.asarray(mat, dtype=np.float64)
    assert mat.shape[-2:] == (3, 3), "Invalid shape matrix {}".format(mat)

    cy = np.sqrt(mat[..., 2, 2] * mat[..., 2, 2] + mat[..., 1, 2] * mat[..., 1, 2])
    condition = cy > _EPS4
    euler = np.empty(mat.shape[:-1], dtype=np.float64)
    euler[..., 2] = np.where(condition,
                             -np.arctan2(mat[..., 0, 1], mat[..., 0, 0]),
                             -np.arctan2(-mat[..., 1, 0], mat[..., 1, 1]))
    euler[..., 1] = np.where(condition,
                             -np.arctan2(-mat[..., 0, 2], cy),
                             -np.arctan2(-mat[..., 0, 2], cy))
    euler[..., 0] = np.where(condition,
                             -np.arctan2(mat[..., 1, 2], mat[..., 2, 2]),
                             0.0)
    return euler

def quat2euler(quat):
    """ Convert Quaternion to Euler Angles.  See rotation.py for notes """
    return mat2euler(quat2mat(quat))


'''

elif task == "rotation":  
            total_angle = 0.
            for i in range(1, len(states)):
                quaternion = states[i].actor_states[0].orientation
                euler = quat2euler(quaternion)
                # from: https://code-examples.net/en/q/d6a4f5
                # webarchive: https://web.archive.org/web/20210818154647/https://code-examples.net/en/q/d6a4f5
                # more info: https://en.wikipedia.org/wiki/Atan2
                # Just like the dot product is proportional to the cosine of the angle,
                # the determinant is proportional to its sine. So you can compute the angle like this:
                #
                # dot = x1*x2 + y1*y2      # dot product between [x1, y1] and [x2, y2]
                # det = x1*y2 - y1*x2      # determinant
                # angle = atan2(det, dot)  # atan2(y, x) or atan2(sin, cos)
                #
                # The function atan2(y,x) (from "2-argument arctangent") is defined as the angle in the Euclidean plane,
                # given in radians, between the positive x axis and the ray to the point (x, y) ≠ (0, 0).

                # u = prev vector
                # v = curr vector
                u: Vector3 = vec_list[i-1]
                v: Vector3 = vec_list[i]

                # if vector is too vertical, fail the fitness
                # (assuming these are unit vectors)
                if abs(u.z) > vertical_limit:
                    return total_angle

                dot = u.x*v.x + u.y*v.y       # dot product between [x1, y1] and [x2, y2]
                det = u.x*v.y - u.y*v.x       # determinant
                delta = math.atan2(det, dot)  # atan2(y, x) or atan2(sin, cos)

                total_angle += delta

            return total_angle


'''
'''        elif task == "rotation":
            total_angle = 0.
            #f = open('rotation.csv', 'w')
            #writer = csv.writer(f)
            #writer.writerow(['euler_z', 'last_euler_z', 'net_angle', 'total_angle'])
            for i in range(len(states)):

                #print(states[i].actor_states[0].orientation)
                quaternion = states[i].actor_states[0].orientation
                if i ==0: # init
                    #last_euler_z = euler_from_quaternion(quaternion[0], quaternion[1], quaternion[2], quaternion[3])[2]
                    last_euler_z = quat2euler(quaternion)[2]
                    euler_z = last_euler_z

                last_euler_z = euler_z
                #euler_z = euler_from_quaternion(quaternion[0], quaternion[1], quaternion[2], quaternion[3])[2]
                euler_z = quat2euler(quaternion)[2]        
                #print("quaternion", quaternion)
                #print("euler z", euler_z)
                net_angle = euler_z - last_euler_z
                if net_angle > math.pi:
                    net_angle = -(2*math.pi - net_angle)
                if net_angle < -math.pi:
                    net_angle = (2*math.pi - abs(net_angle))
                total_angle += net_angle

                #print("last euler z", last_euler_z)
                #print("net_angle", net_angle)
                #print("total angle",total_angle)
                #print("-------------------")
                #row = [euler_z, last_euler_z, net_angle, total_angle]
                #writer.writerow(row)
                
            penalty = math.sqrt(
                (begin_state.position[0] - end_state.position[0]) ** 2
                + ((begin_state.position[1] - end_state.position[1]) ** 2))
            #f.close()
            #sys.quit()
            return total_angle 
'''

'''
        elif task =="rotation":
            euler_z = 0.
            last_euler_z = 0.
            net_angle = 0.
            total_angle = 0.
            for i in range(1, len(states)):
                quat = states[i].actor_states[0].orientation
                last_quat = states[i-1].actor_states[0].orientation
                #print("quat",quat)
                #euler_z = quat2euler(quat)[2]
                euler_z = euler_from_quaternion(quat)[2]
                #print("euler_z",euler_z)
                #print("euler_z_2", euler_z_2)
                last_euler_z = quat2euler(last_quat)[2]
                pi_2 = math.pi / 2.0

                if last_euler_z > pi_2 and euler_z < - pi_2:  # rotating left
                    net_angle = 2.0 * math.pi + euler_z - last_euler_z
                elif (last_euler_z < - pi_2) and (euler_z > pi_2):
                    net_angle = - (2.0 * math.pi - euler_z + last_euler_z)
                else:
                    net_angle = euler_z - last_euler_z
                total_angle += net_angle

            fitness_value: float = total_angle # - factor_orien_ds * robot_manager._dist
            #sys.quit()
            return fitness_value  


'''



'''

V[0] = 2 * (x * z - w * y)
V[1] = 2 * (y * z + w * x)
V[2] = 1 - 2 * (x * x + y * y)
'''