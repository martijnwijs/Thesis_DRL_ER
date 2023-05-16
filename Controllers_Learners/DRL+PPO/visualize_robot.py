"""Visualize and run a modular robot using Mujoco."""

import math
from random import Random
import time
from pyrr import Quaternion, Vector3
from revolve2.actor_controller import ActorController
from revolve2.core.modular_robot import ActiveHinge, Body, Brick, ModularRobot
from revolve2.core.modular_robot.brains import BrainCpgNetworkNeighbourRandom
from revolve2.core.physics.running import ActorControl, Batch, Environment, PosedActor
from revolve2.runners.mujoco import LocalRunner
import modular_robots
import numpy as np
class Simulator:
    """
    Simulator setup.

    Simulates using Mujoco.
    Defines a control function that steps the controller and applies the degrees of freedom the controller provides.
    """

    _controller: ActorController

    async def simulate(self, robot: ModularRobot, control_frequency: float) -> None:
        """
        Simulate a robot.

        :param robot: The robot to simulate.
        :param control_frequency: Control frequency for the simulator.
        """
        batch = Batch(
            simulation_time=1,
            sampling_frequency=0.0001,
            control_frequency=control_frequency,
            control=self._control,
        )

        actor, self._controller = robot.make_actor_and_controller()
        bounding_box = actor.calc_aabb()

        env = Environment()
        env.actors.append(
            PosedActor(
                actor,
                Vector3(
                    [
                        0.0,
                        0.0,
                        bounding_box.size.z / 2.0 - bounding_box.offset.z,
                    ]
                ),
                Quaternion(),
                [0.0 for _ in self._controller.get_dof_targets()],
            )
        )
        batch.environments.append(env)

        runner = LocalRunner()   # this runs the mujoco environment
        await runner.run_batch(batch)  # runs the batch inside the function. batch contains everything sim time, samp freq, contr, freq, control mechanicsm, Environment

    def _control(
        self, environment_index: int, dt: float, control: ActorControl
    ) -> None:
        time.sleep(2)
        self._controller.step(dt)
        print(" control")
        control.set_dof_targets(0, self._controller.get_dof_targets())


async def main() -> None:
    """Run the simulation."""
    rng = Random()
    rng.seed(5)
    # creating the body object
    body= 'snake14'
    body = modular_robots.get(body)


    brain = BrainCpgNetworkNeighbourRandom(rng)
    robot = ModularRobot(body, brain)

    sim = Simulator()
    await sim.simulate(robot, 1000) # modular robot, control frequency
    print("finished simulation")

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
