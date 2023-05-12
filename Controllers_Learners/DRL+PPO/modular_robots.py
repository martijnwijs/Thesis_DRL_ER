"""Standard modular robots."""

from typing import List

import numpy as np
from revolve2.core.modular_robot import ActiveHinge, Body, Brick


def all() -> List[Body]:
    """
    Get a list of all standard module robots.

    :returns: The list of robots.
    """
    return [
        spider6(),
        spider10(),
        spider14(),
        T6(),
        T10(),
        T14(),
        gecko6(),
        gecko10(),
        gecko14(),
        snake6(),
        snake10(),
        snake14(),
        gecko8()
   
    ]


def get(name: str) -> Body:
    """
    Get a robot by name.

    :param name: The name of the robot to get.
    :returns: The robot with that name.
    :raises ValueError: When a robot with that name does not exist.
    """
    if name == "spider6":
        return spider6()
    elif name == "spider10":
        return spider10()
    elif name == "spider14":
        return spider14()
    elif name == "T6":
        return T6()
    elif name == "T10":
        return T10()
    elif name == "T14":
        return T14()
    elif name == "gecko6":
        return gecko6()
    elif name == "gecko10":
        return gecko10()
    elif name == "gecko14":
        return gecko14()
    elif name == "snake6":
        return snake6()
    elif name == "snake10":
        return snake10()
    elif name == "snake14":
        return snake14()
    elif name == "gecko8":
        return gecko8()
    elif name == "gecko7":
        return gecko7()
    else:
        raise ValueError(f"Robot does not exist: {name}")
def gecko7() -> Body:
    """
    Get the gecko modular robot.

    :returns: the robot.
    """
    body = Body()

    body.core.left = ActiveHinge(0.0)
    body.core.left.attachment = Brick(0.0)

    body.core.right = ActiveHinge(0.0)
    body.core.right.attachment = Brick(0.0)

    body.core.back = ActiveHinge(np.pi / 2.0)
    body.core.back.attachment = Brick(-np.pi / 2.0)
    body.core.back.attachment.front = ActiveHinge(np.pi / 2.0)
    body.core.back.attachment.front.attachment = Brick(-np.pi / 2.0)


    body.core.back.attachment.front.attachment.left = ActiveHinge(0.0)
    body.core.back.attachment.front.attachment.left.attachment = Brick(0.0)
    body.core.back.attachment.front.attachment.right = ActiveHinge(0.0)
    body.core.back.attachment.front.attachment.right.attachment = Brick(0.0)

    body.finalize()
    return body

def gecko8() -> Body:
    """
    Get the gecko modular robot.

    :returns: the robot.
    """
    body = Body()

    body.core.left = ActiveHinge(0.0)
    body.core.left.attachment = Brick(0.0)

    body.core.right = ActiveHinge(0.0)
    body.core.right.attachment = Brick(0.0)

    body.core.back = ActiveHinge(np.pi / 2.0)
    body.core.back.attachment = Brick(-np.pi / 2.0)
    body.core.back.attachment.front = ActiveHinge(np.pi / 2.0)
    body.core.back.attachment.front.attachment = Brick(-np.pi / 2.0)
    body.core.back.attachment.front.attachment.front = ActiveHinge(np.pi / 2.0)
    body.core.back.attachment.front.attachment.front.attachment = Brick(-np.pi / 2.0)

    body.core.back.attachment.front.attachment.left = ActiveHinge(0.0)
    body.core.back.attachment.front.attachment.left.attachment = Brick(0.0)
    body.core.back.attachment.front.attachment.right = ActiveHinge(0.0)
    body.core.back.attachment.front.attachment.right.attachment = Brick(0.0)

    body.finalize()
    return body


def spider6() -> Body:
    """
    Get the spider modular robot.

    :returns: the robot.
    """
    body = Body()

    body.core.left = ActiveHinge(np.pi / 2.0)
    body.core.left.attachment = Brick(-np.pi / 2.0)


    body.core.right = ActiveHinge(np.pi / 2.0)
    body.core.right.attachment = Brick(-np.pi / 2.0)


    body.core.front = ActiveHinge(np.pi / 2.0)
    body.core.front.attachment = Brick(-np.pi / 2.0)


    body.core.back = ActiveHinge(np.pi / 2.0)
    body.core.back.attachment = Brick(-np.pi / 2.0)
    body.core.back.attachment.front = ActiveHinge(0.0)
    body.core.back.attachment.front.attachment = Brick(0.0)

    body.finalize()
    return body

def spider10() -> Body:
    """
    Get the spider modular robot.

    :returns: the robot.
    """
    body = Body()

    body.core.left = ActiveHinge(np.pi / 2.0)
    body.core.left.attachment = Brick(-np.pi / 2.0)
    body.core.left.attachment.front = ActiveHinge(0.0)
    body.core.left.attachment.front.attachment = Brick(0.0)

    body.core.right = ActiveHinge(np.pi / 2.0)
    body.core.right.attachment = Brick(-np.pi / 2.0)
    body.core.right.attachment.front = ActiveHinge(0.0)
    body.core.right.attachment.front.attachment = Brick(0.0)

    body.core.front = ActiveHinge(np.pi / 2.0)
    body.core.front.attachment = Brick(-np.pi / 2.0)
    body.core.front.attachment.front = ActiveHinge(0.0)
    body.core.front.attachment.front.attachment = Brick(0.0)
    
    body.core.back = ActiveHinge(np.pi / 2.0)
    body.core.back.attachment = Brick(-np.pi / 2.0)
    body.core.back.attachment.front = ActiveHinge(0.0)
    body.core.back.attachment.front.attachment = Brick(0.0)
    body.core.back.attachment.front.attachment.front = ActiveHinge(np.pi / 2.0)
    body.core.back.attachment.front.attachment.front.attachment = Brick(-np.pi / 2.0)

    body.finalize()
    return body


def spider14() -> Body:
    """
    Get the spider modular robot.

    :returns: the robot.
    """
    body = Body()

    body.core.left = ActiveHinge(np.pi / 2.0)
    body.core.left.attachment = Brick(-np.pi / 2.0)
    body.core.left.attachment.front = ActiveHinge(0.0)
    body.core.left.attachment.front.attachment = Brick(0.0)
    body.core.left.attachment.front.attachment.front = ActiveHinge(np.pi / 2.0)
    body.core.left.attachment.front.attachment.front.attachment = Brick(-np.pi / 2.0)

    body.core.right = ActiveHinge(np.pi / 2.0)
    body.core.right.attachment = Brick(-np.pi / 2.0)
    body.core.right.attachment.front = ActiveHinge(0.0)
    body.core.right.attachment.front.attachment = Brick(0.0)
    body.core.right.attachment.front.attachment.front = ActiveHinge(np.pi / 2.0)
    body.core.right.attachment.front.attachment.front.attachment = Brick(-np.pi / 2.0)

    body.core.front = ActiveHinge(np.pi / 2.0)
    body.core.front.attachment = Brick(-np.pi / 2.0)
    body.core.front.attachment.front = ActiveHinge(0.0)
    body.core.front.attachment.front.attachment = Brick(0.0)
    body.core.front.attachment.front.attachment.front = ActiveHinge(np.pi / 2.0)
    body.core.front.attachment.front.attachment.front.attachment = Brick(-np.pi / 2.0)
    
    body.core.back = ActiveHinge(np.pi / 2.0)
    body.core.back.attachment = Brick(-np.pi / 2.0)
    body.core.back.attachment.front = ActiveHinge(0.0)
    body.core.back.attachment.front.attachment = Brick(0.0)
    body.core.back.attachment.front.attachment.front = ActiveHinge(np.pi / 2.0)
    body.core.back.attachment.front.attachment.front.attachment = Brick(-np.pi / 2.0)
    body.core.back.attachment.front.attachment.front.attachment.front = ActiveHinge(0.0)
    body.core.back.attachment.front.attachment.front.attachment.front.attachment = Brick(0.0)
    body.finalize()
    return body

def T6() -> Body:
    """
    Get the spider modular robot.

    :returns: the robot.
    """
    body = Body()

    body.core.left = ActiveHinge(np.pi / 2.0)
    body.core.left.attachment = Brick(-np.pi / 2.0)

    body.core.right = ActiveHinge(np.pi / 2.0)
    body.core.right.attachment = Brick(-np.pi / 2.0)

    body.core.back = ActiveHinge(np.pi / 2.0)
    body.core.back.attachment = Brick(-np.pi / 2.0)
    body.core.back.attachment.front = ActiveHinge(0.0)
    body.core.back.attachment.front.attachment = Brick(0.0)
    body.core.back.attachment.front.attachment.front = ActiveHinge(np.pi / 2.0)
    body.core.back.attachment.front.attachment.front.attachment = Brick(-np.pi / 2.0)
    body.finalize()
    return body

def T10() -> Body:
    """
    Get the spider modular robot.

    :returns: the robot.
    """
    body = Body()

    body.core.left = ActiveHinge(np.pi / 2.0)
    body.core.left.attachment = Brick(-np.pi / 2.0)
    body.core.left.attachment.front = ActiveHinge(0.0)
    body.core.left.attachment.front.attachment = Brick(0.0)

    body.core.right = ActiveHinge(np.pi / 2.0)
    body.core.right.attachment = Brick(-np.pi / 2.0)
    body.core.right.attachment.front = ActiveHinge(0.0)
    body.core.right.attachment.front.attachment = Brick(0.0)

    body.core.back = ActiveHinge(np.pi / 2.0)
    body.core.back.attachment = Brick(-np.pi / 2.0)
    body.core.back.attachment.front = ActiveHinge(0.0)
    body.core.back.attachment.front.attachment = Brick(0.0)
    body.core.back.attachment.front.attachment.front = ActiveHinge(np.pi / 2.0)
    body.core.back.attachment.front.attachment.front.attachment = Brick(-np.pi / 2.0)
    body.core.back.attachment.front.attachment.front.attachment.front = ActiveHinge(0.0)
    body.core.back.attachment.front.attachment.front.attachment.front.attachment = Brick(0.0)   
    body.core.back.attachment.front.attachment.front.attachment.front.attachment.front = ActiveHinge(np.pi / 2.0)
    body.core.back.attachment.front.attachment.front.attachment.front.attachment.front.attachment = Brick(-np.pi / 2.0)

    body.finalize()
    return body


def T14() -> Body:
    """
    Get the spider modular robot.

    :returns: the robot.
    """
    body = Body()

    body.core.left = ActiveHinge(np.pi / 2.0)
    body.core.left.attachment = Brick(-np.pi / 2.0)
    body.core.left.attachment.front = ActiveHinge(0.0)
    body.core.left.attachment.front.attachment = Brick(0.0)
    body.core.left.attachment.front.attachment.front = ActiveHinge(np.pi / 2.0)
    body.core.left.attachment.front.attachment.front.attachment = Brick(-np.pi / 2.0)

    body.core.right = ActiveHinge(np.pi / 2.0)
    body.core.right.attachment = Brick(-np.pi / 2.0)
    body.core.right.attachment.front = ActiveHinge(0.0)
    body.core.right.attachment.front.attachment = Brick(0.0)
    body.core.right.attachment.front.attachment.front = ActiveHinge(np.pi / 2.0)
    body.core.right.attachment.front.attachment.front.attachment = Brick(-np.pi / 2.0)

    body.core.back = ActiveHinge(np.pi / 2.0)
    body.core.back.attachment = Brick(-np.pi / 2.0)
    body.core.back.attachment.front = ActiveHinge(0.0)
    body.core.back.attachment.front.attachment = Brick(0.0)
    body.core.back.attachment.front.attachment.front = ActiveHinge(np.pi / 2.0)
    body.core.back.attachment.front.attachment.front.attachment = Brick(-np.pi / 2.0)
    body.core.back.attachment.front.attachment.front.attachment.front = ActiveHinge(0.0)
    body.core.back.attachment.front.attachment.front.attachment.front.attachment = Brick(0.0)   
    body.core.back.attachment.front.attachment.front.attachment.front.attachment.front = ActiveHinge(np.pi / 2.0)
    body.core.back.attachment.front.attachment.front.attachment.front.attachment.front.attachment = Brick(-np.pi / 2.0)   
    body.core.back.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front = ActiveHinge(0.0)
    body.core.back.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment = Brick(0.0)       
    body.core.back.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front = ActiveHinge(np.pi / 2.0)
    body.core.back.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment = Brick(-np.pi / 2.0)       
    body.finalize()
    return body

def gecko6() -> Body:
    """
    Get the gecko modular robot.

    :returns: the robot.
    """
    body = Body()

    body.core.left = ActiveHinge(0.0)
    body.core.left.attachment = Brick(0.0)

    body.core.right = ActiveHinge(0.0)
    body.core.right.attachment = Brick(0.0)

    body.core.back = ActiveHinge(np.pi / 2.0)
    body.core.back.attachment = Brick(-np.pi / 2.0)

    body.core.back.attachment.left = ActiveHinge(0.0)
    body.core.back.attachment.left.attachment = Brick(0.0)
    body.core.back.attachment.right = ActiveHinge(0.0)
    body.core.back.attachment.right.attachment = Brick(0.0)

    body.finalize()
    return body

def gecko10() -> Body:
    """
    Get the gecko modular robot.

    :returns: the robot.
    """
    body = Body()

    body.core.left = ActiveHinge(0.0)
    body.core.left.attachment = Brick(0.0)

    body.core.right = ActiveHinge(0.0)
    body.core.right.attachment = Brick(0.0)

    body.core.back = ActiveHinge(np.pi / 2.0)
    body.core.back.attachment = Brick(-np.pi / 2.0)
    body.core.back.attachment.front = ActiveHinge(0.0)
    body.core.back.attachment.front.attachment = Brick(0.0)
    body.core.back.attachment.front.attachment.front = ActiveHinge(np.pi / 2.0)
    body.core.back.attachment.front.attachment.front.attachment = Brick(-np.pi / 2.0)
    body.core.back.attachment.front.attachment.front.attachment.front = ActiveHinge(0.0)
    body.core.back.attachment.front.attachment.front.attachment.front.attachment = Brick(0.0)
    body.core.back.attachment.front.attachment.front.attachment.front.attachment.front = ActiveHinge(np.pi / 2.0)
    body.core.back.attachment.front.attachment.front.attachment.front.attachment.front.attachment = Brick(-np.pi / 2.0)

    body.core.back.attachment.front.attachment.front.attachment.left = ActiveHinge(0.0)
    body.core.back.attachment.front.attachment.front.attachment.left.attachment = Brick(0.0)
    body.core.back.attachment.front.attachment.front.attachment.right = ActiveHinge(0.0)
    body.core.back.attachment.front.attachment.front.attachment.right.attachment = Brick(0.0)

    body.finalize()
    return body

def gecko14() -> Body:
    """
    Get the gecko modular robot.

    :returns: the robot.
    """
    body = Body()

    body.core.left = ActiveHinge(0.0)
    body.core.left.attachment = Brick(0.0)
    body.core.left.attachment.front = ActiveHinge(np.pi / 2.0)
    body.core.left.attachment.front.attachment = Brick(-np.pi / 2.0)

    body.core.right = ActiveHinge(0.0)
    body.core.right.attachment = Brick(0.0)
    body.core.right.attachment.front = ActiveHinge(np.pi / 2.0)
    body.core.right.attachment.front.attachment = Brick(-np.pi / 2.0)

    body.core.back = ActiveHinge(np.pi / 2.0)
    body.core.back.attachment = Brick(-np.pi / 2.0)
    body.core.back.attachment.front = ActiveHinge(0.0)
    body.core.back.attachment.front.attachment = Brick(0.0)
    body.core.back.attachment.front.attachment.front = ActiveHinge(np.pi / 2.0)
    body.core.back.attachment.front.attachment.front.attachment = Brick(-np.pi / 2.0)
    body.core.back.attachment.front.attachment.front.attachment.front = ActiveHinge(0.0)
    body.core.back.attachment.front.attachment.front.attachment.front.attachment = Brick(0.0)
    body.core.back.attachment.front.attachment.front.attachment.front.attachment.front = ActiveHinge(np.pi / 2.0)
    body.core.back.attachment.front.attachment.front.attachment.front.attachment.front.attachment = Brick(-np.pi / 2.0)

    body.core.back.attachment.front.attachment.front.attachment.left = ActiveHinge(0.0)
    body.core.back.attachment.front.attachment.front.attachment.left.attachment = Brick(0.0)
    body.core.back.attachment.front.attachment.front.attachment.left.attachment.front = ActiveHinge(np.pi / 2.0)
    body.core.back.attachment.front.attachment.front.attachment.left.attachment.front.attachment = Brick(-np.pi / 2.0)

    body.core.back.attachment.front.attachment.front.attachment.right = ActiveHinge(0.0)
    body.core.back.attachment.front.attachment.front.attachment.right.attachment = Brick(0.0)
    body.core.back.attachment.front.attachment.front.attachment.right.attachment.front = ActiveHinge(np.pi / 2.0)
    body.core.back.attachment.front.attachment.front.attachment.right.attachment.front.attachment = Brick(-np.pi / 2.0)

    body.finalize()
    return body

def snake6() -> Body:
    """
    Get the snake modular robot.

    :returns: the robot.
    """
    body = Body()

    body.core.back = ActiveHinge(0.0)
    body.core.back.attachment = Brick(0.0)
    body.core.back.attachment.front = ActiveHinge(np.pi / 2.0)
    body.core.back.attachment.front.attachment = Brick(-np.pi / 2.0)
    body.core.back.attachment.front.attachment.front = ActiveHinge(0.0)
    body.core.back.attachment.front.attachment.front.attachment = Brick(0.0)
    body.core.back.attachment.front.attachment.front.attachment.front = ActiveHinge(
        np.pi / 2.0
    )
    body.core.back.attachment.front.attachment.front.attachment.front.attachment = (
        Brick(-np.pi / 2.0)
    )
    body.core.back.attachment.front.attachment.front.attachment.front.attachment.front = ActiveHinge(
        0.0
    )
    body.core.back.attachment.front.attachment.front.attachment.front.attachment.front.attachment = Brick(
        0.0
    )

    body.finalize()
    return body

def snake10() -> Body:
    """
    Get the snake modular robot.

    :returns: the robot.
    """
    body = Body()

    body.core.back = ActiveHinge(0.0)
    body.core.back.attachment = Brick(0.0)
    body.core.back.attachment.front = ActiveHinge(np.pi / 2.0)
    body.core.back.attachment.front.attachment = Brick(-np.pi / 2.0)
    body.core.back.attachment.front.attachment.front = ActiveHinge(0.0)
    body.core.back.attachment.front.attachment.front.attachment = Brick(0.0)
    body.core.back.attachment.front.attachment.front.attachment.front = ActiveHinge(
        np.pi / 2.0
    )
    body.core.back.attachment.front.attachment.front.attachment.front.attachment = (
        Brick(-np.pi / 2.0)
    )
    body.core.back.attachment.front.attachment.front.attachment.front.attachment.front = ActiveHinge(
        0.0
    )
    body.core.back.attachment.front.attachment.front.attachment.front.attachment.front.attachment = Brick(
        0.0
    )
    body.core.back.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front = ActiveHinge(
        np.pi / 2.0
    )
    body.core.back.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment = (
        Brick(-np.pi / 2.0)
    )
    body.core.back.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front = ActiveHinge(
        0.0
    )
    body.core.back.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment = Brick(
        0.0
    )
    body.core.back.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front = ActiveHinge(
        np.pi / 2.0
    )
    body.core.back.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment = (
        Brick(-np.pi / 2.0)
    )
    body.core.back.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front = ActiveHinge(
        0.0
    )
    body.core.back.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment = Brick(
        0.0
    )
    body.finalize()
    return body



    body.finalize()
    return body


def snake14() -> Body:
    """
    Get the snake modular robot.

    :returns: the robot.
    """
    body = Body()

    body.core.back = ActiveHinge(0.0)
    body.core.back.attachment = Brick(0.0)
    body.core.back.attachment.front = ActiveHinge(np.pi / 2.0)
    body.core.back.attachment.front.attachment = Brick(-np.pi / 2.0)
    body.core.back.attachment.front.attachment.front = ActiveHinge(0.0)
    body.core.back.attachment.front.attachment.front.attachment = Brick(0.0)
    body.core.back.attachment.front.attachment.front.attachment.front = ActiveHinge(
        np.pi / 2.0
    )
    body.core.back.attachment.front.attachment.front.attachment.front.attachment = (
        Brick(-np.pi / 2.0)
    )
    body.core.back.attachment.front.attachment.front.attachment.front.attachment.front = ActiveHinge(
        0.0
    )
    body.core.back.attachment.front.attachment.front.attachment.front.attachment.front.attachment = Brick(
        0.0
    )
    body.core.back.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front = ActiveHinge(
        np.pi / 2.0
    )
    body.core.back.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment = (
        Brick(-np.pi / 2.0)
    )
    body.core.back.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front = ActiveHinge(
        0.0
    )
    body.core.back.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment = Brick(
        0.0
    )
    body.core.back.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front = ActiveHinge(
        np.pi / 2.0
    )
    body.core.back.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment = (
        Brick(-np.pi / 2.0)
    )
    body.core.back.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front = ActiveHinge(
        0.0
    )
    body.core.back.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment = Brick(
        0.0
    )
    body.core.back.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front = ActiveHinge(
        np.pi / 2.0
    )
    body.core.back.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment = (
        Brick(-np.pi / 2.0)
    )
    body.core.back.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front = ActiveHinge(
        0.0
    )
    body.core.back.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment = Brick(
        0.0
    )
    body.core.back.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front = ActiveHinge(
        np.pi / 2.0
    )
    body.core.back.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment = (
        Brick(-np.pi / 2.0)
    )
    body.core.back.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front = ActiveHinge(
        0.0
    )
    body.core.back.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment = Brick(
        0.0
    )
    body.core.back.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front = ActiveHinge(
        np.pi / 2.0
    )
    body.core.back.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment = (
        Brick(-np.pi / 2.0)
    )   
    body.finalize()
    return body













