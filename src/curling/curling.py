from __future__ import annotations

import numpy as np
import cv2

from typing import ClassVar, List, Optional, Tuple
from dataclasses import dataclass

from .enums import Colors, DisplayTime, StoneColor, SimulationState
from .constants import CurlingConstants, SimulationConstants

class Curling:
    constants: CurlingConstants = CurlingConstants()
    starting_button_distance: np.floating = np.array(38.405) # distance to button from where stone is released
    pitch_length: np.floating = np.array(45.720)
    pitch_width: np.floating = np.array(4.750)
    hog_line_position: np.floating = np.array(11.888) # distance from back line to hog line
    tee_line_position: np.floating = np.array(5.487) # distance from back line to tee line
    button_position = np.array((0., -tee_line_position))
    target_radii: np.ndarray = np.array((0.152, 0.610, 1.219, 1.829)) # radii of rings in the circle
    house_radius: np.floating = np.array((1.996)) # distance from centre of stone to button
    vertical_lines: np.ndarray = np.array((-.457, 0, .457)) # positioning of vertical lines
    horizontal_lines: np.ndarray = np.array((3.658, tee_line_position, hog_line_position, 33.832, 40.233, 42.062)) # positioning of horizontal lines
    num_stones_per_end: int = 8
    def __init__(self, starting_color: Optional[StoneColor] = None):
        self.reset(starting_color)

    def reset(self, starting_color: Optional[StoneColor] = None):
        self.stones: List[Stone] = []
        self.next_stone_colour = starting_color or np.random.choice([StoneColor.RED, StoneColor.YELLOW])

    def step(self, simulation_constants: SimulationConstants = SimulationConstants()) -> SimulationState:
        finished = SimulationState.FINISHED
        invalid_stone_indices = []
        for i, stone in enumerate(self.stones):
            if stone.step(simulation_constants) == SimulationState.UNFINISHED:
                finished = SimulationState.UNFINISHED
            if self.out_of_bounds(stone):
                invalid_stone_indices.append(i)
        for invalid_index in reversed(invalid_stone_indices):
            self.stones.pop(invalid_index)
        Stone.handle_collisions(self.stones)
        return finished

    def render(self) -> Canvas:
        canvas = Canvas(self, pixels_per_meter=920//self.pitch_length)
        canvas.draw_vertical_lines(self.vertical_lines)
        canvas.draw_targets(buffer=self.tee_line_position, radii=self.target_radii)
        canvas.draw_horizontal_lines(self.horizontal_lines)

        for stone in self.stones:
            canvas.draw_stone(stone)
        return canvas

    def out_of_bounds(self, stone: Stone) -> bool:
        return np.abs(stone.position[0]) > self.pitch_width / 2 or stone.position[1] > 0

    def button_distance(self, stone: Stone) -> float:
        return np.linalg.norm(stone.position - self.button_position)

    def in_house(self, stone: Stone) -> bool:
        return self.button_distance(stone) < self.house_radius

    def in_fgz(self, stone: Stone) -> bool:
        """determines if a stone is in the free guard zone"""
        return (-stone.position[1] >= self.tee_line_position and -stone.position[1] <= self.hog_line_position) and not self.in_house(stone)

    def display(self, constants: SimulationConstants = SimulationConstants()):
        self.render().display(constants)

    def create_stone(self, stone_throw: StoneThrow):
        return Stone(
            color=stone_throw.color,
            velocity=stone_throw.velocity,
            angle=stone_throw.angle,
            spin=stone_throw.spin,
            position=(0, -self.pitch_length+self.hog_line_position),
            curling_constants=self.constants
        )

    def throw(self, stone_throw: StoneThrow, constants: SimulationConstants = SimulationConstants(), display: bool=False):
        assert stone_throw.color == self.next_stone_colour
        self.next_stone_colour = ~self.next_stone_colour
        self.stones.append(
            self.create_stone(
                stone_throw
            )

        )
        while self.step(constants) == SimulationState.UNFINISHED:
            if display:
                self.display(constants)

    def evaluate_position(self):
        stone_distances = [self.button_distance(stone) for stone in self.stones]
        if len(self.stones) == 0:
            distance_ordering = []
        else:
            distance_ordering = np.argsort(stone_distances)
        ordered_stones = [self.stones[index] for index in distance_ordering]
        score = 0
        for stone in ordered_stones:
            if score * stone.color < 0 or not self.in_house(stone):
                break
            score += stone.color
        return score

class Canvas:
    WINDOW_NAME = "Curling"
    DISPLAY_TIME = DisplayTime.TWICE_SPEED
    def __init__(self, curling: Curling, pixels_per_meter: int = 20):
        self.pitch_width = curling.pitch_width
        self.pitch_length = curling.pitch_length
        self.pixels_per_meter = pixels_per_meter
        self.canvas_width = int(self.pitch_width * pixels_per_meter)
        self.canvas_height = int(self.pitch_length * pixels_per_meter)
        self._canvas = np.tile(np.array(Colors.BACKGROUND.value).astype(np.uint8), (self.canvas_height, self.canvas_width, 1))

    def adjust_coordinates(self, xy: Tuple[float, float]) -> Tuple[int, int]:
        return (int((xy[0] + self.pitch_width / 2) * self.pixels_per_meter), int((-xy[1]) * self.pixels_per_meter))

    def convert_radius(self, radius: float) -> float:
        return int(radius * self.pixels_per_meter)

    def draw_target(self, radii: List[float], offset: float):
        TARGET_COLOURS = (Colors.RED, Colors.WHITE, Colors.BLUE, Colors.WHITE)
        for color, radius in zip(TARGET_COLOURS, reversed(sorted((radii)))):
            cv2.circle(self._canvas, center=self.adjust_coordinates((0, offset)), radius=self.convert_radius(radius), color=color.value, thickness=-1)

    def draw_horizontal_lines(self, lines: List[float]):
        for height in lines:
            cv2.line(self._canvas, self.adjust_coordinates((-self.pitch_width, -height)), self.adjust_coordinates((self.pitch_width, -height)), color=Colors.WHITE.value, thickness=1)

    def draw_vertical_lines(self, lines: List[float]):
        for width in lines:
            cv2.line(self._canvas, self.adjust_coordinates((width, 0)), self.adjust_coordinates((width, -self.pitch_length)), color=Colors.WHITE.value, thickness=1)

    def draw_targets(self, radii: List[float], buffer: float):
        self.draw_target(radii=radii, offset=-buffer)
        self.draw_target(radii, buffer-self.pitch_length)

    def draw_stone(self, stone: Stone):
        stone_color = Colors.RED if stone.color == StoneColor.RED else Colors.YELLOW
        cv2.circle(self._canvas, center=self.adjust_coordinates(stone.position), radius=self.convert_radius(stone.outer_radius), color=stone_color.value, thickness=-1)
        cv2.circle(self._canvas, center=self.adjust_coordinates(stone.position), radius=self.convert_radius(stone.outer_radius), color=Colors.GRAY.value, thickness=1)
        handle_offset = stone.outer_radius * (np.cos(stone.angular_position), np.sin(stone.angular_position))
        cv2.line(self._canvas, pt1=self.adjust_coordinates(stone.position + handle_offset), pt2=self.adjust_coordinates(stone.position - handle_offset), color=Colors.GRAY.value, thickness=1)

    def get_canvas(self)-> np.ndarray:
        return self._canvas

    def display(self, constants: SimulationConstants = SimulationConstants()):
        cv2.imshow(self.WINDOW_NAME, self._canvas)
        linear_transform = self.DISPLAY_TIME.value
        cv2.waitKey(int(linear_transform(1000 * constants.dt)))

class Stone:
    mass: np.floating = np.array(19.) # stones between 17.24-19.96
    height: np.floating = np.array(0.1143) # height of the stone
    ring_radius: np.floating = np.array(0.065) # radius of the inner ring
    outer_radius: np.floating = np.array(0.142) # radius of the entire stone
    angular_acceleration: np.floating = np.array(0.) # angular acceleration
    angular_velocity: np.floating = np.array(1.5) # clockwise is negative (rad/s)
    angular_position: np.floating = np.array(0.)
    weight: np.floating = np.array(mass * CurlingConstants.g)
    moment_of_inertia: np.floating = np.array(.5 * mass * outer_radius ** 2) # I = 1/2 mr^2
    coefficient_of_restitution: np.floating = np.array(.5) # coefficient of restitution between stones
    coefficient_of_friction: np.floating = np.array(.2) # coefficient of friction between stones
    acceleration = np.array((0., 0.)) # xy acceleration of the stone
    velocity = np.array((0.01, 2.2)) # xy velocity of the stone
    position = np.array((0., -Curling.starting_button_distance)) # xy position of the stone
    def __init__(self, color: StoneColor, position: Tuple[float, float] = (0, 0), velocity: float = 0, angle: float = 0, spin: float = 0, curling_constants: CurlingConstants = Curling.constants):
        """create a moving stone (equivalent to throwing)

        Args:
            x_position (float): position of the thrower from the centre line (-2, 2)
            velocity (float): velocity of the stone when it is released (?)
            angle (float): angle between the centre line and direction of the stone in radians ()
            spin (float): amount of spin on the stone in radians (-2, 2)
        """
        self.color = color
        self.curling_constants = curling_constants
        self.position = np.array(position)
        self.velocity = np.array((-np.sin(angle), np.cos(angle))) * velocity
        self.angular_position = np.array(0.)
        self.angular_velocity = np.array(spin, dtype=float)

    def step(self, simulation_constants: SimulationConstants=SimulationConstants()) -> SimulationState:
        if np.linalg.norm(self.velocity) < simulation_constants.eps:
            self.angular_velocity = 0
            return SimulationState.FINISHED

        dt = simulation_constants.dt
        theta = np.arange(0, 2 * np.pi, simulation_constants.dtheta)
        relative_point_position = np.array((-np.sin(theta), np.cos(theta))).T * self.ring_radius # position of this point relative to centre of stone
        normalised_tangent = np.array((-np.cos(theta), -np.sin(theta))).T # direction of the tangent normal to the radius from the centre
        phi = theta - np.arctan2(-self.velocity[0], self.velocity[1]) # angle relative to direction of motion

        point_velocity = self.angular_velocity * self.ring_radius * normalised_tangent + self.velocity # speed relative to the ground
        point_speed = np.linalg.norm(point_velocity, axis=-1)

        # divide weight between all the points
        normal_force = self.weight / simulation_constants.num_points_on_circle
        forward_ratio = np.cos(phi) # ratio along the disc used to calculate friction
        mu = self.curling_constants.calculate_friction(point_speed, forward_ratio)
        frictional_force = np.minimum(normal_force * mu, point_speed * (self.mass / simulation_constants.num_points_on_circle) / dt) # F <= mu * N
        # point force in opposite direction to velocity
        frictional_force = np.tile(frictional_force, (2, 1)).T * -point_velocity / (np.tile(point_speed, (2, 1))).T

        # add the frictional force
        net_force = frictional_force.sum(axis=0)
        # torque magnitude
        torque = -np.linalg.det(np.stack((frictional_force, relative_point_position), axis=1)).sum(axis=-1)


        # update values
        self.angular_acceleration = torque / self.moment_of_inertia
        self.angular_velocity += self.angular_acceleration * dt
        self.angular_position += self.angular_velocity * dt
        self.acceleration = net_force / self.mass
        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt
        return SimulationState.UNFINISHED

    @staticmethod
    def handle_collisions(stones: List[Stone], constants: SimulationConstants = SimulationConstants()):
        impulses = np.zeros((len(stones), 2))
        torques = np.zeros((len(stones), ))
        # Could be rewritten more efficiently if stones are sorted by y coordinate and then values are recalculated
        for i in range(len(stones)):
            stone1 = stones[i]
            for j in range(i):
                stone2 = stones[j]
                normal_vector = stone1.position - stone2.position
                distance = np.linalg.norm(normal_vector)
                if distance <= stone1.outer_radius + stone2.outer_radius:
                    normal_vector /= distance
                    tangent_vector = np.array((-normal_vector[1], normal_vector[0]))
                    relative_velocity = stone1.velocity - stone2.velocity
                    relative_normal_velocity = np.dot(relative_velocity, normal_vector) # relative velocity in normal direction
                    relative_tangent_velocity = np.dot(relative_velocity, tangent_vector) # relative velocity in the tangent direction
                    relative_tangent_velocity -= stone1.angular_velocity * stone1.outer_radius + stone2.angular_velocity * stone2.outer_radius

                    impulse = -(1 + Stone.coefficient_of_restitution) * relative_normal_velocity / (1 / stone1.mass + 1 / stone2.mass)
                    impulse *= normal_vector
                    impulses[i] += impulse
                    impulses[j] -= impulse

                    # tangent impulse is limited by the relative tangent velocity
                    tangent_impulse = min(
                        Stone.coefficient_of_friction * np.linalg.norm(impulse),
                        (1 + Stone.coefficient_of_restitution) * np.abs(relative_tangent_velocity) / (stone1.outer_radius ** 2 / stone1.moment_of_inertia + 1 / stone1.mass + stone2.outer_radius ** 2 / stone2.moment_of_inertia + 1 / stone2.mass)
                    ) * np.sign(relative_tangent_velocity)

                    torques[i] += tangent_impulse * stone1.outer_radius
                    torques[j] -= tangent_impulse * stone2.outer_radius

                    tangent_impulse *= -tangent_vector
                    impulses[i] += tangent_impulse
                    impulses[j] -= tangent_impulse

        dt = constants.dt
        for stone, impulse, torque in zip(stones, impulses, torques):
            if (torque != 0).any() or (impulse != 0).any():
                # undo previous steps
                stone.position -= stone.velocity * dt
                stone.velocity -= stone.acceleration * dt
                stone.angular_position -= stone.angular_velocity * dt
                stone.angular_velocity -= stone.angular_acceleration * dt

                # update to next state
                stone.velocity += impulse / stone.mass
                stone.position += stone.velocity * dt
                stone.angular_velocity += torque / stone.moment_of_inertia

@dataclass
class StoneThrow:
    bounds: ClassVar[np.ndarray] = np.array([
        (1.3, 2.),
        (-.1, .1),
        (-4, 4)
    ]).astype(float)
    color: StoneColor
    velocity: float
    angle: float
    spin: float
    def __post_init__(self):
        self.velocity **= 2