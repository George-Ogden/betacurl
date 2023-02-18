import numpy as np

from src.curling import Curling, Stone, StoneColor, StoneThrow, SimulationConstants
from src.curling.enums import SimulationState

approx_constants = SimulationConstants(dt=.1, num_points_on_circle=10)
accurate_constants = SimulationConstants(dt=.02)

def get_short_curling(stone_on_button=False):
    short_curling = Curling(StoneColor.RED)
    short_curling.starting_button_distance = short_curling.tee_line_position
    short_curling.pitch_length = short_curling.tee_line_position * 2
    if stone_on_button:
        short_curling.stones.append(
            Stone(
                StoneColor.YELLOW,
                position=(0, -short_curling.tee_line_position),
                velocity=0,
                angle=0,
                spin=0,
            )
        )
    return short_curling

def test_energy_decrease():
    curling = Curling(StoneColor.RED)
    curling.stones.append(
        curling.create_stone(
            StoneThrow(
                StoneColor.RED,
                spin=1.5,
                angle=-.02,
                velocity=2.1
            )
        )
    )
    speed = np.linalg.norm(curling.stones[0].velocity)
    spin_speed = np.abs(curling.stones[0].angular_velocity)
    while curling.step(accurate_constants) == SimulationState.UNFINISHED:
        new_speed = np.linalg.norm(curling.stones[0].velocity)
        new_spin_speed = np.abs(curling.stones[0].angular_velocity)
        assert new_speed <= speed
        assert new_spin_speed <= spin_speed
        speed = new_speed
        spin_speed = new_spin_speed

def test_reasonable_throw_default():
    curling = Curling(StoneColor.RED)
    curling.throw(StoneThrow(
        StoneColor.RED,
        spin=1.5,
        angle=-.02,
        velocity=2.25
    ), constants=approx_constants)
    assert len(curling.stones) == 1
    assert np.linalg.norm(curling.stones[0].position - curling.button_position) < curling.target_radii[-1]

def test_reasonable_throw_accurate():
    curling = Curling(StoneColor.RED)
    curling.throw(StoneThrow(
        StoneColor.RED,
        spin=1.5,
        angle=-.02,
        velocity=2.25
    ), constants=accurate_constants)
    assert len(curling.stones) == 1
    assert np.linalg.norm(curling.stones[0].position - curling.button_position) < curling.target_radii[-1]

def test_straight_throw():
    curling = Curling(StoneColor.RED)
    curling.throw(StoneThrow(
        StoneColor.RED,
        spin=0,
        angle=0,
        velocity=2.25
    ), constants=approx_constants)
    assert len(curling.stones) == 1
    assert np.linalg.norm(curling.stones[0].position - curling.button_position) < curling.target_radii[-1]
    assert np.abs(curling.stones[0].position[0] - curling.button_position[0]) < 1e-3

def test_gentle_collision():
    curling = Curling(StoneColor.RED)
    curling.throw(StoneThrow(
        StoneColor.RED,
        spin=0,
        angle=0,
        velocity=2.1
    ), constants=approx_constants)
    position = curling.stones[0].position.copy()
    curling.throw(StoneThrow(
        StoneColor.YELLOW,
        spin=0,
        angle=0,
        velocity=2.1
    ), constants=approx_constants)
    assert len(curling.stones) == 2
    assert np.linalg.norm(curling.stones[0].position - position) < curling.target_radii[1]
    assert np.linalg.norm(curling.stones[1].position - position) < curling.target_radii[1]
    assert curling.stones[1].position[1] < curling.stones[0].position[1]

def test_hard_slightly_offcenter_collision():
    curling = Curling(StoneColor.RED)
    curling.throw(StoneThrow(
        StoneColor.RED,
        spin=0,
        angle=0,
        velocity=2.25
    ), constants=approx_constants)
    stone = curling.stones[0]
    curling.throw(StoneThrow(
        StoneColor.YELLOW,
        spin=0,
        angle=1.5e-3,
        velocity=4
    ), constants=accurate_constants)
    assert len(curling.stones) == 0
    assert np.linalg.norm(stone.velocity) > 1

def test_negative_angle_moves_right():
    curling = Curling(StoneColor.RED)
    curling.throw(StoneThrow(
        StoneColor.RED,
        angle=-.01,
        velocity=2.25,
        spin=0
    ), constants=approx_constants)
    assert len(curling.stones) == 1
    assert curling.stones[0].position[0] > curling.target_radii[0]
    assert np.linalg.norm(curling.stones[0].position - curling.button_position) < curling.target_radii[-1]

def test_positive_angle_moves_left():
    curling = Curling(StoneColor.RED)
    curling.throw(StoneThrow(
        StoneColor.RED,
        angle=.01,
        velocity=2.25,
        spin=0
    ), constants=approx_constants)
    assert len(curling.stones) == 1
    assert curling.stones[0].position[0] < curling.target_radii[0]
    assert np.linalg.norm(curling.stones[0].position - curling.button_position) < curling.target_radii[-1]

def test_positive_spin_moves_left():
    curling = Curling(StoneColor.RED)
    curling.throw(StoneThrow(
        StoneColor.RED,
        angle=0,
        velocity=2.25,
        spin=2
    ), constants=approx_constants)
    assert len(curling.stones) == 1
    assert curling.stones[0].position[0] < -curling.target_radii[0]

def test_negative_spin_moves_right():
    curling = Curling(StoneColor.RED)
    curling.throw(StoneThrow(
        StoneColor.RED,
        angle=0,
        velocity=2.25,
        spin=-2
    ), constants=approx_constants)
    assert len(curling.stones) == 1
    assert curling.stones[0].position[0] > curling.target_radii[0]

def test_slow_head_on_collision():
    curling = get_short_curling(True)
    curling.stones.append(
        curling.create_stone(
            StoneThrow(
                StoneColor.RED,
                spin=0,
                angle=0,
                velocity=1.2
        ))
    )
    while curling.step(accurate_constants) == SimulationState.UNFINISHED:
        assert len(curling.stones) == 2
        assert abs(curling.stones[0].position[0]) < 1e-2
        assert abs(curling.stones[1].position[0]) < 1e-2
        assert abs(curling.stones[0].velocity[0]) < 1e-3
        assert abs(curling.stones[1].velocity[0]) < 1e-3
        assert abs(curling.stones[0].angular_velocity) < 1e-2
        assert abs(curling.stones[1].angular_velocity) < 1e-2

def test_slow_off_centre_collision():
    curling = get_short_curling(True)
    curling.stones.append(
        curling.create_stone(
            StoneThrow(
                StoneColor.RED,
                spin=0,
                angle=0.03,
                velocity=1.2
        ))
    )
    while curling.step(accurate_constants) == SimulationState.UNFINISHED:
        assert len(curling.stones) == 2
        assert curling.stones[0].position[0] > -1e-3
        assert np.linalg.norm(curling.stones[0].velocity) < 1e-1 or curling.stones[0].velocity[0] > -1e-3

        assert curling.stones[1].position[0] < 1e-2 or curling.stones[1].velocity[0] < 1e-3

def test_dual_collision():
    curling = get_short_curling(False)
    curling.stones.append(
        Stone(
            StoneColor.YELLOW,
            position=(-0.2, -curling.tee_line_position),
            velocity=0,
            angle=0.00,
            spin=0,
        )
    )
    curling.stones.append(
        Stone(
            StoneColor.YELLOW,
            position=(0.2, -curling.tee_line_position),
            velocity=0,
            angle=0.00,
            spin=0,
        )
    )
    curling.stones.append(
        curling.create_stone(
            StoneThrow(
                StoneColor.RED,
                spin=0,
                angle=0,
                velocity=1.5
        ))
    )
    while curling.step(accurate_constants) == SimulationState.UNFINISHED:
        assert len(curling.stones) == 3
        assert curling.stones[0].position[0] < -1e-1
        assert curling.stones[0].velocity[0] < 1e-3 or np.linalg.norm(curling.stones[0].velocity) < 1e-1

        assert curling.stones[1].position[0] > 1e-1
        assert curling.stones[1].velocity[0] > -1e-3 or np.linalg.norm(curling.stones[1].velocity) < 1e-1

    assert curling.stones[0].position[0] < -curling.target_radii[1]
    assert curling.stones[1].position[0] > curling.target_radii[1]
    assert np.abs(curling.stones[2].position[0]) < curling.target_radii[0]

def test_single_collision_multiple_stones():
    curling = get_short_curling(False)
    curling.stones.append(
        Stone(
            StoneColor.YELLOW,
            position=(-0.2, -curling.tee_line_position),
            velocity=0,
            angle=0.00,
            spin=0,
        )
    )
    curling.stones.append(
        Stone(
            StoneColor.YELLOW,
            position=(0.2, -curling.tee_line_position),
            velocity=0,
            angle=0.00,
            spin=0,
        )
    )
    curling.stones.append(
        curling.create_stone(
            StoneThrow(
                StoneColor.RED,
                spin=0,
                angle=0.05,
                velocity=1.5
        ))
    )
    while curling.step(accurate_constants) == SimulationState.UNFINISHED:
        assert len(curling.stones) == 3
        assert np.linalg.norm(curling.stones[1].velocity) < 1e-6

def test_conservation_of_momentum():
    curling = get_short_curling(True)
    curling.stones.append(
        curling.create_stone(
            StoneThrow(
                StoneColor.RED,
                spin=0,
                angle=0.02,
                velocity=2
        ))
    )
    momentum = sum([stone.mass * stone.velocity for stone in curling.stones])
    while curling.step(accurate_constants) == SimulationState.UNFINISHED:
        if len(curling.stones) < 2:
            break
        assert curling.stones[0].position[0] > -1e-3
        assert curling.stones[0].velocity[0] > -1e-3

        assert curling.stones[1].position[0] < 1e-2 or curling.stones[1].velocity[0] < 1e-3

        new_momentum = sum([stone.mass * stone.velocity for stone in curling.stones])
        assert (np.minimum(np.abs(1 - new_momentum / momentum), np.abs(new_momentum - momentum)) < .1).all()
        momentum = new_momentum

def test_conservation_of_angular_momentum():
    curling = get_short_curling(True)
    curling.stones.append(
        curling.create_stone(
            StoneThrow(
                StoneColor.RED,
                spin=5,
                angle=-0.03,
                velocity=2
        ))
    )
    angular_momentum = sum([stone.moment_of_inertia * stone.angular_velocity for stone in curling.stones])
    while curling.step(accurate_constants) == SimulationState.UNFINISHED:
        if len(curling.stones) < 2:
            break
        assert curling.stones[0].position[0] < 1e-3
        assert curling.stones[0].velocity[0] < 1e-3

        assert curling.stones[1].position[0] > -1e-2 or curling.stones[1].velocity[0] < -1e-3

        new_momentum = sum([stone.moment_of_inertia * stone.angular_velocity for stone in curling.stones])
        assert (np.minimum(np.abs(1 - new_momentum / angular_momentum), np.abs(new_momentum - angular_momentum)) < .1).all()
        angular_momentum = new_momentum

def test_angle_causes_no_spin():
    curling = Curling(StoneColor.RED)
    curling.throw(
        StoneThrow(
            StoneColor.RED,
            spin=0,
            angle=0.01,
            velocity=2.25
        )
    )

    assert np.abs(curling.stones[0].angular_position) < 1e-1

def test_left_collision_causes_negative_spin():
    curling = get_short_curling(True)
    curling.stones.append(
        curling.create_stone(
            StoneThrow(
                StoneColor.RED,
                spin=0,
                angle=0.04,
                velocity=1.25
        ))
    )
    while curling.step(accurate_constants) == SimulationState.UNFINISHED:
        assert len(curling.stones) == 2
        assert np.linalg.norm(curling.stones[0].position[0]) < 1e-3 or \
            np.linalg.norm(curling.stones[0].velocity) < 1e-2 or \
            (curling.stones[0].angular_velocity > 0 and curling.stones[1].angular_velocity < -1e-3)

def test_right_collision_causes_positive_spin():
    curling = get_short_curling(True)
    curling.stones.append(
        curling.create_stone(
            StoneThrow(
                StoneColor.RED,
                spin=0,
                angle=-0.04,
                velocity=1.25
        ))
    )
    while curling.step(accurate_constants) == SimulationState.UNFINISHED:
        assert len(curling.stones) == 2
        assert np.linalg.norm(curling.stones[0].position[0]) > -1e-3 or \
            np.linalg.norm(curling.stones[0].velocity) < 1e-2 or \
            (curling.stones[0].angular_velocity < 0 and curling.stones[1].angular_velocity > 1e-3)

def test_time_frame_is_reasonable():
    curling = Curling(StoneColor.RED)
    curling.stones.append(
        curling.create_stone(
            StoneThrow(
                StoneColor.RED,
                spin=1.5,
                angle=-.02,
                velocity=2.1
        ))
    )
    time = 0
    while curling.step(accurate_constants) == SimulationState.UNFINISHED:
        time += accurate_constants.dt
    assert 20 <= time <= 30

def test_close_interaction():
    curling = get_short_curling(True)
    curling.stones.append(Stone(StoneColor.YELLOW, (0, -curling.tee_line_position - Stone.outer_radius * (2 + 1e-5))))
    curling.throw(stone_throw=StoneThrow(
        color=curling.next_stone_colour,
        velocity=2.00,
        spin=0,
        angle=.0
    ))
    assert len(curling.stones) > 1