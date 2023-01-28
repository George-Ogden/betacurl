from src.curling import Curling, StoneThrow, Stone, StoneColor, SimulationConstants
from src.curling.enums import SimulationState, Colors, LinearTransform, DisplayTime
from src.curling.curling import Canvas
import pytest

import numpy as np
import cv2

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

def test_red_initialisation():
    curling = Curling(StoneColor.RED)
    assert len(curling.stones) == 0
    assert curling.next_stone_colour == StoneColor.RED

def test_yellow_initialisation():
    curling = Curling(StoneColor.YELLOW)
    assert len(curling.stones) == 0
    assert curling.next_stone_colour == StoneColor.YELLOW

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

def test_colour_changes():
    curling = Curling(StoneColor.RED)
    curling.throw(StoneThrow(
        StoneColor.RED,
        spin=0,
        angle=0,
        velocity=1
    ), constants=approx_constants)
    with pytest.raises(AssertionError) as e:
        curling.throw(StoneThrow(
            StoneColor.RED,
            spin=0,
            angle=0,
            velocity=1
        ), constants=approx_constants)

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

def test_display():
    curling = Curling(StoneColor.YELLOW)
    curling.throw(StoneThrow(
        StoneColor.YELLOW,
        angle=.01,
        velocity=2,
        spin=0
    ), constants=approx_constants)
    curling.display()
    assert cv2.getWindowProperty(Canvas.WINDOW_NAME, cv2.WND_PROP_VISIBLE) != -1
    
    # cleanup
    cv2.destroyAllWindows()

def test_image():
    curling = Curling(StoneColor.YELLOW)
    curling.throw(StoneThrow(
        StoneColor.YELLOW,
        angle=.01,
        velocity=2.1,
        spin=0
    ), constants=approx_constants)
    image = curling.render().get_canvas()
    assert np.abs(image.shape[0] / image.shape[1] - curling.pitch_length / curling.pitch_width) < 0.1
    assert (image == Colors.YELLOW.value).any()
    assert (image == Colors.BACKGROUND.value).sum() > np.prod(image.shape[:-1]) / 2

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

def test_evaluate_with_single_stone():
    curling = Curling(StoneColor.RED)
    curling.throw(
        StoneThrow(
            StoneColor.RED,
            velocity=2.25,
            angle=0,
            spin=0
        )
    )
    assert curling.evaluate_position() == StoneColor.RED

    curling.reset(StoneColor.YELLOW)
    curling.throw(
        StoneThrow(
            StoneColor.YELLOW,
            velocity=2.25,
            angle=0,
            spin=0
        )
    )
    assert curling.evaluate_position() == StoneColor.YELLOW

def test_evaluate_with_double_stone():
    curling = Curling(StoneColor.RED)
    curling.throw(
        StoneThrow(
            StoneColor.RED,
            velocity=2.25,
            angle=0,
            spin=0
        )
    )

    curling.throw(
        StoneThrow(
            StoneColor.YELLOW,
            velocity=2.25,
            angle=0.05,
            spin=0
        )
    )

    curling.throw(
        StoneThrow(
            StoneColor.RED,
            velocity=2.25,
            angle=0.02,
            spin=0
        )
    )
    assert curling.evaluate_position() == StoneColor.RED * 2

def test_evaluate_after_collision():
    curling = Curling(StoneColor.RED)
    curling.throw(
        StoneThrow(
            StoneColor.RED,
            velocity=2.25,
            angle=0,
            spin=0
        )
    )

    curling.throw(
        StoneThrow(
            StoneColor.YELLOW,
            velocity=2.5,
            angle=0,
            spin=0
        )
    )

    curling.throw(
        StoneThrow(
            StoneColor.RED,
            velocity=4,
            angle=0.5,
            spin=0
        )
    )

    curling.throw(
        StoneThrow(
            StoneColor.YELLOW,
            velocity=2.25,
            angle=0,
            spin=0
        )
    )
    assert curling.evaluate_position() == StoneColor.YELLOW * 2

def test_evaluate_with_split_stones():
    curling = Curling(StoneColor.RED)
    curling.throw(
        StoneThrow(
            StoneColor.RED,
            velocity=2.25,
            angle=0,
            spin=0
        )
    )

    curling.throw(
        StoneThrow(
            StoneColor.YELLOW,
            velocity=2.25,
            angle=0.02,
            spin=0
        )
    )

    curling.throw(
        StoneThrow(
            StoneColor.RED,
            velocity=2.25,
            angle=0.05,
            spin=0
        )
    )
    assert curling.evaluate_position() == StoneColor.RED

def test_linear_transform():
    transform = LinearTransform(10, 5)
    assert transform(6) == 65

def test_display_times():
    for time in DisplayTime:
        if time == DisplayTime.FOREVER:
            assert int(time.value(1000 * accurate_constants.dt)) == 0
        else:
            assert int(time.value(1000 * accurate_constants.dt)) > 0
    
def test_horizontal_lines_are_symmetric():
    line_sums = Curling.horizontal_lines + Curling.horizontal_lines[::-1]
    assert (line_sums == Curling.pitch_length).all()

def test_vertical_lines_are_symmetric():
    line_sums = Curling.vertical_lines + Curling.vertical_lines[::-1]
    assert (line_sums == 0).all()
        
def test_free_guard_zone():
    curling = Curling()
    stone = Stone(color=StoneColor.RED, position=(0,0))
    assert not curling.in_fgz(stone)
    
    stone.position = (0, -curling.tee_line_position + curling.target_radii[0])
    assert not curling.in_fgz(stone)
    
    stone.position = (1, -curling.tee_line_position - curling.target_radii[0])
    assert not curling.in_fgz(stone)
    
    stone.position = (0, -curling.tee_line_position - curling.target_radii[-1])
    assert not curling.in_fgz(stone)

    stone.position = (0, -curling.tee_line_position - curling.target_radii[-1] * 2)
    assert curling.in_fgz(stone)
    
    stone.position = (-1, -curling.hog_line_position + curling.target_radii[0])
    assert curling.in_fgz(stone)
    
    stone.position = (0, -curling.hog_line_position - curling.target_radii[0])
    assert not curling.in_fgz(stone)
    
    stone.position = (1, -curling.horizontal_lines[-2])
    assert not curling.in_fgz(stone)

def test_in_house():
    curling = Curling()
    stone = Stone(color=StoneColor.RED, position=(0,0))
    assert not curling.in_house(stone)
    
    stone.position = (0, -curling.tee_line_position + curling.target_radii[0])
    assert curling.in_house(stone)
    
    stone.position = (1, -curling.tee_line_position - curling.target_radii[0])
    assert curling.in_house(stone)
    
    stone.position = (curling.target_radii[0], -curling.tee_line_position + curling.target_radii[-2])
    assert curling.in_house(stone)
    
    stone.position = (-1, -curling.tee_line_position - curling.target_radii[-1])
    assert not curling.in_house(stone)
    
    stone.position = (-curling.target_radii[-1], -curling.tee_line_position)
    assert curling.in_house(stone)
    
    stone.position = (-curling.hog_line_position)
    assert not curling.in_house(stone)