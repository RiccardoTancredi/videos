import numpy as np
import sys
import os

lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "./../../"))
sys.path.append(lib_path)
from manim_imports_ext import *

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from network import Network


def get_opposite_color(color):
    """Restituisce il colore opposto di un dato colore in formato RGB."""
    rgb = np.array(color_to_rgb(color))  # Ottieni il colore in [0,1]
    opposite_rgb = 1 - rgb  # Complemento in [0,1]
    return rgb_to_color(opposite_rgb)  # Converti di nuovo in colore Manim



def lorenz_system(t, state, sigma=10, rho=28, beta=8 / 3):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

def hopf(t, state, alpha=1, omega=2, beta=0.9, gamma=1, delta=0.1):
    x, y, z = state
    r2 = x**2 + y**2  # Distanza quadratica dall'origine
    dx = alpha * x - omega * y - beta * r2 * x
    dy = omega * x + alpha * y - beta * r2 * y
    dz = gamma * (r2 - 1)  # Nuovo termine che regola z
    return [dx, dy, dz]


def ode_solution_points(function, state0, time, dt=0.01):
    solution = solve_ivp(
        function,
        t_span=(0, time),
        y0=state0,
        t_eval=np.arange(0, time, dt)
    )
    return solution.y.T



class KoopmanOperatorApplication(ThreeDScene):
    def construct(self):
        # Left scene
        background_rect = Rectangle(height=3.5, width=3.5, color=get_opposite_color(YELLOW))
        background_rect.set_fill(opacity=0)  # Leggero riempimento
        background_rect.set_stroke(get_opposite_color(YELLOW), width=3)
        background_rect.to_edge(LEFT)  # Posizionato a sinistra
        # background_rect.rotate(PI / 2, axis=RIGHT)
        # background_rect.rotate(PI / 2, axis=LEFT)
        # background_rect.rotate(-PI / 2, axis=OUT)

        # Testo sopra il riquadro
        label = Tex(R"\mathrm{Non-linear,\enspace finite-dim.}", 
            font_size=28)
        label.next_to(background_rect, UP, buff=0.3)

        background_rect.fix_in_frame()
        label.fix_in_frame()
        # self.add_foreground_mobjects(background_rect, label)


        lorenz_axes = ThreeDAxes(
            x_range=(-1., 1., .5),
            y_range=(-1., 1., .5),
            z_range=(-1.8, 1.8, .5),
            width=6,
            height=6,
            depth=5,
        )
        # lorenz_axes.set_width(FRAME_WIDTH)
        # lorenz_axes.center()
        # lorenz_axes.reorient(43, 76, 1, IN, 10)
        lorenz_axes.scale(0.45)
        lorenz_axes.move_to(background_rect.get_center()+UP*3+RIGHT*0.65)  # Centra nel rettangolo


        # self.frame.reorient(43, 76, 1, IN, 10)
        # self.frame.add_updater(lambda m, dt: m.increment_theta(dt * 3 * DEGREES))
        

        # Compute a set of solutions
        epsilon = 1 # e-5
        evolution_time = 30
        n_points = 10
        states = [
            [1, 1, 1 + n * epsilon + np.random.uniform(-0.5, 0.5)]
            for n in range(n_points)
        ]
        colors = color_gradient([BLUE_E, BLUE_A], len(states))

        curves = VGroup()
        for state, color in zip(states, colors):
            points = ode_solution_points(hopf, state, evolution_time) # lorenz_system
            curve = VMobject().set_points_as_corners(lorenz_axes.c2p(*points.T))
            curve.set_stroke(get_opposite_color(color), 2, opacity=0.25)
            curves.add(curve)

        curves.scale(0.3).set_stroke(width=5, opacity=1).move_to(lorenz_axes.get_center())

        # Aggiunta del rettangolo e titolo (ruotati nello spazio)
        # self.add(lorenz_group)

        # Add arrow
        arrow_text = Tex(R"\text{Lift}", font_size=28)
        arrow = Arrow(LEFT, RIGHT, buff=0.1).scale(0.5)  # Crea una freccia
        arrow.next_to(background_rect, RIGHT)  # Posiziona la freccia sopra il testo
        arrow_text.next_to(arrow, UP)
        arrow_text.fix_in_frame()
        arrow.fix_in_frame()

 

        # Central scene
        background_rect1 = Rectangle(height=3.5, width=3.5, color=get_opposite_color(YELLOW))
        background_rect1.set_fill(opacity=0)  # Leggero riempimento
        background_rect1.set_stroke(get_opposite_color(YELLOW), width=3)
        background_rect1.center().next_to(arrow)  # Posizionato a sinistra
        
        # Testo sopra il riquadro
        label1 = Tex(R"\mathrm{Linear}\enspace\infty-\mathrm{dim.}", 
            font_size=33)
        label1.next_to(background_rect1, UP, buff=0.3)

        background_rect1.fix_in_frame()
        label1.fix_in_frame()
        

        # Lista di assi traslati
        axes_group = VGroup()

        axes = ThreeDAxes(
            x_range=(-15, 15, 5),
            y_range=(-15, 15, 5),
            z_range=(-0, 15, 5),
            width=16,
            height=16,
            depth=8
        )
        axes_group.add(axes)

        # Numero di rette aggiuntive
        n_lines = 10
        max_length = 12  
        delta_theta = PI / 3  # Passo per l'angolo azimutale
        delta_phi = PI / 3

        lines = VGroup()
        index = 0
        for theta in np.arange(0, TAU, delta_theta):  # Angolo azimutale (0 a 2π)
            for phi in np.arange(0, PI, delta_phi):  # Angolo polare (0 a π)
                # Coordinate cartesiane dalla sfera
                x = np.sin(phi) * np.cos(theta)
                y = np.sin(phi) * np.sin(theta)
                z = np.cos(phi)

                end_point = max_length * np.array([x, y, z])
                line = Line(ORIGIN, end_point, color=WHITE, stroke_width=2)
                lines.add(line)
                index += 1

        axes_group.add(lines)

        axes_group.scale(0.15)
        axes_group.move_to(background_rect1.get_center()+UP+RIGHT*0.5)  # Centra nel rettangolo
        # self.add(axes_group)

        # Add arrow
        arrow_text1 = Tex(R"\text{Approx.}", font_size=28)
        arrow1 = Arrow(LEFT, RIGHT, buff=0.1).scale(0.5)  # Crea una freccia
        arrow1.next_to(background_rect1, RIGHT)  # Posiziona la freccia sopra il testo
        arrow_text1.next_to(arrow1, UP)
        arrow_text1.fix_in_frame()
        arrow1.fix_in_frame()


        # Right scene
        background_rect2 = Rectangle(height=3.5, width=3.5, color=get_opposite_color(YELLOW))
        background_rect2.set_fill(opacity=0)  # Leggero riempimento
        background_rect2.set_stroke(get_opposite_color(YELLOW), width=3)
        background_rect2.center().next_to(arrow1)  # Posizionato a sinistra
        
        # Testo sopra il riquadro
        label2 = Tex(R"\mathrm{Linear\enspace finite-dim.}", 
            font_size=30)
        label2.next_to(background_rect2, UP, buff=0.3)

        background_rect2.fix_in_frame()
        label2.fix_in_frame()

        radii = range(10)
        
        # Creazione dei cerchi concentrici
        circles = VGroup(*[Circle(radius=r, color=get_opposite_color(RED)) for r in radii])

        circles.scale(0.15)
        circles.move_to(background_rect2.get_center())
        circles.fix_in_frame()

        self.wait(3)

        self.add(background_rect, label)
        self.add(lorenz_axes)
        self.add(background_rect1, label1)
        self.add(background_rect2, label2)


        self.play(
            *(
                ShowCreation(curve, rate_func=linear, run_time=10)
                for curve in curves
            ),
            Write(arrow, run_time=1),
            Write(arrow_text, run_time=1),
            ShowCreation(axes_group, run_time=10),
            Write(arrow1, run_time=1),
            Write(arrow_text1, run_time=1),
            ShowCreation(circles, run_time=10),
        )

        self.wait(5)