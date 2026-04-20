from scipy.integrate import solve_ivp
import numpy as np
import sys
import os

lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "./../../"))
sys.path.append(lib_path)
# lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../_2024/"))
# sys.path.append(lib_path)


from manim_imports_ext import *

def get_opposite_color(color):
    """Restituisce il colore opposto di un dato colore in formato RGB."""
    rgb = np.array(color_to_rgb(color))  # Ottieni il colore in [0,1]
    opposite_rgb = 1 - rgb  # Complemento in [0,1]
    return rgb_to_color(opposite_rgb)  # Converti di nuovo in colore Manim


def discrete_spectrum(t, z, mu, gamma):
    x1, x2 = z
    dzdt = [mu * x1, gamma*(x2 - x1**2)]
    return dzdt


def linear_discrete_spectrum(t, z, mu, gamma):
    x1, x2 = z
    dzdt = [mu * x1, gamma*x2]
    return dzdt


def solve(func, t_span, z0, t_eval, args=()):
    sol = solve_ivp(func, t_span, z0, args=args, t_eval=t_eval, method='RK45')
    return sol.y.T


def create_ic(size, folder_path, save=False):
    if os.path.exists(f'{folder_path}/ic.npy'):
        states = np.load(f'{folder_path}/ic.npy')
        print(f'Loading ic: {states.shape}')
        return states

    ic_x = np.random.uniform(low=-0.45, high=-0.1, size=(size//2, 1))
    ic_x = np.concatenate((ic_x, -ic_x))
    ic_y = []
    ii = 0
    while ii < ic_x.size:
        yy = np.random.uniform(low=-0.5, high=0.5, size=1)
        if yy > -0.13:
            ic_y.append(yy)
        else:
            if ic_x[ii] > -0.25:
                ic_y.append(yy)
            else:
                continue
        ii += 1

    states = np.column_stack((ic_x, np.array(ic_y)))
    print(f'Creating ic: {states.shape}')
    if save:
        np.save(f'{folder_path}/ic', states)

    return states


class DiscreteSpectrum(InteractiveScene):
    def construct(self):
        # Add the equations
        t2c = {
            "x_1": get_opposite_color(RED),
            "x_2": get_opposite_color(BLUE),
        }
        equations = Tex(
            R"""
            \begin{aligned}
            \frac{\mathrm{d} x_1}{\mathrm{~d} t} & =\mu x_1 \\ \\
            \frac{\mathrm{d} x_2}{\mathrm{~d} t} & =\gamma \big(x_2-x_1^2\big)
            \end{aligned}
            """,
            t2c=t2c,
            font_size=50
        )

        equations.fix_in_frame()
        equations.set_backstroke()
        self.play(Write(equations), run_time=1)

        box1 = SurroundingRectangle(equations[0:10], buff=0.1, color=get_opposite_color(RED))
        box2 = SurroundingRectangle(equations[11:], buff=0.1, color=get_opposite_color(BLUE))
        box_tot = SurroundingRectangle(equations, buff=0.1, color=get_opposite_color(PURPLE))

        self.play(ShowCreation(box1), run_time=1)
        self.play(ShowCreation(box2), run_time=1)

        self.play(
            Transform(box1, box_tot, run_time=1),
            Transform(box2, box_tot, run_time=1)
        )
        self.wait(0.5)

        self.play(
            FadeOut(box1),
            FadeOut(box2),
            # FadeOut(equations),
            FadeOut(box_tot),
        )

        self.wait(1.5)

        # Set up Axes
        axes = Axes(
            x_range=[-0.5, 0.5, 0.1],
            y_range=[-0.5, 0.5, 0.1],
            width=FRAME_WIDTH*0.9,
            height=FRAME_WIDTH*0.55,
            axis_config=dict(
                include_tip=False,
                # tip_shape='StealthTip',
                stroke_color=WHITE,
                # numbers_to_exclude=[0],
            ),
        )
        # axes.add_coordinate_labels(
        #     font_size=8,
        #     num_decimal_places=1,
        # )
        # axes.set_width(FRAME_WIDTH*0.9)

        # self.frame.reorient(43, 76, 1, IN, 10)
        axes.center()  # to_edge(UL)
        # self.frame.add_updater(lambda m, dt: m.increment_theta(dt * 3 * DEGREES))
        labels = axes.get_axis_labels("x_1", "x_2")
        label_colors = [get_opposite_color(RED), get_opposite_color(BLUE), get_opposite_color(GREEN)]
        for ii, label in enumerate(labels, start=1):
            label.set_color_by_tex(f'x_{ii}', label_colors[ii-1])
            if ii == 2:
                label.shift((DOWN+RIGHT)*0.1)


        # grid = NumberPlane(
        #     x_range=axes.x_range, 
        #     y_range=axes.y_range,
        #     width=axes.get_width(), 
        #     height=axes.get_height(),
        #     background_line_style=dict(
        #         stroke_color=BLUE_D,
        #         stroke_width=0.5,
        #         stroke_opacity=0.9,
        #     ),
        # ) 
        # # grid.set_width(FRAME_WIDTH*0.9)
        # # grid.set_height(FRAME_WIDTH*0.55)
        # grid.center()


        self.add(axes, labels) # , grid
        self.play(
            equations.animate.scale(0.5).to_corner(DL),
            ShowCreation(axes),
            ShowCreation(labels),
            # ShowCreation(grid),
            # axes.animate.center(),
            run_time=2.5,
        )

        # Load data & define hyperparameters
        dt = 0.05
        t_max = 100+dt
        time = np.arange(0, t_max, dt)
        points_per_traj = time.size
        t_span = (0, t_max)

        mu_0 = -0.05
        gamma = -1

        size = 40
        folder_path = '/Users/riccardotancredi/Documents/videos3b1b/_2024/Thesis/Data/DiscreteSpectrum'
        
        states = create_ic(size, folder_path, save=True)
        colors = color_gradient([TEAL_E, TEAL_A], states.shape[0])

        # Create manim VGroup object
        curves = VGroup()
        for state, color in zip(states, colors):
            points = solve(discrete_spectrum, t_span,
                           state, time, args=(mu_0, gamma))
            curve = VMobject().set_points_smoothly(axes.c2p(*points.T))
            curve.set_stroke(get_opposite_color(color), 5)
            curves.add(curve)

        # curves.set_stroke(width=5, opacity=0.7)

        # Display dots moving along those trajectories
        dots = Group(GlowDot(color=get_opposite_color(color), radius=0.3)
                     for color in colors)

        def update_dots(dots, curves=curves):
            for dot, curve in zip(dots, curves):
                dot.move_to(curve.get_end())

        dots.add_updater(update_dots)

        dots_states = Group(GlowDot(color=get_opposite_color(color), radius=0.25)
                            for color in colors)

        zeros_column = np.zeros((states.shape[0], 1))
        new_states = np.hstack((states, zeros_column))
        for dot, state in zip(dots_states, new_states):
            dot.move_to(axes.c2p(*state.T))

        # tail = VGroup(
        #     TracingTail(dot, time_traced=10).match_color(dot)
        #     for dot in dots
        # )

        self.add(dots_states)
        # self.add(tail)
        # curves.set_opacity(0)
        self.add(equations)
        self.wait(1)
        self.play(
            *(
                ShowCreation(curve)
                for curve in curves
            ),
            rate_func=rush_into,
            run_time=t_max // 10,
        )

        # Add invariant parabola graph
        invariant_graph = axes.get_graph(
            lambda x: x**2,
            x_range=axes.x_range[:2],
            color=get_opposite_color(TEAL),
        )
        graph_label = axes.get_graph_label(
            invariant_graph,
            Tex("x_2 = x_1^2",
                t2c=t2c,
                ),  # In this way I force the label to be a Tex object, adding t2c attribut
            x=0.5,
            color=WHITE,  # here to force manim to use the t2c dict and not the graph color
        )

        # Convert to dashed graph
        dashed_graph = DashedVMobject(invariant_graph,
                                      num_dashes=30,
                                      positive_space_ratio=0.8,)
        true_params_note = Tex(R"""
            \begin{aligned}
            \mu_{\mathrm{True}} &= -0.05 \\ 
            \gamma_{\mathrm{True}} &=-1
            \end{aligned}
            """,
            font_size=25)
        true_params_note.to_corner(DR)
        self.play(
            Write(true_params_note),
            ShowCreation(dashed_graph),
            FadeIn(graph_label, UR),
        )
        self.wait(2)

        # Fadeout Scene
        to_FadeOut = (equations,
                      true_params_note,
                      # dots, 
                      labels,
                      dashed_graph, graph_label,
                      axes)

        # self.play(
        #     *(FadeOut(fading) for fading in to_FadeOut)
        # )

        ###################
        ### New 3D Idea ###     # wrong
        ###################

        # Changing scene
        # new_text = Text("This system actually allows a straightforward\n\nlinear embedding in 3D space:",
        #                 font_size=36,  
        #                 )
        self.wait(3)

        self.play(
            FadeOut(curves),
            *(ApplyMethod(fading.set_opacity, 0.05) for fading in to_FadeOut),
            # Write(new_text),
        )




        # Clean the scene
        self.remove(*to_FadeOut)

        # # Add the equations
        # t2c3d = {
        #     "x_1": RED,
        #     "x_2": BLUE,
        #     "y_1": RED,
        #     "y_2": BLUE,
        #     "y_3": GREEN,
        # }
        
        # equations3d = Tex(
        #     R"""
        #     \begin{aligned}
        #     y_1 &= x_1 \\ \\
        #     y_2 &= x_2 \\ \\
        #     y_3 &= x_1^2
        #     \end{aligned}
        #     """,
        #     t2c=t2c3d,
        #     font_size=50
        # )

        # self.play(
        #     FadeTransform(new_text, equations3d),
        #     run_time=3)
        # self.wait(2)

        # t2c3d = {
        #     "y_1": RED,
        #     "y_2": BLUE,
        #     "y_3": GREEN,
        # }

        # # 3D equations
        # new_3d_equations = Tex(
        #     R"""
        #     \begin{aligned}
        #     \frac{\mathrm{d} y_1}{\mathrm{~d} t} & =\mu y_1 \\ \\
        #     \frac{\mathrm{d} y_2}{\mathrm{~d} t} & =\gamma (y_2 -y_3)\\ \\
        #     \frac{\mathrm{d} y_3}{\mathrm{~d} t} & =2\mu y_3 \\ \\
        #     \end{aligned}
        #     """,
        #     t2c=t2c3d,
        #     font_size=50
        # )

        # self.play(
        #     FadeTransform(equations3d, new_3d_equations),
        #     run_time=3,
        # )
        # self.wait(3)

        # # Don't need of 3d space
        # aha_moment = Text("Actually, there is no need to embed the system\n\nin a 3d space, if we can think of a way\n\nto learn the two coordinates:",
        #                   font_size=30,
        #                   )
        # aha_moment.to_corner(UL)
        # t2c = {
        #     R"\mu": RED,
        #     R"\gamma": BLUE,
        #     R"\beta": GREEN,
        # }

        note = Tex(R"\* \quad \beta = \frac{\gamma}{\gamma-2\mu}",
                   t2c=t2c,
                   font_size=40)
        note.to_corner(DR)

        convert_eq = Tex(
            R"""
            \begin{aligned}
            y_1 &= x_1 \\ \\
            y_2 &= x_2- \beta x_1^2 \\ \\
            \end{aligned} """,
            t2c=t2c,
            font_size=40
        )
        convert_eq.move_to(2*LEFT)

        new_2d_equations = Tex(
            R"""
                \to
                \begin{pmatrix}
                \dot{y_1} \\
                \dot{y_2} 
                \end{pmatrix}
                = \underbrace{\begin{pmatrix}
                \mu & 0 \\
                0 & \gamma 
                \end{pmatrix}}_{\mathcal{K}}
                \begin{pmatrix}
                y_1 \\
                y_2
                \end{pmatrix}
            """,
            t2c=t2c,
            font_size=40
        )
        new_2d_equations.next_to(convert_eq)
        self.play(
            # Write(aha_moment),
            # FadeTransform(new_3d_equations, convert_eq),
            Write(convert_eq),
            Write(note),
            Write(new_2d_equations),
            run_time=3,
        )
        self.wait(2)

        # Clean scene
        to_remove = (new_2d_equations, convert_eq, note) # , aha_moment
        self.play(*(FadeOut(rem) for rem in to_remove))
        self.wait(2)

        # New 2DScene -> 2D-Embedding
        # Add the linear equations
        linear_equations = Tex(
            R"""
            \begin{aligned}
            \frac{\mathrm{d} y_1}{\mathrm{~d} t} & =\mu y_1 \\ \\
            \frac{\mathrm{d} y_2}{\mathrm{~d} t} & =\gamma y_2
            \end{aligned}
            """,
            t2c={
                "y_1": get_opposite_color(RED),
                "y_2": get_opposite_color(BLUE),
            },
            font_size=50
        )

        linear_equations.fix_in_frame()
        # linear_equations.to_corner(DL)
        linear_equations.set_backstroke()
        self.play(Write(linear_equations), run_time=1)

        labels = axes.get_axis_labels("y_1", "y_2")
        label_colors = [get_opposite_color(RED), get_opposite_color(BLUE), get_opposite_color(GREEN)]
        for ii, label in enumerate(labels, start=1):
            label.set_color_by_tex(f'y_{ii}', label_colors[ii-1])
            if ii == 2:
                label.shift((DOWN+RIGHT)*0.1)
        axes.set_opacity(1.0)
        convert_eq = Tex(
            R"""
            \begin{aligned}
            y_1 &= x_1 \\
            y_2 &= x_2- \beta x_1^2 \\ 
            \beta &= \frac{\gamma}{\gamma-2\mu}
            \end{aligned} """,
            t2c={
                "y_1": get_opposite_color(RED),
                "y_2": get_opposite_color(BLUE),
                R"\beta": get_opposite_color(GREEN)
            },
            font_size=30
        )
        convert_eq.to_corner(UR*0.5)
        
        self.add(axes, labels, convert_eq)
        self.play(
            linear_equations.animate.scale(0.7).to_corner(DL),
            ShowCreation(axes),
            ShowCreation(labels),
            Write(convert_eq),
            run_time=2.5,
        )

        self.wait(2)

        # Create manim VGroup object
        curves = VGroup()
        for state, color in zip(states, colors):
            points = solve(linear_discrete_spectrum, t_span,
                           state, time, args=(mu_0, gamma))
            curve = VMobject().set_points_smoothly(axes.c2p(*points.T))
            curve.set_stroke(get_opposite_color(color), 5)
            curves.add(curve)

        # Display dots moving along those trajectories
        dots = Group(GlowDot(color=get_opposite_color(color), radius=0.3)
                     for color in colors)

        def update_dots(dots, curves=curves):
            for dot, curve in zip(dots, curves):
                dot.move_to(curve.get_end())

        dots.add_updater(update_dots)

        self.add(linear_equations)        
        true_params_note = Tex(R"""
            \begin{aligned}
            \mu_{\mathrm{True}} &= -0.05 \\ 
            \gamma_{\mathrm{True}} &=-1
            \end{aligned}
            """,
            font_size=25)
        true_params_note.to_corner(DR)
        self.play(
            Write(true_params_note),
            )
        self.play(
            *(
                ShowCreation(curve)
                for curve in curves
            ),
            rate_func=rush_into,
            run_time=t_max // 10,
        )
        