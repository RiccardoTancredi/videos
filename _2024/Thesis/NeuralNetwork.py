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


class NetworkMobject(VGroup):
    def __init__(self, neural_network, output_labels=False, **kwargs):
        VGroup.__init__(self, **kwargs)
        self.neural_network = neural_network
        self.layer_sizes = neural_network.sizes

        # Converting CONFIG (setup) into self
        self.neuron_radius = 0.15
        self.neuron_to_neuron_buff = MED_SMALL_BUFF
        self.layer_to_layer_buff = LARGE_BUFF
        self.neuron_stroke_color = get_opposite_color(BLUE)
        self.neuron_stroke_width = 3
        self.neuron_fill_color = get_opposite_color(GREEN)
        self.edge_color = get_opposite_color(GREY_B)
        self.edge_stroke_width = 2
        self.edge_propogation_color = get_opposite_color(YELLOW)
        self.edge_propogation_time = 1
        self.max_shown_neurons = 10
        self.brace_for_large_layers = False
        self.average_shown_activation_of_large_layer = True
        self.include_output_labels = output_labels


        self.add_neurons()
        self.add_edges()

    def add_neurons(self):
        layers = VGroup(*[
            self.get_layer(size)
            for size in self.layer_sizes
        ])
        layers.arrange(RIGHT, buff = self.layer_to_layer_buff)
        self.layers = layers
        self.add(self.layers)
        if self.include_output_labels:
            self.add_output_labels()


    def get_layer(self, size):
        layer = VGroup()
        n_neurons = size
        if n_neurons > self.max_shown_neurons:
            n_neurons = self.max_shown_neurons
        neurons = VGroup(*[
            Circle(
                radius = self.neuron_radius,
                stroke_color = self.neuron_stroke_color,
                stroke_width = self.neuron_stroke_width,
                fill_color = self.neuron_fill_color,
                fill_opacity = 0,
            )
            for x in range(n_neurons)
        ])   
        neurons.arrange(
            DOWN, buff = self.neuron_to_neuron_buff
        )
        for neuron in neurons:
            neuron.edges_in = VGroup()
            neuron.edges_out = VGroup()
        layer.neurons = neurons
        layer.add(neurons)

        if size > n_neurons:
            dots = Tex(R"\vdots")
            dots.move_to(neurons)
            VGroup(*neurons[:len(neurons) // 2]).next_to(
                dots, UP, MED_SMALL_BUFF
            )
            VGroup(*neurons[len(neurons) // 2:]).next_to(
                dots, DOWN, MED_SMALL_BUFF
            )
            layer.dots = dots
            layer.add(dots)
            if self.brace_for_large_layers:
                brace = Brace(layer, LEFT)
                brace_label = brace.get_tex(str(size))
                layer.brace = brace
                layer.brace_label = brace_label
                layer.add(brace, brace_label)

        return layer

    def add_edges(self):
        self.edge_groups = VGroup()
        for l1, l2 in zip(self.layers[:-1], self.layers[1:]):
            edge_group = VGroup()
            for n1, n2 in it.product(l1.neurons, l2.neurons):
                edge = self.get_edge(n1, n2)
                edge_group.add(edge)
                n1.edges_out.add(edge)
                n2.edges_in.add(edge)
            self.edge_groups.add(edge_group)
        self.add_to_back(self.edge_groups)

    def get_edge(self, neuron1, neuron2):
        return Line(
            neuron1.get_center(),
            neuron2.get_center(),
            buff = self.neuron_radius,
            stroke_color = self.edge_color,
            stroke_width = self.edge_stroke_width,
        )

    def get_active_layer(self, layer_index, activation_vector):
        layer = self.layers[layer_index].deepcopy()
        self.activate_layer(layer, activation_vector)
        return layer

    def activate_layer(self, layer, activation_vector):
        n_neurons = len(layer.neurons)
        av = activation_vector
        def arr_to_num(arr):
            return (np.sum(arr > 0.1) / float(len(arr)))**(1./3)

        if len(av) > n_neurons:
            if self.average_shown_activation_of_large_layer:
                indices = np.arange(n_neurons)
                indices *= int(len(av)/n_neurons)
                indices = list(indices)
                indices.append(len(av))
                av = np.array([
                    arr_to_num(av[i1:i2])
                    for i1, i2 in zip(indices[:-1], indices[1:])
                ])
            else:
                av = np.append(
                    av[:n_neurons/2],
                    av[-n_neurons/2:],
                )
        for activation, neuron in zip(av, layer.neurons):
            neuron.set_fill(
                color = self.neuron_fill_color,
                opacity = activation
            )
        return layer

    def activate_layers(self, input_vector):
        activations = self.neural_network.get_activation_of_all_layers(input_vector)
        for activation, layer in zip(activations, self.layers):
            self.activate_layer(layer, activation)

    def deactivate_layers(self):
        all_neurons = VGroup(*it.chain(*[
            layer.neurons
            for layer in self.layers
        ]))
        all_neurons.set_fill(opacity = 0)
        return self

    def get_edge_propogation_animations(self, index):
        edge_group_copy = self.edge_groups[index].copy()
        edge_group_copy.set_stroke(
            self.edge_propogation_color,
            width = 1.5*self.edge_stroke_width
        )
        return [ShowCreationThenDestruction(
            edge_group_copy, 
            run_time = self.edge_propogation_time,
            lag_ratio = 0.5
        )]

    def add_output_labels(self):
        self.output_labels = VGroup()
        for n, neuron in enumerate(self.layers[-1].neurons):
            label = self.include_output_labels[n]
            label.set_height(0.75*neuron.get_height())
            label.move_to(neuron)
            label.shift(neuron.get_width()*1.5*RIGHT)
            self.output_labels.add(label)
        self.add(self.output_labels)





class BrainHighlightThanks(Scene):
    def construct(self):
        # Brain display
        folder = 'C:/Users/ricta/Documents/videos3b1b/_2024/Thesis/Data'
        brainfolder = f'{folder}/BrainSvg'
        brain = SVGMobject(file_name=f'{brainfolder}/brain1.svg')
        brain.set_height(1.6)
        brain.set_fill(GREY_B)
        brain_outline = brain.copy()
        brain_outline.set_fill(opacity=0)
        brain_outline.set_stroke(BLUE_B, 3)
        brain.center()
        brain_outline.center()
        
        santa_cap = SVGMobject(file_name=f'{brainfolder}/santa_cap2.svg')
        santa_cap.rotate(-PI/6-5*PI/180) # set_height(1.6)
        santa_cap.set_fill('#c60f0f')
        santa_cap.center().scale(0.5)
        santa_cap.center()
        santa_cap.shift(brain.get_right()+UP*0.55)
        
        # Add people to thank
        people = []
        with open(f'{folder}/people_to_thank.txt', 'r') as file:
            lines = file.readlines()
        for line in lines:
            people.append(Tex(line.replace('\n', ''), font_size=50))

        all_people_left = VGroup(*people[::2])
        all_people_left.arrange(DOWN, buff=MED_LARGE_BUFF)
        all_people_right = VGroup(*people[1::2])
        all_people_right.arrange(DOWN, buff=MED_LARGE_BUFF)

        all_people_left.to_edge(LEFT)
        all_people_right.to_edge(RIGHT)

        thank_you = Tex(R'Thank\enspace you\enspace all!', 
            font_size=50, color=BLUE)
        thank_you.to_edge(UP)

        self.add(brain.set_color("#404340"), santa_cap)

        for ii in range(len(people)//2):
            if ii == 0:
                self.play(
                    brain.animate.scale(1.5),
                    brain_outline.animate.scale(1.5),
                    # very bad code here below: -> look for a solution 
                    # that preserves positions when scaling
                    santa_cap.animate.scale(1.5).shift(brain.get_right()+LEFT*0.5+UP*0.2),
                    Write(thank_you),
                    self.camera.frame.animate.scale(1.5),
                )

            self.play(
                ShowPassingFlash(
                    brain_outline.set_color("#613325"),
                    time_width=0.5,
                    run_time=2,
                ),
                Write(all_people_left[ii]),
                Write(all_people_right[ii]),
                run_time=1,
            )
            self.wait(5) if ii == 0 else None

        self.wait(5)


        # Add feedback window
        tamaloo = Tex(str('Tamaloo ?!'), font_size=50)
        feedback = Tex(str('Feedback !?'), font_size=50)

        tamaloo.to_edge(UP)
        feedback.to_edge(UP)

        self.play(
            self.camera.frame.animate.scale(1./1.5),
            *(FadeOut(pp) for pp in all_people_left),
            *(FadeOut(pp) for pp in all_people_right),
            FadeOut(thank_you),
            Transform(santa_cap, tamaloo),
            run_time=0.5,
        )

        for ii in range(10):
            if ii == 0:
                self.play(
                    Transform(santa_cap, feedback),
                    run_time=2,
                )

            self.play(
                ShowPassingFlash(
                    brain_outline,
                    time_width=0.5,
                    run_time=2,
                ),
            )



class HemispheresHighlight(Scene):
    def construct(self):
        # Brain display
        folder = 'C:/Users/ricta/Documents/videos3b1b/_2024/Thesis/Data'
        brainfolder = f'{folder}/HemispheresSvg'
        brain = SVGMobject(file_name=f'{brainfolder}/brain1.svg')
        brain.set_height(1.6)
        brain.set_fill(get_opposite_color(GREY_B))
        brain_outline = brain.copy()
        brain_outline.set_fill(opacity=0)
        brain_outline.set_stroke(get_opposite_color(BLUE_B), 3)
        brain.center()
        brain_outline.center()

        print(len(brain.submobjects))
        # self.add(brain)

        right_hemisphere = brain.submobjects[0].copy()
        right_hemisphere_smth = brain.submobjects[1].copy()
        brain_outline_right = brain_outline.submobjects[0].copy()
        brain_outline_right_smth = brain_outline.submobjects[1].copy()
        
        left_hemisphere = brain.submobjects[2].copy()
        left_hemisphere_smth = brain.submobjects[3].copy()
        brain_outline_left = brain_outline.submobjects[2].copy()
        brain_outline_left_smth = brain_outline.submobjects[3].copy()

        left_hemisphere.set_fill(get_opposite_color(BLUE_E), opacity=0.8)
        left_hemisphere_smth.set_fill(get_opposite_color(BLUE_E), opacity=0.8)
        right_hemisphere.set_fill(get_opposite_color(RED_E), opacity=0.8)
        right_hemisphere_smth.set_fill(get_opposite_color(RED_E), opacity=0.8)

        left_hemisphere.shift(LEFT * 0.3)
        brain_outline_left.shift(LEFT * 0.3)
        left_hemisphere_smth.shift(LEFT * 0.3)
        brain_outline_left_smth.shift(LEFT * 0.3)

        right_hemisphere.shift(RIGHT * 0.3)
        brain_outline_right.shift(RIGHT * 0.3)    
        right_hemisphere_smth.shift(RIGHT * 0.3)
        brain_outline_right_smth.shift(RIGHT * 0.3)

        self.play(self.camera.frame.animate.scale(.25)) 

        self.play(
            FadeIn(left_hemisphere),
            FadeIn(left_hemisphere_smth), 
            FadeIn(right_hemisphere),
            FadeIn(right_hemisphere_smth) 
            )
        
        FC_text = Tex(R'\overset{FC}{\Leftrightarrow}', 
            font_size=42, color=get_opposite_color(BLUE))
        FC_text.center() # move_to(brain_outline, UP).shift(UP*0.7)

        brain_outline_right.set_stroke(width=10)
        brain_outline_right_smth.set_stroke(width=10)
        brain_outline_left.set_stroke(width=10)
        brain_outline_left_smth.set_stroke(width=10)

        self.add(FC_text) # brain, 
        self.wait(5)

        # PassingFlash
        for ii in range(20):
            self.play(
                ShowPassingFlash(
                    brain_outline_right,
                    time_width=0.5,
                    run_time=2,
                ),
                ShowPassingFlash(
                    brain_outline_right_smth,
                    time_width=0.5,
                    run_time=2,
                ),
                ShowPassingFlash(
                    brain_outline_left,
                    time_width=0.5,
                    run_time=2,
                ),
                ShowPassingFlash(
                    brain_outline_left_smth,
                    time_width=0.5,
                    run_time=2,
                )
            )
            
        self.wait(5)



class fMRIScan(Scene):
    def construct(self):
        # Set up Axes
        axes = Axes(
            x_range=[-1, 10, 2],
            y_range=[-1, 1, 0.4],
            width=FRAME_WIDTH*0.5,
            height=FRAME_WIDTH*0.5,
            axis_config=dict(
                include_tip=True,
                # tip_shape='StealthTip',
                stroke_color=WHITE,
                # numbers_to_exclude=[0],
            ),
        )
        # self.frame.reorient(43, 76, 1, IN, 10)
        axes.center()  # to_edge(UL)
        # self.frame.add_updater(lambda m, dt: m.increment_theta(dt * 3 * DEGREES))
        labels = axes.get_axis_labels(R"Time", R"BOLD\enspace signal",)
        # for ii, label in enumerate(labels, start=1):
        #     label.set_color(WHITE)
        #     label[ii].align_to(axes, LEFT).shift(DOWN * 0.1)
        labels[0].align_to(axes, RIGHT).shift(DOWN * 0.1)
        labels[1].align_to(axes, RIGHT*0.5).shift(LEFT)


        title = Tex(R"\# ROI", font_size=50)
        title.next_to(axes, LEFT).shift(UP) 


        def noisy_sine(x):
            return -0.5*np.sin(2 * np.pi * x / 5) + 0.3 * np.random.uniform(-0.5, 0.5)

        # Creazione del plot con la funzione noisy_sine
        signal = axes.get_graph(noisy_sine, x_range=[0, 10], color=YELLOW)

        self.add(axes, labels, title)
        self.play(
            ShowCreation(axes),
            ShowCreation(labels),
            Write(title),
            ShowCreation(signal, run_time=3),
            run_time=2.5,
        )



class BrainHighlight(Scene):
    def construct(self):
        # Brain display
        folder = 'C:/Users/ricta/Documents/videos3b1b/_2024/Thesis/Data'
        brainfolder = f'{folder}/BrainSvg'
        brain = SVGMobject(file_name=f'{brainfolder}/brain1.svg')
        brain.set_height(1.6)
        brain.set_fill(get_opposite_color(BLACK)) # GREY_B
        brain_outline = brain.copy()
        brain_outline.set_fill(opacity=0)
        brain_outline.set_stroke(get_opposite_color(BLUE_B), 3)
        brain.center()
        brain_outline.center()

        self.play(self.camera.frame.animate.scale(.35))

        people = [
            Tex(R"Prof.\enspace M.\enspace Allegra", font_size=20), 
            Tex(R"Prof.\enspace G.\enspace Deco", font_size=20)
        ]
        people[0].center().move_to(DL)
        people[1].center().move_to(DR)

        # thank_you = Tex(R'Thank\enspace you\enspace all!', 
        #     font_size=50, color=get_opposite_color(BLUE))
        # thank_you.to_edge(UP)

        self.add(brain.set_color(get_opposite_color("#404340"))) # , thank_you
        self.play(
            *(Write(pep) for pep in people),
            run_time=1.5
        ) 
        for ii in range(30):
            self.play(
                ShowPassingFlash(
                    brain_outline, #.set_color(get_opposite_color("#613325")),
                    time_width=0.5,
                    run_time=2,
                ),
                run_time=1,
            )

        self.wait(5)



class BrainHighlightROI(Scene):
    def construct(self):
        # Brain display
        folder = 'C:/Users/ricta/Documents/videos3b1b/_2024/Thesis/Data'
        brainfolder = f'{folder}/BrainSvg'
        brain = SVGMobject(file_name=f'{brainfolder}/brain1.svg')
        brain.set_height(1.6)
        brain.set_fill(get_opposite_color(BLACK)) # GREY_B
        brain_outline = brain.copy()
        brain_outline.set_fill(opacity=0)
        brain_outline.set_stroke(get_opposite_color(BLUE_B), 3)
        brain.center()
        brain_outline.center()

        self.play(self.camera.frame.animate.scale(.35))
        self.add(brain.set_color(get_opposite_color("#404340")))

        num_ROIs = 15
        circles = [         # ROIs
            Circle(
                radius = 0.04,
                stroke_color = get_opposite_color(BLUE_E),
                # stroke_width = 10,
                # fill_color = None,
                fill_opacity = 0
            )
        for _ in range(num_ROIs)
        ]

        self.play(
            ShowPassingFlash(
                brain_outline, #.set_color(get_opposite_color("#613325")),
                time_width=0.5,
                run_time=3
            )
        )

        self.wait(2)
        
        self.play(
            brain.animate.shift(LEFT),
            brain_outline.animate.shift(LEFT),
            run_time=3
        )

        center = brain_outline.get_center()
        # shift these ROIs on the brain surface
        for circ in circles:
            circ.move_to(center + np.random.normal(loc=0.03, scale=0.28, size=3))
            # circ.fix_in_frame()

        self.play(*(ShowCreation(circ) for circ in circles))

        random_selection = np.random.randint(num_ROIs)
        selected_circle = circles[random_selection]

        self.wait(1.5)

        target_point = np.array([0.5, 0.1, 0])
        arrow = Arrow(start=selected_circle.get_center(), end=target_point, 
            color=WHITE, stroke_width=0.01, buff=0)
        
        # Create plot with rs-fMRI time behaviour
        # Set up Axes
        axes = Axes(
            x_range=[-1, 10, 2],
            y_range=[-1, 1, 0.4],
            width=FRAME_WIDTH*0.3,
            height=FRAME_WIDTH*0.3,
            axis_config=dict(
                include_tip=True,
                # tip_shape='StealthTip',
                stroke_color=WHITE,
                # numbers_to_exclude=[0],
            ),
        )
        # self.frame.reorient(43, 76, 1, IN, 10)
        # self.frame.add_updater(lambda m, dt: m.increment_theta(dt * 3 * DEGREES))
        labels = axes.get_axis_labels(R"Time", R"BOLD\enspace signal",)
        # for ii, label in enumerate(labels, start=1):
        #     label.set_color(WHITE)
        #     label[ii].align_to(axes, LEFT).shift(DOWN * 0.1)
        labels[0].align_to(axes, RIGHT).shift(DOWN * 0.1)
        labels[1].align_to(axes, RIGHT)#.shift(LEFT)
        for label in labels:
            label.scale(0.7)


        title = Tex(R"\# ROI", font_size=50)
        title.next_to(axes, UP*0.7).shift(LEFT*0.7) 


        def noisy_sine(x):
            return -0.5*np.sin(2 * np.pi * x / 5) + 0.3 * np.random.uniform(-0.5, 0.5)

        # Creazione del plot con la funzione noisy_sine
        signal = axes.get_graph(noisy_sine, x_range=[0, 9.5], color=get_opposite_color(BLUE_E))

        group = VGroup(axes, labels, title, signal)
        group.scale(0.3)
        group.move_to(target_point).shift(RIGHT*0.7+UP*0.2)
        
        # self.add(axes, labels, title)
        # self.add(group)
        self.play(
            ShowCreation(arrow),
            ShowCreation(axes),
            ShowCreation(labels),
            Write(title),
            ShowPassingFlash(
                brain_outline, #.set_color(get_opposite_color("#613325")),
                time_width=0.5,
                run_time=3
            ),
            ShowCreation(signal, run_time=3),
            run_time=2.5,
        )

        self.wait(3)

        for ii in range(10):
            selected_circle = circles[-ii]
            new_arrow = Arrow(start=selected_circle.get_center(), end=target_point, 
                color=WHITE, stroke_width=0.01, buff=0)
            new_signal = axes.get_graph(noisy_sine, x_range=[0, 9.5], color=get_opposite_color(BLUE_E))
            self.play(
                ShowPassingFlash(
                    brain_outline, #.set_color(get_opposite_color("#613325")),
                    time_width=0.5,
                    run_time=2,
                ),
                Transform(arrow, new_arrow),
                # FadeOut(arrow),
                FadeOut(signal),
                ShowCreation(new_signal, 
                    run_time=2),
                run_time=1,
            )
            if ii != 9:
                self.remove(arrow)
            arrow = new_arrow
            signal = new_signal

        self.wait(5)



class NetworkScene(InteractiveScene):
    def construct(self):
        # # Encoder network
        # self.layer_sizes = [8, 6, 6, 4]
        # self.network_mob_config = {}
        # encoder = self.add_network()
        # self.add_nn_label(encoder, nn_name='\\varphi(x)')
        # # self.remove(encoder)

        # # Koopman network
        # self.layer_sizes = [4, 4]
        # koopman = self.add_network()
        # koopman.next_to(encoder)
        # self.add_nn_label(koopman, nn_name='\\mathcal{K}')
        # # self.remove(koopman)

        # Encoder + Koopman + Decoder network
        self.layer_sizes = [256, 8, 6, 4, 4, 4, 4, 6, 8, 256]
        self.network_mob_config = {}
        self.whole_network = self.add_network().scale(0.85)

        input_nn = Tex(R"x_{t} \to")
        output_nn = Tex(R"\to x_{t+1}")
        input_nn.next_to(self.whole_network, LEFT)
        output_nn.next_to(self.whole_network, RIGHT)
        
        self.show_network()
        self.play(
            Write(input_nn),
            Write(output_nn)
        )

        # Add boxes
        self.rects_widths, self.rects_heights, self.rectangles = [], [], []
        self.add_box_around_layers(self.whole_network.layers[:4])
        self.add_box_around_layers(self.whole_network.layers[4:6])
        self.add_box_around_layers(self.whole_network.layers[6:])

        # Add all sub-networks labels
        self.labels_tex, self.labels = [], [] 
        self.braces_tex, self.braces = [], []
        self.add_nn_labels(
            ['\\varphi', '\\mathcal{K}', '\\psi'],
            ['Encoder', 'Koopman', 'Decoder']
        )

        # perform animations
        self.animate_rects_labels()
        self.show_math()

        # clean up the scene:
        self.play(
            # *(FadeOut(rect) for rect in self.rectangles),
            *(FadeOut(bra) for bra in  self.braces),
            *(FadeOut(lab) for lab in self.labels),
            FadeOut(input_nn), FadeOut(output_nn)
        )

        # Forward/Backward propagation
        in_vect = np.random.random(self.network.sizes[0])
        self.feed_forward(in_vect)
        self.show_learning()


        # Koopman network - auxiliary network
        self.play(
            # focus on the Koopman layers
            self.rectangles[1].animate.set_fill(color=get_opposite_color(GREY), opacity=0.989),
            # self.camera.frame.animate.scale(1/2.).move_to(
            #     self.whole_network.get_center()
            # )
        )
        self.wait(1.5)

        # Add some Tex
        t2c={R"\mu": get_opposite_color(BLUE), R"\omega": get_opposite_color(RED), }
        line1 = Tex(
            R"\text{Use } \mathcal{K} \text{ to model the eigenvalues of a non-linear}",
            font_size=32)
        line2 = Tex(    
            R"\begin{raggedright}\text{dynamics, as those of a linear one.}\end{raggedright}", 
            font_size=32) 
        note = VGroup(line1, line2)
        note.arrange(DOWN, buff=MED_LARGE_BUFF)

        blockStructure = Tex(
            R"""
                \mathcal{B}_{j}(\mu_{j}, \omega_{j}) = 
                \underbrace{e^{\mu_{j}\: dt}
                \begin{pmatrix}
                \cos(\omega_{j}\: dt) & - \sin(\omega_{j}\: dt) \\
                \sin(\omega_{j}\: dt) & \cos(\omega_{j}\: dt)
                \end{pmatrix}}_{\text{block structure}}
            """,
            t2c=t2c,
            font_size = 32
        )

        blockEigenvalues = Tex(
            R"""
                \mathbb{E}(\mathcal{B}_{j}) = \mu_{j} \pm i\omega_{j}
            """,
            t2c=t2c,
            font_size = 32
        )

        
        koopman_matrix = Tex(
            R"""
                \mathcal{K} = 
                \begin{pmatrix}
                \mathcal{B}_{1} & 0 & \cdots & 0 \\
                0 & \mathcal{B}_{2} & \cdots & 0 \\
                0 & \cdots &  & 0 \\
                0 & \cdots & \cdots &  \mathcal{B}_{\lfloor N/2 \rfloor}
                \end{pmatrix}
            """,
            t2c={
                R"\mu": get_opposite_color(BLUE),
                R"\omega": get_opposite_color(RED),
            },
            font_size = 32
        )
        ellipsis = Tex(R"\cdots").rotate(-PI/4)
        title = Tex(R"\mathrm{Auxiliary-network}:", font_size=38)
        title.to_edge(UL)
        
        # koopman layers
        self.layer_sizes = [100, 30, 2]
        self.network_mob_config = {}
        blockStructure.center()
        note.to_edge(2*UP).next_to(blockStructure, UP)
        koopman_matrix.to_edge(3.5*DOWN).next_to(blockStructure, DOWN)
        blockEigenvalues.center().to_edge(RIGHT).shift(DOWN*0.7)
        ellipsis.move_to(koopman_matrix[22].shift(RIGHT*0.72)).scale(0.7)

        # add NN label and brace
        # self.reset_lists()
        output_labels = [
            Tex(R"\mu", t2c={R"\mu": get_opposite_color(BLUE),}),
            Tex(R"\omega", t2c={R"\omega": get_opposite_color(RED)})
        ]
        self.koopman_network = self.add_network(
            output_labels=output_labels, add=False).scale(0.85)
        self.add_box_around_layers(self.koopman_network)
        nn_input = Tex(R"\varphi(\boldsymbol{x}_{t})\to")
        nn_input.next_to(self.koopman_network, LEFT)
        self.add_nn_labels([R'\Lambda(\varphi(\boldsymbol{x}_{t}))'], [R'\mathrm{Auxiliary-network}'], [self.rectangles[-1]])
        self.add(self.koopman_network)
        self.play(
            self.camera.frame.animate.scale(1).move_to(
                self.koopman_network.get_center()
            ),
            Write(nn_input),
            Transform(self.whole_network, self.koopman_network),
            FadeOut(self.whole_network),
            *(FadeOut(rect) for rect in self.rectangles[:-1]),
            *(FadeOut(bra) for bra in  self.braces_tex[:-1]),
            *(FadeOut(lab) for lab in self.labels_tex[:-1]),
        )
        self.animate_rect_label(index=-1) # add last added rect and label
        self.wait(5)
        self.play(
            self.koopman_network.animate.to_edge(LEFT),
            *(FadeOut(obj, run_time=1.2) for obj in [
                self.rectangles[-1], self.braces_tex[-1],
                self.braces[-1], self.labels_tex[-1], 
                self.labels[-1], 
                nn_input
                ]),
            Write(note), Write(blockStructure), 
            Write(blockEigenvalues), Write(title),
            Write(koopman_matrix), Write(ellipsis), 
            run_time=3,        
        )
        self.wait(3)



    def add_box_around_layers(self, layers):
        layer_bounding_box = layers.get_bounding_box()
        offset = 0.3 # to have the rect to surround completely the edges 
        self.rects_widths.append(layer_bounding_box[-1][0] - layer_bounding_box[0][0] + offset)
        self.rects_heights.append(layer_bounding_box[-1][1] - layer_bounding_box[0][1] + offset)
        rect = Rectangle(
            width=min(self.rects_widths[0], self.rects_widths[-1]),
            height=max(self.rects_heights[0], self.rects_heights[-1]),
            color=get_opposite_color(YELLOW),
            fill_color=get_opposite_color(WHITE),
            fill_opacity=0.03, 
        )
        rect.move_to(layers.get_center())
        self.rectangles.append(rect)


    def add_nn_labels(self, tex_nn_names, nn_names, rects=None):
        if rects == None:
            rects = self.rectangles

        for ii, rect in enumerate(rects):
            brace_tex = Brace(rect, DOWN)
            label_tex = brace_tex.get_tex(tex_nn_names[ii])
            brace = Brace(rect, UP)
            label = brace.get_tex(nn_names[ii])

            self.labels_tex.append(label_tex)
            self.labels.append(label)
            self.braces_tex.append(brace_tex)
            self.braces.append(brace)



    def animate_rect_label(self, index=0):
        self.add(
                self.labels_tex[index],
                self.labels[index],
                self.braces[index],
                self.braces_tex[index],
                self.rectangles[index]
            )
        self.play(
            ShowCreation(self.rectangles[index]),
            ShowCreation(self.braces_tex[index]),
            ShowCreation(self.braces[index]),
            Write(self.labels_tex[index]),
            Write(self.labels[index]),
            run_time=0.8,
        )


    def animate_rects_labels(self):
        for ii in range(len(self.rectangles)):
            self.add(
                self.labels_tex[ii],
                self.labels[ii],
                self.braces[ii],
                self.braces_tex[ii],
                self.rectangles[ii]
            )

            self.play(
                ShowCreation(self.rectangles[ii]),
                ShowCreation(self.braces_tex[ii]),
                ShowCreation(self.braces[ii]),
                Write(self.labels_tex[ii]), 
                Write(self.labels[ii]),
                run_time=0.8,
            )

        self.wait(2)


    def add_network(self, output_labels=False, add=True):
        self.network = Network(sizes = self.layer_sizes)
        self.network_mob = NetworkMobject(
            self.network, output_labels,
            **self.network_mob_config
        )
        if add:
            self.add(self.network_mob)
        return self.network_mob

    def show_network(self):
        network_mob = self.network_mob

        self.play(ShowCreation(
            network_mob.edge_groups,
            lag_ratio = 0.5,
            run_time = 3.5,
            rate_func=linear,
        ))
        self.wait()



    def feed_forward(self, input_vector, false_confidence = False, added_anims = None):
        self.play(
            self.camera.frame.animate.scale(1./1.2).move_to(
                self.whole_network.get_center()
            )
        )
        if added_anims is None:
            added_anims = []
        activations = self.network.get_activation_of_all_layers(
            input_vector
        )
        if false_confidence:
            i = np.argmax(activations[-1])
            activations[-1] *= 0
            activations[-1][i] = 1.0
        for i, activation in enumerate(activations):
            self.show_activation_of_layer(i, activation, added_anims)
            added_anims = []

    def show_activation_of_layer(self, layer_index, activation_vector, added_anims = None):
        if added_anims is None:
            added_anims = []
        layer = self.network_mob.layers[layer_index]
        active_layer = self.network_mob.get_active_layer(
            layer_index, activation_vector
        )
        anims = [Transform(layer, active_layer, run_time=0.1)]
        if layer_index > 0:
            anims += self.network_mob.get_edge_propogation_animations(
                layer_index-1
            )
        anims += added_anims
        self.play(*anims, run_time=0.03)
        self.wait()

    def remove_random_edges(self, prop = 0.9):
        for edge_group in self.network_mob.edge_groups:
            for edge in list(edge_group):
                if np.random.random() < prop:
                    edge_group.remove(edge)


    def show_math(self):
        equation = Tex(
            R"\varphi(\boldsymbol{x}_{t+1}) = \mathcal{K} \varphi(\boldsymbol{x}_{t})", 
            t2c={R"t" : get_opposite_color(GREEN),},
        )

        # self.camera.frame.scale(1.2)
        # equation.move_to(self.network_mob.get_corner(RIGHT))
        equation.move_to(DOWN)
        equation.shift(DOWN*3.4)
        self.equation = equation

        self.play(
            self.camera.frame.animate.scale(1.2).move_to(
                self.whole_network.get_center()
            ),
            Write(self.equation, run_time = 2),
            )
        self.wait(5)


    def show_learning(self):
        # word = self.words[0][1].copy()
        # rect = SurroundingRectangle(word, color = YELLOW)
        self.network_mob.neuron_fill_color = get_opposite_color(YELLOW)

        layer = self.network_mob.layers[-1]
        activation = np.zeros(len(layer.neurons))
        activation[1] = 1.0
        active_layer = self.network_mob.get_active_layer(
            -1, activation
        )
        # word_group = VGroup(word, rect)
        # word_group.generate_target()
        # word_group.target.move_to(self.equation, LEFT)
        # word_group.target[0].set_color(YELLOW)
        # word_group.target[1].set_stroke(width = 0)

        # self.play(ShowCreation(rect))
        self.play(
            Transform(layer, active_layer, run_time=0.3),
            FadeOut(self.equation),
            # MoveToTarget(word_group),
        )
        for edge_group in reversed(self.network_mob.edge_groups):
            edge_group.generate_target()
            for edge in edge_group.target:
                edge.set_stroke(
                    get_opposite_color(YELLOW), 
                    width = 4*np.random.random()**2
                )
            self.play(MoveToTarget(edge_group), run_time=0.3)
        self.wait()

        # self.learning_word = word


    def reset_lists(self):
        self.labels_tex, self.labels = [], [] 
        self.braces_tex, self.braces = [], []
        self.rects_widths, self.rects_heights, self.rectangles = [], [], []
        