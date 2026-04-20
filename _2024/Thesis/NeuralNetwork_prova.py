import numpy as np
import sys
import os

lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "./../../"))
sys.path.append(lib_path)
from manim_imports_ext import *

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from network import Network


class NetworkMobject(VGroup):
    # CONFIG = {
    #     "neuron_radius" : 0.15,
    #     "neuron_to_neuron_buff" : MED_SMALL_BUFF,
    #     "layer_to_layer_buff" : LARGE_BUFF,
    #     "neuron_stroke_color" : BLUE,
    #     "neuron_stroke_width" : 3,
    #     "neuron_fill_color" : GREEN,
    #     "edge_color" : GREY_B,
    #     "edge_stroke_width" : 2,
    #     "edge_propogation_color" : YELLOW,
    #     "edge_propogation_time" : 1,
    #     "max_shown_neurons" : 16,
    #     "brace_for_large_layers" : True,
    #     "average_shown_activation_of_large_layer" : True,
    #     "include_output_labels" : False,
    # }
    def __init__(self, neural_network, **kwargs):
        VGroup.__init__(self, **kwargs)
        self.neural_network = neural_network
        self.layer_sizes = neural_network.sizes

        # Converting CONFIG (setup) into self
        self.neuron_radius = 0.15
        self.neuron_to_neuron_buff = MED_SMALL_BUFF
        self.layer_to_layer_buff = LARGE_BUFF
        self.neuron_stroke_color = BLUE
        self.neuron_stroke_width = 3
        self.neuron_fill_color = GREEN
        self.edge_color = GREY_B
        self.edge_stroke_width = 2
        self.edge_propogation_color = YELLOW
        self.edge_propogation_time = 1
        self.max_shown_neurons = 16
        self.brace_for_large_layers = True
        self.average_shown_activation_of_large_layer = True
        self.include_output_labels = False


        self.add_neurons()
        print(self.visible_indices, self.layer_sizes)
        self.add_edges()

    # def add_neurons(self):
    #     layers = VGroup(*[
    #         self.get_layer(size)
    #         for size in self.layer_sizes
    #     ])
    #     layers.arrange(RIGHT, buff = self.layer_to_layer_buff)
    #     self.layers = layers
    #     self.add(self.layers)
    #     if self.include_output_labels:
    #         self.add_output_labels()

    def add_neurons(self):
        layers = VGroup()
        max_visible_layers = 4
        if len(self.layer_sizes) > max_visible_layers:
            visible_indices = list(range(max_visible_layers // 2)) + \
                              list(range(len(self.layer_sizes) - max_visible_layers // 2, len(self.layer_sizes)))
            self.visible_indices = visible_indices
            print(visible_indices)
            for i in range(len(self.layer_sizes)):
                if i in visible_indices:
                    layers.add(self.get_layer(self.layer_sizes[i]))
                else:
                    # if isinstance(layers[i-1], Tex):
                    #     # avoid adding consecutive "\cdots"
                    #     layers.add(Tex(""))
                    #     continue
                    dots = Tex(R"\cdots")
                    dots.move_to(layers[i-1].get_center()) 
                    print(layers[i-1].get_center())
                    layers.add(dots)
        else:
            for size in self.layer_sizes:
                layers.add(self.get_layer(size))
        
        layers.arrange(RIGHT, buff=self.layer_to_layer_buff)
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
            if isinstance(l1, Tex) or isinstance(l2, Tex):
                continue
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
            label = Tex(str(n))
            label.set_height(0.75*neuron.get_height())
            label.move_to(neuron)
            label.shift(neuron.get_width()*RIGHT)
            self.output_labels.add(label)
        self.add(self.output_labels)


class BrainHighlight(Scene):
    def construct(self):
        # Brain display
        brainfolder = 'C:/Users/ricta/Documents/videos3b1b/_2024/Thesis/Data/BrainSvg'
        brain = SVGMobject(file_name=f'{brainfolder}/brain.svg')
        brain.set_height(1.6)
        brain.set_fill(GREY_B)
        brain_outline = brain.copy()
        brain_outline.set_fill(opacity=0)
        brain_outline.set_stroke(BLUE_B, 3)

        brain.to_edge(DR)
        brain_outline.to_edge(DR)
        # how = Tex(r"How?!?")
        # how.scale(2)
        # how.next_to(brain, UP)

        self.add(brain)
        # self.play(Write(how))
        for _ in range(2):
            self.play(
                ShowPassingFlash(
                    brain_outline,
                    time_width=0.5,
                    run_time=2
                )
            )
        self.wait()



class NetworkScene(InteractiveScene):
    def construct(self):
        # NN scene
        self.layer_sizes = [8, 6, 6, 6, 6, 5, 4]
        self.network_mob_config = {}
        self.add_network()

        # perform animations
        self.show_network()
        self.show_math()
        self.show_learning()


    def add_network(self):
        self.network = Network(sizes = self.layer_sizes)
        self.network_mob = NetworkMobject(
            self.network,
            **self.network_mob_config
        )
        self.add(self.network_mob)

    def show_network(self):
        network_mob = self.network_mob

        self.play(ShowCreation(
            network_mob.edge_groups,
            lag_ratio = 0.5,
            run_time = 2,
            rate_func=linear,
        ))
        in_vect = np.random.random(self.network.sizes[0])
        self.feed_forward(in_vect)



    def feed_forward(self, input_vector, false_confidence = False, added_anims = None):
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
        anims = [Transform(layer, active_layer)]
        if layer_index > 0:
            anims += self.network_mob.get_edge_propogation_animations(
                layer_index-1
            )
        anims += added_anims
        self.play(*anims)

    def remove_random_edges(self, prop = 0.9):
        for edge_group in self.network_mob.edge_groups:
            for edge in list(edge_group):
                if np.random.random() < prop:
                    edge_group.remove(edge)


    def show_math(self):
        equation = Tex(
            "\\textbf{a}_{l+1}", "=",  
            "\\sigma(",
                "W_l", "\\textbf{a}_l", "+", "b_l",
            ")"
        )
        equation.set_color_by_tex_to_color_map({
            "\\textbf{a}" : GREEN,
        })
        equation.move_to(self.network_mob.get_corner(UP+RIGHT))
        equation.to_edge(UP)

        self.play(Write(equation, run_time = 2))
        self.wait()

        self.equation = equation



    def show_learning(self):
        # word = self.words[0][1].copy()
        # rect = SurroundingRectangle(word, color = YELLOW)
        self.network_mob.neuron_fill_color = YELLOW

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
            Transform(layer, active_layer),
            FadeOut(self.equation),
            # MoveToTarget(word_group),
        )
        for edge_group in reversed(self.network_mob.edge_groups):
            edge_group.generate_target()
            for edge in edge_group.target:
                edge.set_stroke(
                    YELLOW, 
                    width = 4*np.random.random()**2
                )
            self.play(MoveToTarget(edge_group))
        self.wait()

        # self.learning_word = word
