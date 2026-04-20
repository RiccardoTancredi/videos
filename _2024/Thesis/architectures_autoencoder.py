import numpy as np
import sys
import os

lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "./../../"))
sys.path.append(lib_path)
from manim_imports_ext import *

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


def get_opposite_color(color):
    """Restituisce il colore opposto di un dato colore in formato RGB."""
    rgb = np.array(color_to_rgb(color))  # Ottieni il colore in [0,1]
    opposite_rgb = 1 - rgb  # Complemento in [0,1]
    return rgb_to_color(opposite_rgb)  # Converti di nuovo in colore Manim


class NetworkArchitecture(Scene):
    def construct(self):
        # --- Funzione helper per creare componenti ---
        def create_component(shape, inside_str, label_str, text_color=WHITE):
            # Testo interno con specifiche (numero di layer, neuroni, ecc.)
            inside_text = Tex(inside_str, font_size=30).set_color(text_color)
            inside_text.move_to(shape.get_center())
            # if shift_type is not None:
            # 	inside_text.shift(shift_type*0.2)

            # Ottieni la larghezza della forma per avere un underbrace "della stessa misura"
            width = shape.get_width()
            # Crea il testo con underbrace in LaTeX: qui usiamo \hspace con larghezza approssimata dalla forma.
            # Nota: la conversione tra unità di Manim e spazi in LaTeX non è esatta,
            # quindi potresti dover regolare manualmente il valore (ad es. in cm).
            label = Tex(R"\underbrace{\hspace{%0.2fcm}}_{\mathrm{%s}}" % (width, label_str), font_size=40)
            label.next_to(shape, DOWN, buff=0.2)

            return VGroup(shape, inside_text, label)

        def create_connected_component(num_layers, inside_strs, label_str, text_color=WHITE,
            shape_width=1.5, shape_height=1, spacing=0.2, color=WHITE, fill_color='#AD3B4C'):
            """
            Crea un gruppo di rettangoli connessi, ognuno con un testo all'interno, e un'unica underbrace sotto.
            
            Args:
                num_layers (int): Numero di rettangoli (layer).
                inside_strs (list of str): Lista di stringhe da inserire dentro ogni rettangolo.
                label_str (str): Testo da mettere sotto l'underbrace.
                shape_width (float): Larghezza di ogni rettangolo.
                shape_height (float): Altezza di ogni rettangolo.
                spacing (float): Distanza tra i rettangoli.
            
            Returns:
                VGroup: Gruppo contenente tutti i rettangoli, i testi interni e l'underbrace con etichetta.
            """
            
            assert len(inside_strs) == num_layers, "Il numero di stringhe deve corrispondere al numero di layer."

            # Creazione dei rettangoli allineati orizzontalmente
            shapes = VGroup()
            for i in range(num_layers):
                rect = Rectangle(width=shape_width, height=shape_height, color=color).set_fill(fill_color, opacity=1)
                rect.move_to(RIGHT * (i * (shape_width + spacing)))  # Allinea i rettangoli in orizzontale
                # rect.rotate(-PI/2)
                text = Tex(R"n = {%s}" % inside_strs[i], font_size=30).set_color(text_color).move_to(rect.get_center())  # Testo dentro il rettangolo
                shapes.add(VGroup(rect, text))

            # Centrare la componente nel punto (0,0)
            shapes.move_to(ORIGIN)

            # Calcolare la larghezza totale della componente
            total_width = num_layers * shape_width + (num_layers - 1) * spacing

            # Creare l'underbrace sotto tutta la larghezza della componente
            underbrace = Tex(R"\underbrace{\hspace{%0.2fcm}}_{\mathrm{%s}}" % (total_width, label_str), font_size=40)
            
            # Posizionare l'underbrace sotto il gruppo di rettangoli
            underbrace.next_to(shapes, DOWN, buff=0.2)

            return VGroup(shapes, underbrace)



        # --- Input e Output (rettangoli con testo al centro) ---
        input_rect = Rectangle(width=3, height=1, color=WHITE).set_fill(GREY_E, opacity=1)
        # input_text = Tex(R"Input\\newline [Dim: ...]", font_size=50)
        # input_text.move_to(input_rect.get_center())
        # input_group = VGroup(input_rect, input_text)
        input_component = create_component(
        	input_rect, 
        	R"M = 100",
        	"Input"
        )

        output_rect = Rectangle(width=3, height=1, color=WHITE).set_fill(GREY_E, opacity=1)
        # output_text = Tex("Output\\newline [Dim: ...]", font_size=50)
        # output_text.move_to(output_rect.get_center())
        # output_group = VGroup(output_rect, output_text)
        output_component = create_component(
        	output_rect, 
        	R"M = 100", 
        	"Output"
        )

        # --- Encoder: Triangolo rosso orientato a destra ---
        # encoder_shape.rotate(-PI/2)  # Punta a destra
        # encoder_shape.scale(1.5) # 0.8
        encoder_component = create_connected_component(
            num_layers=3, 
            inside_strs=["60", "30", "2"],
            label_str="Encoder",
            fill_color=get_opposite_color("#376996"),
            text_color=BLACK
        )

        # --- Auxiliary Network (Koopman): Rettangolo con base più stretta dell'altezza ---
        aux_shape = Rectangle(width=1.5, height=1, # color=get_opposite_color("#6290C8"), 
            fill_color=get_opposite_color("#6290C8"), fill_opacity=1)
        aux_component = create_component(
            aux_shape, 
            "HL:2\nN:[300, 12]", 
            "Koopman"
        )

        # --- Decoder: Triangolo verde orientato a sinistra ---
        # decoder_shape = Triangle(color="#EC9274", fill_color="#EC9274", fill_opacity=1)
        # decoder_shape.rotate(PI/2)  # Punta a sinistra
        # decoder_shape.scale(0.8) 
        decoder_shape = Rectangle(width=1.5, height=1, # color=get_opposite_color("#829CBC"), 
            fill_color=get_opposite_color("#829CBC"), fill_opacity=1)
        decoder_component = create_component(
            decoder_shape, 
            "n = 100", 
            "Decoder",
            text_color=BLACK
        )

        # --- Disposizione dei componenti in sequenza ---
        components = VGroup(
            # input_group,
            input_component,
            encoder_component,
            # aux_component,
            decoder_component,
            output_component
        )
        components.arrange(RIGHT, buff=1)

        # --- Creazione delle frecce che collegano i componenti ---
        arrows = VGroup()
        for i in range(len(components) - 1):
            arrow = Arrow(
                start=components[i][0].get_right(),
                end=components[i+1][0].get_left(),
                buff=0.1,
                stroke_width=2,
            )
            arrows.add(arrow)

        # Aggiungi tutto alla scena
        self.play(self.camera.frame.animate.scale(1.25))
        self.add(components, arrows)
        self.wait(5)

