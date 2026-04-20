import numpy as np
from scipy.stats import pearsonr
import sys
import os

lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "./../"))
sys.path.append(lib_path)

from manim_imports_ext import *


def get_opposite_color(color):
    """Restituisce il colore opposto di un dato colore in formato RGB."""
    rgb = np.array(color_to_rgb(color))  # Ottieni il colore in [0,1]
    opposite_rgb = 1 - rgb  # Complemento in [0,1]
    return rgb_to_color(opposite_rgb)  # Converti di nuovo in colore Manim


class GaussianTaperExplanation(Scene):
    def construct(self):
        # ==================== PARAMETRI ====================
        self.TIMEPOINTS = 795
        self.WINDOW_SIZE = 75
        self.STEP_SIZE = 28
        self.N_ROIS = 3  # Solo 2 ROI per chiarezza
        
        # Genera dati BOLD con transizione netta (per mostrare artefatti)
        self.raw_data = self.generate_transitional_data()
        
        self.setup_scene()
        self.frame.set_width(20)
        self.animate_taper_comparison()
        
    def generate_transitional_data(self):
        """Genera dati con una transizione netta per evidenziare artefatti."""
        t = np.linspace(0, 4*np.pi, self.TIMEPOINTS)
        data = np.zeros((self.TIMEPOINTS, self.N_ROIS))
        
        for i in range(self.N_ROIS):
            # Prima metà: pattern A
            # Seconda metà: pattern B completamente diverso
            freq = 0.5 + i * 0.2
            
            signal = np.sin(freq * t) * 0.6
            
            # Transizione netta al centro
            if i == 0:
                signal[self.TIMEPOINTS//2:] = np.cos(0.8 * t[self.TIMEPOINTS//2:]) * 0.6
            
            data[:, i] = signal + np.random.normal(0, 0.08, self.TIMEPOINTS)
        
        return data
    
    def setup_scene(self):
        """Setup della scena."""
        # Titolo
        # title = Text("Gaussian Taper vs Boxcar Window", font_size=40)
        # title.to_edge(UP, buff=0.3)
        # self.add(title)
        
        # Plot temporali (alto)
        self.time_axes = Axes(
            x_range=[0, self.TIMEPOINTS, 20],
            y_range=[-2, 2, 1],
            width=9, height=2.5,
            axis_config={"include_tip": True, "color": GREY_C}
        )
        self.time_axes.shift(UP * 2)
        
        # Label assi
        self.time_label = Tex(R"Time\enspace (s)", font_size=24).next_to(self.time_axes.x_axis, RIGHT, buff=0.2).shift(DOWN*0.7+LEFT)
        self.bold_label = Tex(R"BOLD\enspace signal", font_size=24).next_to(self.time_axes.y_axis, UP+RIGHT, buff=0.2)
        
        self.add(self.time_axes, self.time_label, self.bold_label)
        
        # Plot delle due ROI
        self.create_bold_plots()
        
        # Finestra scorrevole iniziale
        self.window_rect = Rectangle(
            width=self.WINDOW_SIZE * (9 / self.TIMEPOINTS),
            height=2.7,
            stroke_color=WHITE,
            stroke_width=3,
            fill_color=GREY_E,
            fill_opacity=0.3
        )
        self.window_rect.move_to(self.time_axes.get_left() + 
                                RIGHT * self.WINDOW_SIZE/2 * (9 / self.TIMEPOINTS))
        self.add(self.window_rect)
        
        # Pannello pesi (centro)
        self.weight_axes = Axes(
            x_range=[-self.WINDOW_SIZE/2, self.WINDOW_SIZE/2, 10],
            y_range=[0, 0.11, 0.4],
            width=6, height=3,
            axis_config={"include_tip": True, "color": GREY_C}
        )
        self.weight_axes.shift(DOWN * 3 + LEFT * 4.5)
        
        self.weight_label = Tex(R"Window\enspace Weights", font_size=28).next_to(self.weight_axes, UP)
        self.add(self.weight_axes, self.weight_label)
        
        # Confronto FC (basso)
        self.fc_axes = Axes(
            x_range=[0, (self.TIMEPOINTS - self.WINDOW_SIZE) // self.STEP_SIZE, 2],
            y_range=[-1, 0.7, 0.2],
            width=9, height=2.5,
            axis_config={"include_tip": True, "color": GREY_C}
        )
        self.fc_axes.shift(DOWN * 3 + RIGHT * 4.5)
        
        self.fc_label = Tex(R"Dynamic\enspace FC", font_size=28).next_to(self.fc_axes, UP)
        self.fc_xlabel = Tex(R"Time\enspace (s)", font_size=28).next_to(self.fc_axes, RIGHT).shift(DOWN*0.5+LEFT)
        self.fc_ylabel = Tex(R"\rho", font_size=28).next_to(self.fc_axes, LEFT)
        self.add(self.fc_axes, self.fc_label, self.fc_xlabel, self.fc_ylabel)
        
        # Legenda
        legend = VGroup(
            Line(color=get_opposite_color(RED)).scale(0.5).next_to(Tex(R"Boxcar", font_size=24), LEFT),
            Line(color=get_opposite_color(GREEN)).scale(0.5).next_to(Tex(r"Gaussian", font_size=24), LEFT)
        )
        legend.arrange(RIGHT, buff=1)
        legend.next_to(self.fc_axes, DOWN, buff=0.3)
        self.add(legend)
        
    def create_bold_plots(self):
        """Crea i plot BOLD delle 2 ROI."""
        colors = [get_opposite_color(BLUE_D), get_opposite_color(YELLOW_D)]
        
        for roi_idx in range(2):  # Solo 2 ROI per chiarezza
            signal = self.raw_data[:, roi_idx]
            
            points = []
            for i, val in enumerate(signal):
                point = self.time_axes.c2p(i, val + roi_idx * 1.5)  # Spaziati verticalmente
                if np.all(np.isfinite(point)):
                    points.append(point)
            
            line = VMobject(color=colors[roi_idx], stroke_width=3)
            line.set_points_smoothly(points)
            self.add(line)
            
            # Etichetta ROI
            label = Tex(Rf"ROI\enspace {roi_idx+1}", font_size=20, color=colors[roi_idx])
            label.next_to(self.time_axes, LEFT, buff=0.1).shift(UP * roi_idx)
            self.add(label)
    
    def animate_taper_comparison(self):
        """Animazione principale del confronto."""
        # Pre-calcola le curve dFC per entrambi i metodi
        boxcar_fc, gaussian_fc = self.compute_both_fcs()
        
        # LINEE FINALI (placeholder)
        self.boxcar_line = VMobject(color=get_opposite_color(RED), stroke_width=3)
        self.gaussian_line = VMobject(color=get_opposite_color(GREEN), stroke_width=3)
        self.add(self.boxcar_line, self.gaussian_line)
        
        # PUNTI ILLUMINATI
        self.boxcar_dot = Dot(color=get_opposite_color(RED), radius=0.08)
        self.gaussian_dot = Dot(color=get_opposite_color(GREEN), radius=0.08)
        self.add(self.boxcar_dot, self.gaussian_dot)
        
        # PESI (placeholder)
        self.boxcar_weight_graph = VMobject(color=get_opposite_color(RED), stroke_width=4)
        self.gaussian_weight_graph = VMobject(color=get_opposite_color(GREEN), stroke_width=4)
        self.add(self.boxcar_weight_graph, self.gaussian_weight_graph)
        
        # Slide la finestra e aggiorna tutto
        n_windows = len(boxcar_fc)
        
        for w_idx in range(n_windows):
            # 1. Sposta finestra
            window_x = w_idx * self.STEP_SIZE + self.WINDOW_SIZE/2
            window_pos = self.time_axes.c2p(window_x, 0)
            
            self.play(
                self.window_rect.animate.move_to(window_pos),
                run_time=0.15
            )
            
            # 2. Mostra pesi nella finestra
            self.update_weights_visual(w_idx)
            
            # 3. Illumina punto corrente sulle curve dFC
            self.update_fc_points(w_idx, boxcar_fc, gaussian_fc)
            
            # 4. Aggiungi segmenti alle linee
            if w_idx > 0:
                self.update_fc_lines(w_idx, boxcar_fc, gaussian_fc)
            
            self.wait(0.05)
        
        # Messaggio finale
        # conclusion = Text(
        #     "Gaussian taper = meno artefatti, più smooth!",
        #     font_size=28, color=GREEN
        # )
        # conclusion.to_edge(DOWN, buff=0.5)
        # self.play(Write(conclusion, run_time=2))
        self.wait(3)
    
    def compute_both_fcs(self):
        """Calcola FC con entrambi i metodi."""
        n_windows = (self.TIMEPOINTS - self.WINDOW_SIZE) // self.STEP_SIZE + 1
        boxcar_fc = np.zeros(n_windows)
        gaussian_fc = np.zeros(n_windows)
        
        for w in range(n_windows):
            start = w * self.STEP_SIZE
            window_data = self.raw_data[start:start + self.WINDOW_SIZE, :2]  # Solo prime 2 ROI
            
            # Boxcar: correlazione semplice
            corr, _ = pearsonr(window_data[:, 0], window_data[:, 1])
            boxcar_fc[w] = corr
            
            # Gaussian: correlazione pesata
            weights = self.gaussian_weights()
            weighted_corr = self.weighted_pearsonr(window_data[:, 0], window_data[:, 1], weights)
            gaussian_fc[w] = weighted_corr
        
        return boxcar_fc, gaussian_fc
    
    def gaussian_weights(self):
        """Genera pesi gaussiani per la finestra."""
        x = np.arange(self.WINDOW_SIZE) - self.WINDOW_SIZE/2
        sigma = self.WINDOW_SIZE / 6  # 3 sigma ≈ finestra
        weights = np.exp(-0.5 * (x / sigma)**2)
        return weights / np.sum(weights)  # Normalizza
    
    def weighted_pearsonr(self, x, y, weights):
        """Calcola correlazione di Pearson pesata."""
        # Centra i dati usando media pesata
        x_centered = x - np.average(x, weights=weights)
        y_centered = y - np.average(y, weights=weights)
        
        # Covarianza e varianze pesate
        cov = np.sum(weights * x_centered * y_centered)
        var_x = np.sum(weights * x_centered**2)
        var_y = np.sum(weights * y_centered**2)
        
        return cov / np.sqrt(var_x * var_y)
    
    def update_weights_visual(self, w_idx):
        """Aggiorna la visualizzazione dei pesi nella finestra."""
        # Boxcar: linea piatta
        start = w_idx * self.STEP_SIZE
        boxcar_points = []
        
        for i in range(self.WINDOW_SIZE):
            x = self.weight_axes.c2p(i - self.WINDOW_SIZE/2, 0.1)
            boxcar_points.append(x)
        
        self.boxcar_weight_graph.set_points_smoothly(boxcar_points)
        
        # Gaussian: curva a campana
        gaussian_points = []
        weights = self.gaussian_weights()
        
        for i, w in enumerate(weights):
            x = self.weight_axes.c2p(i - self.WINDOW_SIZE/2, w * 1.2)  # Scala per visibilità
            gaussian_points.append(x)
        
        self.gaussian_weight_graph.set_points_smoothly(gaussian_points)
        
        # Etichetta metodo
        boxcar_label = Tex(R"Boxcar", font_size=24, color=get_opposite_color(RED)).next_to(self.boxcar_weight_graph, LEFT+UP, buff=0.2)
        gauss_label = Tex(R"Gaussian", font_size=24, color=get_opposite_color(GREEN)).next_to(self.gaussian_weight_graph, LEFT+UP, buff=0.2)
        
        if w_idx == 0:
            self.add(boxcar_label, gauss_label)
    
    def update_fc_points(self, w_idx, boxcar_fc, gaussian_fc):
        """Illumina i punti correnti sulle curve dFC."""
        # Punti nel grafico finale
        boxcar_point = self.fc_axes.c2p(w_idx, boxcar_fc[w_idx])
        gaussian_point = self.fc_axes.c2p(w_idx, gaussian_fc[w_idx])
        
        # Animazione "flash" su entrambi i punti
        self.play(
            self.boxcar_dot.animate.move_to(boxcar_point).set_color(get_opposite_color(YELLOW)),
            self.gaussian_dot.animate.move_to(gaussian_point).set_color(get_opposite_color(YELLOW)),
            run_time=0.1
        )
        
        # Torna al colore normale
        self.play(
            self.boxcar_dot.animate.set_color(get_opposite_color(RED)),
            self.gaussian_dot.animate.set_color(get_opposite_color(GREEN)),
            run_time=0.1
        )
    
    def update_fc_lines(self, w_idx, boxcar_fc, gaussian_fc):
        """Aggiunge segmenti alle linee dFC."""
        # Segmente boxcar
        p1 = self.fc_axes.c2p(w_idx - 1, boxcar_fc[w_idx - 1])
        p2 = self.fc_axes.c2p(w_idx, boxcar_fc[w_idx])
        
        if np.all(np.isfinite([p1, p2])):
            segment = Line(p1, p2, color=get_opposite_color(RED), stroke_width=3)
            self.boxcar_line.add(segment)
        
        # Segmento gaussian
        p1 = self.fc_axes.c2p(w_idx - 1, gaussian_fc[w_idx - 1])
        p2 = self.fc_axes.c2p(w_idx, gaussian_fc[w_idx])
        
        if np.all(np.isfinite([p1, p2])):
            segment = Line(p1, p2, color=get_opposite_color(GREEN), stroke_width=3)
            self.gaussian_line.add(segment)
