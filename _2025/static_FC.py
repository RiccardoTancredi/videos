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


class FunctionalConnectivityConstruction(Scene):
    def construct(self):
        # ==================== PARAMETRI ====================
        self.N_ROIS_DISPLAY = 5
        self.TIMEPOINTS = 200
        
        np.random.seed(0)
        self.data_matrix, self.data_fake_matrix = self.generate_simulated_data()
        # self.roi_labels = [f"ROI_{i}" for i in range(N_ROIS_DISPLAY)]
        
        # Colormap 3B1B
        self.cmap_colors = [get_opposite_color(BLUE_E), get_opposite_color(WHITE), get_opposite_color(RED_E)]
        self.vmin, self.vmax = -1, 1
        
        self.setup_scene()
        self.frame.set_width(15)
        self.animate_fc_construction()
        
    def generate_simulated_data(self, TR=0.8, filepath='/Users/riccardotancredi/Documents/videos3b1b/_2025/data/data.npy'):
        """Dati BOLD simulati con pattern realistici."""
        data = np.load(filepath)
        rois = data.shape[-1]
        random_rois = np.random.choice(rois, self.N_ROIS_DISPLAY, replace=False)
        real_data = data[3, :self.TIMEPOINTS, :self.N_ROIS_DISPLAY] # (first_subj, #timepoints, #rois)
        
        """Genera dati BOLD simulati con alcune correlazioni artificiali."""
        t = np.linspace(0, 4*np.pi, self.TIMEPOINTS)
        
        # Crea pattern temporali correlati
        data = np.zeros((self.N_ROIS_DISPLAY, self.TIMEPOINTS))
        for i in range(self.N_ROIS_DISPLAY):
            # Frequenza base + armoniche
            freq = 0.5 + i * 0.1
            phase = (i if i != 3 else 0) * np.pi / 4
            
            # Segnale base
            signal = np.sin(freq * t + phase) * 0.5
            
            # Aggiungi correlazioni artificiali tra alcune ROI
            if i == 2:
                signal += 2*np.sin(freq * t + phase)
            if i == 3:
                signal += 0
            if i == 5:
                signal += 0.3 * np.cos(t)
            if i == 6:
                signal -= 0.3 * np.sin(t)  # Anti-correlata con ROI 2
                
            # Aggiungi rumore
            if i != 3:
                signal += np.random.normal(0, 0.1, self.TIMEPOINTS)
            
            data[i] = signal

        return real_data, data.T


    def setup_scene(self):
        """Layout ottimizzato: NO box bianco, grafici visibili."""
        # Titolo
        # title = Text("Costruzione della FC Matrix", font_size=36)
        # title.to_edge(UP, buff=0.3)
        # self.add(title)

        self.data_min = np.min(self.data_matrix)
        self.data_max = np.max(self.data_matrix)
        margin = (self.data_max - self.data_min) * 0.1
        
        # Range dinamico degli assi
        y_min = self.data_min - margin
        y_max = self.data_max + margin
        y_step = (y_max - y_min) / 4
        
        # === GRAFICI TEMPORALI (Sinistra) ===
        # Grafico SUPERIORE (ROI n)
        self.axes1 = Axes(
            x_range=[0, self.TIMEPOINTS+10, 50],
            y_range=[y_min, y_max, y_step],
            width=6.5,
            height=2.5,
            axis_config={
                "include_tip": True,
                "color": WHITE,
                "stroke_width": 2,
            }
        )
        self.axes1.shift(LEFT * 4 + UP * 1.8)
        
        # Grafico INFERIORE (ROI m)
        self.axes2 = self.axes1.copy()
        self.axes2.shift(DOWN * 3.5)  # Distanza verticale
        
        # LABEL ASSI (solo "Time" sotto)
        # labels1 = self.axes1.get_axis_labels(R"Time (s)", R"BOLD\enspace signal",)
        # labels2 = self.axes2.get_axis_labels(R"Time (s)", R"BOLD\enspace signal",)
        # labels1[0].align_to(self.axes1, RIGHT).shift(DOWN * 0.1)
        # labels1[1].align_to(self.axes1, RIGHT*0.5).shift(LEFT)
        # labels2[0].align_to(self.axes2, RIGHT).shift(DOWN * 0.1)
        # labels2[1].align_to(self.axes2, RIGHT*0.5).shift(LEFT)
        self.time_label1 = Tex(R"Time\enspace (s)", font_size=24).next_to(self.axes1.x_axis, 0.05*RIGHT+DOWN)
        self.time_label2 = Tex(R"Time\enspace (s)", font_size=24).next_to(self.axes2.x_axis, 0.05*RIGHT+DOWN)
        
        # LABEL ROI sull'asse Y (GRANDI!)
        self.roi_label1 = Tex(r"ROI\enspace 1", font_size=40, color=WHITE)
        self.roi_label1.next_to(self.axes1.y_axis, UP+RIGHT, buff=0.1)

        self.roi_label2 = Tex(r"ROI \enspace 1", font_size=40, color=WHITE)
        self.roi_label2.next_to(self.axes2.y_axis, UP+RIGHT, buff=0.1)
        
        # Placeholder per i grafici
        self.graph1 = VMobject()
        self.graph2 = VMobject()
        
        # === MATRICE FC (Destra) ===
        matrix_width = 6
        matrix_frame = Rectangle(
            width=matrix_width,
            height=matrix_width,
            stroke_color=WHITE,
            stroke_width=2
        )
        matrix_frame.move_to(RIGHT * 3.5 + UP * 0.5)  # Spostata leggermente in alto
        
        self.matrix_cells = VGroup()
        self.matrix_values = VGroup()
        self.n_rois = self.data_matrix.shape[1]
        
        cell_size = matrix_width / self.n_rois
        
        for i in range(self.n_rois):
            for j in range(self.n_rois):
                cell = Square(side_length=cell_size)
                cell.set_stroke(WHITE, width=0.8)
                cell.set_fill(BLACK, opacity=1)
                cell.move_to(matrix_frame.get_center() + 
                           RIGHT * (j - self.n_rois/2 + 0.5) * cell_size +
                           UP * (self.n_rois/2 - i - 0.5) * cell_size)
                self.matrix_cells.add(cell)
                
                value_text = Text("", font_size=18)
                value_text.move_to(cell)
                self.matrix_values.add(value_text)
        
        # Etichetta matrice
        # matrix_label = Text("FC Matrix", font_size=28)
        # matrix_label.next_to(matrix_frame, UP, buff=0.2)
        
        # === COLORBAR (Sotto la matrice, STESSA LARGHEZZA) ===
        colorbar_height = 0.5
        colorbar_frame = Rectangle(
            width=matrix_width,  # STESSA larghezza della matrice!
            height=colorbar_height,
            stroke_color=get_opposite_color(GREY_E),
            stroke_width=2
        )
        colorbar_frame.next_to(matrix_frame, DOWN*1.3, buff=0.4)
        
        colorbar = VGroup()
        n_steps = 150  # Alta risoluzione
        
        for i in range(n_steps):
            rect = Rectangle(
                width=matrix_width / n_steps,
                height=colorbar_height,
                stroke_width=0
            )
            value = -1 + 2 * i / (n_steps - 1)
            color = self.value_to_color(value)
            rect.set_fill(color, opacity=1)
            rect.move_to(colorbar_frame.get_left() + 
                       RIGHT * (i + 0.5) * matrix_width / n_steps)
            colorbar.add(rect)
        
        # Etichette colorbar
        labels = VGroup(
            Tex(R"-1.0", font_size=24).next_to(colorbar_frame, LEFT, buff=0.2),
            Tex(R"0.0", font_size=24).next_to(colorbar_frame, DOWN, buff=0.1),
            Tex(R"FC_{i,j}", font_size=24).next_to(colorbar_frame, UP, buff=0.1),
            Tex(R"+1.0", font_size=24).next_to(colorbar_frame, RIGHT, buff=0.2)
        )
        
        self.colorbar_group = VGroup(colorbar_frame, colorbar, labels)
        
        # Aggiungi tutto
        self.add(
            self.axes1, self.axes2,
            self.time_label1, self.time_label2,
            # labels1, labels2,
            self.roi_label1, self.roi_label2,
            matrix_frame, # matrix_label,
            self.matrix_cells, self.matrix_values,
            self.colorbar_group
        )
        
    def value_to_color(self, value):
        """Colormap 3B1B."""
        normalized = (value - self.vmin) / (self.vmax - self.vmin)
        
        if normalized < 0.5:
            t = normalized * 2
            color = interpolate_color(self.cmap_colors[0], self.cmap_colors[1], t)
        else:
            t = (normalized - 0.5) * 2
            color = interpolate_color(self.cmap_colors[1], self.cmap_colors[2], t)
        
        return color
    
    def animate_fc_construction(self):
        """Animazione velocizzata."""
        fc_matrix = self.compute_fc_matrix()
        # cells_nm, cells_mn, values = [], [], []
        # connection_liness, highlights_mn, highlights_nm = [], [], []
        # target_colors, values_str = [], []


        for n in range(self.n_rois):
            for m in range(n, self.n_rois):
                
                # Aggiorna grafici temporali
                self.update_timeseries_plots(n, m)

                # Celle target
                cell_nm = self.matrix_cells[n * self.n_rois + m]
                cell_mn = self.matrix_cells[m * self.n_rois + n]
                value = fc_matrix[n, m]
                
                # Linee connessione
                connection_lines = self.create_connection_lines(n, m, cell_nm)
                
                # Highlights
                highlight_nm = SurroundingRectangle(cell_nm, color=get_opposite_color(YELLOW_E), buff=0.02, stroke_width=4)
                highlight_mn = SurroundingRectangle(cell_mn, color=get_opposite_color(YELLOW_E), buff=0.02, stroke_width=4)
                
                # Colora e aggiorna valori
                target_color = self.value_to_color(value)
                value_str = f"{value:.2f}"
                
                self.play(
                    ShowCreation(connection_lines),
                    ShowCreation(highlight_nm),
                    ShowCreation(highlight_mn),

                    cell_nm.animate.set_fill(target_color, opacity=0.85),
                    cell_mn.animate.set_fill(target_color, opacity=0.85),
                    Transform(
                        self.matrix_values[n * self.n_rois + m],
                        Text(value_str, font_size=18).move_to(cell_nm)
                    ),
                    Transform(
                        self.matrix_values[m * self.n_rois + n],
                        Text(value_str, font_size=18).move_to(cell_mn)
                    ),

                    FadeOut(highlight_nm),
                    FadeOut(highlight_mn),
                    FadeOut(connection_lines),
                    rate_func=linear,
                    run_time=0.25
                )
                
                # self.wait(0.01)  # Pausa microscopica
        
        # Zoom finale su matrice
        self.wait(2)
        final_group = VGroup(self.matrix_cells, self.matrix_values, self.colorbar_group)
        self.play(
            *(FadeOut(rect) for rect in [
                self.axes1, self.axes2,
                self.time_label1, self.time_label2,
                self.roi_label1, self.roi_label2,
                self.graph1, self.graph2
            ]),
            self.camera.frame.animate.scale(1).move_to(
                final_group.get_center()
            ),
            run_time=2
        )
        self.wait(2)
    
    def update_timeseries_plots(self, roi_n, roi_m):
        """AGGIORNAMENTO CORRETTO con scaling dei dati reali."""
        # Rimuovi vecchi grafici
        self.remove(self.graph1, self.graph2)
        
        # Crea NUOVE linee con coordinate CONVERTITE
        points1 = self.axes1.c2p(np.arange(0, self.TIMEPOINTS), self.data_matrix[:, roi_n])
        points2 = self.axes2.c2p(np.arange(0, self.TIMEPOINTS), self.data_matrix[:, roi_m])
        
        # Crea le linee
        self.graph1 = VMobject(color=get_opposite_color(BLUE_D), stroke_width=3)
        self.graph1.set_points_smoothly(points1)
        
        self.graph2 = VMobject(color=get_opposite_color(YELLOW_D), stroke_width=3)
        self.graph2.set_points_smoothly(points2)
        self.graph2.shift(DOWN * 3.5 * 2)
        
        # Aggiorna label ROI
        new_label1 = Tex(rf"ROI \enspace {roi_n+1}", font_size=40, color=get_opposite_color(BLUE_D))
        new_label1.next_to(self.axes1.y_axis, RIGHT+UP, buff=0.1)
        
        new_label2 = Tex(rf"ROI \enspace {roi_m+1}", font_size=40, color=get_opposite_color(YELLOW_D))
        new_label2.next_to(self.axes2.y_axis, RIGHT+UP, buff=0.1)
        
        # Anima la transizione
        self.play(
            Transform(self.roi_label1, new_label1),
            Transform(self.roi_label2, new_label2),
            ShowCreation(self.graph1),
            ShowCreation(self.graph2),
            run_time=0.1
        )
    
    def create_connection_lines(self, roi_n, roi_m, target_cell):
        """Linee curve al CENTRO della cella."""
        start1 = self.axes1.get_right() + RIGHT * 0.2
        start2 = self.axes2.get_right() + RIGHT * 0.2
        end = target_cell.get_center()
        
        line1 = CubicBezier(
            start1, start1 + RIGHT * 2.5,
            end + LEFT * 2.5, end,
            color=get_opposite_color(BLUE_E), stroke_width=2.5
        )
        
        line2 = CubicBezier(
            start2, start2 + RIGHT * 2.5,
            end + LEFT * 2.5, end,
            color=get_opposite_color(YELLOW_D), stroke_width=2.5
        )
        
        return VGroup(line1, line2)
    

    def z_score_normalize(self, data, axis=0, ddof=0):
        mean = np.mean(data, axis=axis)
        std = np.std(data, axis=axis, ddof=ddof)
        return (data - mean) / std

    def fisher_r_to_z(self, r):
        r = np.clip(r, -0.999999, 0.999999)
        # In z-space, for i.i.d. samples, the distribution is closer to a normal distribution 
        # with variance ≈ 1/(L−3), r-independent: variance stabilization
        return 0.5 * np.log((1 + r) / (1 - r)) # atanh(r)


    def compute_fc_matrix(self):
        """Calcola correlazioni."""
        ts = self.z_score_normalize(self.data_fake_matrix, axis=0, ddof=0)
        
        fc_matrix = np.corrcoef(ts, rowvar=False)   # (N, N)
        # fc_matrix = self.fisher_r_to_z(fc_matrix)
        return fc_matrix
