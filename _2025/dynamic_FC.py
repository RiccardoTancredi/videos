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


class DynamicFunctionalConnectivity(Scene):
    def construct(self):
        # ==================== PARAMETRI ====================
        self.N_ROIS_DISPLAY = 5
        self.TIMEPOINTS = 795
        self.WINDOW_SIZE = 75
        self.STEP_SIZE = 28
        
        # Carica dati reali: (W, N, N)
        # self.dfc_windows, self.raw_data = self.load_dfc_data()
        self.dfc_windows, self.raw_data = self.simulate_dfc_data()
        self.previous_matrices = []  # ← Accumula matrici vecchie per fadeout
        self.vmin, self.vmax = -1., 1.

        self.setup_scene()
        self.frame.set_width(20)
        self.animate_dfc_construction()
        
    def load_dfc_data(self, folder='/Users/riccardotancredi/Documents/videos3b1b/_2025/data/'):
        raw_data = np.load(f'{folder}data.npy')
        dfc_windows = np.load(f'{folder}dFC_data.npy')
        return dfc_windows[3, :, :, :], raw_data[3, :, :]
        
    def simulate_dfc_data(self):
        """Simula dFC realistica."""
        t = np.linspace(0, 4*np.pi, self.TIMEPOINTS)
        raw_data = np.zeros((self.TIMEPOINTS, self.N_ROIS_DISPLAY))
        
        for i in range(self.N_ROIS_DISPLAY):
            freq = 0.5 + i * 0.1
            signal = np.sin(freq * t) * 0.6 + np.random.normal(0, 0.1, self.TIMEPOINTS)
            raw_data[:, i] = signal
        
        n_windows = (self.TIMEPOINTS - self.WINDOW_SIZE) // self.STEP_SIZE + 1
        dfc_windows = np.zeros((n_windows, self.N_ROIS_DISPLAY, self.N_ROIS_DISPLAY))
        
        for w in range(n_windows):
            start = w * self.STEP_SIZE
            window = raw_data[start:start + self.WINDOW_SIZE, :]
            
            for i in range(self.N_ROIS_DISPLAY):
                for j in range(self.N_ROIS_DISPLAY):
                    corr, _ = pearsonr(window[:, i], window[:, j])
                    dfc_windows[w, i, j] = corr
        
        return dfc_windows, raw_data
    
    def setup_scene(self):
        """Layout iniziale."""
        # Sinistra: Plot temporali
        self.time_axes = Axes(
            x_range=[0, self.TIMEPOINTS+50, 20],
            y_range=[-2, 2, 1],
            width=8, height=3,
            axis_config={"include_tip": True, "color": WHITE}
        )
        self.time_axes.shift(LEFT * 4.5 + UP * 3.5)
        
        # Label assi
        self.time_label = Tex(R"Time\enspace (s)", font_size=28).next_to(self.time_axes.x_axis, RIGHT+DOWN, buff=0.2)
        self.bold_label = Tex(R"BOLD\enspace signal", font_size=28).next_to(self.time_axes.y_axis, UP, buff=0.2)
        
        # Plot temporali
        self.create_timeseries_plots()
        
        # Sliding window
        self.window_rect = Rectangle(
            width=self.WINDOW_SIZE * (8 / self.TIMEPOINTS), height=3.2,
            stroke_color=WHITE, stroke_width=3,
            fill_color=GREY_E, fill_opacity=0.3
        )
        self.window_rect.move_to(self.time_axes.get_left() + 
                                RIGHT * self.WINDOW_SIZE/2 * (8 / self.TIMEPOINTS))
        
        # Destra: Plot dFC
        self.dfc_axes = Axes(
            x_range=[0, len(self.dfc_windows), 2],
            y_range=[-1, 1, 0.5],
            width=8, height=5,
            axis_config={"include_tip": True, "color": WHITE}
        )
        self.dfc_axes.shift(RIGHT * 5.5 + UP * 0.5)
        
        self.dfc_label = Tex(R"Dynamic\enspace FC", font_size=32).next_to(self.dfc_axes, UP)
        self.dfc_time_label = Tex(R"Time\enspace (s)", font_size=28).next_to(self.dfc_axes.x_axis, DOWN, buff=0.2)
        self.dfc_y_label = Tex(R"\overline\rho", font_size=28).next_to(self.dfc_axes.y_axis, LEFT, buff=0.2).shift(UP+LEFT*0.2)

        # self.dfc_line = VMobject(color=get_opposite_color(BLUE_E), stroke_width=3)
        self.dfc_points = []
        
        # 🎯 TRACCIA CONTINUA (inizialmente vuota)
        self.dfc_trail = VMobject(color=get_opposite_color(BLUE_E), stroke_width=3)
        
        # 🎯 GRUPPO PUNTI PERMANENTI
        self.dfc_dots = VGroup()
        
        # 🎯 PUNTO ILLUMINATO TEMPORANEO
        self.dfc_highlight_dot = Dot(color=get_opposite_color(YELLOW), radius=0.08)
        self.dfc_highlight_dot.set_opacity(0)  # Invisibile all'inizio
        
        # self.add(
        #     self.time_axes, self.time_label, self.bold_label,
        #     self.window_rect,
        #     self.dfc_axes, self.dfc_label, self.dfc_time_label,
        #     self.dfc_trail, self.dfc_dots, self.dfc_highlight_dot,
        #     self.current_matrix
        # )

        # Matrice corrente (unica che si sposta)
        self.current_matrix = self.create_matrix_group()
        self.current_matrix.next_to(self.window_rect, DOWN, buff=0.8)
        
        self.add(
            self.time_axes, self.time_label, self.bold_label,
            self.window_rect,
            self.dfc_axes, self.dfc_label, self.dfc_time_label, self.dfc_y_label,
            self.current_matrix
        )
    
    def create_timeseries_plots(self):
        """Crea i plot temporali."""
        t = np.arange(self.TIMEPOINTS)
        
        for roi_idx in range(self.N_ROIS_DISPLAY):
            signal = self.raw_data[:, roi_idx]
            
            # points = []
            points = self.time_axes.c2p(np.arange(0, self.TIMEPOINTS), signal)
            # for i, val in enumerate(signal):
            #     # PER DATI REALI: self.time_axes.c2p(i, val)
            #     point = self.time_axes.c2p(i, val)
                # if np.all(np.isfinite(point)):
                #     points.append(point)
            
            colors = [get_opposite_color(col) for col in [BLUE_D, GREEN_D, RED_D, YELLOW_D, PURPLE_D]]
            line = VMobject(color=colors[roi_idx], stroke_width=2, stroke_opacity=0.7)
            line.set_points_smoothly(points)
            self.add(line)
    
    def create_matrix_group(self):
        """Crea matrice con colorbar e titolo integrati."""
        width = 3.2
        
        frame = Rectangle(
            width=width, height=width,
            stroke_color=WHITE, stroke_width=2
        )
        
        cells = VGroup()
        values = VGroup()
        cell_size = width / self.N_ROIS_DISPLAY
        
        for i in range(self.N_ROIS_DISPLAY):
            for j in range(self.N_ROIS_DISPLAY):
                cell = Square(side_length=cell_size)
                cell.set_stroke(WHITE, width=0.8)
                cell.set_fill(BLACK, opacity=1)
                cell.move_to(frame.get_center() + 
                           RIGHT * (j - self.N_ROIS_DISPLAY/2 + 0.5) * cell_size +
                           UP * (self.N_ROIS_DISPLAY/2 - i - 0.5) * cell_size)
                cells.add(cell)
                
                value_text = Tex("", font_size=16)
                value_text.move_to(cell)
                values.add(value_text)
        
        # Colorbar sotto
        colorbar_height = 0.4
        colorbar_frame = Rectangle(
            width=width, height=colorbar_height,
            stroke_color=WHITE, stroke_width=1
        )
        colorbar_frame.next_to(frame, DOWN, buff=0.15)
        
        colorbar = VGroup()
        n_steps = 30
        
        for i in range(n_steps):
            rect = Rectangle(
                width=width / n_steps, height=colorbar_height,
                stroke_width=0
            )
            value = -1 + 2 * i / (n_steps - 1)
            color = interpolate_color(BLUE_E, RED_E, (value + 1) / 2)
            rect.set_fill(color, opacity=1)
            rect.move_to(colorbar_frame.get_left() + 
                       RIGHT * (i + 0.5) * width / n_steps)
            colorbar.add(rect)
        
        labels = VGroup(
            Tex(R"-1.0", font_size=18).next_to(colorbar_frame, LEFT, buff=0.1),
            Tex(R"+1.0", font_size=18).next_to(colorbar_frame, RIGHT, buff=0.1),
            Tex(R"\rho", font_size=18).next_to(colorbar_frame, DOWN, buff=0.1)
        )
        
        title = Tex("FC_{i,j}^{w=0}", font_size=24)
        title.next_to(frame, UP)
        
        return VGroup(frame, cells, values, colorbar_frame, colorbar, labels, title)
    
    def animate_dfc_construction(self):
        """Animazione: matrici opache accumulate, nuova sempre sopra."""
        for w_idx in range(len(self.dfc_windows)):
            # ✅ 1. SPOSTA finestra temporale
            window_x = w_idx * self.STEP_SIZE + self.WINDOW_SIZE/2
            window_pos = self.time_axes.c2p(window_x, 0)
            
            # ✅ 2. SPOSTA la matrice corrente con la finestra
            target_matrix_pos = window_pos + DOWN * 4.5
            
            self.play(
                self.window_rect.animate.move_to(window_pos),
                self.current_matrix.animate.move_to(target_matrix_pos),
                run_time=0.2
            )
            
            # ✅ 3. CREA copia opaca della matrice corrente PRIMA di aggiornarla
            if w_idx > 0:
                opaque_copy = self.current_matrix.copy()
                opaque_copy.set_opacity(0.05)  # ← Oppure 0.05 per meno visibilità
                
                # Rimuovi titolo e valori dalla copia opaca (per pulizia)
                opaque_copy[2].set_opacity(0)   # Valori numerici invisibili
                opaque_copy[-1].set_opacity(0.1)  # Titolo semi-trasparente
                opaque_copy[3].set_opacity(0.03)    # Colorbar invisibile
                opaque_copy[4].set_opacity(0.03)    # Colorbar invisibile

                 
                self.add(opaque_copy)
                self.previous_matrices.append(opaque_copy)
            
            # ✅ 4. AGGIORNA la matrice corrente con nuovi valori
            # Il titolo e la colorbar si aggiornano INSIEME alla matrice
            self.update_matrix(self.dfc_windows[w_idx], w_idx)
            
            # ✅ 5. PORTA la matrice corrente in PRIMO PIANO (sopra le opache)
            self.bring_to_front(self.current_matrix)
            
            # ✅ 6. CONNESSIONI
            self.create_window_connections(window_pos, self.current_matrix)
            
            # ✅ 7. PLOT dFC
            self.update_dfc_plot(w_idx)
            # fc_mean = np.mean(self.dfc_windows[w_idx])
            # new_point = self.dfc_axes.c2p(w_idx, fc_mean)
            # self.dfc_points.append(new_point)
            
            # new_line = VMobject(color=get_opposite_color(GREEN), stroke_width=3)
            # if len(self.dfc_points) > 1:
            #     new_line.set_points_smoothly(self.dfc_points)
            # else:
            #     new_line.set_points_smoothly([self.dfc_points[0], self.dfc_points[0]])
            
            # self.play(
            #     self.dfc_line.animate.become(new_line),
            #     run_time=0.15
            # )
            
            self.wait(0.05)

    def update_dfc_plot(self, w_idx):
        """🎯 AGGIUNGE SOLO L'ULTIMO PUNTO ALLA CURVA E LO ILLUMINA."""
        fc_mean = np.mean(self.dfc_windows[w_idx])
        new_point = self.dfc_axes.c2p(w_idx, fc_mean)
        self.dfc_points.append(new_point)
        
        # Aggiungi il punto permanente al gruppo
        new_dot = Dot(color=get_opposite_color(BLUE_E), radius=0.06)
        new_dot.move_to(new_point)
        self.dfc_dots.add(new_dot)
        
        # Se è il primo punto, mostralo
        if w_idx == 0:
            self.play(FadeIn(new_dot, run_time=0.1))
            return
        
        # Altrimenti: Aggiungi segmento + illumina punto finale
        new_segment = Line(
            self.dfc_points[-2],  # Punto precedente
            self.dfc_points[-1],  # Nuovo punto
            color=get_opposite_color(BLUE_E),
            stroke_width=3
        )
        
        # Illumina temporaneamente il nuovo punto
        highlight_dot = Dot(color=get_opposite_color(YELLOW), radius=0.1)
        highlight_dot.move_to(new_point)
        
        self.play(
            ShowCreation(new_segment, run_time=0.1),
            Transform(highlight_dot, new_dot, run_time=0.15),  # Dal giallo illuminato al verde normale
        )
    
    
    def update_matrix(self, fc_matrix, window_idx):
        """Aggiorna valori, colori e titolo."""
        # Aggiorna titolo (è il 7° elemento del gruppo)
        new_title = Tex(f"FC_{{i,j}}^{{w={window_idx+1}}}", font_size=24)
        new_title.next_to(self.current_matrix[0], UP)
        self.current_matrix[-1].become(new_title)
        
        # Aggiorna celle
        cells = self.current_matrix[1]
        values = self.current_matrix[2]
        
        for i in range(self.N_ROIS_DISPLAY):
            for j in range(self.N_ROIS_DISPLAY):
                cell = cells[i * self.N_ROIS_DISPLAY + j]
                value_text = values[i * self.N_ROIS_DISPLAY + j]
                
                value = fc_matrix[i, j]
                normalized = (value - self.vmin) / (self.vmax - self.vmin)
                color = interpolate_color(get_opposite_color(BLUE_E), get_opposite_color(RED_E), normalized)
                cell.set_fill(color, opacity=0.85)
                
                new_value = Tex(f"{value:.2f}", font_size=16)
                new_value.move_to(cell)
                value_text.become(new_value)
    
    def create_window_connections(self, window_pos, target_matrix):
        """Linee di connessione."""
        if hasattr(self, 'connection_lines'):
            self.remove(self.connection_lines)
        
        start_left = window_pos + LEFT * self.WINDOW_SIZE/2 * (8 / self.TIMEPOINTS) + DOWN * 1.6
        start_right = window_pos + RIGHT * self.WINDOW_SIZE/2 * (8 / self.TIMEPOINTS) + DOWN * 1.6
        
        target_left = target_matrix[0].get_corner(UL)
        target_right = target_matrix[0].get_corner(UR)
        
        line_left = CubicBezier(
            start_left, start_left + DOWN * 1.5 + RIGHT * 2,
            target_left + UP * 1.5 + LEFT * 2, target_left,
            color=get_opposite_color(BLUE_E), stroke_width=2.5
        )
        
        line_right = CubicBezier(
            start_right, start_right + DOWN * 1.5 + RIGHT * 2,
            target_right + UP * 1.5 + LEFT * 2, target_right,
            color=get_opposite_color(BLUE_E), stroke_width=2.5
        )
        
        self.connection_lines = VGroup(line_left, line_right)
        self.add(self.connection_lines)
        self.play(FadeOut(self.connection_lines, run_time=0.1))