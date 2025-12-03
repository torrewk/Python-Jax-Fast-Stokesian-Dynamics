import sys
import os
import subprocess
import threading
import tempfile
from vispy import scene
from vispy.scene import visuals
from vispy.visuals.transforms import STTransform
from pathlib import Path
import toml
import numpy as np
from PySide6.QtCore import Qt, Signal, Slot, QTimer  # <-- add QTimer here
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QTextEdit,
    QMessageBox,
    QTabWidget,
    QFormLayout,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QCheckBox,
    QGroupBox,
    QRadioButton,
    QButtonGroup,
    QScrollArea,
    QSlider,          
)
from PySide6.QtGui import QTextCursor




class JFSDGui(QWidget):
    log_message = Signal(str)
    run_finished = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("JFSD – Fast Stokesian Dynamics")
        self.setMinimumSize(900, 550)

	        # --- Visualization state ---
        self.trajectory = None   # np.ndarray (T, N, 3)
        self.num_frames = 0
        self.current_frame = 0

        # camera parameters
        self.azim = 45.0
        self.elev = 30.0
        self.zoom = 1.0

        # bounding box (for zoom)
        self.xmin_base = self.xmax_base = 0.0
        self.ymin_base = self.ymax_base = 0.0
        self.zmin_base = self.zmax_base = 0.0

        # VisPy objects (created in _build_visualize_tab)
        self.canvas_vis = None
        self.view = None
        self.marker = None
        self.spheres = []     # list of Sphere visuals
        self.max_spheres = 1000  # max number of spheres to draw (for speed)


        # play timer
        self.playing = False
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._on_timer_tick)

        # track temporary configs
        self._temp_cfg_files = set()

        # build the UI (creates tabs, including visualize tab)
        self._build_ui()

        # connect signals
        self.log_message.connect(self._append_log)
        self.run_finished.connect(self._on_run_finished)

        self._current_thread = None

        
    def _on_load_trajectory_clicked(self):
        traj_path_str = self.traj_edit.text().strip()
        if not traj_path_str:
            QMessageBox.warning(self, "No file selected", "Please select a trajectory .npy file.")
            return

        traj_file = Path(traj_path_str)
        if not traj_file.is_file():
            QMessageBox.warning(
                self,
                "File not found",
                f"Could not find:\n{traj_path_str}",
            )
            return

        try:
            traj = np.load(traj_file)
        except Exception as e:
            QMessageBox.critical(self, "Error loading trajectory", f"Could not load file:\n{e}")
            return

        self.log_message.emit(f"Loaded trajectory from {traj_file}, shape = {traj.shape}")

        if traj.ndim != 3 or traj.shape[2] != 3:
            QMessageBox.critical(
                self,
                "Invalid trajectory array",
                f"Expected array of shape (T, N, 3), got {traj.shape}",
            )
            return

        self.trajectory = traj
        self.num_frames, num_particles, _ = traj.shape
        self.current_frame = 0
        
        # setup slider range
        if self.num_frames > 0:
            self.frame_slider.setEnabled(True)
            self.frame_slider.setMinimum(0)
            self.frame_slider.setMaximum(self.num_frames - 1)
            self.frame_slider.setValue(0)
        else:
            self.frame_slider.setEnabled(False)
            self.frame_slider.setMinimum(0)
            self.frame_slider.setMaximum(0)
            self.frame_slider.setValue(0)

        # update label
        self._update_frame_label()


        # base extents for zoom and camera distance
        all_coords = traj.reshape(-1, 3)
        self.xmin_base = float(all_coords[:, 0].min())
        self.xmax_base = float(all_coords[:, 0].max())
        self.ymin_base = float(all_coords[:, 1].min())
        self.ymax_base = float(all_coords[:, 1].max())
        self.zmin_base = float(all_coords[:, 2].min())
        self.zmax_base = float(all_coords[:, 2].max())

        self.zoom = 1.0
        self.azim = 45.0
        self.elev = 30.0

        self._update_plot()


    def _draw_sphere(self, x0, y0, z0, R=1.0, res=10):
        """Draw one sphere at (x0,y0,z0) with radius R using a low-res mesh."""
        import numpy as _np

        u = _np.linspace(0, 2 * _np.pi, res)
        v = _np.linspace(0, _np.pi, res)

        xs = x0 + R * _np.outer(_np.cos(u), _np.sin(v))
        ys = y0 + R * _np.outer(_np.sin(u), _np.sin(v))
        zs = z0 + R * _np.outer(_np.ones_like(u), _np.cos(v))

        self.ax3d.plot_surface(xs, ys, zs, color="blue", linewidth=0, shade=True, alpha=0.8)

    def _on_slider_changed(self, value: int):
        """User moved the slider: jump to that frame."""
        if self.trajectory is None or self.num_frames == 0:
            return
        # clamp just in case
        value = max(0, min(value, self.num_frames - 1))
        self.current_frame = value
        self._update_plot()

    def _update_frame_label(self):
        if self.num_frames <= 1:
            self.frame_label.setText("Frame: 0 / 0 (0.00)")
            return
        frac = self.current_frame / (self.num_frames - 1)
        self.frame_label.setText(
            f"Frame: {self.current_frame} / {self.num_frames - 1} ({frac:.2f})"
        )

    def _update_plot(self):
        # basic checks
        if self.trajectory is None:
            self.log_message.emit("DEBUG: _update_plot: self.trajectory is None")
            return
        if self.view is None:
            self.log_message.emit("DEBUG: _update_plot: no VisPy view yet")
            return
        if self.num_frames == 0:
            self.log_message.emit("DEBUG: _update_plot: num_frames == 0")
            return

        # clamp frame
        self.current_frame = max(0, min(self.current_frame, self.num_frames - 1))

        coords = self.trajectory[self.current_frame]  # (N, 3)
        if coords.ndim != 2 or coords.shape[1] != 3:
            self.log_message.emit(f"DEBUG: coords has unexpected shape {coords.shape}")
            return

        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        N = len(x)

        # DEBUG info
        self.log_message.emit(
            f"DEBUG: frame {self.current_frame}/{self.num_frames-1}, "
            f"x ∈ [{x.min():.3g}, {x.max():.3g}], "
            f"y ∈ [{y.min():.3g}, {y.max():.3g}], "
            f"z ∈ [{z.min():.3g}, {z.max():.3g}], N = {N}"
        )

        # how many spheres to actually draw
        n_draw = min(N, self.max_spheres)
        if N > self.max_spheres:
            self.log_message.emit(
                f"DEBUG: drawing only first {n_draw} spheres out of {N} for performance."
            )

        # lazily create Sphere visuals if needed
        from vispy.scene import visuals as _visuals

        # create more spheres if we don't have enough
        while len(self.spheres) < n_draw:
            sph = _visuals.Sphere(
                radius=1.0,          # REAL radius in world units
                rows=6, cols=6,    # lower = faster
                method="latitude",
                parent=self.view.scene,
                shading="smooth",
                color=(0.2, 0.4, 1.0, 1.0),
            )
            sph.transform = STTransform()  # we'll set translate each frame
            self.spheres.append(sph)

        # show only the first n_draw spheres, hide the rest (if any)
        for i, sph in enumerate(self.spheres):
            if i < n_draw:
                sph.visible = True
                xi, yi, zi = coords[i]
                # translate to particle center; radius is already 1.0
                sph.transform.translate = (xi, yi, zi)
            else:
                sph.visible = False

        # update camera center & distance based on zoom
        Lx = max(self.xmax_base - self.xmin_base, 1e-9)
        Ly = max(self.ymax_base - self.ymin_base, 1e-9)
        Lz = max(self.zmax_base - self.zmin_base, 1e-9)
        Lmax = max(Lx, Ly, Lz)

        cx = 0.5 * (self.xmin_base + self.xmax_base)
        cy = 0.5 * (self.ymin_base + self.ymax_base)
        cz = 0.5 * (self.zmin_base + self.zmax_base)
        self.view.camera.center = (cx, cy, cz)

        box_radius = 0.5 * Lmax
        dist = max(box_radius * 3.0 / self.zoom, 1e-3)
        self.view.camera.distance = dist

        self.view.camera.azimuth = self.azim
        self.view.camera.elevation = self.elev

        # keep slider & label in sync
        if self.frame_slider is not None and self.num_frames > 0:
            # avoid triggering slider handler recursively
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(self.current_frame)
            self.frame_slider.blockSignals(False)
            self._update_frame_label()

        if self.canvas_vis is not None:
            self.canvas_vis.update()






    def _update_axes_limits(self):
        # If we haven't loaded anything, nothing to do
        if self.trajectory is None:
            return

        # Base ranges
        cx = (self.xmin_base + self.xmax_base) * 0.5
        cy = (self.ymin_base + self.ymax_base) * 0.5
        cz = (self.zmin_base + self.zmax_base) * 0.5

        rx = (self.xmax_base - self.xmin_base) * 0.5 * self.zoom
        ry = (self.ymax_base - self.ymin_base) * 0.5 * self.zoom
        rz = (self.zmax_base - self.zmin_base) * 0.5 * self.zoom

        r = max(rx, ry, rz, 1e-9)  # avoid zero

        self.ax3d.set_xlim(cx - r, cx + r)
        self.ax3d.set_ylim(cy - r, cy + r)
        self.ax3d.set_zlim(cz - r, cz + r)

    def _rotate_view(self, d_azim=0.0, d_elev=0.0):
        self.azim = (self.azim + d_azim) % 360.0
        self.elev = max(-89.0, min(89.0, self.elev + d_elev))
        self._update_plot()

    def _zoom(self, factor: float):
        # factor < 1 → zoom in, factor > 1 → zoom out
        self.zoom *= factor
        self.zoom = max(0.1, min(self.zoom, 10.0))
        self._update_plot()


    def _on_prev_frame(self):
        if self.trajectory is None:
            return
        self.current_frame -= 1
        if self.current_frame < 0:
            self.current_frame = self.num_frames - 1  # loop
        self._update_plot()

    def _on_next_frame(self):
        if self.trajectory is None:
            return
        self.current_frame += 1
        if self.current_frame >= self.num_frames:
            self.current_frame = 0  # loop
        self._update_plot()

    def _on_play_pause(self):
        if self.trajectory is None:
            return
        if not self.playing:
            # start playing
            self.playing = True
            self.play_btn.setText("⏸ Pause")
            self.play_timer.start(100)  # ms per frame
        else:
            # pause
            self.playing = False
            self.play_btn.setText("▶ Play")
            self.play_timer.stop()

    def _on_timer_tick(self):
        # Called by QTimer while playing
        self._on_next_frame()


    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        main_layout = QVBoxLayout(self)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self.run_tab = QWidget()
        self._build_run_tab()
        self.tabs.addTab(self.run_tab, "Run simulation")

        self.config_tab = QWidget()
        self._build_config_tab()
        self.tabs.addTab(self.config_tab, "Create config")
        
        # NEW: Visualize tab
        self.visual_tab = QWidget()
        self._build_visualize_tab()
        self.tabs.addTab(self.visual_tab, "Visualize")

    def _build_run_tab(self):
        layout = QVBoxLayout(self.run_tab)

        # Config file row
        cfg_row = QHBoxLayout()
        cfg_label = QLabel("Config (.toml):")
        self.cfg_edit = QLineEdit()
        self.cfg_edit.setPlaceholderText("Path to configuration .toml")
        cfg_browse = QPushButton("Browse…")
        cfg_browse.clicked.connect(self._browse_cfg)
        cfg_row.addWidget(cfg_label)
        cfg_row.addWidget(self.cfg_edit)
        cfg_row.addWidget(cfg_browse)
        layout.addLayout(cfg_row)

        # Positions file row (optional)
        pos_row = QHBoxLayout()
        pos_label = QLabel("Initial positions (.npy, optional):")
        self.pos_edit = QLineEdit()
        self.pos_edit.setPlaceholderText("Path to positions .npy (optional)")
        pos_browse = QPushButton("Browse…")
        pos_browse.clicked.connect(self._browse_pos)
        pos_row.addWidget(pos_label)
        pos_row.addWidget(self.pos_edit)
        pos_row.addWidget(pos_browse)
        layout.addLayout(pos_row)

        # Output directory row
        out_row = QHBoxLayout()
        out_label = QLabel("Output directory:")
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("Folder where trajectory data will be stored")
        out_browse = QPushButton("Browse…")
        out_browse.clicked.connect(self._browse_out)
        out_row.addWidget(out_label)
        out_row.addWidget(self.out_edit)
        out_row.addWidget(out_browse)
        layout.addLayout(out_row)

        # Option: delete temporary config after run
        opts_row = QHBoxLayout()
        opts_row.addStretch(1)
        self.delete_temp_after_run_check = QCheckBox("Delete temporary config after run")
        self.delete_temp_after_run_check.setChecked(True)
        opts_row.addWidget(self.delete_temp_after_run_check)
        layout.addLayout(opts_row)

        # Run button
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self.run_button = QPushButton("Run simulation")
        self.run_button.clicked.connect(self._on_run_clicked)
        btn_row.addWidget(self.run_button)
        layout.addLayout(btn_row)

        # Log output
        log_label = QLabel("Log:")
        layout.addWidget(log_label)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setLineWrapMode(QTextEdit.NoWrap)
        layout.addWidget(self.log, stretch=1)

    def _build_config_tab(self):
        # ---- Outer layout (to host the scroll area) ----
        outer_layout = QVBoxLayout(self.config_tab)

        # ---- Scroll area ----
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)      # important
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        outer_layout.addWidget(scroll)

        # ---- Inside scroll area, we put a single container widget ----
        container = QWidget()
        scroll.setWidget(container)

        # Layout inside container
        layout = QVBoxLayout(container)

        # Form layout containing all parameters
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignTop)

        # ---------------- [general] ----------------
        self.n_steps_spin = QSpinBox()
        self.n_steps_spin.setRange(1, 10**9)
        self.n_steps_spin.setValue(1000)
        form.addRow("[general] n_steps:", self.n_steps_spin)

        self.n_particles_spin = QSpinBox()
        self.n_particles_spin.setRange(0, 10**9)
        self.n_particles_spin.setValue(1000)
        form.addRow("n_particles:", self.n_particles_spin)

        self.dt_spin = QDoubleSpinBox()
        self.dt_spin.setRange(1e-12, 1e3)
        self.dt_spin.setDecimals(8)
        self.dt_spin.setValue(0.005)
        form.addRow("dt:", self.dt_spin)

        # ---------------- [initialization] ----------------
        self.position_source_combo = QComboBox()
        self.position_source_combo.addItems(["random_hardsphere", "file"])
        form.addRow("[initialization] position_source_type:", self.position_source_combo)

        self.init_seed_edit = QLineEdit("7238659235")
        form.addRow("init_seed:", self.init_seed_edit)

        # ---------------- [physics] ----------------
        self.dynamics_combo = QComboBox()
        self.dynamics_combo.addItems(["brownian", "rpy", "stokesian"])
        self.dynamics_combo.setCurrentText("stokesian")
        form.addRow("[physics] dynamics_type:", self.dynamics_combo)

        self.bc_combo = QComboBox()
        self.bc_combo.addItems(["periodic", "open"])
        form.addRow("boundary_conditions:", self.bc_combo)

        self.kT_spin = QDoubleSpinBox()
        self.kT_spin.setRange(0.0, 1e3)
        self.kT_spin.setDecimals(9)
        self.kT_spin.setValue(0.005305165)
        form.addRow("kT:", self.kT_spin)

        self.interaction_strength_spin = QDoubleSpinBox()
        self.interaction_strength_spin.setRange(-1e6, 1e6)
        self.interaction_strength_spin.setDecimals(6)
        form.addRow("interaction_strength:", self.interaction_strength_spin)

        self.interaction_cutoff_spin = QDoubleSpinBox()
        self.interaction_cutoff_spin.setRange(0.0, 1e6)
        self.interaction_cutoff_spin.setDecimals(6)
        form.addRow("interaction_cutoff:", self.interaction_cutoff_spin)

        self.shear_rate_spin = QDoubleSpinBox()
        self.shear_rate_spin.setRange(0.0, 1e6)
        self.shear_rate_spin.setDecimals(6)
        form.addRow("shear_rate:", self.shear_rate_spin)

        self.shear_freq_spin = QDoubleSpinBox()
        self.shear_freq_spin.setRange(0.0, 1e6)
        self.shear_freq_spin.setDecimals(6)
        form.addRow("shear_frequency:", self.shear_freq_spin)

        self.friction_coeff_spin = QDoubleSpinBox()
        self.friction_coeff_spin.setRange(0.0, 1e6)
        self.friction_coeff_spin.setDecimals(6)
        form.addRow("friction_coefficient:", self.friction_coeff_spin)

        self.friction_range_spin = QDoubleSpinBox()
        self.friction_range_spin.setRange(0.0, 1e6)
        self.friction_range_spin.setDecimals(6)
        form.addRow("friction_range:", self.friction_range_spin)

        # constant_force
        self.const_fx = QDoubleSpinBox()
        self.const_fy = QDoubleSpinBox()
        self.const_fz = QDoubleSpinBox()
        for spin in (self.const_fx, self.const_fy, self.const_fz):
            spin.setRange(-1e12, 1e12)
            spin.setDecimals(6)
            spin.setMinimumHeight(26)

        cf_row = QHBoxLayout()
        cf_row.addWidget(self.const_fx)
        cf_row.addWidget(self.const_fy)
        cf_row.addWidget(self.const_fz)
        cf_widget = QWidget()
        cf_widget.setLayout(cf_row)
        form.addRow("constant_force:", cf_widget)

        # constant_torque
        self.const_tx = QDoubleSpinBox()
        self.const_ty = QDoubleSpinBox()
        self.const_tz = QDoubleSpinBox()
        for spin in (self.const_tx, self.const_ty, self.const_tz):
            spin.setRange(-1e12, 1e12)
            spin.setDecimals(6)
            spin.setMinimumHeight(26)

        ct_row = QHBoxLayout()
        ct_row.addWidget(self.const_tx)
        ct_row.addWidget(self.const_ty)
        ct_row.addWidget(self.const_tz)
        ct_widget = QWidget()
        ct_widget.setLayout(ct_row)
        form.addRow("constant_torque:", ct_widget)

        self.buoyancy_check = QCheckBox("Enable buoyancy")
        form.addRow("buoyancy:", self.buoyancy_check)

        # ---------------- [box] ----------------
        self.lx_spin = QSpinBox(); self.lx_spin.setRange(1, 10**6); self.lx_spin.setValue(24)
        self.ly_spin = QSpinBox(); self.ly_spin.setRange(1, 10**6); self.ly_spin.setValue(24)
        self.lz_spin = QSpinBox(); self.lz_spin.setRange(1, 10**6); self.lz_spin.setValue(24)

        box_row = QHBoxLayout()
        box_row.addWidget(self.lx_spin)
        box_row.addWidget(self.ly_spin)
        box_row.addWidget(self.lz_spin)
        box_widget = QWidget(); box_widget.setLayout(box_row)
        form.addRow("[box] lx, ly, lz:", box_widget)

        self.max_strain_spin = QDoubleSpinBox()
        self.max_strain_spin.setRange(0.0, 10.0)
        self.max_strain_spin.setDecimals(4)
        self.max_strain_spin.setValue(0.5)
        form.addRow("max_strain:", self.max_strain_spin)

        # ---------------- [seeds] ----------------
        self.rfd_seed_edit = QLineEdit("43175675")
        self.ffwave_seed_edit = QLineEdit("6357442")
        self.ffreal_seed_edit = QLineEdit("6474524532")
        self.nf_seed_edit = QLineEdit("325425435")

        form.addRow("[seeds] rfd:", self.rfd_seed_edit)
        form.addRow("ffwave:", self.ffwave_seed_edit)
        form.addRow("ffreal:", self.ffreal_seed_edit)
        form.addRow("nf:", self.nf_seed_edit)

        # ---------------- [output] ----------------
        self.store_stresslet_check = QCheckBox()
        self.store_velocity_check = QCheckBox()
        self.store_orientation_check = QCheckBox()

        form.addRow("[output] store_stresslet:", self.store_stresslet_check)
        form.addRow("store_velocity:", self.store_velocity_check)
        form.addRow("store_orientation:", self.store_orientation_check)

        self.writing_period_spin = QSpinBox()
        self.writing_period_spin.setRange(1, 10**9)
        self.writing_period_spin.setValue(10)
        form.addRow("writing_period:", self.writing_period_spin)

        self.thermal_test_combo = QComboBox()
        self.thermal_test_combo.addItems(["none", "far-field", "lubrication"])
        form.addRow("thermal_fluctuation_test:", self.thermal_test_combo)

        # add form to layout
        layout.addLayout(form)

        # -------- config file mode --------
        save_group = QGroupBox("Config file mode")
        save_layout = QVBoxLayout(save_group)

        self.temp_radio = QRadioButton("Create temporary config (auto-delete after run)")
        self.perm_radio = QRadioButton("Save config to a chosen file")
        self.temp_radio.setChecked(True)

        save_layout.addWidget(self.temp_radio)
        save_layout.addWidget(self.perm_radio)

        layout.addWidget(save_group)

        # Create button
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        btn = QPushButton("Create config and switch to Run tab")
        btn.clicked.connect(self._on_create_config_clicked)
        btn_row.addWidget(btn)

        layout.addLayout(btn_row)

    def _browse_traj_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select trajectory file",
            "",
            "NumPy arrays (*.npy);;All files (*)",
        )
        if path:
            self.traj_edit.setText(path)
        
    def _build_visualize_tab(self):
        layout = QVBoxLayout(self.visual_tab)

        # --- Trajectory file selector ---
        traj_row = QHBoxLayout()
        traj_label = QLabel("Trajectory file (.npy):")
        self.traj_edit = QLineEdit()
        self.traj_edit.setPlaceholderText("Path to trajectory.npy")
        traj_browse = QPushButton("Browse…")
        traj_browse.clicked.connect(self._browse_traj_file)
        traj_row.addWidget(traj_label)
        traj_row.addWidget(self.traj_edit, 1)
        traj_row.addWidget(traj_browse)
        layout.addLayout(traj_row)

        # Load button
        load_row = QHBoxLayout()
        load_row.addStretch(1)
        load_btn = QPushButton("Load trajectory")
        load_btn.clicked.connect(self._on_load_trajectory_clicked)
        load_row.addWidget(load_btn)
        layout.addLayout(load_row)

        # --- VisPy canvas ---
        self.canvas_vis = scene.SceneCanvas(
            keys=None,              # no keyboard shortcuts
            show=False,
            bgcolor="white",
            size=(800, 500),
        )
        self.view = self.canvas_vis.central_widget.add_view()

        # Turntable-like camera: rotates around center, no translation in our code
        from vispy.scene import cameras
        self.view.camera = cameras.TurntableCamera(
            azimuth=self.azim,
            elevation=self.elev,
            distance=10.0,
            fov=60.0,
        )

        # Markers visual for particles
        self.marker = visuals.Markers()
        self.view.add(self.marker)

        # embed VisPy canvas into Qt layout
        layout.addWidget(self.canvas_vis.native, stretch=1)

        # --- Controls ---

        # playback controls
        controls_row = QHBoxLayout()
        self.prev_btn = QPushButton("⏮ Prev")
        self.play_btn = QPushButton("▶ Play")
        self.next_btn = QPushButton("⏭ Next")
        self.prev_btn.clicked.connect(self._on_prev_frame)
        self.play_btn.clicked.connect(self._on_play_pause)
        self.next_btn.clicked.connect(self._on_next_frame)
        controls_row.addWidget(self.prev_btn)
        controls_row.addWidget(self.play_btn)
        controls_row.addWidget(self.next_btn)
        layout.addLayout(controls_row)

        # --- NEW: frame slider ---
        slider_row = QHBoxLayout()
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)     # will be updated after loading traj
        self.frame_slider.setEnabled(False)
        self.frame_slider.setSingleStep(1)
        self.frame_slider.setPageStep(10)
        self.frame_slider.valueChanged.connect(self._on_slider_changed)

        self.frame_label = QLabel("Frame: 0 / 0 (0.00)")
        slider_row.addWidget(self.frame_slider, 1)
        slider_row.addWidget(self.frame_label)
        layout.addLayout(slider_row)


        # rotation controls
        rot_row = QHBoxLayout()
        rot_left_btn = QPushButton("⟲ Left")
        rot_right_btn = QPushButton("⟳ Right")
        rot_up_btn = QPushButton("↑ Up")
        rot_down_btn = QPushButton("↓ Down")
        rot_left_btn.clicked.connect(lambda: self._rotate_view(d_azim=-10, d_elev=0))
        rot_right_btn.clicked.connect(lambda: self._rotate_view(d_azim=+10, d_elev=0))
        rot_up_btn.clicked.connect(lambda: self._rotate_view(d_azim=0, d_elev=+5))
        rot_down_btn.clicked.connect(lambda: self._rotate_view(d_azim=0, d_elev=-5))
        rot_row.addWidget(rot_left_btn)
        rot_row.addWidget(rot_right_btn)
        rot_row.addWidget(rot_up_btn)
        rot_row.addWidget(rot_down_btn)
        layout.addLayout(rot_row)

        # zoom controls
        zoom_row = QHBoxLayout()
        zoom_in_btn = QPushButton("🔍 +")
        zoom_out_btn = QPushButton("🔍 −")
        zoom_in_btn.clicked.connect(lambda: self._zoom(factor=0.8))
        zoom_out_btn.clicked.connect(lambda: self._zoom(factor=1.25))
        zoom_row.addWidget(zoom_in_btn)
        zoom_row.addWidget(zoom_out_btn)
        layout.addLayout(zoom_row)



    # ------------------------------------------------------------------ Browsers
    def _browse_traj_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select trajectory file",
            "",
            "NumPy arrays (*.npy);;All files (*)",
        )
        if path:
            self.traj_edit.setText(path)

    def _browse_cfg(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select configuration file",
            "",
            "TOML files (*.toml);;All files (*)",
        )
        if path:
            self.cfg_edit.setText(path)

    def _browse_pos(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select positions file",
            "",
            "NumPy arrays (*.npy);;All files (*)",
        )
        if path:
            self.pos_edit.setText(path)

    def _browse_out(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "Select output directory",
            "",
        )
        if path:
            self.out_edit.setText(path)

    # ------------------------------------------------------------------ Config creation

    def _parse_int_from_lineedit(self, widget: QLineEdit, name: str) -> int:
        text = widget.text().strip()
        if text == "":
            raise ValueError(f"{name} cannot be empty.")
        try:
            return int(text)
        except ValueError:
            raise ValueError(f"{name} must be an integer, got: {text!r}")

    def _build_config_dict(self) -> dict:
        # [general]
        n_steps = int(self.n_steps_spin.value())
        n_particles_val = int(self.n_particles_spin.value())
        dt = float(self.dt_spin.value())

        general = {
            "n_steps": n_steps,
            "dt": dt,
        }
        # treat 0 as "omit n_particles" (let JFSD infer from positions file)
        if n_particles_val > 0:
            general["n_particles"] = n_particles_val

        # [initialization]
        position_source_type = self.position_source_combo.currentText()
        init_seed = self._parse_int_from_lineedit(self.init_seed_edit, "init_seed")

        initialization = {
            "position_source_type": position_source_type,
            "init_seed": init_seed,
        }

        # [physics]
        physics = {
            "dynamics_type": self.dynamics_combo.currentText(),
            "boundary_conditions": self.bc_combo.currentText(),
            "kT": float(self.kT_spin.value()),
            "interaction_strength": float(self.interaction_strength_spin.value()),
            "interaction_cutoff": float(self.interaction_cutoff_spin.value()),
            "shear_rate": float(self.shear_rate_spin.value()),
            "shear_frequency": float(self.shear_freq_spin.value()),
            "friction_coefficient": float(self.friction_coeff_spin.value()),
            "friction_range": float(self.friction_range_spin.value()),
            "constant_force": [
                float(self.const_fx.value()),
                float(self.const_fy.value()),
                float(self.const_fz.value()),
            ],
            "constant_torque": [
                float(self.const_tx.value()),
                float(self.const_ty.value()),
                float(self.const_tz.value()),
            ],
            "buoyancy": bool(self.buoyancy_check.isChecked()),
        }

        # [box]
        box = {
            "lx": int(self.lx_spin.value()),
            "ly": int(self.ly_spin.value()),
            "lz": int(self.lz_spin.value()),
            "max_strain": float(self.max_strain_spin.value()),
        }

        # [seeds]
        seeds = {
            "rfd": self._parse_int_from_lineedit(self.rfd_seed_edit, "seeds.rfd"),
            "ffwave": self._parse_int_from_lineedit(self.ffwave_seed_edit, "seeds.ffwave"),
            "ffreal": self._parse_int_from_lineedit(self.ffreal_seed_edit, "seeds.ffreal"),
            "nf": self._parse_int_from_lineedit(self.nf_seed_edit, "seeds.nf"),
        }

        # [output]
        output = {
            "store_stresslet": bool(self.store_stresslet_check.isChecked()),
            "store_velocity": bool(self.store_velocity_check.isChecked()),
            "store_orientation": bool(self.store_orientation_check.isChecked()),
            "writing_period": int(self.writing_period_spin.value()),
            "thermal_fluctuation_test": self.thermal_test_combo.currentText(),
        }

        config = {
            "general": general,
            "initialization": initialization,
            "physics": physics,
            "box": box,
            "seeds": seeds,
            "output": output,
        }

        return config

    def _on_create_config_clicked(self):
        try:
            config = self._build_config_dict()
        except ValueError as e:
            QMessageBox.critical(self, "Invalid configuration", str(e))
            return

        # decide file path
        if self.temp_radio.isChecked():
            tmp = tempfile.NamedTemporaryFile(prefix="jfsd_cfg_", suffix=".toml", delete=False)
            cfg_path = tmp.name
            tmp.close()
            is_temp = True
        else:
            cfg_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save configuration file",
                "files/config_gui.toml",
                "TOML files (*.toml);;All files (*)",
            )
            if not cfg_path:
                return
            is_temp = False

        # write TOML
        try:
            Path(cfg_path).parent.mkdir(parents=True, exist_ok=True)
            with open(cfg_path, "w", encoding="utf-8") as f:
                toml.dump(config, f)
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error writing config",
                f"Could not write configuration file:\n{e}",
            )
            return

        if is_temp:
            self._temp_cfg_files.add(cfg_path)

        # set on Run tab and switch to it
        self.cfg_edit.setText(cfg_path)
        self.tabs.setCurrentWidget(self.run_tab)

        QMessageBox.information(
            self,
            "Config created",
            f"Configuration written to:\n{cfg_path}",
        )

    # ------------------------------------------------------------------ Run handling

    def _on_run_clicked(self):
        cfg = self.cfg_edit.text().strip()
        out_dir = self.out_edit.text().strip()
        pos = self.pos_edit.text().strip()

        if not cfg:
            QMessageBox.warning(self, "Missing config", "Please select a configuration .toml file.")
            return
        if not Path(cfg).is_file():
            QMessageBox.warning(self, "Invalid config", "Configuration file does not exist.")
            return
        if not out_dir:
            QMessageBox.warning(self, "Missing output folder", "Please select an output directory.")
            return

        out_path = Path(out_dir)
        if not out_path.exists():
            try:
                out_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not create output directory:\n{e}")
                return
        
        # Update visualize tab label
        if hasattr(self, "vis_out_label"):
            self.vis_out_label.setText(out_dir)
            self.vis_out_label.setStyleSheet("color: gray;")
            
        if pos and not Path(pos).is_file():
            QMessageBox.warning(self, "Invalid positions file", "Positions file does not exist.")
            return

        self.run_button.setEnabled(False)
        self.log.clear()
        self.log_message.emit("Starting jfsd...\n")

        self._current_thread = threading.Thread(
            target=self._run_jfsd_subprocess,
            args=(cfg, pos, out_dir),
            daemon=True,
        )
        self._current_thread.start()

    def _run_jfsd_subprocess(self, cfg, pos, out_dir):
        cmd = ["jfsd", "-c", cfg, "-o", out_dir]
        if pos:
            cmd.extend(["-s", pos])

        self.log_message.emit(f"Command: {' '.join(cmd)}\n\n")

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except FileNotFoundError:
            self.log_message.emit(
                "ERROR: Could not find 'jfsd' in PATH. "
                "Is the package installed in this environment?\n"
            )
            self.run_finished.emit(1)
            return
        except Exception as e:
            self.log_message.emit(f"ERROR starting process: {e}\n")
            self.run_finished.emit(1)
            return

        assert process.stdout is not None
        for line in process.stdout:
            self.log_message.emit(line.rstrip("\n"))

        return_code = process.wait()
        self.run_finished.emit(return_code)

    # ------------------------------------------------------------------ Slots

    @Slot(str)
    def _append_log(self, text: str):
        self.log.append(text)
        # Move cursor to end using the enum from QTextCursor
        self.log.moveCursor(QTextCursor.End)

    @Slot(int)
    def _on_run_finished(self, return_code: int):
        self.run_button.setEnabled(True)

        # delete temporary config if requested
        if self.delete_temp_after_run_check.isChecked():
            cfg_path = self.cfg_edit.text().strip()
            if cfg_path in self._temp_cfg_files and os.path.exists(cfg_path):
                try:
                    os.remove(cfg_path)
                    self.log_message.emit(f"\nTemporary config deleted: {cfg_path}")
                except Exception as e:
                    self.log_message.emit(
                        f"\nWARNING: Could not delete temporary config ({cfg_path}): {e}"
                    )
                self._temp_cfg_files.discard(cfg_path)

        if return_code == 0:
            QMessageBox.information(self, "Simulation finished", "JFSD completed successfully.")
        else:
            QMessageBox.warning(
                self,
                "Simulation finished with errors",
                f"JFSD exited with return code {return_code}. Check the log above.",
            )


def main():
    app = QApplication(sys.argv)
    window = JFSDGui()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

