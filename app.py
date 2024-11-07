import cv2
import numpy as np
import pygame
import pyaudio
from collections import deque
import time
import torch
import traceback
from pygame.locals import *
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import threading
import pystray
from PIL import Image, ImageDraw

# Initialize Pygame font
pygame.font.init()

# Print CUDA availability
print(f"PyTorch CUDA: {torch.cuda.is_available()} - {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")


class LightweightAudioReader:
    def __init__(self, device_index=None, chunk_size=256):
        self.chunk_size = chunk_size
        self.buffer = deque(maxlen=3)  # Reduced buffer size

        self.p = pyaudio.PyAudio()
        self.device_index = device_index

        # Use lower sample rate for better performance
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=22050,  # Reduced from 44100
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=chunk_size,
            stream_callback=self._audio_callback
        )
        self.stream.start_stream()

        # Add smoothing for amplitude
        self.smoothed_amplitude = 0
        self.smoothing_factor = 0.3

    def _audio_callback(self, in_data, frame_count, time_info, status):
        try:
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            self.buffer.append(audio_data)
        except Exception as e:
            print(f"Audio callback error: {e}")
        return (None, pyaudio.paContinue)

    def get_amplitude(self):
        """Get smoothed audio amplitude."""
        if len(self.buffer) > 0:
            try:
                audio_chunk = self.buffer[-1]  # Just use latest chunk
                current_amplitude = np.mean(np.abs(audio_chunk))
                if np.isnan(current_amplitude):
                    current_amplitude = 0.0

                # Apply smoothing
                self.smoothed_amplitude = (self.smoothing_factor * current_amplitude +
                                           (1 - self.smoothing_factor) * self.smoothed_amplitude)
                return self.smoothed_amplitude
            except Exception as e:
                print(f"Amplitude calculation error: {e}")
                return 0.0
        return 0.0

    def cleanup(self):
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'p'):
            self.p.terminate()


class AudioDeviceManager:
    def __init__(self, pyaudio_instance):
        self.p = pyaudio_instance
        self.devices = self.get_input_devices()

    def get_input_devices(self):
        devices = []
        for i in range(self.p.get_device_count()):
            try:
                device_info = self.p.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:  # Only list input devices
                    devices.append({
                        'index': i,
                        'name': device_info['name'],
                        'channels': device_info['maxInputChannels'],
                        'sample_rate': int(device_info['defaultSampleRate'])
                    })
            except Exception as e:
                print(f"Error accessing device {i}: {e}")
                continue
        return devices

    def list_devices(self):
        print("\nAvailable Audio Input Devices:")
        for device in self.devices:
            print(f"{device['index']}: {device['name']}")

    def cleanup(self):
        pass  # Currently, no resources to clean up


class WaveNeuron:
    def __init__(self, radius, angle, frequency=2.0, sensitivity=0.4):
        self.radius = radius
        self.angle = angle
        self.frequency = frequency
        self.phase = torch.rand(1).item() * 2 * np.pi
        self.energy = 0.0
        self.response = 0.0
        self.memory = deque(maxlen=10)
        self.sensitivity = sensitivity

    def process(self, input_value, edge_strength, audio_amplitude):
        # Scale factors for higher ranges
        time_scale = 4.0  # Faster time-based movement
        audio_scale = 0.2  # Scale down audio to prevent oversaturation

        # Make waves more dynamic
        radial_factor = torch.sin(torch.tensor(
            self.radius * 0.4 - time.time() * time_scale
        )).item()

        wave = torch.sin(torch.tensor(
            2 * np.pi * self.frequency * input_value +
            self.phase +
            radial_factor +
            audio_amplitude * audio_scale
        )).item()

        # Enhanced response with better scaling
        distance_factor = torch.exp(torch.tensor(-self.radius * 0.05)).item()
        self.response = wave * distance_factor * (1 + edge_strength * self.sensitivity) * (1 + audio_amplitude * audio_scale)

        # More energetic movement
        self.energy = 0.8 * self.energy + 0.2 * edge_strength * (1 + audio_amplitude * audio_scale) * distance_factor
        self.phase += 0.2 * (1 + edge_strength + audio_amplitude * audio_scale) * distance_factor

        return self.response, self.energy


class WaveVisualizer:
    def __init__(self, width=854, height=480, camera_index=0, audio_device_index=None):
        # Initialize Pygame
        pygame.init()
        pygame.mixer.quit()  # Don't need pygame's audio

        # Set display mode
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode(
            (width, height),
            pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        pygame.display.set_caption("Wave Pond PyTorch")

        # Check if display is properly initialized
        if not pygame.display.get_init():
            raise RuntimeError("Failed to initialize Pygame display")

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize background source
        self.background_source = "webcam"  # Default
        self.background_image = None
        self.background_video_path = ""
        self.cap = None
        self.set_background_source("webcam", camera_index)

        # Initialize audio
        self.audio = LightweightAudioReader(
            device_index=audio_device_index,
            chunk_size=256
        )

        # Wave parameters with higher limits
        self.num_rings = 6
        self.neurons_per_ring = 12
        self.wave_speed = 4.0
        self.wave_amplitude = 3.0
        self.ripple_decay = 0.005
        self.audio_multiplier = 20.0
        self.blend_ratio = 0.7

        # Maximum limits
        self.MAX_WAVE_SPEED = 50.0
        self.MAX_WAVE_AMPLITUDE = 50.0
        self.MAX_AUDIO_MULTIPLIER = 200.0
        self.MAX_BLEND_RATIO = 1.0

        # Control step sizes
        self.SPEED_STEP = 2.0
        self.AMPLITUDE_STEP = 2.0
        self.AUDIO_STEP = 10.0
        self.BLEND_STEP = 0.1

        # Performance monitoring
        self.show_fps = True
        self.frame_times = deque(maxlen=30)
        self.start_time = time.time()

        # View controls
        self.zoom_factor = 1.0
        self.min_zoom = 0.5
        self.max_zoom = 3.0
        self.paused = False
        self.show_help = False

        # Setup visualization
        self.setup_neurons()
        self.setup_distance_matrices()

        # Enhanced color palettes with more contrast
        self.palettes = {
            'ocean': [(0, 0, 96), (0, 128, 255), (192, 255, 255)],
            'fire': [(96, 0, 0), (255, 128, 0), (255, 255, 128)],
            'plasma': [(96, 0, 96), (255, 0, 255), (255, 192, 255)],
            'matrix': [(0, 96, 0), (0, 255, 64), (192, 255, 192)],
            'sunset': [(96, 0, 32), (255, 64, 0), (255, 255, 128)]
        }
        self.current_palette = 'ocean'

        # Menu setup
        self.menu_button = pygame.Rect(self.width - 110, 10, 100, 30)
        self.menu_font = pygame.font.SysFont(None, 24)

        # Print controls
        print("\nControls:")
        print("UP/DOWN: Wave amplitude (0-50)")
        print("LEFT/RIGHT: Wave speed (0-50)")
        print("A/Z: Audio sensitivity (0-200)")
        print("S/X: Ripple size")
        print("D/C: Effect visibility (0-1)")
        print("SPACE: Change color palette")
        print("F: Toggle FPS display")
        print("H: Toggle Help Section")
        print("F11: Toggle Fullscreen")
        print("P: Pause/Resume")
        print("Mouse Wheel: Zoom in/out")
        print("ESC: Quit")
        print("Click 'Menu' button for Settings")

    def open_settings(self):
        """Opens the settings GUI to adjust options like background, webcam, audio source, etc."""
        if hasattr(self, 'settings_gui'):
            self.settings_gui.show()
        else:
            print("Settings GUI not found. Ensure it is initialized properly.")

    def set_background_source(self, source, camera_index=0, bg_path=None):
        if self.cap:
            self.cap.release()
            self.cap = None

        self.background_source = source
        if source == "webcam":
            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened():
                raise ValueError(f"Failed to open camera {camera_index}")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.background_image = None
        elif source == "image" and bg_path:
            img = cv2.imread(bg_path)
            if img is not None:
                self.background_image = cv2.resize(img, (self.width, self.height))
            else:
                raise ValueError(f"Failed to load image: {bg_path}")
        elif source == "video" and bg_path:
            self.cap = cv2.VideoCapture(bg_path)
            if not self.cap.isOpened():
                raise ValueError(f"Failed to open video: {bg_path}")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.background_image = None
        else:
            raise ValueError("Invalid background source. Choose from 'webcam', 'image', 'video'.")

    def setup_neurons(self):
        self.neurons = []
        max_radius = min(self.width / 2, self.height / 2)

        # Move coordinate generation to GPU
        angles = torch.linspace(0, 2 * np.pi, self.neurons_per_ring, device=self.device)
        radii = torch.linspace(max_radius / self.num_rings, max_radius, self.num_rings, device=self.device)

        self.coordinates = []
        center = torch.tensor([self.width / 2, self.height / 2], device=self.device)

        for radius in radii:
            for angle in angles:
                neuron = WaveNeuron(radius.item(), angle.item())
                self.neurons.append(neuron)
                pos = center + radius * torch.tensor([torch.cos(angle), torch.sin(angle)], device=self.device)
                neuron.position = (int(pos[0].item()), int(pos[1].item()))
                self.coordinates.append(neuron.position)

    def setup_distance_matrices(self):
        y, x = torch.meshgrid(torch.arange(self.height, device=self.device),
                              torch.arange(self.width, device=self.device))
        self.dist_matrices = []

        for coord in self.coordinates:
            dist = torch.sqrt((x - coord[0]) ** 2 + (y - coord[1]) ** 2)
            self.dist_matrices.append(dist)

    def apply_gradient(self, ripple_tensor, palette):
        gradient = np.zeros((256, 3), dtype=np.uint8)
        num_colors = len(palette)
        for i in range(num_colors - 1):
            start = np.array(palette[i], dtype=np.float32)
            end = np.array(palette[i + 1], dtype=np.float32)
            for j in range(int(256 / (num_colors - 1))):
                t = j / (256 / (num_colors - 1))
                idx = i * int(256 / (num_colors - 1)) + j
                if idx < 256:
                    gradient[idx] = (start * (1 - t) + end * t).astype(np.uint8)
        return gradient[ripple_tensor]

    def draw_help_section(self):
        help_text = [
            "Controls:",
            "UP/DOWN: Wave amplitude (0-50)",
            "LEFT/RIGHT: Wave speed (0-50)",
            "A/Z: Audio sensitivity (0-200)",
            "S/X: Ripple size",
            "D/C: Effect visibility (0-1)",
            "SPACE: Change color palette",
            "F: Toggle FPS display",
            "H: Toggle Help Section",
            "F11: Toggle Fullscreen",
            "P: Pause/Resume",
            "Mouse Wheel: Zoom in/out",
            "ESC: Quit",
            "Click 'Menu' button for Settings"
        ]

        # Create semi-transparent overlay
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))

        # Render help text
        font = pygame.font.SysFont(None, 24)
        y_pos = 20
        for line in help_text:
            text = font.render(line, True, (255, 255, 255))
            overlay.blit(text, (20, y_pos))
            y_pos += 30

        self.screen.blit(overlay, (0, 0))



    def handle_node_dragging(self):
        if hasattr(self, 'dragging') and self.dragging and hasattr(self, 'dragged_node') and self.dragged_node:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            # Adjust for zoom
            adjusted_x = int(mouse_x / self.zoom_factor)
            adjusted_y = int(mouse_y / self.zoom_factor)
            self.dragged_node.position = (adjusted_x - self.mouse_offset[0], adjusted_y - self.mouse_offset[1])

    def apply_fullscreen(self):
        self.fullscreen = not getattr(self, 'fullscreen', False)
        if self.fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            # Update resolution variables
            self.width, self.height = self.screen.get_size()
        else:
            self.screen = pygame.display.set_mode((self.width, self.height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        # Resize background accordingly
        if self.background_source == "image" and self.background_image is not None:
            self.background_image = cv2.resize(self.background_image, (self.width, self.height))
        elif self.background_source == "video" and self.background_video_path:
            if self.cap:
                self.cap.release()
            try:
                self.set_background_source("video", bg_path=self.background_video_path)
            except Exception as e:
                print(f"Failed to set background video: {e}")
        elif self.background_source == "webcam":
            if self.cap:
                self.cap.release()
            try:
                self.set_background_source("webcam", camera_index=0)
            except Exception as e:
                print(f"Failed to set webcam: {e}")
        # Recompute distance matrices
        self.setup_distance_matrices()
        # Re-render menu button
        self.menu_button = pygame.Rect(self.width - 110, 10, 100, 30)

    def toggle_pause(self):
        self.paused = not self.paused
        print(f"Visualization {'Paused' if self.paused else 'Resumed'}")

    def process_frame(self):
        if self.paused:
            return True  # Skip processing when paused

        # Get frame from source
        if self.background_source in ["webcam", "video"]:
            ret, frame = self.cap.read()
            if not ret:
                if self.background_source == "video":
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if not ret:
                        return False
                else:
                    return False
            frame = cv2.resize(frame, (self.width, self.height))
            frame = cv2.flip(frame, 1)
        else:
            frame = self.background_image.copy() if self.background_image is not None else np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Convert to GPU tensor
        frame_tensor = torch.from_numpy(frame).to(self.device).float()

        # Convert to grayscale
        gray_tensor = (frame_tensor[:, :, 0] * 0.299 + frame_tensor[:, :, 1] * 0.587 + frame_tensor[:, :, 2] * 0.114)

        # Edge detection
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], device=self.device).float()
        sobel_y = torch.tensor([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]], device=self.device).float()

        edges_x = torch.nn.functional.conv2d(gray_tensor.unsqueeze(0).unsqueeze(0),
                                           sobel_x.unsqueeze(0).unsqueeze(0),
                                           padding=1)
        edges_y = torch.nn.functional.conv2d(gray_tensor.unsqueeze(0).unsqueeze(0),
                                           sobel_y.unsqueeze(0).unsqueeze(0),
                                           padding=1)
        edges = torch.sqrt(edges_x ** 2 + edges_y ** 2).squeeze()

        # Get audio amplitude
        audio_amplitude = self.audio.get_amplitude() * self.audio_multiplier

        # Process ripples
        ripple_tensor = torch.zeros((self.height, self.width), device=self.device)

        # Process each neuron
        for neuron, coord, dist_matrix in zip(self.neurons, self.coordinates, self.dist_matrices):
            x, y = coord
            region_size = 10
            x1, x2 = max(0, x - region_size), min(self.width, x + region_size)
            y1, y2 = max(0, y - region_size), min(self.height, y + region_size)

            if x1 < x2 and y1 < y2:
                edge_strength = edges[y1:y2, x1:x2].mean().item() / 255.0
                intensity = gray_tensor[y1:y2, x1:x2].mean().item() / 255.0

                response, _ = neuron.process(intensity, edge_strength, audio_amplitude)

                # Calculate ripple contribution with time-based movement
                current_time = time.time() * self.wave_speed
                time_factor = torch.sin(dist_matrix * 0.1 + current_time)
                ripple_contribution = response * torch.exp(-dist_matrix * self.ripple_decay) * (1 + time_factor * 0.5)
                ripple_tensor += ripple_contribution * self.wave_amplitude

        # Normalize ripples
        if torch.max(ripple_tensor) != torch.min(ripple_tensor):
            ripple_tensor = (ripple_tensor - torch.min(ripple_tensor)) / (torch.max(ripple_tensor) - torch.min(ripple_tensor))
        ripple_tensor = (ripple_tensor * 255).clamp(0, 255).byte()

        # Apply color palette
        colored_ripples = self.apply_gradient(ripple_tensor.cpu().numpy(), self.palettes[self.current_palette])

        # Blend with original frame
        frame_np = frame_tensor.cpu().numpy().astype(np.uint8)
        result = (frame_np * (1 - self.blend_ratio) + colored_ripples * self.blend_ratio).astype(np.uint8)

        # Convert to pygame surface
        surface = pygame.surfarray.make_surface(result.swapaxes(0, 1))

        # Apply zoom if needed
        if self.zoom_factor != 1.0:
            scaled_width = int(self.width * self.zoom_factor)
            scaled_height = int(self.height * self.zoom_factor)
            surface = pygame.transform.scale(surface, (scaled_width, scaled_height))
            x_pos = (self.width - scaled_width) // 2
            y_pos = (self.height - scaled_height) // 2
            self.screen.blit(surface, (x_pos, y_pos))
        else:
            self.screen.blit(surface, (0, 0))

        # Draw menu button
        pygame.draw.rect(self.screen, (50, 50, 50), self.menu_button)
        menu_text = self.menu_font.render("Menu", True, (255, 255, 255))
        self.screen.blit(menu_text, (self.menu_button.x + 10, self.menu_button.y + 5))

        # Update FPS display
        if self.show_fps:
            current_time = time.time()
            self.frame_times.append(current_time)
            if len(self.frame_times) > 1:
                fps = 1.0 / (self.frame_times[-1] - self.frame_times[-2])
                pygame.display.set_caption(f"Wave Pond PyTorch - FPS: {fps:.1f}")

        pygame.display.flip()
        return True

    def run(self):
        running = True
        clock = pygame.time.Clock()

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        mouse_x, mouse_y = event.pos
                        # Check if menu button is clicked
                        if self.menu_button.collidepoint(event.pos):
                            self.open_settings()
                            continue

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        palettes = list(self.palettes.keys())
                        current_idx = palettes.index(self.current_palette)
                        self.current_palette = palettes[(current_idx + 1) % len(palettes)]
                    elif event.key == pygame.K_f:
                        self.show_fps = not self.show_fps
                    elif event.key == pygame.K_UP:
                        self.wave_amplitude = min(self.MAX_WAVE_AMPLITUDE,
                                                 self.wave_amplitude + self.AMPLITUDE_STEP)
                        print(f"Wave Amplitude: {self.wave_amplitude:.1f}")
                    elif event.key == pygame.K_DOWN:
                        self.wave_amplitude = max(0.1, self.wave_amplitude - self.AMPLITUDE_STEP)
                        print(f"Wave Amplitude: {self.wave_amplitude:.1f}")
                    elif event.key == pygame.K_RIGHT:
                        self.wave_speed = min(self.MAX_WAVE_SPEED,
                                              self.wave_speed + self.SPEED_STEP)
                        print(f"Wave Speed: {self.wave_speed:.1f}")
                    elif event.key == pygame.K_LEFT:
                        self.wave_speed = max(0.3, self.wave_speed - self.SPEED_STEP)
                        print(f"Wave Speed: {self.wave_speed:.1f}")
                    elif event.key == pygame.K_a:
                        self.audio_multiplier = min(self.MAX_AUDIO_MULTIPLIER,
                                                   self.audio_multiplier + self.AUDIO_STEP)
                        print(f"Audio Sensitivity: {self.audio_multiplier:.1f}")
                    elif event.key == pygame.K_z:
                        self.audio_multiplier = max(1.0, self.audio_multiplier - self.AUDIO_STEP)
                        print(f"Audio Sensitivity: {self.audio_multiplier:.1f}")
                    elif event.key == pygame.K_s:
                        self.ripple_decay = min(0.02, self.ripple_decay + 0.001)
                        print(f"Ripple Size: {1 / self.ripple_decay:.1f}")
                    elif event.key == pygame.K_x:
                        self.ripple_decay = max(0.001, self.ripple_decay - 0.001)
                        print(f"Ripple Size: {1 / self.ripple_decay:.1f}")
                    elif event.key == pygame.K_d:
                        self.blend_ratio = min(self.MAX_BLEND_RATIO,
                                               self.blend_ratio + self.BLEND_STEP)
                        print(f"Effect Visibility: {self.blend_ratio:.1f}")
                    elif event.key == pygame.K_c:
                        self.blend_ratio = max(0.1, self.blend_ratio - self.BLEND_STEP)
                        print(f"Effect Visibility: {self.blend_ratio:.1f}")
                    elif event.key == pygame.K_h:
                        self.show_help = not self.show_help
                    elif event.key == pygame.K_F11:
                        self.apply_fullscreen()
                    elif event.key == pygame.K_p:
                        self.toggle_pause()
                    elif event.key == pygame.K_ESCAPE:
                        running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        mouse_x, mouse_y = event.pos
                        # Check if menu button is clicked
                        if self.menu_button.collidepoint(event.pos):
                            self.open_settings()
                            continue
                        # Adjust for zoom
                        adjusted_x = int(mouse_x / self.zoom_factor)
                        adjusted_y = int(mouse_y / self.zoom_factor)
                        for neuron in self.neurons:
                            nx, ny = neuron.position
                            if (adjusted_x - nx) ** 2 + (adjusted_y - ny) ** 2 <= 25:  # Radius 5
                                self.dragging = True
                                self.dragged_node = neuron
                                self.mouse_offset = (mouse_x - nx * self.zoom_factor, mouse_y - ny * self.zoom_factor)
                                break
                        else:
                            self.dragged_node = None
                    elif event.button == 4:  # Scroll up
                        self.zoom_factor = min(self.max_zoom, self.zoom_factor + 0.1)
                    elif event.button == 5:  # Scroll down
                        self.zoom_factor = max(self.min_zoom, self.zoom_factor - 0.1)
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:  # Left click release
                        self.dragging = False
                        self.dragged_node = None
                elif event.type == pygame.MOUSEMOTION:
                    if hasattr(self, 'dragging') and self.dragging and hasattr(self, 'dragged_node') and self.dragged_node:
                        mouse_x, mouse_y = event.pos
                        new_x = int((mouse_x - self.mouse_offset[0]) / self.zoom_factor)
                        new_y = int((mouse_y - self.mouse_offset[1]) / self.zoom_factor)
                        self.dragged_node.position = (new_x, new_y)

            if not self.process_frame():
                running = False

            if self.show_help:
                self.draw_help_section()
                pygame.display.flip()

            clock.tick(30)

    def create_settings_window(visualizer):
        window = tk.Tk()
        window.title("Wave Visualizer Settings")
        window.geometry("400x700")
        window.resizable(False, False)

        # Background Source Selection
        tk.Label(window, text="Background Source:").pack(pady=5)
        bg_source_var = tk.StringVar(value=visualizer.background_source)

        def update_bg_source():
            source = bg_source_var.get()
            if source == "webcam":
                bg_file_entry.config(state='disabled')
                bg_file_button.config(state='disabled')
                bg_file_entry.delete(0, tk.END)
                bg_file_entry.insert(0, "Webcam Selected")
            else:
                bg_file_entry.config(state='normal')
                bg_file_button.config(state='normal')
                if source == "image":
                    bg_file_entry.delete(0, tk.END)
                    bg_file_entry.insert(0, "Select Image")
                elif source == "video":
                    bg_file_entry.delete(0, tk.END)
                    bg_file_entry.insert(0, "Select Video")

        tk.Radiobutton(window, text="Webcam", variable=bg_source_var, value="webcam", command=update_bg_source).pack()
        tk.Radiobutton(window, text="Image", variable=bg_source_var, value="image", command=update_bg_source).pack()
        tk.Radiobutton(window, text="Video", variable=bg_source_var, value="video", command=update_bg_source).pack()

        # Background File Selection
        bg_file_frame = tk.Frame(window)
        bg_file_frame.pack(pady=10)

        bg_file_label = tk.Label(bg_file_frame, text="Background File:")
        bg_file_label.pack(side=tk.LEFT)

        bg_file_entry = tk.Entry(bg_file_frame, width=30)
        bg_file_entry.pack(side=tk.LEFT, padx=5)
        if visualizer.background_source == "image":
            bg_file_entry.insert(0, "Select Image")
        elif visualizer.background_source == "video":
            bg_file_entry.insert(0, "Select Video")
        else:
            bg_file_entry.insert(0, "Webcam Selected")
            bg_file_entry.config(state='disabled')

        def browse_bg_file():
            source = bg_source_var.get()
            if source == "image":
                file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
            elif source == "video":
                file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
            else:
                file_path = ""
            if file_path:
                bg_file_entry.delete(0, tk.END)
                bg_file_entry.insert(0, file_path)

        bg_file_button = tk.Button(bg_file_frame, text="Browse", command=browse_bg_file)
        bg_file_button.pack(side=tk.LEFT)

        # Resolution Setting
        tk.Label(window, text="Resolution:").pack(pady=5)
        resolution_frame = tk.Frame(window)
        resolution_frame.pack(pady=5)

        tk.Label(resolution_frame, text="Width:").pack(side=tk.LEFT)
        width_entry = tk.Entry(resolution_frame, width=10)
        width_entry.pack(side=tk.LEFT, padx=5)
        width_entry.insert(0, str(visualizer.width))

        tk.Label(resolution_frame, text="Height:").pack(side=tk.LEFT)
        height_entry = tk.Entry(resolution_frame, width=10)
        height_entry.pack(side=tk.LEFT, padx=5)
        height_entry.insert(0, str(visualizer.height))

        # Audio Device Selection
        tk.Label(window, text="Audio Input Device:").pack(pady=5)
        audio_device_var = tk.StringVar()
        audio_manager = AudioDeviceManager(pyaudio_instance=pyaudio.PyAudio())
        device_names = [f"{device['index']}: {device['name']}" for device in audio_manager.get_input_devices()]
        device_names.insert(0, "Default")
        audio_device_dropdown = tk.OptionMenu(window, audio_device_var, *device_names)
        audio_device_dropdown.pack()
        # Set current selection
        current_device = visualizer.audio.device_index
        if current_device is not None:
            for name in device_names:
                if name.startswith(str(current_device)):
                    audio_device_var.set(name)
                    break
        else:
            audio_device_var.set("Default")

        # Node Count Selection
        tk.Label(window, text="Number of Rings:").pack(pady=5)
        num_rings_entry = tk.Entry(window, width=10)
        num_rings_entry.pack()
        num_rings_entry.insert(0, str(visualizer.num_rings))

        tk.Label(window, text="Neurons per Ring:").pack(pady=5)
        neurons_per_ring_entry = tk.Entry(window, width=10)
        neurons_per_ring_entry.pack()
        neurons_per_ring_entry.insert(0, str(visualizer.neurons_per_ring))

        # Save Button
        def save_settings():
            # Update background source
            visualizer.background_source = bg_source_var.get()
            if visualizer.background_source == "image":
                bg_path = bg_file_entry.get()
                if os.path.exists(bg_path):
                    visualizer.background_image = cv2.imread(bg_path)
                    visualizer.background_image = cv2.resize(visualizer.background_image, (visualizer.width, visualizer.height))
                    visualizer.background_video_path = ""
                    if visualizer.cap:
                        visualizer.cap.release()
                        visualizer.cap = None
                    try:
                        visualizer.set_background_source("image", bg_path=bg_path)
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to set background image: {e}")
                        return
                else:
                    messagebox.showerror("Error", "Selected image file does not exist.")
                    return
            elif visualizer.background_source == "video":
                bg_path = bg_file_entry.get()
                if os.path.exists(bg_path):
                    visualizer.background_video_path = bg_path
                    try:
                        visualizer.set_background_source("video", bg_path=bg_path)
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to set background video: {e}")
                        return
                    visualizer.background_image = None
                else:
                    messagebox.showerror("Error", "Selected video file does not exist.")
                    return
            elif visualizer.background_source == "webcam":
                try:
                    visualizer.set_background_source("webcam", camera_index=0)
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to set webcam: {e}")
                    return
                visualizer.background_image = None
                visualizer.background_video_path = ""

            # Update resolution
            try:
                new_width = int(width_entry.get())
                new_height = int(height_entry.get())
                if new_width <= 0 or new_height <= 0:
                    raise ValueError
                visualizer.width = new_width
                visualizer.height = new_height
                # Resize Pygame display
                pygame.display.set_mode((visualizer.width, visualizer.height), pygame.HWSURFACE | pygame.DOUBLEBUF)
                # Resize background image/video
                if visualizer.background_source == "image" and visualizer.background_image is not None:
                    visualizer.background_image = cv2.resize(visualizer.background_image, (visualizer.width, visualizer.height))
                elif visualizer.background_source == "video" and visualizer.background_video_path:
                    if visualizer.cap:
                        visualizer.cap.release()
                    try:
                        visualizer.set_background_source("video", bg_path=visualizer.background_video_path)
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to set background video: {e}")
                        return
                elif visualizer.background_source == "webcam":
                    if visualizer.cap:
                        visualizer.cap.release()
                    try:
                        visualizer.set_background_source("webcam", camera_index=0)
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to set webcam: {e}")
                        return
                # Recompute distance matrices
                visualizer.setup_distance_matrices()
                # Re-render menu button
                visualizer.menu_button = pygame.Rect(visualizer.width - 110, 10, 100, 30)
            except ValueError:
                messagebox.showerror("Error", "Resolution must be positive integers.")
                return

            # Update audio device
            audio_selection = audio_device_var.get()
            if audio_selection == "Default":
                visualizer.audio.device_index = None
            else:
                device_index = int(audio_selection.split(":")[0])
                visualizer.audio.device_index = device_index

            # Reinitialize audio with new device
            visualizer.audio.cleanup()
            visualizer.audio = LightweightAudioReader(
                device_index=visualizer.audio.device_index,
                chunk_size=256
            )

            # Update node counts
            try:
                num_rings = int(num_rings_entry.get())
                neurons_per_ring = int(neurons_per_ring_entry.get())
                if num_rings <= 0 or neurons_per_ring <= 0:
                    raise ValueError
                visualizer.num_rings = num_rings
                visualizer.neurons_per_ring = neurons_per_ring
                visualizer.setup_neurons()
                visualizer.setup_distance_matrices()
            except ValueError:
                messagebox.showerror("Error", "Number of rings and neurons per ring must be positive integers.")
                return

            # Close the settings window
            messagebox.showinfo("Settings", "Settings saved successfully!")
            window.destroy()

        save_button = tk.Button(window, text="Save Settings", command=save_settings)
        save_button.pack(pady=20)

        window.mainloop()


    def create_system_tray_icon(visualizer, settings_gui):
        def on_quit(icon, item):
            icon.stop()
            visualizer.paused = True
            cleanup(visualizer)
            os._exit(0)

        def on_toggle_help(icon, item):
            visualizer.show_help = not visualizer.show_help

        def on_pause_resume(icon, item):
            visualizer.toggle_pause()

        def on_open_settings(icon, item):
            settings_gui.show()

        # Create an icon image (64x64) with a simple design
        img = Image.new('RGB', (64, 64), color=(0, 0, 0))
        d = ImageDraw.Draw(img)
        d.ellipse((16, 16, 48, 48), fill=(0, 128, 255))

        icon = pystray.Icon("WaveVisualizer", img, "Wave Pond PyTorch", menu=pystray.Menu(
            pystray.MenuItem("Settings", on_open_settings),
            pystray.MenuItem("Toggle Help", on_toggle_help),
            pystray.MenuItem("Pause/Resume", on_pause_resume),
            pystray.MenuItem("Exit", on_quit)
        ))
        threading.Thread(target=icon.run, daemon=True).start()


    def cleanup(visualizer):
        if hasattr(visualizer, 'cap') and visualizer.cap:
            visualizer.cap.release()
        if hasattr(visualizer, 'audio'):
            visualizer.audio.cleanup()
        pygame.quit()


class SettingsGUI:
    def __init__(self, visualizer):
        self.visualizer = visualizer
        self.window = None

    def show(self):
        if self.window is not None:
            self.window.lift()  # Bring window to the front if already open
            return
        self.window = tk.Tk()
        self.window.title("Wave Visualizer Settings")
        self.window.geometry("400x700")
        self.window.resizable(False, False)
        
        # Setup UI components for selecting background, audio source, etc.
        tk.Label(self.window, text="Background Source:").pack(pady=5)
        bg_source_var = tk.StringVar(value=self.visualizer.background_source)

        def update_bg_source():
            source = bg_source_var.get()
            if source == "webcam":
                bg_file_entry.config(state='disabled')
                bg_file_button.config(state='disabled')
            else:
                bg_file_entry.config(state='normal')
                bg_file_button.config(state='normal')

        tk.Radiobutton(self.window, text="Webcam", variable=bg_source_var, value="webcam", command=update_bg_source).pack()
        tk.Radiobutton(self.window, text="Image", variable=bg_source_var, value="image", command=update_bg_source).pack()
        tk.Radiobutton(self.window, text="Video", variable=bg_source_var, value="video", command=update_bg_source).pack()

        bg_file_frame = tk.Frame(self.window)
        bg_file_frame.pack(pady=10)

        bg_file_label = tk.Label(bg_file_frame, text="Background File:")
        bg_file_label.pack(side=tk.LEFT)

        bg_file_entry = tk.Entry(bg_file_frame, width=30)
        bg_file_entry.pack(side=tk.LEFT, padx=5)
        
        def browse_bg_file():
            source = bg_source_var.get()
            filetypes = [("Image Files", "*.jpg *.jpeg *.png")] if source == "image" else [("Video Files", "*.mp4 *.avi *.mov")]
            file_path = filedialog.askopenfilename(filetypes=filetypes)
            if file_path:
                bg_file_entry.delete(0, tk.END)
                bg_file_entry.insert(0, file_path)

        bg_file_button = tk.Button(bg_file_frame, text="Browse", command=browse_bg_file)
        bg_file_button.pack(side=tk.LEFT)

        tk.Label(self.window, text="Audio Input Device:").pack(pady=5)
        audio_device_var = tk.StringVar()
        audio_manager = AudioDeviceManager(pyaudio_instance=pyaudio.PyAudio())
        device_names = [f"{device['index']}: {device['name']}" for device in audio_manager.get_input_devices()]
        device_names.insert(0, "Default")
        audio_device_dropdown = tk.OptionMenu(self.window, audio_device_var, *device_names)
        audio_device_dropdown.pack()

        def save_settings():
            source = bg_source_var.get()
            bg_file = bg_file_entry.get()
            if source == "image" and os.path.exists(bg_file):
                self.visualizer.set_background_source("image", bg_path=bg_file)
            elif source == "video" and os.path.exists(bg_file):
                self.visualizer.set_background_source("video", bg_path=bg_file)
            elif source == "webcam":
                self.visualizer.set_background_source("webcam", camera_index=0)
            else:
                messagebox.showerror("Error", "Invalid background source or file not found.")
                return

            # Update audio device
            audio_selection = audio_device_var.get()
            if audio_selection == "Default":
                self.visualizer.audio.device_index = None
            else:
                device_index = int(audio_selection.split(":")[0])
                self.visualizer.audio.device_index = device_index
            self.visualizer.audio.cleanup()
            self.visualizer.audio = LightweightAudioReader(device_index=self.visualizer.audio.device_index, chunk_size=256)

            messagebox.showinfo("Settings", "Settings saved successfully!")
            self.window.destroy()
            self.window = None

        save_button = tk.Button(self.window, text="Save Settings", command=save_settings)
        save_button.pack(pady=20)

        self.window.mainloop()



def create_system_tray_icon(visualizer, settings_gui):
    def on_quit(icon, item):
        icon.stop()
        visualizer.paused = True
        cleanup(visualizer)
        os._exit(0)

    def on_toggle_help(icon, item):
        visualizer.show_help = not visualizer.show_help

    def on_pause_resume(icon, item):
        visualizer.toggle_pause()

    def on_open_settings(icon, item):
        settings_gui.show()

    # Create an icon image (64x64) with a simple design
    img = Image.new('RGB', (64, 64), color=(0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((16, 16, 48, 48), fill=(0, 128, 255))

    icon = pystray.Icon("WaveVisualizer", img, "Wave Pond PyTorch", menu=pystray.Menu(
        pystray.MenuItem("Settings", on_open_settings),
        pystray.MenuItem("Toggle Help", on_toggle_help),
        pystray.MenuItem("Pause/Resume", on_pause_resume),
        pystray.MenuItem("Exit", on_quit)
    ))
    threading.Thread(target=icon.run, daemon=True).start()


class SystemTrayIcon:
    def __init__(self, visualizer, settings_gui):
        self.visualizer = visualizer
        self.settings_gui = settings_gui
        create_system_tray_icon(self.visualizer, self.settings_gui)


def cleanup(visualizer):
    if hasattr(visualizer, 'cap') and visualizer.cap:
        visualizer.cap.release()
    if hasattr(visualizer, 'audio'):
        visualizer.audio.cleanup()
    pygame.quit()


def main():
    try:
        # Initialize audio device manager
        pyaudio_instance = pyaudio.PyAudio()
        audio_manager = AudioDeviceManager(pyaudio_instance)

        # Create visualizer with default audio device
        visualizer = WaveVisualizer(
            width=854,
            height=480,
            camera_index=0,
            audio_device_index=None  # Default device
        )

        # Create settings GUI and assign it to visualizer
        settings_gui = SettingsGUI(visualizer)
        visualizer.settings_gui = settings_gui  # Assign settings GUI to visualizer

        # Create system tray icon
        tray_icon = SystemTrayIcon(visualizer, settings_gui)

        # Run the visualizer
        visualizer.run()

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()  # Print full error trace
    finally:
        if 'visualizer' in locals():
            cleanup(visualizer)
        if 'audio_manager' in locals():
            audio_manager.cleanup()
        if 'pyaudio_instance' in locals():
            pyaudio_instance.terminate()

if __name__ == "__main__":
    main()
