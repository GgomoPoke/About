import pygame
import librosa
import numpy as np
import sys

def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr

def extract_beats(audio, sr):
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onset_times = librosa.frames_to_time(onsets, sr=sr)
    # print(onset_times)
    return beat_times, onset_env, tempo

def main(file_path):
    pygame.init()

    WIDTH, HEIGHT = 600, 400
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Visual')

    audio, sr = load_audio(file_path)
    # print(sr) Hz
    beat_times, onset_env, tempo = extract_beats(audio, sr)

    pygame.mixer.init(frequency=sr)
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()

    clock = pygame.time.Clock()
    running = True
    beat_index = 0
    active_beats = [] 
    base_radius = 50

    MORANDI_COLORS = [
        (255, 160, 130),  # 紅
        (255, 190, 100),  # 暖黃
        (120, 180, 250),  # 藍色
        (160, 215, 120),  # 綠色
        (160, 140, 210),  # 紫
        (240, 120, 160),  # 粉
    ]

    def get_color_by_strength(strength):
        num_colors = len(MORANDI_COLORS) - 1
        index = int(strength * num_colors)  
        frac = (strength * num_colors) - index

        index = min(index, num_colors - 1)
        c1 = MORANDI_COLORS[index]
        c2 = MORANDI_COLORS[index + 1]
        r = int(c1[0] + (c2[0] - c1[0]) * frac)
        g = int(c1[1] + (c2[1] - c1[1]) * frac)
        b = int(c1[2] + (c2[2] - c1[2]) * frac)

        return (r, g, b)

    latest_onset_strength = 0
    onset_values = []
    # beat_counter = 0  
    while running:
        screen.fill((20, 20, 20))

        current_time = pygame.mixer.music.get_pos() / 1000  

        frame_index = librosa.time_to_frames(current_time, sr=sr)
        frame_index = min(frame_index, len(onset_env) - 1)  
        onset_strength = onset_env[frame_index]

        strength_percentage = np.clip(onset_strength / np.max(onset_env), 0, 1)  
        color = get_color_by_strength(strength_percentage)

        onset_values.append(onset_strength)

        # 限制條形圖數據不會無限增長
        if len(onset_values) > WIDTH - 20: 
            onset_values.pop(0)  # 刪除最舊的數據

        # min_onset_value = np.min(onset_values)
        #max_onset_value = np.max(onset_values)

        #scale_width = WIDTH - 30  # 將刻度放在右邊
        #scale_height = 100

        #for i in range(0, scale_height + 1, 20):  
            #y_position = HEIGHT - 50 - i
            #scaled_value = min_onset_value + (max_onset_value - min_onset_value) * (i / scale_height) 
            #pygame.draw.line(screen, (255, 255, 255), (scale_width, y_position), (scale_width + 10, y_position), 2)  

            #font = pygame.font.SysFont("Arial", 12)
            #label = font.render(f"{scaled_value:.2f}", True, (255, 255, 255))
            #screen.blit(label, (scale_width + 12, y_position - 10)) 

        # 折線圖
        for i in range(1, len(onset_values)):
            start_pos = (
                20 + (i - 1) * (WIDTH - 40) / (len(onset_values) - 1),
                HEIGHT - 50 - (onset_values[i - 1] / np.max(onset_env)) * 100
            )
            end_pos = (
                20 + i * (WIDTH - 40) / (len(onset_values) - 1),
                HEIGHT - 50 - (onset_values[i] / np.max(onset_env)) * 100
            )

            start_pos = (float(start_pos[0]), float(start_pos[1]))
            end_pos = (float(end_pos[0]), float(end_pos[1]))
            if isinstance(start_pos, tuple) and isinstance(end_pos, tuple):
                pygame.draw.line(
                    screen,
                    color,
                    start_pos,
                    end_pos,
                    2 # 寬度
                )
       
        # beat time
        if beat_index < len(beat_times) and current_time >= beat_times[beat_index]:
            frame_index = librosa.time_to_frames(beat_times[beat_index], sr=sr)
            frame_index = min(frame_index, len(onset_env) - 1) 
            onset_strength = onset_env[frame_index]
            # latest_beat_time = beat_times[beat_index]
            latest_onset_strength = onset_strength
            strength_percentage = np.clip(onset_strength / np.max(onset_env), 0, 1)
            fade_duration =(strength_percentage)
           
            color = get_color_by_strength(strength_percentage) # 強度
            scale = 1.0 + strength_percentage * 1.5 
            next_beat_time = beat_times[beat_index + 1] if beat_index + 1 < len(beat_times) else current_time + 1
            #if beat_counter % 2 == 0: 
            active_beats.append({
                "start_time": current_time,
                "color": color,
                "scale": scale,
                "end_time": next_beat_time
            })
            # beat_counter += 1
            beat_index += 1  
            
        # 動畫
        for beat in active_beats[:]:
            elapsed = current_time - beat["start_time"]
            fade_duration = beat["end_time"] - beat["start_time"] + 0.6
            if elapsed > fade_duration:
                active_beats.remove(beat)
                continue
            
            radius = int(base_radius * beat["scale"] * (1 + 0.2 * elapsed))  
            alpha = max(0.6, int(255 * (1 - elapsed / 0.5)))  

            surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            pygame.draw.circle(surface, (*beat["color"], alpha), (WIDTH // 2, HEIGHT // 2), radius)
            screen.blit(surface, (0, 0))
            
        font = pygame.font.SysFont("Arial", 24)
        screen.blit(font.render(f"Time: {current_time:.2f}s", True, (255, 255, 255)), (20, 20))
        screen.blit(font.render(f"BPM: {int(tempo.item())}", True, (255, 255, 255)), (20, 50))
        screen.blit(font.render(f"Strength: {latest_onset_strength:.2f}", True, (255, 255, 255)), (20, 80))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        print("Error: You must provide the music file path as an argument.")
        sys.exit(1) 
    main(file_path)
