import sys
import pygame
import socket
from pygame.locals import *

# 初始化 Pygame
pygame.init()

# 設置畫面寬高和顏色
screen_info = pygame.display.Info()
screen_width, screen_height = screen_info.current_w, screen_info.current_h

# 初始化 Pygame 畫面
screen = pygame.display.set_mode((screen_width, screen_height), flags=pygame.DOUBLEBUF)
x_center, y_center = screen_width // 2, screen_height // 2
pygame.display.set_caption("SSVEP Stimulus")
pygame.mouse.set_visible(False)

# 設置頻閃頻率（以 Hz 為單位）
fps = 50
frequencies = [8.33, 10.00, 12.50]
flashing_states = [False] * len(frequencies)
frequency_timers = [0] * len(frequencies)

# 設置時鐘
clock = pygame.time.Clock()

# 設置物件參數
background_color = (0, 0, 0)
object_color = (189, 192, 186)
offset = screen_height // 8 + 15
inter_offset = 70
top_circle = (x_center, 3 / 4 * screen_height - (3 ** 0.5) / 4 * screen_width + offset + inter_offset)
circle_position = (top_circle, (x_center // 2, y_center + y_center // 2 + offset - inter_offset),
                   (x_center + x_center // 2, y_center + y_center // 2 + offset - inter_offset))
cross_thickness, cross_len = 5, 10
cross_position = (((x_center - cross_len, y_center), (x_center + cross_len, y_center)),
                  ((x_center, y_center - cross_len), (x_center, y_center + cross_len)))
text_position = (x_center, y_center)

# 設置 image
image = pygame.image.load("./source/black_white_checkered_pattern.png")
target_width = 300
scale_factor = target_width / image.get_width()
target_height = int(image.get_height() * scale_factor)
image = pygame.transform.scale(image, (target_width, target_height))

# 設置 record 參數
cycle = 10
classes = ['front', 'left', 'right']
pipline_period = [2, 5, 2]
pipline_timer, pipline_index = 0, 0


def toggle_flash(index):
    flashing_states[index] = not flashing_states[index]


def draw_cross():
    pygame.draw.line(screen, object_color, cross_position[0][0], cross_position[0][1], cross_thickness)
    pygame.draw.line(screen, object_color, cross_position[1][0], cross_position[1][1], cross_thickness)


def draw_ssvep():
    for idx, (postion, frequency) in enumerate(zip(circle_position, frequencies)):
        target_frames = int(fps // frequency)
        frame_open = target_frames // 2 if target_frames % 2 == 0 else target_frames // 2 + 1
        frame_close = target_frames - frame_open

        # 判斷是否閃爍
        if flashing_states[idx]:
            image_rect = image.get_rect(center=postion)
            screen.blit(image, image_rect)

        # 更新計時器
        frequency_timers[idx] += 1

        # 根據頻率控制閃爍狀態
        frame_cont = frame_open if flashing_states[idx] else frame_close
        if frequency_timers[idx] == frame_cont:
            toggle_flash(idx)
            frequency_timers[idx] = 0


# def draw_ssvep():
#     for idx, (postion, frequency) in enumerate(zip(circle_position, frequencies)):
#         if idx != 1:
#             continue

#         target_frames = int(fps // frequency)
#         frame_open = target_frames // 2 if target_frames % 2 == 0 else target_frames // 2 + 1
#         frame_close = target_frames - frame_open

#         # 判斷是否閃爍
#         if flashing_states[idx]:
#             image_rect = image.get_rect(center=(x_center, y_center))
#             screen.blit(image, image_rect)

#         # 更新計時器
#         frequency_timers[idx] += 1

#         # 根據頻率控制閃爍狀態
#         frame_cont = frame_open if flashing_states[idx] else frame_close
#         if frequency_timers[idx] == frame_cont:
#             toggle_flash(idx)
#             frequency_timers[idx] = 0

def draw_fixation():
    for postion in circle_position:
        image_rect = image.get_rect(center=postion)
        screen.blit(image, image_rect)
    draw_cross()


def draw_hint(message):
    for postion in circle_position:
        image_rect = image.get_rect(center=postion)
        screen.blit(image, image_rect)
    font = pygame.font.Font(None, 48)
    text_surface = font.render(message, True, object_color)
    text_rect = text_surface.get_rect()
    text_rect.center = text_position
    screen.blit(text_surface, text_rect)


def draw_front():
    draw_hint('Please look at the top circle')


def draw_left():
    draw_hint('Please look at the left circle')


def draw_right():
    draw_hint('Please look at the right circle')


def draw_screen(piplines, period=None, pipline_timer=None, pipline_index=None, need_label=None, client_socket=None):
    current_class_stat, init_timer = False, 0.0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # 刷新畫面
        screen.fill(background_color)

        if period is not None and pipline_timer is not None and pipline_index is not None and need_label is not None and client_socket is not None:
            piplines[pipline_index]()
            pipline_timer += clock.get_time()

            if need_label[pipline_index] != -1 and not current_class_stat:
                print(f'send label: {pipline_timer}')
                init_timer = pipline_timer
                client_socket.send(classes[need_label[pipline_index]].encode())
                current_class_stat = True

            if (pipline_timer - init_timer) / 1000 >= period[pipline_index]:
                if current_class_stat:
                    print(f'send close: {pipline_timer}')
                    client_socket.send('close labeling'.encode())
                    current_class_stat = False
                pipline_timer = 0
                pipline_index += 1

            if pipline_index > len(period) - 1:
                running = False

        else:
            for pipline in piplines:
                pipline()

        # 更新畫面
        pygame.display.flip()
        # clock.tick(fps)
        clock.tick_busy_loop(fps)


def create_client():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(("127.0.0.1", 7777))
    return client_socket


def real_time():
    realtime_piplines = [draw_ssvep]
    draw_screen(realtime_piplines)


def session_record():
    # 建立 client 並連接 server
    client_socket = create_client()
    client_socket.send('create session'.encode())

    piplines = [draw_front, draw_ssvep, draw_fixation, draw_left, draw_ssvep, draw_fixation, draw_right, draw_ssvep,
                draw_fixation]
    need_label = [-1, 0, -1, -1, 1, -1, -1, 2, -1]
    draw_screen(piplines * cycle, pipline_period * len(classes) * cycle, pipline_timer, pipline_index,
                need_label * cycle, client_socket)

    # 退出 Pygame、關閉 client 連接
    client_socket.send('close session'.encode())
    client_socket.close()


# session_record()
real_time()

pygame.quit()
sys.exit()