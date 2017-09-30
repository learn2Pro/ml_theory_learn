import pygame
from pygame.locals import *


# 定义Player对象 调用super赋予它属性和方法
# 我们画在屏幕上的surface 现在是player的一个属性
class Player(pygame.sprite.Sprite):
    def __init__(self):
        super(Player, self).__init__()
        self.surf = pygame.Surface((75, 25))
        self.surf.fill((255, 255, 255))
        self.rect = self.surf.get_rect()


pygame.init()
screen = pygame.display.set_mode((800, 600))
player = Player()
running = True
x = 100
# 主循环！
while running:
    # for 循环遍历事件队列
    for event in pygame.event.get():
        # 检测 KEYDOWN 事件: KEYDOWN 是 pygame.locals 中定义的常量，pygame.locals文件开始已经导入
        if event.type == KEYDOWN:
            # 如果按下 Esc 那么主循环终止
            if event.key == K_ESCAPE:
                running = False
            if event.key == K_SPACE:
                print("space")
            if event.key == K_UP:
                x += 100
                screen.blit(player.surf, (x, 300))
                pygame.display.flip()
        # 检测 QUIT : 如果 QUIT, 终止主循环
        elif event.type == QUIT:
            running = False
