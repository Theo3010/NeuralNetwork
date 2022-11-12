import threading
import time
import pygame

from Neuralnetwork import NeuralNetwork
import matplotlib.pyplot as plt


magnification = 20

window_height = 28 * magnification
window_width = 28 * magnification


def draw(screen, color: tuple, pos: tuple) -> None:
    screen.fill(color, (pos[0], pos[1], magnification, magnification))
    pygame.display.flip()


def main():
    pixels = []
    pygame.init()
    NN = NeuralNetwork()
    pygame.display.set_caption("Drawning Game")

    screen = pygame.display.set_mode((window_width, window_height))
    clock = pygame.time.Clock()

    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        pos = pygame.mouse.get_pos()
        if pygame.mouse.get_pressed()[0]:
            draw(screen, (255, 0, 0), pos)

        if pygame.key.get_pressed()[pygame.K_SPACE]:
            screen.fill((0, 0, 0))
            pygame.display.flip()

        if pygame.key.get_pressed()[pygame.K_ESCAPE]:
            for x in range(0, window_width, magnification):
                for y in range(0, window_height, magnification):
                    pixels.append(1 if screen.get_at(
                        (y, x)) == (255, 0, 0) else 0)
            screen.fill((0, 0, 0))

            # convert pixels to 28x28
            pixels = [pixels[x:x+28] for x in range(0, len(pixels), 28)]

            # draw pixels
            for x in range(len(pixels)):
                for y in range(len(pixels[x])):
                    if pixels[x][y] == 1:
                        draw(screen, (255, 0, 0), (x, y))

            plt.imshow(pixels, cmap='gray')
            plt.show()

            # get prediction
            prediction = NN.get_prediction(pixels)
            print(prediction)

            pixels = []

            pygame.display.flip()

        clock.tick(60)
    pygame.quit()


class drawingGame:

    def __init__(self) -> None:
        self.screen = pygame.display.set_mode((window_width, window_height))
        self.clock = pygame.time.Clock()
        self.pixels = []
        self.nn = NeuralNetwork()
        self.done = False
        self.prediction = 0
        threading.Thread(target=self.get_prediction).start()

    def get_prediction(self) -> int:
        while not self.done:
            if self.pixels:
                pixels = [self.pixels[x:x+28]
                          for x in range(0, len(self.pixels), 28)]
                self.prediction = self.nn.get_prediction(pixels)
                print(self.prediction)
            time.sleep(.5)

    def main(self) -> None:
        pygame.init()
        while not self.done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.done = True

            pos = pygame.mouse.get_pos()
            if pygame.mouse.get_pressed()[0]:
                self.screen.fill(
                    (255, 0, 0), (pos[0], pos[1], magnification, magnification))
                pygame.display.flip()

            if pygame.key.get_pressed()[pygame.K_SPACE]:
                self.screen.fill((0, 0, 0))
                pygame.display.flip()

            self.pixels = []
            for x in range(0, window_width, magnification):
                for y in range(0, window_height, magnification):
                    self.pixels.append(1 if self.screen.get_at(
                        (y, x)) == (255, 0, 0) else 0)

            self.clock.tick(60)


drawingGame().main()
