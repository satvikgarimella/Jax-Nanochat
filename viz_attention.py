from manim import *
import numpy as np

class AttentionVisualizer(Scene):
    def construct(self):
        matrix = np.load("real_attn.npy")
        title = Text("Causal Attention", font_size=32).to_edge(UP)
        self.add(title)

        grid = VGroup(*[
            Square(side_length=0.5, fill_opacity=0, stroke_opacity=0.2, stroke_color=GRAY)
            for _ in range(100)
        ]).arrange_in_grid(rows=10, cols=10, buff=0.05)
        self.play(Create(grid), run_time=1.5)

        for row in range(10):
            anims = []
            for col in range(10):
                if col <= row:
                    val = matrix[row, col]
                    anims.append(
                        grid[row * 10 + col].animate.set_fill(BLUE, opacity=val)
                        .set_stroke(BLUE_A, opacity=1, width=2)
                    )
            self.play(*anims, run_time=0.4)
        self.wait(2)
