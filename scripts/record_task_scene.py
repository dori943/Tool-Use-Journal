"""Record an explicitly labelled local scene preview without motion planning."""
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("task")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import run_m1 as m1
    import numpy as np
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont

    np.random.seed(0)
    spec = m1.task_spec(args.task)
    env = m1.suite.make(env_name=spec["env_name"], robots="UR5e",
        use_camera_obs=False, has_offscreen_renderer=True, has_renderer=False,
        ignore_done=True, seed=0, render_camera="agentview")
    try:
        env.reset()
        width, height, fps = 1280, 720, 24
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 28)
        small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 22)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(str(args.output), fps=fps, codec="libx264",
                quality=8, macro_block_size=1, ffmpeg_params=["-movflags", "+faststart"]) as writer:
            for i in range(fps * 8):
                # Camera inspection only: task dynamics and robot are not executed.
                frame = env.sim.render(
                    width=width, height=height, camera_name="agentview"
                )[::-1].copy()
                img = Image.fromarray(frame)
                draw = ImageDraw.Draw(img)
                draw.rectangle((0, 0, width, 88), fill="#13202d")
                draw.text((24, 10), f"{args.task.upper()} | M1 SCENE PREVIEW | SEED 0", font=font, fill="white")
                draw.text((24, 50), "Diagonal-fit packing: objects, box and lid", font=small, fill="#c6d8e6")
                draw.rectangle((0, height-48, width, height), fill="#13202d")
                draw.text((24, height-37), "Scene inspection only | M2-M5 task execution not performed", font=small, fill="#ffd38a")
                writer.append_data(np.asarray(img))
                if i == fps * 4:
                    img.save(args.output.with_suffix(".png"))
        print(args.output, flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
