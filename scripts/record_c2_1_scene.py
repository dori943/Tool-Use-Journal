"""Record real C2-1 physics before a motion plan is available."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reason', default='M1 complete / motion plan not executed')
    args = p.parse_args()
    import numpy as np
    import mujoco
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    import run_m1 as m1
    np.random.seed(0)
    env = m1.suite.make(env_name='C2_1_ObjectSorting', robots='UR5e',
        use_camera_obs=False, has_offscreen_renderer=True, has_renderer=False,
        camera_names='agentview', camera_heights=720, camera_widths=1280,
        ignore_done=True)
    env.reset()
    cid = env.sim.model.camera_name2id('agentview')
    env.sim.model.cam_fovy[cid] = 60
    m1.fit_camera_to_points(env, cid, m1.object_bound_points(env, m1.task_spec('c2_1')), margin=.76)
    env.sim.forward()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 25)
    small = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 20)
    fps = 24
    with imageio.get_writer(str(args.output), fps=fps, codec='libx264',
            quality=8, macro_block_size=1, ffmpeg_params=['-movflags', '+faststart']) as writer:
        for i in range(fps * 8):
            if i < fps * 5:
                for _ in range(round(1 / fps / env.sim.model.opt.timestep)):
                    env.sim.step()
            frame = env.sim.render(width=1280, height=720, camera_name='agentview')[::-1].copy()
            img = Image.fromarray(frame)
            draw = ImageDraw.Draw(img)
            draw.rectangle((0, 0, 1280, 88), fill='#13202d')
            draw.text((24, 12), 'C2-1 | M1 SCENE CAPTURE | SEED 0', font=font, fill='white')
            draw.text((24, 49), args.reason, font=small, fill='#ffd38a')
            draw.rectangle((0, 679, 1280, 720), fill='#13202d')
            draw.text((24, 688), 'Actual MuJoCo scene / passive physics only / no pick-and-place executed', font=small, fill='white')
            if i == fps * 4:
                img.save(args.output.with_suffix('.png'))
            writer.append_data(np.asarray(img))
    env.close()
    print(args.output, flush=True)

if __name__ == '__main__':
    main()
