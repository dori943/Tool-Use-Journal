"""Label and encode the actual prefix replay for delivery."""
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

out = Path(__file__).resolve().parents[1] / 'output/c2_1'
source = out / 'c2_1_wide_raw.mp4'
target = out / 'c2_1_failure.mp4'
font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 24)
small = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 20)
with imageio.get_reader(str(source)) as reader:
    fps = reader.get_meta_data()['fps']
    with imageio.get_writer(str(target), fps=fps, codec='libx264', quality=8,
            macro_block_size=1, ffmpeg_params=['-movflags', '+faststart']) as writer:
        for index, frame in enumerate(reader):
            image = Image.fromarray(frame)
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, 1280, 71), fill='#13202d')
            draw.text((22, 9), 'C2-1 | M1-M4 COMPLETE | M5 PLANNING FAILED', font=font, fill='white')
            draw.text((22, 40), 'PICK_TOOL request rejected by Gemini: 400 INVALID_ARGUMENT', font=small, fill='#ffd38a')
            draw.rectangle((0, 681, 1280, 720), fill='#13202d')
            draw.text((22, 690), 'Actual controller replay: 2-finger gripper attachment only. Object transfer not executed.', font=small, fill='white')
            if index in (0, 120, 240, 314):
                image.save(out / f'failure_frame_{index}.jpg')
            writer.append_data(np.asarray(image))
print(target)
