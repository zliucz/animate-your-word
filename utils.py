
import torch
import pydiffvg
import numpy as np
import imageio
import collections.abc
import os
import matplotlib.pyplot as plt

import os.path as osp
import cairosvg
from ttf import font_string_to_svgs, normalize_letter_size
import save_svg

# ==================================
# ====== video realted utils =======
# ==================================

def check_and_create_dir(path):
    pathdir = osp.split(path)[0]
    if osp.isdir(pathdir):

        pass
    else:

        os.makedirs(pathdir)

def save_image(img, filename, gamma=1):
    check_and_create_dir(filename)
    imshow = img.detach().cpu()
    pydiffvg.imwrite(imshow, filename, gamma=gamma)

def check_length(letter, font):
    if font == "KaushanScript-Regular":
        if letter in ['A','D','P','R','d','e','o','q','s','K','i','j']:
               length=2
        elif letter in ['B']:
               length=3
        else:
               length=1
    elif font in ["Roboto-Light","Roboto-Bold"]:
        if letter in ['A','D','O','P','Q','R','a','b','d','e','g','i','j','o','p','q']:
               length=2
        elif letter in ['B']:
               length=3
        else:
               length=1
    elif font == "segoepr":
        if letter in ['A','D','O','P','Q','a','b','d','e','g','i','j','o','p','q']:
               length=2
        else:
               length=1
    else:   # other fonts
        length=0


    return length

def get_letter_ids(letter, word, shape_groups, font):
    id = 0
    for i,l in enumerate(word):
        if l != letter:
            id+=check_length(l, font)
        else:
            start_shape_ind=id
            end_shape_ind=id+check_length(l, font)
            return list(range(start_shape_ind, end_shape_ind))

           
def compute_last_frame_size(experiment_dir,step,i):
    svg_result = os.path.join(experiment_dir, f"svg_step{step}/frame{i:03d}.svg")
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(svg_result)
    return shapes

def combine_word(word, letter, font,experiment_dir,step,i,resize_by_last_frame=False,last_frame_shapes=None):
    word_svg_scaled = f"./data/init/{font}_{word}_scaled.svg"

    canvas_width_word, canvas_height_word, shapes_word, shape_groups_word = pydiffvg.svg_to_scene(word_svg_scaled)

    letter_ids = []
    for l in letter:
        letter_ids += get_letter_ids(l, word, shape_groups_word, font)
    w_min, w_max = min([torch.min(shapes_word[ids].points[:, 0]) for ids in letter_ids]), max(
        [torch.max(shapes_word[ids].points[:, 0]) for ids in letter_ids])
    h_min, h_max = min([torch.min(shapes_word[ids].points[:, 1]) for ids in letter_ids]), max(
        [torch.max(shapes_word[ids].points[:, 1]) for ids in letter_ids])

    c_w = (-w_min + w_max) / 2
    c_h = (-h_min + h_max) / 2
    output_subfolder="svg_step"+str(step)

    svg_result = os.path.join(experiment_dir, f"svg_step{step}/frame{i:03d}.svg")
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(svg_result)

    out_w_min, out_w_max = min([torch.min(p.points[:, 0]) for p in shapes]), max(
        [torch.max(p.points[:, 0]) for p in shapes])
    out_h_min, out_h_max = min([torch.min(p.points[:, 1]) for p in shapes]), max(
        [torch.max(p.points[:, 1]) for p in shapes])

    if resize_by_last_frame:

        out_w_min, out_w_max = min([torch.min(p.points[:, 0]) for p in last_frame_shapes]), max(
            [torch.max(p.points[:, 0]) for p in last_frame_shapes])
        out_h_min, out_h_max = min([torch.min(p.points[:, 1]) for p in last_frame_shapes]), max(
            [torch.max(p.points[:, 1]) for p in last_frame_shapes])
    out_c_w = (-out_w_min + out_w_max) / 2
    out_c_h = (-out_h_min + out_h_max) / 2

    scale_canvas_w = (w_max - w_min) / (out_w_max - out_w_min)
    scale_canvas_h = (h_max - h_min) / (out_h_max - out_h_min)

    if scale_canvas_h > scale_canvas_w:
        wsize = int((out_w_max - out_w_min) * scale_canvas_h)
        scale_canvas_w = wsize / (out_w_max - out_w_min)
        shift_w = -out_c_w * scale_canvas_w + c_w
    else:
        hsize = int((out_h_max - out_h_min) * scale_canvas_w)
        scale_canvas_h = hsize / (out_h_max - out_h_min)
        shift_h = -out_c_h * scale_canvas_h + c_h

    for num, p in enumerate(shapes):
        p.points[:, 0] = p.points[:, 0] * scale_canvas_w
        p.points[:, 1] = p.points[:, 1] * scale_canvas_h
        if scale_canvas_h > scale_canvas_w:
            p.points[:, 0] = p.points[:, 0] - out_w_min * scale_canvas_w + w_min + shift_w
            p.points[:, 1] = p.points[:, 1] - out_h_min * scale_canvas_h + h_min
        else:
            p.points[:, 0] = p.points[:, 0] - out_w_min * scale_canvas_w + w_min
            p.points[:, 1] = p.points[:, 1] - out_h_min * scale_canvas_h + h_min + shift_h

    for j, s in enumerate(letter_ids):
        shapes_word[s] = shapes[j]
    directory1=f"{experiment_dir}/svg_step{step}/concate_svg"
    directory2=f"{experiment_dir}/svg_step{step}/concate_png"
    if not os.path.exists(directory1):
        os.makedirs(directory1)
    if not os.path.exists(directory2):
        os.makedirs(directory2)
    
    
    save_svg.save_svg(
        f"{experiment_dir}/svg_step{step}/concate_svg/{font}_{word}_{letter}_{i:03d}.svg", canvas_width, canvas_height, shapes_word,
        shape_groups_word,word,font)

    render = pydiffvg.RenderFunction.apply
    scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, canvas_height, shapes_word, shape_groups_word)
    img = render(canvas_width, canvas_height, 2, 2, 0, None, *scene_args)
    img = img[:, :, 3:4] * img[:, :, :3] + \
               torch.ones(img.shape[0], img.shape[1], 3, device="cuda:0") * (1 - img[:, :, 3:4])
    img = img[:, :, :3]
    save_image(img, f"{experiment_dir}/svg_step{step}/concate_png/{font}_{word}_{letter}_{i:03d}.png")

def frames_to_vid(video_frames, output_vid_path):
    """
    Saves an mp4 file from the given frames
    """
    writer = imageio.get_writer(output_vid_path, fps=8)
    for im in video_frames:
        writer.append_data(im)
    writer.close()

def render_frames_to_tensor(frames_shapes, frames_shapes_grous, w, h, render, device):
    """
    Given a list with the points parameters, render them frame by frame and return a tensor of the rasterized frames ([16, 256, 256, 3])
    """
    frames_init = []
    for i in range(len(frames_shapes)):
        shapes = frames_shapes[i]
        shape_groups = frames_shapes_grous[i]
        scene_args = pydiffvg.RenderFunction.serialize_scene(w, h, shapes, shape_groups)
        cur_im = render(w, h, 2, 2, 0, None, *scene_args)
    
        cur_im = cur_im[:, :, 3:4] * cur_im[:, :, :3] + \
               torch.ones(cur_im.shape[0], cur_im.shape[1], 3, device=device) * (1 - cur_im[:, :, 3:4])
        cur_im = cur_im[:, :, :3]
        frames_init.append(cur_im)
    return torch.stack(frames_init)

def save_mp4_from_tensor(frames_tensor, output_vid_path):
    # input is a [16, 256, 256, 3] video
    frames_copy = frames_tensor.clone()
    frames_output = []
    for i in range(frames_copy.shape[0]):
        cur_im = frames_copy[i]
        cur_im = cur_im[:, :, :3].detach().cpu().numpy()
        cur_im = (cur_im * 255).astype(np.uint8)
        frames_output.append(cur_im)
    frames_to_vid(frames_output, output_vid_path=output_vid_path)

def save_png_to_video(input_folder, output_video):
    file_paths = [os.path.join(input_folder, f) for f in sorted(os.listdir(input_folder)) if os.path.isfile(os.path.join(input_folder, f))]
    file_paths.sort()
    
    with imageio.get_reader(file_paths[0]) as img_reader:
        meta_data = img_reader.get_meta_data()
        fps = 8
        size = (320,320)

    with imageio.get_writer(output_video, fps=fps) as video_writer:
        for file_path in file_paths:
            image = imageio.imread(file_path)
            video_writer.append_data(image)

    print(f"Video saved to {output_video}")


def save_vid_svg(frames_svg, output_folder, step, w, h):
    if not os.path.exists(f"{output_folder}/svg_step{step}"):
        os.mkdir(f"{output_folder}/svg_step{step}")
    for i in range(len(frames_svg)):
        pydiffvg.save_svg(f"{output_folder}/svg_step{step}/frame{i:03d}.svg", w, h, frames_svg[i][0], frames_svg[i][1])

def svg_to_png(path_to_svg_files, dest_path):
    svgs = sorted(os.listdir(path_to_svg_files))
    filenames = [k for k in svgs if ".svg" in k]
    for filename in filenames:        
        dest_path_ = f"{dest_path}/{os.path.splitext(filename)[0]}.png"
        cairosvg.svg2png(url=f"{path_to_svg_files}/{filename}", write_to=dest_path_, scale=4, background_color="white")
  
def save_gif_from_pngs(path_to_png_files, gif_dest_path):
    pngs = sorted(os.listdir(path_to_png_files))
    filenames = [k for k in pngs if "png" in k]
    images = []
    length = len(filenames)
    for i,filename in enumerate(filenames):
        if i<length-1 and i<24:
            im = imageio.imread(f"{path_to_png_files}/{filename}")
            images.append(im)
        else:
            pass
    imageio.mimsave(f"{gif_dest_path}", images, 'GIF', loop=4, fps=8)

def save_hq_video(path_to_outputs, iter_=1000):
    dest_path_png = f"{path_to_outputs}/png_files_ite{iter_}"
    os.makedirs(dest_path_png, exist_ok=True)

    svg_to_png(f"{path_to_outputs}/svg_logs/svg_step{iter_}", dest_path_png)

    gif_dest_path = f"{path_to_outputs}/HQ_gif.gif"
    save_gif_from_pngs(dest_path_png, gif_dest_path)
    print(f"GIF saved to [{gif_dest_path}]")

def save_hq_video_concate(path_to_outputs, iter_=1000):
    dest_path_png = f"{path_to_outputs}/png_logs/{iter_}"
    os.makedirs(dest_path_png, exist_ok=True)

    svg_to_png(f"{path_to_outputs}/svg_logs/svg_step{iter_}/concate_svg", dest_path_png)

    gif_dest_path = f"{path_to_outputs}/mp4_logs/{iter_}_HQ_gif.gif"
    save_gif_from_pngs(dest_path_png, gif_dest_path)
    print(f"GIF saved to [{gif_dest_path}]")

def normalize_tensor(tensor: torch.Tensor, canvas_size: int = 256):
    range_value = float(canvas_size)# / 2
    normalized_tensor = tensor / range_value
    return normalized_tensor

def preprocess(font, word, letter, level_of_cc=1):

    if level_of_cc == 0:
        target_cp = None
    else:
        target_cp = {"A": 120, "B": 120, "C": 100, "D": 100,
                     "E": 120, "F": 120, "G": 120, "H": 120,
                     "I": 35, "J": 80, "K": 100, "L": 80,
                     "M": 100, "N": 100, "O": 100, "P": 120,
                     "Q": 120, "R": 130, "S": 110, "T": 90,
                     "U": 100, "V": 100, "W": 100, "X": 130,
                     "Y": 120, "Z": 120,
                     "a": 120, "b": 120, "c": 100, "d": 100,
                     "e": 120, "f": 120, "g": 120, "h": 120,
                     "i": 35, "j": 80, "k": 100, "l": 80,
                     "m": 100, "n": 100, "o": 100, "p": 120,
                     "q": 120, "r": 130, "s": 110, "t": 90,
                     "u": 100, "v": 100, "w": 100, "x": 130,
                     "y": 120, "z": 120
                     }
        target_cp = {k: v * level_of_cc for k, v in target_cp.items()}

    print(f"======= {font} =======")
    font_path = f"data/fonts/{font}.ttf"
    init_path = f"data/init"
    subdivision_thresh = None
    font_string_to_svgs(init_path, font_path, word, target_control=target_cp,
                        subdivision_thresh=subdivision_thresh)
    normalize_letter_size(init_path, font_path, word)

    # optimaize two adjacent letters
    if len(letter) > 1:
        subdivision_thresh = None
        font_string_to_svgs(init_path, font_path, letter, target_control=target_cp,
                            subdivision_thresh=subdivision_thresh)
        normalize_letter_size(init_path, font_path, letter)

    print("Done preprocess")

def update(d, u):
    """https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth"""
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

if __name__ == '__main__':
    parent_directory = "ablation_videos"  # Adjust this to your parent directory path

    for root, dirs, files in os.walk(parent_directory):
        if root.endswith('svg_step1000'):
            save_hq_video(root.rsplit('/svg_logs', 1)[0], '1000')
