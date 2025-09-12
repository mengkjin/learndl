import requests
import zipfile
import shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

FONT_DIR = Path(__file__).parent.parent / "fonts"
IMAGE_DIR = Path(__file__).parent.parent / "images"

def get_font() -> Path:
    """
    download desired font from dafont.com
    """

    font_file = "animeace2bb_tt/animeace2_bld.ttf"
    font_path = FONT_DIR / font_file.split("/")[-1]
    temp_dir = FONT_DIR / "temp"
    
    if font_path.exists(): 
        return font_path
    temp_dir.mkdir(exist_ok=True)
    url = "https://dl.dafont.com/dl/?f=anime_ace_bb"
    response = requests.get(url, allow_redirects=True)
    
    if response.status_code == 200:
        zip_path = temp_dir / "anime_ace_bb.zip"
        
        with open(zip_path, 'wb') as f:
            f.write(response.content)
    else:
        print(f"download failed: {response.status_code}")
        print("please download the font file and put it in the fonts directory")
        raise Exception("download failed")
    
    # 2. unzip
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    if not temp_dir.joinpath(font_file).exists():
        # find other font files
        font_files = list(temp_dir.rglob("*.ttf")) + list(temp_dir.rglob("*.otf"))
        if font_files:
            font_file_path = font_files[0]
            return font_file_path
        else:
            raise Exception("no font file found")
    else:
        shutil.copy(temp_dir.joinpath(font_file), font_path)

    # 3. remove temp file
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    return font_path

def draw_rocket():
    """draw a realistic rocket"""
    
    # create transparent background
    width, height = 200, 300
    image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    
    # define colors
    rocket_body = (255, 100, 100)      # rocket body - red
    rocket_tip = (255, 200, 200)       # rocket tip - light red
    rocket_fin = (200, 80, 80)         # rocket fin - dark red
    window_color = (100, 200, 255)     # window - blue
    flame_color = (255, 165, 0)        # flame - orange
    
    # rocket body (ellipse)
    body_rect = [60, 80, 140, 220]
    draw.ellipse(body_rect, fill=rocket_body, outline=(200, 80, 80), width=2)
    
    # rocket tip (triangle) 
    tip_points = [(100, 40), (70, 80), (130, 80)]
    draw.polygon(tip_points, fill=rocket_tip, outline=(200, 80, 80), width=2)
    
    # rocket fin (triangle)
    # left fin
    left_fin = [(60, 200), (40, 240), (80, 220)]
    draw.polygon(left_fin, fill=rocket_fin, outline=(150, 60, 60), width=2)
    
    # right fin
    right_fin = [(140, 200), (160, 240), (120, 220)]
    draw.polygon(right_fin, fill=rocket_fin, outline=(150, 60, 60), width=2)
    
    # rocket window (circle)
    window_center = (100, 130)
    window_radius = 15
    draw.ellipse([window_center[0]-window_radius, window_center[1]-window_radius,
                   window_center[0]+window_radius, window_center[1]+window_radius], 
                  fill=window_color, outline=(80, 160, 200), width=2)
    
    # add window highlight
    highlight_radius = 8
    draw.ellipse([window_center[0]-highlight_radius+3, window_center[1]-highlight_radius+3,
                   window_center[0]+highlight_radius-3, window_center[1]+highlight_radius-3], 
                  fill=(150, 220, 255), outline=None)
    
    # rocket flame
    flame_points = [
        (80, 220),   # left flame
        (70, 260),
        (90, 240),
        (100, 220),  # middle flame
        (95, 260),
        (105, 240),
        (120, 220),  # right flame
        (130, 260),
        (110, 240)
    ]
    
    # draw flame
    for i in range(0, len(flame_points), 3):
        if i + 2 < len(flame_points):
            flame_triangle = [flame_points[i], flame_points[i+1], flame_points[i+2]]
            draw.polygon(flame_triangle, fill=flame_color, outline=(200, 120, 0), width=1)
    
    # add some decorative details
    # lines on rocket body
    draw.line([(70, 100), (130, 100)], fill=(200, 80, 80), width=2)
    draw.line([(70, 180), (130, 180)], fill=(200, 80, 80), width=2)
    
    # decorative details on rocket tip
    draw.ellipse([95, 50, 105, 60], fill=(255, 255, 255), outline=None)
    
    return image

def create_icon(recreate = False):
    """create an icon"""
    assert not recreate , "recreate must be False"
    if not recreate and IMAGE_DIR.joinpath("icon.png").exists(): 
        return
    image = draw_rocket()
    image.save(IMAGE_DIR / "icon.png")
    return image

def create_banner(recreate = False):
    """
    generate image with font
    """
    if not recreate and IMAGE_DIR.joinpath("banner.png").exists(): 
        return

    # text and font
    text = "Learn  Deep  Learning"
    font_path = get_font()
    font_size = 100
    font = ImageFont.truetype(font_path, font_size) 
    
    # calculate text size
    temp_img = Image.new("RGBA", (1, 1))
    temp_draw = ImageDraw.Draw(temp_img)
    text_bbox = temp_draw.textbbox((0, 0), text, font=font)
    
    # get text size
    text_width :int = int(text_bbox[2] - text_bbox[0])
    text_height :int = int(text_bbox[3] - text_bbox[1]) + 50 # add padding
    
    # add padding
    padding_x = 10
    padding_y = 20
    
    # create canvas
    canvas_width :int = int(text_width + padding_x)
    canvas_height :int = int(text_height + padding_y)
    image = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))
    ImageDraw.Draw(image)
    
    # rainbow gradient
    colors = [(255, 0, 0), (255, 127, 0), (255, 255, 0),
                (0, 255, 0), (0, 0, 255), (75, 0, 130), (148, 0, 211)]
    gradient = np.zeros((text_height, text_width, 3), dtype=np.uint8)
    
    for x in range(text_width):
        t = x / text_width * (len(colors) - 1)
        i = int(t)
        f = t - i
        c1, c2 = colors[i], colors[min(i + 1, len(colors) - 1)]
        gradient[:, x] = [int(c1[j] + f * (c2[j] - c1[j])) for j in range(3)]
    
    gradient_img = Image.fromarray(gradient, 'RGB')
    
    # create text mask
    mask = Image.new("L", (text_width, text_height), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.text((0, 0), text, font=font, fill=255)
    
    # calculate center position
    x_offset = (canvas_width - text_width) // 2
    y_offset = (canvas_height - text_height) // 2
    
    # paste gradient to image
    image.paste(gradient_img, (x_offset, y_offset), mask)
    
    # save image
    image.save(IMAGE_DIR / "banner.png")

def create_logo(icon_path="icon.png", banner_path="banner.png", 
                output_path="logo.png", spacing=True, alignment='center', recreate=False):
    """
    - icon_path: icon path
    - banner_path: banner path  
    - output_path: output logo path
    - spacing: spacing between icon and banner
    - alignment: vertical alignment ('top', 'center', 'bottom')
    """
    if not recreate and IMAGE_DIR.joinpath(output_path).exists(): 
        return

    create_icon(False)
    create_banner(recreate)

    icon = Image.open(IMAGE_DIR / icon_path)
    banner = Image.open(IMAGE_DIR / banner_path)
    
    # convert to RGBA mode
    if icon.mode != 'RGBA':
        icon = icon.convert('RGBA')
    if banner.mode != 'RGBA':
        banner = banner.convert('RGBA')
    
    # get size
    icon_width, icon_height = icon.size
    banner_width, banner_height = banner.size
    
    # calculate target height
    target_height = max(icon_height, banner_height)
    
    # resize image
    if icon_height != target_height:
        icon_ratio = target_height / icon_height
        new_icon_width = int(icon_width * icon_ratio)
        icon = icon.resize((new_icon_width, target_height), Image.Resampling.LANCZOS)
    
    if banner_height != target_height:
        banner_ratio = target_height / banner_height
        new_banner_width = int(banner_width * banner_ratio)
        banner = banner.resize((new_banner_width, target_height), Image.Resampling.LANCZOS)
    
    # get adjusted size
    icon_width, icon_height = icon.size
    banner_width, banner_height = banner.size
    
    # create canvas
    total_width = icon_width + spacing * icon_width + banner_width
    logo = Image.new('RGBA', (total_width, target_height), (0, 0, 0, 0))
    
    # calculate vertical position
    if alignment == 'top':
        icon_y = 0
        banner_y = 0
    elif alignment == 'center':
        icon_y = (target_height - icon_height) // 2
        banner_y = (target_height - banner_height) // 2
    else:  # bottom
        icon_y = target_height - icon_height
        banner_y = target_height - banner_height
    
    # paste image
    logo.paste(icon, (0, icon_y), icon)
    logo.paste(banner, (icon_width + spacing * icon_width, banner_y), banner)
    
    # save image
    logo.save(IMAGE_DIR / output_path, 'PNG')

    return logo

def get_logo() -> dict:
    """get the logo"""
    create_logo()
    return {
        "image" : IMAGE_DIR / "logo.png" ,
        "icon_image" : IMAGE_DIR / "icon.png"
    }

if __name__ == "__main__":
    create_logo(recreate=True)