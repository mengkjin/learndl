import os , argparse , subprocess
import ipywidgets as widgets

from typing import Any
from pathlib import Path
from IPython.display import display

def get_python_path():
    if os.name == 'posix':
        return 'python3.10'
    else:
        return 'python'

def get_terminal_cmd(cmd : str):
    if os.name == 'posix':
        return f'gnome-terminal -- bash -c "{cmd}; exec bash"'
    else:
        return f'start cmd /k {cmd}'

def get_argparse_dict():
    parser = argparse.ArgumentParser(description='Run daily update script.')
    parser.add_argument('--source', type=str, default='', help='Source of the script call')
    parser.add_argument('--email', type=int, default=1, help='Send email or not')
    parser.add_argument('--param', type=str, default='', help='Extra parameters for the script')
    args , _ = parser.parse_known_args()
    return args.__dict__

def get_script_header(file_path , verbose = False):
    header_dict = {}
    try:
        header_char = '#'
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                stripped_line = line.strip()
                if stripped_line.startswith('#!'): continue
                if stripped_line.startswith(header_char):
                    components = [s.strip() for s in stripped_line.removeprefix(header_char).strip().split(':')]
                    assert len(components) <= 2 , f'header format error : {stripped_line}'
                    header_dict[components[0]] = components[1] if len(components) == 2 else ''
                else:
                    break
        if 'description' not in header_dict:
            header_dict['description'] = file_path
    except FileNotFoundError:
        header_dict['disabled'] = True
        header_dict['description'] = 'file not found'
        header_dict['content'] = f'file not found : {file_path}'
        if verbose: print(f'file not found : {file_path}')
    except Exception as e:
        header_dict['disabled'] = True
        header_dict['description'] = 'read file error'
        header_dict['content'] = f'error info : {e}'
        if verbose: print(f'read file error : {e}')
    
    return header_dict

def get_caption(txt : str):
    return ' '.join([s.capitalize() for s in txt.split(' ')])

def run_script(script : str | Path , source = 'button' , email = 1 , param = ''):
    print(f'Script path  : {script}')
    print(f'Script source: {source}')

    main_cmd = f'{get_python_path()} {str(script)} --source {source} --email {email}'
    if param: main_cmd += f' --param {param}'
    terminal_cmd = get_terminal_cmd(main_cmd)
    process = subprocess.Popen(terminal_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True , encoding='utf-8')
    process.wait() 


def get_script_button(script : str | Path, prefix = '' , **kwargs):
    def func(b): run_script(script , **kwargs)

    header = get_script_header(script , verbose = False)

    desc = get_caption(header['description']) + prefix 
    disabled = header.get('disabled' , False)
    button = widgets.Button(description=desc, 
                            layout=widgets.Layout(width='auto', min_width='200px'), 
                            style={'button_color': 'lightgrey' , 'font_weight': 'bold'} , 
                            disabled=disabled)
    button.on_click(func)

    value = "<br>".join([value for key , value in header.items() if key in ['content' , 'TODO']])
    style = 'font-size: 12px; text-align: left; margin: 0.2px; padding: 0px; line-height: 1.5;'
    text = widgets.HTML(value=f'<p style="{style}"><em>{value}</em></p>' ,
                        layout=widgets.Layout(width='auto', min_width='300px' , border_top = '2px solid grey'))

    button = get_vertical_box(button, text , border = '2px solid grey')
    return button

def get_button_grids(*buttons , max_columns = 3):
    columns = min(len(buttons) , max_columns)
    layout = widgets.Layout(
        grid_template_columns=f'repeat({columns}, 1fr)',
        width='100%',
        border='2px solid grey', 
    )
    gridbox = widgets.GridBox(buttons, layout=layout)
    return gridbox

def get_vertical_box(*boxes , tight_layout = True , border = None):
    kwargs = {'padding' : '0px' , 'margin' : '0px' , 'spacing' : '0px'} if tight_layout else {}
    vbox_layout = widgets.Layout(
        display='flex',
        flex_flow='column',
        align_items='center',
        width='100%',
        border = border ,
        **kwargs
    )
    vbox = widgets.VBox(boxes, layout=vbox_layout)
    return vbox

def get_title(text : str , level : int):
    assert level in [1,2,3] , f'level must be 1,2,3'
    text = text.replace('_' , ' ').strip()
    def title_style(color : str | None = None , size : int | None = None , bold : bool = False):
        style = ''
        if color is not None: style += f'color: {color};'
        if size is not None:  style += f'font-size: {size}px;'
        if bold:              style += f'font-weight: bold;'
        style += 'text-align: center;'
        return style
    if level == 1:
        style = title_style(color = '#007bff' , size = 18 , bold = True)
    elif level == 2:
        style = title_style(size = 16 , bold = True)
    elif level == 3:
        style = title_style(size = 14 , bold = True)
    return widgets.HTML(f'<div style="{style}"><em>{get_caption(text)}</em></div>')

def folder_box(folder : str | Path , level : int , exclude_self = True):
    dir_boxes , scp_boxes = [] , []
    email = 1 if 'autorun' in str(folder) else 0
    self_path = Path(__file__).absolute() if exclude_self else None
    
    if level > 0: dir_boxes.append(get_title(f'{Path(folder).name} scripts' , min(level , 3)))

    for path in Path(folder).iterdir():
        if path.name.startswith(('.' , '_')): continue
        if path.is_dir(): 
            dir_boxes.append(folder_box(path , level + 1 , exclude_self = exclude_self))
        else:
            if self_path is not None and path.absolute() == self_path: continue
            scp_boxes.append(get_script_button(str(path) , email = email))

    if scp_boxes: dir_boxes.append(get_button_grids(*scp_boxes))
    return get_vertical_box(*dir_boxes)

if __name__ == '__main__':
    project = folder_box('src_runs' , 0)
    display(project)
