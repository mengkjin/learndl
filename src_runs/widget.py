import os , argparse , subprocess
import ipywidgets as widgets

from typing import Any
from pathlib import Path
from IPython.display import display

def python_path():
    if os.name == 'posix':
        return 'python3.10'
    else:
        return 'python'

def terminal_cmd(script : str | Path , params : dict = {}):
    if isinstance(script , Path): script = str(script.absolute())
    args = ' '.join([f'--{k} {str(v).replace(" ", "")}' for k , v in params.items()])

    if os.name == 'posix':
        cmd = f'python3.10 {script} {args}'
        return f'gnome-terminal -- bash -c "{cmd}; exec bash"'
    else:
        cmd = f'python {script} {args}'
        return f'start cmd /k {cmd}'
    
def run_script(script : str | Path , **kwargs):
    print(f'Script path : {script}')
    print(f'Script args : {kwargs}')

    cmd = terminal_cmd(script , kwargs)
    process = subprocess.Popen(cmd, shell=True, encoding='utf-8')
    process.wait() 

def argparse_dict(**kwargs):
    parser = argparse.ArgumentParser(description='Run daily update script.')
    parser.add_argument('--source', type=str, default=kwargs.get('source',''), help='Source of the script call')
    parser.add_argument('--email', type=int, default=kwargs.get('email',1), help='Send email or not')
    parser.add_argument('--param', type=str, default=kwargs.get('param',''), help='Extra parameters for the script')
    args , _ = parser.parse_known_args()
    return args.__dict__

def script_header(file_path , verbose = False):
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

def caption(txt : str):
    return ' '.join([s.capitalize() for s in txt.split(' ')])

def layout_grids(*boxes , max_columns = 3):
    columns = min(len(boxes) , max_columns)
    layout = widgets.Layout(
        grid_template_columns=f'repeat({columns}, 1fr)',
        width='100%',
        border='2px solid grey', 
    )
    gridbox = widgets.GridBox(boxes, layout=layout)
    return gridbox

def layout_vertical(*boxes , tight_layout = True , border = None):
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
    return widgets.HTML(f'<div style="{style}"><em>{caption(text)}</em></div>')

def get_script_box(script : str | Path, prefix = '' , **kwargs):
    header = script_header(script , verbose = False)
    boxes = []

    desc = caption(header['description']) + prefix 
    disabled = header.get('disabled' , False)
    button = widgets.Button(description=desc, 
                            layout=widgets.Layout(width='auto', min_width='200px'), 
                            style={'button_color': 'lightgrey' , 'font_weight': 'bold'} , 
                            disabled=disabled)
    boxes.append(button)

    value = "<br>".join([value for key , value in header.items() if key in ['content' , 'TODO']])
    style = 'font-size: 12px; text-align: left; margin: 0.2px; padding: 0px; line-height: 1.5;'
    text = widgets.HTML(value=f'<p style="{style}"><em>{value}</em></p>' ,
                        layout=widgets.Layout(width='auto', min_width='300px' , border_top = '2px solid grey'))
    boxes.append(text)

    if header.get('param_input' , False):
        param_input = widgets.Textarea(
            placeholder=header.get('param_placeholder' , 'type parameters here...'),
            layout=widgets.Layout(width='auto', min_width='300px', height='50px')
        )
        boxes.append(param_input)
        def func(b):
            run_script(script, **kwargs , param = param_input.value.strip())
    else:
        def func(b):
            run_script(script, **kwargs)
    
    button.on_click(func)
    return layout_vertical(*boxes , border = '2px solid grey')

def get_folder_box(folder : str | Path , level : int , exclude_self = True):
    dir_boxes , file_boxes = [] , []
    email = 1 if 'autorun' in str(folder) else 0
    self_path = Path(__file__).absolute() if exclude_self else None
    
    if level > 0: dir_boxes.append(get_title(f'{Path(folder).name} scripts' , min(level , 3)))

    for path in sorted(Path(folder).iterdir(), key=lambda x: x.name):
        if path.name.startswith(('.' , '_')): continue
        if path.is_dir(): 
            dir_boxes.append(get_folder_box(path , level + 1 , exclude_self = exclude_self))
        else:
            if self_path is not None and path.absolute() == self_path: continue
            file_boxes.append(get_script_box(str(path) , email = email))

    if file_boxes: dir_boxes.append(layout_grids(*file_boxes))
    return layout_vertical(*dir_boxes)

def main():
    print('Initiating project interface...')
    project = get_folder_box('src_runs' , 0)
    print('Project interface initiated successfully.')
    display(project)

if __name__ == '__main__':
    main()
