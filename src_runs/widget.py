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

def terminal_cmd(script : str | Path , params : dict = {} , close_after_run = False):
    if isinstance(script , Path): script = str(script.absolute())
    args = ' '.join([f'--{k} {str(v).replace(" ", "")}' for k , v in params.items() if v != ''])
    cmd = f'{python_path()} {script} {args}'
    if os.name == 'posix':
        cmd = f'gnome-terminal -- bash -c "{cmd}"'
        if not close_after_run: cmd += '; exec bash'
    else:
        # cmd = f'start cmd /k {cmd}'
        if not close_after_run: 
            cmd = f'start cmd /k {cmd}'
        pass
    return cmd
    
def run_script(script : str | Path , close_after_run = False , **kwargs):
    cmd = terminal_cmd(script , kwargs , close_after_run = close_after_run)
    print(f'Script cmd : {cmd}')
    
    process = subprocess.Popen(cmd, shell=True, encoding='utf-8')
    if close_after_run:
        process.communicate()
    else:
        process.wait()

class ScriptRunner:
    def __init__(self , script : Path | str , **kwargs):
        self.script = Path(script).absolute()
        self.kwargs = kwargs
        self.header = self.script_header()
        self.text_area = None

    def __repr__(self):
        return f'ScriptRunner(script_path={self.script} , kwargs={self.kwargs})'

    def script_header(self , verbose = False):
        header_dict = {}
        try:
            header_char = '#'
            with open(self.script, 'r', encoding='utf-8') as file:
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
                header_dict['description'] = self.script
        except FileNotFoundError:
            header_dict['disabled'] = True
            header_dict['description'] = 'file not found'
            header_dict['content'] = f'file not found : {self.script}'
            if verbose: print(f'file not found : {self.script}')
        except Exception as e:
            header_dict['disabled'] = True
            header_dict['description'] = 'read file error'
            header_dict['content'] = f'error info : {e}'
            if verbose: print(f'read file error : {e}')

        return header_dict

    def __getitem__(self , key):
        return self.header[key]

    def get(self , key , default = None):
        return self.header.get(key , default)

    def get_func(self , text_area : widgets.Textarea | None = None , text_key = 'param'):
        params = {'email' : int(eval(self.get('email' , '0'))) , 
                  'close_after_run' : bool(eval(self.get('close_after_run' , 'False')))}
        params.update(self.kwargs)

        def func(b):
            if text_area is not None:
                params[text_key] = text_area.value.strip()
            run_script(self.script, **params)

        return func

    def button(self):
        button = widgets.Button(description=caption(self.header['description']), 
                                layout=widgets.Layout(width='auto', min_width='200px'), 
                                style={'button_color': 'lightgrey' , 'font_weight': 'bold'} , 
                                disabled=self.header.get('disabled' , False))
        return button
    
    def desc(self):
        infos =  '<br>'.join([self.get(key) for key in ['content' , 'TODO'] if key in self.header])
        style = 'font-size: 12px; text-align: left; margin: 0.2px; padding: 0px; line-height: 1.5;'
        desc = widgets.HTML(value=f'<p style="{style}"><em>{infos}</em></p>' ,
                            layout=widgets.Layout(width='auto', min_width='300px' , border_top = '2px solid grey'))
        return desc
    
    def input_area(self):
        if self.header.get('param_input' , False):
            input_area = widgets.Textarea(
                placeholder=self.header.get('param_placeholder' , 'type parameters here...'),
                layout=widgets.Layout(width='auto', min_width='300px', height='50px')
            )
        else:
            input_area = None
        return input_area
    
    def boxes(self):
        button = self.button()
        desc = self.desc()
        input_area = self.input_area()

        button.on_click(self.get_func(input_area))
        boxes = [button , desc]
        if input_area is not None: boxes.append(input_area)
        return boxes


def argparse_dict(**kwargs):
    parser = argparse.ArgumentParser(description='Run daily update script.')
    parser.add_argument('--source', type=str, default='', help='Source of the script call')
    parser.add_argument('--email', type=int, default=0, help='Send email or not')
    parser.add_argument('--param', type=str, default='', help='Extra parameters for the script')
    args , _ = parser.parse_known_args()
    return kwargs | args.__dict__

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

def get_script_box(script : str | Path , **kwargs):
    boxes = ScriptRunner(script , **kwargs).boxes()
    return layout_vertical(*boxes , border = '2px solid grey')

def get_folder_box(folder : str | Path , level : int , exclude_self = True):
    dir_boxes , file_boxes = [] , []
    self_path = Path(__file__).absolute() if exclude_self else None
    
    if level > 0: dir_boxes.append(get_title(f'{Path(folder).name} scripts' , min(level , 3)))

    for path in sorted(Path(folder).iterdir(), key=lambda x: x.name):
        if path.name.startswith(('.' , '_')): continue
        if path.is_dir(): 
            dir_boxes.append(get_folder_box(path , level + 1 , exclude_self = exclude_self))
        else:
            if self_path is not None and path.absolute() == self_path: continue
            file_boxes.append(get_script_box(str(path)))

    if file_boxes: dir_boxes.append(layout_grids(*file_boxes))
    return layout_vertical(*dir_boxes)

def main():
    print('Initiating project interface...')
    project = get_folder_box('src_runs' , 0)
    print('Project interface initiated successfully.')
    display(project)

if __name__ == '__main__':
    main()
