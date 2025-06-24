import os , argparse , subprocess , platform , yaml , re
import ipywidgets as widgets

from typing import Any , Literal
from pathlib import Path
from IPython.display import display

def python_path():
    if platform.system() == 'Linux' and os.name == 'posix':
        return 'python3.10'
    elif platform.system() == 'Darwin':
        return 'source /Users/mengkjin/workspace/learndl/.venv/bin/activate; python'
    else:
        return 'python'

def terminal_cmd(script : str | Path , params : dict | None = None , close_after_run = False):
    params = params or {}
    if isinstance(script , Path): script = str(script.absolute())
    args = ' '.join([f'--{k} {str(v).replace(" ", "")}' for k , v in params.items() if v != ''])
    cmd = f'{python_path()} {script} {args}'
    if platform.system() == 'Linux' and os.name == 'posix':
        if not close_after_run: cmd += '; exec bash'
        cmd = f'gnome-terminal -- bash -c "{cmd}"'
    elif platform.system() == 'Windows':
        # cmd = f'start cmd /k {cmd}'
        if not close_after_run: 
            cmd = f'start cmd /k {cmd}'
        pass
    elif platform.system() == 'Darwin':
        if not close_after_run:
            cmd += '; exec bash'
        cmd = f'''osascript -e 'tell application "Terminal" to do script "{cmd}"' '''
    else:
        raise ValueError(f'Unsupported platform: {platform.system()}')
    return cmd
    
def run_script(script : str | Path , close_after_run = False , **kwargs):
    cmd = terminal_cmd(script , kwargs , close_after_run = close_after_run)
    print(f'Script cmd : {cmd}')
    
    process = subprocess.Popen(cmd, shell=True, encoding='utf-8')
    process.communicate()
    #if close_after_run:
    #    process.communicate()
    #else:
    #    process.wait()

class OutOfRange(Exception): pass
class Unspecified(Exception): pass
class PopupWindow:
    def __init__(self , title : str = 'Input Error'):
        self.title = widgets.HTML(f'<div style="font-weight: 900; color: #0078D7; padding: 0px 5px 0px 5px ; background: #f0f0f0;">{title}</div>')
        self.close = widgets.Button(description='X', layout={'width': 'auto' , 'background': 'solid #d3d3d3'})
        self.content = widgets.HTML(layout={'margin': '0 5px'})

        self.window = widgets.VBox([
            widgets.HBox([self.title , self.close], layout={'justify_content': 'space-between', 'align_items': 'center'}),
            self.content
        ], layout={
            'border': '2px solid #d3d3d3',
            'width': 'auto' , 'min_width' : '200px' ,
            'display': 'none',
        })
        self.close.on_click(lambda b: setattr(self.window.layout , 'display' , 'none'))
        self.window.layout.display = 'none'

    def popup(self , message : str):
        self.content.value = f'<div style="font-size: 11px; line-height: 1.5; margin-top: 0px; padding: 0px 10px 0px 10px; font-weight: 700; color: red;">{message}</div>'
        self.window.layout.display = 'block'

class InputArea:
    Layout = {'width' : 'auto' , 'min_width' : '200px' , 'height' : '28px'}
    Style = {'font_size': '11px' , 'font_weight': '400'}
    BooleanWidget : Literal['toggle' , 'checkbox'] = 'toggle'

    def __init__(self , name : str , type : type | list , **kwargs):
        self.pname = name
        self.ptype = type
        self.layout = self.Layout | (kwargs.get('layout' , {}))
        self.style = self.Style | (kwargs.get('style' , {}))
        self.kwargs = kwargs

        self.widget = self.get_widget()

    @classmethod
    def create(cls , pname : str , pcomponents : dict[str , Any]):
        assert isinstance(pcomponents , dict) , f'param components must be a dict, got {type(pcomponents)}' 
        ptype = pcomponents.pop("type") if "type" in pcomponents else pcomponents.pop("enum")
        assert ptype is not None and isinstance(ptype , (str , list , tuple)) , f'param type must be str, list, or tuple, got {ptype}'
        if isinstance(ptype , str):
            ptype = eval(ptype)
            assert ptype in [str , int , float , bool] , f'param type must be str, int, float, or bool , but got {ptype}'
        else:
            ptype = list(ptype)

        return cls(pname , ptype , **pcomponents)
    
    @property
    def enum(self):
        return [f'{self.prefix}{e}' for e in self.ptype] if isinstance(self.ptype , list) else []
    
    @property
    def prefix(self):
        return self.kwargs.get('prefix' , '')
    
    @property
    def placeholder(self):
        placeholder = self.kwargs.get('desc' , self.pname)
        if self.required: placeholder = f'**{placeholder}'
        return placeholder
    
    @property
    def required(self):
        return self.kwargs.get('required' , False)
        
    @property
    def widget_kwargs(self):
        return self.kwargs | {
            'name' : self.pname , 
            'enum' : self.enum ,
            'type' : self.ptype ,
            'placeholder' : self.placeholder ,
            'layout' : self.layout , 
            'style' : self.style ,
            'required' : self.required ,
        }

    def get_widget(self):
        if isinstance(self.ptype , list):
            return self._dropdown(**self.widget_kwargs)
        elif self.ptype == bool:
            if self.BooleanWidget == 'toggle':
                return self._toggle(**self.widget_kwargs)
            else:
                return self._checkbox(**self.widget_kwargs)
        elif self.ptype in [str , int , float]:
            return self._text(**self.widget_kwargs)
        else:
            raise ValueError(f'Unsupported param type: {self.ptype}')
        
    def get_value(self):
        pvalue = self.widget.value
        if pvalue in self.enum: pvalue = self.enum_to_value(pvalue)
        if isinstance(self.widget , widgets.Dropdown) and pvalue == self.placeholder:
            if self.required:
                raise Unspecified(f'PLEASE SELECT A VALID VALUE! "{pvalue}" is only a placeholder')
            else:
                pvalue = None
        elif isinstance(self.widget , widgets.Textarea) and (pvalue == '' or pvalue is None):
            if self.required:
                raise Unspecified(f'PLEASE INPUT A VALID VALUE! "{pvalue}" is only a placeholder')
            else:
                pvalue = None
        if (pmin := self.kwargs.get('min')) is not None and type(pmin)(pvalue) < pmin: 
            raise OutOfRange(f'TOO SMALL! [{self.pname}] should be >= {pmin} , got {pvalue}')
        if (pmax := self.kwargs.get('max')) is not None and type(pmax)(pvalue) > pmax: 
            raise OutOfRange(f'TOO LARGE! [{self.pname}] should be <= {pmax} , got {pvalue}')
        return pvalue
    
    def enum_to_value(self , enum_value : str | Any):
        index = self.enum.index(enum_value)
        assert isinstance(self.ptype , list) , f'param type must be list, got {type(self.ptype)}'
        return self.ptype[index]

    @classmethod
    def _text(cls , type , placeholder , layout , style , default = None , required = False , **kwargs):
        assert type in [str , int , float] , f'param type must be str, int, float, got {type}'
        if default is not None: default = str(default)
        style = style | {'background' : 'lightyellow'}
        widget = widgets.Textarea(value=default, placeholder=placeholder, layout=layout, style=style)
        if required: 
            widget.observe(lambda change: cls._text_change_color(widget , change), names='value')
            if default is None: widget.style.background = '#ffcccb'
        return widget
    
    @staticmethod
    def _text_change_color(text_area : widgets.Textarea | Any , change : dict):
        if change['new'] == '':
            text_area.style.background = '#ffcccb'
        else:
            text_area.style.background = 'lightyellow'
    
    @classmethod
    def _toggle(cls , type , desc , layout = None , style = None , default = None , **kwargs):
        assert type == bool , f'param type must be bool, got {type}'
        default = bool(eval(default)) if default is not None else False
        icon = 'toggle-on' if default else 'toggle-off'
        widget = widgets.ToggleButton(value=default, description=desc, icon=icon, layout=layout, style=style)
        widget.observe(lambda change: cls._toggle_change_icon(widget , change), names='value')
        return widget
        
    @staticmethod
    def _toggle_change_icon(toggle_button : widgets.ToggleButton | Any , change : dict):
        if change['new']:
            toggle_button.icon = 'toggle-on'
        else:
            toggle_button.icon = 'toggle-off'

    @classmethod
    def _checkbox(cls , type , desc , layout , style , default = None , **kwargs):
        assert type == bool , f'param type must be bool, got {type}'
        default = bool(eval(default)) if default is not None else False
        layout = layout | {'border': '1px solid #ddd', 'align_content': 'center', 'justify_content': 'center'}
        widget = widgets.Checkbox(value=default, description=desc, indent=False, layout=layout, style=style)
        return widget

    @classmethod
    def _dropdown(cls , enum , placeholder , layout , style , default = None , **kwargs):
        assert isinstance(enum , list) , f'param enum must be list, got {enum}'
        default = str(default) if default is not None else placeholder
        options = [placeholder] + enum
        style = style | {'description_width' : '5px' , 'padding' : '0px' , 
                         'dropdown': {'background': '#f0f8ff'}}
        layout = layout | {'background': '#f0f8ff'}
        widget = widgets.Dropdown(options=options, value=default, description='-',layout=layout,style=style)
        return widget
    
class ScriptRunner:
    def __init__(self , script : Path | str , **kwargs):
        self.script = Path(script).absolute()
        self.kwargs = kwargs
        self.header = self.script_header()
        self.text_area = None

    def __repr__(self):
        return f'ScriptRunner(script_path={self.script} , kwargs={self.kwargs})'

    def script_header(self , verbose = False , include_starter = '#' , exit_starter = '' , ignore_starters = ('#!' , '# coding:')):
        header_dict = {}
        yaml_lines : list[str] = []
        try:
            with open(self.script, 'r', encoding='utf-8') as file:
                for line in file:
                    stripped_line = line.strip()
                    if stripped_line.startswith(ignore_starters): 
                        continue
                    elif stripped_line.startswith(include_starter):
                        yaml_lines.append(stripped_line)
                    elif stripped_line.startswith(exit_starter):
                        break

            yaml_str = '\n'.join(line.removeprefix(include_starter) for line in yaml_lines)
            header_dict = yaml.safe_load(yaml_str)
        except FileNotFoundError:
            header_dict['disabled'] = True
            header_dict['description'] = 'file not found'
            header_dict['content'] = f'file not found : {self.script}'
            if verbose: print(f'file not found : {self.script}')
        except yaml.YAMLError as e:
            header_dict['disabled'] = True
            header_dict['description'] = 'YAML parsing error'
            header_dict['content'] = f'error info : {e}'
            if verbose: print(f'YAML parsing error : {e}')
        except Exception as e:
            header_dict['disabled'] = True
            header_dict['description'] = 'read file error'
            header_dict['content'] = f'error info : {e}'
            if verbose: print(f'read file error : {e}')

        if 'description' not in header_dict:
            header_dict['description'] = self.script
        return header_dict

    def __getitem__(self , key):
        return self.header[key]

    def get(self , key , default = None):
        return self.header.get(key , default)

    def get_func(self , input_areas : list[InputArea] | None = None , popup : PopupWindow | Any = None):
        params = {'email' : int(self.get('email' , 0)) , 
                  'close_after_run' : bool(self.get('close_after_run' , False))}
        params.update(self.kwargs)
        input_areas = input_areas or []
        def func(b):
            try:
                params.update({input_area.pname : input_area.get_value() for input_area in input_areas})
                run_script(self.script, **params)
            except (OutOfRange , Unspecified) as e:
                popup.popup(str(e))
                return
        return func

    def button(self):
        button = widgets.Button(description=self.header['description'].title(), 
                                layout=widgets.Layout(width='auto', min_width='200px'), 
                                style={'button_color': 'lightgrey' , 'font_weight': 'bold'} , 
                                disabled=self.header.get('disabled' , False))
        return button
    
    def desc(self):
        infos =  '<br>'.join([self.get(key) for key in ['content' , 'TODO'] if key in self.header])
        style = 'font-size: 12px; text-align: left; margin: 0.2px; padding: 0px; line-height: 1.5;'
        desc = widgets.HTML(value=f'<p style="{style}"><em>{infos}</em></p>' ,
                            layout=widgets.Layout(width='auto', min_width='200px' , border_top = '2px solid grey'))
        return desc
    
    def input_areas(self):
        if param_inputs := self.header.get('param_inputs' , None):
            input_areas = [InputArea.create(pname , pcomponents) for pname , pcomponents in param_inputs.items()]
            return input_areas

    def boxes(self):
        button = self.button()
        desc = self.desc()
        input_areas = self.input_areas()
        popup = PopupWindow() if input_areas else None
        button.on_click(self.get_func(input_areas , popup))
        boxes = [button , desc]
        if input_areas: [boxes.append(input_area.widget) for input_area in input_areas]
        if popup: boxes.append(popup.window)
        return boxes

def argparse_dict(**kwargs):
    parser = argparse.ArgumentParser(description='Run daily update script.')
    parser.add_argument('--source', type=str, default='', help='Source of the script call')
    parser.add_argument('--email', type=int, default=0, help='Send email or not')
    args , unknown = parser.parse_known_args()
    return kwargs | args.__dict__ | unknown_args(unknown)

def unknown_args(unknown):
    args = {}
    for ua in unknown:
        if ua.startswith('--'):
            key = ua[2:]
            if key not in args:
                args[key] = None
            else:
                raise ValueError(f'Duplicate argument: {key}')
        else:
            if args[key] is None:
                args[key] = ua
            elif isinstance(args[key] , tuple):
                args[key] = args[key] + (ua,)
            else:
                args[key] = (args[key] , ua)
    return args

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

def folder_title(folder : str | Path , level : int):
    assert level in [1,2,3] , f'level must be 1,2,3'
    texts = Path(folder).name.strip().replace('_' , ' ').split(' ') + ['scripts']
    if texts[0].isdigit(): texts.pop(0)
    text = ' '.join(texts)
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
    return widgets.HTML(f'<div style="{style}"><em>{text.title()}</em></div>')

def get_script_box(script : str | Path , **kwargs):
    boxes = ScriptRunner(script , **kwargs).boxes()
    return layout_vertical(*boxes , border = '2px solid grey')

def get_folder_box(folder : str | Path , level : int , exclude_self = True):
    dir_boxes , file_boxes = [] , []
    self_path = Path(__file__).absolute() if exclude_self else None
    
    if level > 0: dir_boxes.append(folder_title(folder , min(level , 3)))

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
    return project

if __name__ == '__main__':
    main()
