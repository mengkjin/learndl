import ipywidgets as widgets
import os
from IPython.display import display
from IPython.core.getipython import get_ipython

python = 'python3.10' if os.name == 'posix' else 'python'

def html_title(text : str , color : str | None = None , size : int | None = None , bold : bool = False):
    style = ''
    if color is not None:
        style += f'color: {color};'
    if size is not None:
        style += f'font-size: {size}px;'
    if bold:
        style += f'font-weight: bold;'
    style += 'text-align: center;'

    return widgets.HTML(f'<div style="{style}"><em>{text}</em></div>')

vbox_layout = widgets.Layout(
        display='flex',
        flex_flow='column',
        align_items='center',
        width='100%'
)

def run_script(script , source = 'button' , email = 1 , param = ''):
    shell = get_ipython()
    cmd = f'{python} {script} --source {source} --email {email}'
    if param:
        cmd += f' --param {param}'
    if shell is not None:
        shell.system(cmd)
    else:
        print('IPython shell not found')

def script_button(script : str , button_text : str | None = None , description : str = '' , **kwargs):
    def func(b): run_script(script , **kwargs)
    button = widgets.Button(description=script if button_text is None else button_text, 
                           layout=widgets.Layout(width='auto', min_width='200px'))
    button.on_click(func)
    text = widgets.Textarea(value=description, disabled=True)
    return widgets.VBox([button, text] , layout=widgets.Layout(align_items='center'))

def autorun_box():
    title_widget = html_title('Autorun Scripts' , color = '#007bff' , size = 18 , bold = True)

    button1 = script_button(f'src_runs/autorun/daily_update.py' , 'Run Daily Update'  , 
                         f'Daily updates of data , factors and preds')
    button2 = script_button(f'src_runs/autorun/weekly_update.py' , 'Run Weekly Update' , 
                         f'Weekly updates of ai models training')
    button3 = script_button(f'src_runs/autorun/monthly_update.py' , 'Run Monthly Update' , 
                         f'Not yet implemented.')

    buttons_grid = widgets.GridBox(
        [button1, button2, button3],
        layout=widgets.Layout(
            grid_template_columns='repeat(3, 1fr)',
            grid_gap='10px',
            width='100%'
        )
    )
    
    box = widgets.VBox([title_widget, buttons_grid],layout=vbox_layout)
    return box

def research_box():
    title_widget = html_title('Research Scripts' , color = '#007bff' , size = 18 , bold = True)

    vbox_layout = widgets.Layout(
            display='flex',
            flex_flow='column',
            align_items='center',
            width='100%'
        )

    title_factor = html_title('Update Factors' , size = 14)
    button_factors_1 = script_button('src_runs/research/update_factors.py' , 'Update Factors (100 groups)' , 
                                  f'Update linear factors of current definitions' , email = 0 , param = 100)
    button_factors_2 = script_button('src_runs/research/update_factors.py' , 'Update Factors (500 groups)' , 
                                  f'Update linear factors of current definitions' , email = 0 , param = 500)
    button_factors_3 = script_button('src_runs/research/update_factors.py' , 'Update Factors (unlimited)' , 
                                  f'Update linear factors of current definitions' , email = 0 , param = 10000)
    buttons_factor = widgets.GridBox(
        [button_factors_1, button_factors_2, button_factors_3],
        layout=widgets.Layout(
            grid_template_columns='repeat(3, 1fr)',
            grid_gap='10px',
            width='100%'
        )
    )
    box_factors = widgets.VBox([title_factor, buttons_factor] , layout = vbox_layout)

    title_model = html_title('Update AI Models' , size = 14)
    button_hiddens = script_button('src_runs/research/update_hiddens.py'  , 'Update Hiddens Extraction' , 
                                f'Extract hidden states from hidden feature models' , email = 0)

    button_preds = script_button('src_runs/research/update_preds.py' , 'Update ModelPredictions' , 
                              f'Update model predictions of registered models' , email = 0)
    
    button_models = script_button('src_runs/research/update_models.py' , 'Update AI Models' , 
                               f'Update ai model training of registered models' , email = 0)

    buttons_model = widgets.GridBox(
        [button_hiddens, button_preds, button_models],
        layout=widgets.Layout(grid_template_columns='repeat(3, 1fr)',grid_gap='10px',width='100%',))
    
    box_model = widgets.VBox([title_model, buttons_model] , layout = vbox_layout)

    title_train = html_title('Train AI Models' , size = 14)
    button_train = script_button('src_runs/research/train_model.py' , 'Train Model' , 
                              f'Train ai models of current configs' , email = 0)
    box_train = widgets.VBox([title_train, button_train] , layout = vbox_layout)

    box = widgets.VBox([title_widget, box_factors, box_model, box_train] , layout = vbox_layout)
    return box

def project_box():
    title_widget = widgets.VBox([autorun_box(), research_box()] , layout = vbox_layout)
    return display(title_widget)