"""Stdout/stderr capture to memory, logs, HTML, markdown, and warning interception."""
from __future__ import annotations
import sys , platform
import pandas as pd

from datetime import datetime
from functools import cached_property

from typing import Any , TYPE_CHECKING
from src.proj.env import PATH , MACHINE , Proj
from src.proj.core import Duration
from src.proj.log import Logger

from .basic import OutputCatcher , TimedOutput , DeflectorGroup , get_html_templates

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = ['HtmlCatcher']

class HtmlCatcher(OutputCatcher):
    """
    Html catcher for stdout, stderr, dataframe, and image (use Logger.display to display), export to html file at exit
    example:
        catcher = HtmlCatcher()
        with catcher:
            Logger.stdout('This will be caught')
        contents = catcher.contents
    """
    export_dir = PATH.logs.joinpath('catcher' , 'html')
    export_suffix : str = '.html'

    PrimaryInstance : HtmlCatcher | None = None
    Capturing : bool = True

    def __init__(self, title: str | None = None , category : str = 'miscelaneous', init_time: datetime | None = None , 
                 add_time_to_title: bool = True, **kwargs):
        self.category = category
        self.init_time = init_time if init_time else datetime.now()
        self.title = title
        self.add_time_to_title = add_time_to_title

        self.add_export_file(self.export_dir.joinpath(self.category.replace(' ' , '_') , self.filename))
        
        self.outputs: list[TimedOutput] = []
        self.kwargs = kwargs
        
    def __bool__(self):
        return True

    def keyword_repr(self):
        return f'title="{self.full_title}",primary={self.is_primary}'

    @cached_property
    def title(self) -> str | None:
        """Get the export file list of the catcher"""
        return None

    @property
    def full_title(self) -> str:
        """Get the full title of the catcher"""
        title = self.title or 'html_catcher'
        if self.add_time_to_title:
            time_str = self.init_time.strftime("%Y%m%d%H%M%S")
            title = f'{title} at {time_str}'
        return title

    @property
    def filename(self) -> str:
        """Get the filename of the catcher"""
        return f'{self.full_title.replace(" " , "_")}.html'

    @property
    def is_running(self):
        """Check if the catcher is running"""
        return self.PrimaryInstance is not None

    @property
    def is_primary(self):
        """Check if the catcher is the primary instance"""
        return self.__class__.PrimaryInstance is self

    @property
    def last_output(self) -> TimedOutput | None:
        """Get the last output of the catcher"""
        return self.outputs[-1] if self.outputs else None

    def set_attrs(self , title : str | None = None , category : str | None = None):
        """
        Set the attributes of the catcher even after initialization
        title : str , the title of the catcher
        category : str , the category of the catcher
        """
        instance = self.PrimaryInstance if self.PrimaryInstance is not None else self
        if title: 
            instance.title = title
        if category: 
            instance.category = category
        return self

    def set_instance(self):
        """Set the instance of the catcher, if the catcher is already running, block the new instance"""
        if self.__class__.PrimaryInstance is None:
            self.__class__.PrimaryInstance = self

    def clear_instance(self):
        """Clear the instance of the catcher if the catcher is the current instance"""
        if self.__class__.PrimaryInstance is self:
            self.__class__.PrimaryInstance = None

    def __enter__(self):
        self.set_instance()
        self.start_time = datetime.now()
        if self.is_primary:
            self.deflectors = DeflectorGroup(self , self.keep_original).start_catching()
            self.redirect_display_function()
            self.start_cursor = None
        else:
            assert self.PrimaryInstance is not None , f"Primary instance is not set when entering {self}"
            self.start_cursor = self.PrimaryInstance.last_output
            self.start_point = len(self.PrimaryInstance.outputs)

        self.logger.remark(f"{self.keyword_repr()}, Capturing Start" , vb_level = 1 if self.is_primary else 2)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.remark(f"{self.keyword_repr()}, Capturing Finished, cost {Duration(since = self.start_time)}" , vb_level = 1 if self.is_primary else 2)
        self.export()
        if self.is_primary:
            self.deflectors.end_catching()
            self.restore_display_function()
        self.clear_instance()

    def export(self):
        """Export the catcher to all paths in the export file list"""
        # log first and then export
        for export_path in self.export_file_list:
            self.logger.footnote(f"result saved to {export_path}")
        if self.is_primary and self.export_file_list:
            Proj.exit_files.insert(0 , self.export_file_list[0])
        
        html_content = self.generate_html()
        for export_path in self.export_file_list:
            export_path.parent.mkdir(exist_ok=True,parents=True)
            export_path.write_text(html_content, encoding='utf-8')
        
    def redirect_display_function(self):
        """redirect Logger.Display to catcher"""
        Logger.set_display_callbacks([self.add_output , self.stop_capturing] , [self.start_capturing])

    def restore_display_function(self):
        """restore Logger.Display functions"""
        Logger.reset_display_callbacks()
    
    def generate_html(self):
        """generate html file with time ordered outputs"""
        assert self.PrimaryInstance is not None , f"Primary instance is not set when generating html"

        if self.start_cursor is None or self.start_cursor not in self.PrimaryInstance.outputs:
            start_point = 0
        else:
            start_point = self.PrimaryInstance.outputs.index(self.start_cursor) + 1
        templates = [output.to_template() for output in self.PrimaryInstance.outputs[start_point:]]
        templates = [template for template in templates if template is not None]
        rows = [templates.substitute(index=i) for i, templates in enumerate(templates)]

        return ''.join([self.html_head() , *rows , self.html_tail()])
 
    def add_output(self, content: str | pd.DataFrame | pd.Series | Figure | Any , output_type: str | None = None):
        """add output to time ordered list"""
        if not self.Capturing: 
            return
        
        output = TimedOutput.create(content , output_type)
        if not output or (self.outputs and output.equivalent(self.outputs[-1])): 
            return

        self.outputs.append(output)

    @classmethod
    def stop_capturing(cls , *args, **kwargs):
        """Stop the capturing of the catcher , class level (stop all catchers)"""
        cls.Capturing = False

    @classmethod
    def start_capturing(cls , *args, **kwargs):
        """Start the capturing of the catcher , class level (start all catchers)"""
        cls.Capturing = True

    def write_stdout(self, text: str):
        """Write stdout to the catcher"""
        if text := text.strip('\n'):
            self.add_output(text, 'stdout')

    def write_stderr(self, text: str):
        """Write stderr to the catcher"""
        if text := text.strip('\n'):
            self.add_output(text, 'stderr')

    def get_contents(self):
        """Get the contents of the html catcher"""
        return self.generate_html()
       
    def html_head(self) -> str:
        """Generate the html head , including the styles of the html file, and basic information of the catcher"""
        title = self.full_title.title()
        
        key_width = 80
        if self.kwargs:
            key_width = max(int(max(len(key) for key in list(self.kwargs.keys())) * 5.5) + 10 , key_width)
        finish_time = datetime.now()

        script_infos = {
            'Machine' : MACHINE.name,
            'Python' : f"{platform.python_version()}-{platform.machine()}",
            'Command' : ' '.join(sys.argv),
            'Start at' : f'{self.start_time.strftime("%Y-%m-%d %H:%M:%S")}',
            'Finish at' : f'{finish_time.strftime("%Y-%m-%d %H:%M:%S")}',
            'Duration' : Duration((finish_time - self.start_time).total_seconds()).fmtstr,
        }
        other_types : list[str] = list(set([output.type_str for output in self.outputs if output.type not in ['stdout' , 'stderr' , 'dataframe' , 'image']]))
        output_infos = {
            'Total #' : len(self.outputs),
            'Stdout #' : sum(1 for output in self.outputs if output.type == 'stdout'),
            'Stderr #' : sum(1 for output in self.outputs if output.type == 'stderr'),
            'Dataframe #' : sum(1 for output in self.outputs if output.type == 'dataframe'),
            'Image #' : sum(1 for output in self.outputs if output.type == 'image'),
            **{type.title() + ' #' : sum(1 for output in self.outputs if output.type_str == type) for type in other_types},
        }
        output_infos = {key: value for key, value in output_infos.items() if value > 0}
        infos_script = '<div class="add-infos add-title"> INFORMATION </div>' + \
            '\n'.join([f'<div class="add-infos"><span class="add-key">{key}</span><span class="add-seperator">:</span><span class="add-value">{value}</span></div>' 
                       for key, value in script_infos.items()])
        infos_outputs = '<div class="add-infos add-title"> NUMBER OF OUTPUTS </div>' + \
            '\n'.join([f'<div class="add-infos"><span class="add-key">{key}</span><span class="add-seperator">:</span><span class="add-value">{value}</span></div>' 
                        for key, value in output_infos.items()])
        head = get_html_templates('head').substitute(
            title=title,
            key_width=key_width,
            infos_script=infos_script,
            infos_outputs=infos_outputs,
        )
        return head

    @classmethod
    def html_tail(cls) -> str:
        """Generate the html tail , including the end of the html file"""
        return get_html_templates('tail').substitute() 
