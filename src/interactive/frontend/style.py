import streamlit as st
from string import Template
from src.proj.env import PATH

class CustomCSS:
    Templates : dict[str, Template] = PATH.load_templates('css' , 'interactive')
    def __init__(self) -> None:
        self.css_list : list[str] = [getattr(self , css)() for css in ['basic' , 'special_expander' , 'classic_remove' , 'multi_select' , 'custom']]

    def apply(self):
        css_str = '\n'.join(self.css_list)
        st.markdown(f'''
        <style>
        {css_str}
        </style>
        ''' , unsafe_allow_html = True)

    def basic(self):
        return self.Templates['basic'].substitute()

    def special_expander(self):
        return self.Templates['special_expander'].substitute()

    def classic_remove(self):
        return self.Templates['classic_remove'].substitute()

    def custom(self):
        return self.Templates['custom'].substitute()
    
    def multi_select(self , label_size = 16 , item_size = 16 , popover_size = 14):
        return self.Templates['multi_select'].substitute(label_size = label_size , item_size = item_size , popover_size = popover_size)

def style():
    css = CustomCSS()
    css.apply()
