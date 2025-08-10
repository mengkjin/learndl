import streamlit as st

class CustomCSS:
    def __init__(self , add_css = ['basic' , 'special_expander' , 'classic_remove' , 'multi_select']) -> None:
        self.css_list : list[str] = []
        for css in add_css:
            self.add(getattr(self , css)())

    def add(self , css : str):
        self.css_list.append(css)

    def apply(self):
        css_str = '\n'.join(self.css_list)
        st.markdown(f'''
        <style>
        {css_str}
        </style>
        ''' , unsafe_allow_html = True)

    def basic(self):
        return '''
        h1 {
            font-size: 48px !important;
            font-weight: 900 !important;
            padding: 10px !important;
            letter-spacing: 5px !important;
            border-bottom: 2px solid #1E90FF !important;
        }
        h2 {
            font-size: 36px !important;
            font-weight: 900 !important;
            letter-spacing: 3px !important;
            white-space: nowrap !important;
            color: darkblue !important;
        }
        h3 {
            font-size: 24px !important;
            font-weight: 900 !important;
            letter-spacing: 3px !important;
            white-space: nowrap !important;
        }
        button {
            align-items: center;
            justify-content: center;
            margin: 0px !important;
            min-height: 10px !important;
        }
        button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 12px rgba(30, 58, 138, 0.15) !important;
            background-color: lightblue !important;
            border: none !important;
            color: white !important;
        }
        .stCaptionContainer {
            margin: -10px !important;
        }
        .stSelectbox {
            width: 100% !important;
        }
        .stSelectbox > div > div {
            height: 28px;
            width: 100%;
        }
        .stSelectbox > div > div > div {
            align-self: center;
        }
        .stTextInput {
            width: 100%;
        }
        .stTextInput > div {
            height: 28px;
        }
        .stTextInput > div > div {
            align-self: center !important;
        }
        .stNumberInput {
            width: 100%;
        }
        .stNumberInput > div {
            height: 28px;
            width: 100%;
        }
        .stNumberInput > div > div {
            height: inherit;
            align-self: center !important;
        }
        .stNumberInput button {
            width: 20px !important;
            align-self: flex-end !important;
            margin-top: 0px !important;
        }
        .element-container {
            margin-bottom: 0px;
            display: flex;
        }
        .stMarkdown {
            display: flex;
        }
        .stMarkdown p {
        }
        .stMetric div {
            font-size: 18px !important;
            color: #1E3A8A !important;
        }
        .stMetric > label > div > div {
            font-size: 20px !important;
        }
        .stMetric > div {
            
        }
        .stContainer {
            padding-top: 0px;
            padding-bottom: 0px;
        } 
        .stExpander .stElementContainer {
            margin-bottom: -10px !important;
            padding-bottom: 0px !important;
        }
        .stExpander summary {
            padding-top: 4px !important;
            padding-bottom: 4px !important;
        }
        .stCode code {
            font-size: 12px !important;
        }
        .stAlert > div {
            min-height: 18px !important;
            display: flex !important;
            line-height: 1.0 s!important;
            align-items: center;
            justify-content: right;
            font-size: 14px !important;
            padding: 0.25rem 0.5rem !important;
            margin-top: 0px !important;
            margin-bottom: 0px !important;
        }
        .stColumn {
            display: flex;
        }
        .stExpander {
            summary p {
                font-size: 16px !important;
                font-weight: bold !important;
            }
        }
        '''

    def special_expander(self):
        return '''
        [class*="special-expander"] > div > [data-testid="stExpander"] > details {
            background: transparent !important;
            border: none !important;
        }
        [class*="special-expander"] > div > [data-testid="stExpander"] > details > div {
            border: 1px solid #D1D5DB !important;
            border-radius: 12px !important;
            margin-top: -1px !important;
            margin-bottom: 10px !important;
            padding-top: 20px !important;
            padding-bottom: 20px !important;
        }       
        [class*="special-expander"] > div > [data-testid="stExpander"] > details > summary {
            font-size: 16px !important;
            font-weight: 900 !important;
            color: #1E3A8A !important;
            background: linear-gradient(135deg, #F3F4F6 0%, #E5E7EB 100%) !important;
            border-radius: 12px !important;
            padding: 14px 18px !important;
            border: 2px solid #D1D5DB !important;
            letter-spacing: 1px !important;
            text-transform: uppercase !important;
        }
        [class*="special-expander"] > div > [data-testid="stExpander"] > details > summary:hover {
            background: linear-gradient(135deg, #DBEAFE 0%, #BFDBFE 100%) !important;
            border-color: #1E3A8A !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 12px rgba(30, 58, 138, 0.15) !important;
        }
        '''

    def classic_remove(self):
        return '''
        [class*="classic-remove"] button {
            height: 32px !important;
            width: 32px !important;
            background-color: red !important; 
            fill: white !important; 
            color: white !important; 
            margin: 0px !important;
        }
        '''
    
    def multi_select(self , label_size = 16 , item_size = 16 , popover_size = 14):
        return f"""
        [data-baseweb="popover"] li {{
            font-size: {popover_size}px !important;
            line-height: 1.0 !important;
            min-height: 10px !important;
        }}
        .stRadio label span {{
            font-size: {label_size}px !important;
        }}
        .stMultiSelect {{
            width: 100% !important;
        }}
        .stMultiSelect span {{
            font-size: {label_size}px !important;
        }}
        .stMultiSelect > div div  {{
            font-size: {item_size}px !important;
            min-height: 20px !important;
        }}
        .stMultiSelect > div span  {{
            font-size: {item_size}px !important;
        }}
        .stSelectbox{{
            width: 100% !important;
        }}
        .stSelectbox span {{
            font-size: {label_size}px !important;
        }}
        .stSelectbox > div div  {{
            font-size: {item_size}px !important;
            min-height: 20px !important;
        }}
        .stSelectbox > div span  {{
            font-size: {item_size}px !important;
        }}
        """


def style():
    css = CustomCSS(add_css = ['basic' , 'special_expander' , 'classic_remove' , 'multi_select'])
    css.add("""
    .stVerticalBlock[class*="queue-header-buttons"] {
        div {justify-content: flex-start !important;}
        * {
            font-weight: bold !important;
            font-size: 24px !important;
        }
        button {
            height: 36px !important; 
        }
        [class*="-sync"] button {
            color: #1E88E5 !important;
            &:hover {
                background-color: #1E88E5 !important;
                color: white !important;
            }
        } 
        [class*="-refresh"] button {
            color: green !important;
            &:hover {
                background-color: green !important;
                color: white !important;
            }
        }   
        [class*="-delist"] button {
            color: violet !important;
            &:hover {
                background-color: violet !important;
                color: white !important;
            }
        }
        [class*="-remove"] button {
            color: red !important;
            &:hover {
                background-color: red !important;
                color: white !important;
            }
        }
        [class*="-restore"] button {
            color: darkgreen !important;
            &:hover {
                background-color: darkgreen !important;
                color: white !important;
            }
        }
    }
    .stVerticalBlock[class*="queue-item-container"] {
        margin-bottom: -10px !important;
        
        .stElementContainer[class*="queue-item"] {
            * {
                border: none !important;
                border-radius: 10px !important;
                line-height: 32px !important;
                font-weight: bold !important;
                font-size: 20px !important;
                justify-content: flex-end !important;
                align-items: flex-end !important;
            }     
            button {
                justify-content: center !important;
                padding: 0 !important;
                width: 32px !important;
                height: 32px !important;
            }
        }
        .stElementContainer[class*="-remove"] {
            button:hover {background-color: #ff8080 !important;}
        } 
        .stElementContainer[class*="-delist"] {
            button:hover {background-color: #8080ff !important;}
        }
        [class*="click-content"] button {
            justify-content: flex-start !important;
            text-align: left !important;
            padding-left: 6px !important;
            p {font-size: 16px !important;}
        }
    }
    .stVerticalBlock[class*="developer-info"] {
        button {
            border-radius: 10px !important;
            border: 1px solid lightgray !important;
            min-width: 100px !important;
            height: 36px !important; 
            p {
                font-weight: 900 !important;
            }
        }
        [class*="-sync"] button {
            color: #1E88E5 !important;
            &:hover {
                background-color: #1E88E5 !important;
                color: white !important;
            }
        } 
        [class*="-refresh"] button {
            color: green !important;
            &:hover {
                background-color: green !important;
                color: white !important;
            }
        }   
        [class*="-delist"] button {
            color: lightcoral !important;
            &:hover {
                background-color: lightcoral !important;
                color: white !important;
            }
        }
        [class*="-remove"] button {
            color: red !important;
            &:hover {
                background-color: red !important;
                color: white !important;
            }
        }
    }
    }
    .stElementContainer[class*="choose-item-select"] {
        button {
            justify-content: flex-start !important;
            text-align: left !important;
            padding-left: 12px !important;
        }
        &[class*="-selected"] {
            button {
                background-color: #1E88E5 !important;
                color: white !important;
            }
            p {
                font-weight: bold !important;
            }
        }
    }
    .stVerticalBlock[class*="choose-item-remove"] {
        .stTooltipIcon {justify-content: flex-end !important;}
        .stElementContainer[class*="remove-button"] {
            * {
                border: none !important;
                border-radius: 10px !important;
                line-height: 32px !important;
                font-weight: bold !important;
                font-size: 24px !important;
            }     
            button {
                width: 32px !important;
                height: 32px !important;
                color: red !important;
                &:hover {
                    color: white !important;
                    background-color: red !important;
                }
            }
        }
    }
    .stVerticalBlock[class*="sidebar-script-menu"] {
        .stPageLink {
            margin-top: 0px !important;
            margin-bottom: -10px !important;
            padding-top: 0px !important;
            padding-bottom: 0px !important;
        }
    } 
    .stVerticalBlock[class*="script-structure"] {
        margin-top: 0px !important;
        margin-bottom: -10px !important;
        &[class*="-level-1"] button {margin-left: 45px !important;}
        &[class*="-level-2"] button {margin-left: 90px !important;}
        &[class*="-level-3"] button {margin-left: 135px !important;}
        .stElementContainer[class*="-runner-expand"] {
            button {
                min-width: 400px !important;
                justify-content: flex-start !important;
                padding-left: 20px !important;
            }       
            p {
                font-size: 16px !important;
                letter-spacing: 2px !important;
            }
            &[class*="-selected"] button {
                background-color: #1E88E5 !important;
                color: white !important;
            }  
        }
    } 
    .stElementContainer[class*="script-runner-run"] {
        button {
            min-width: 70px !important;
            height: 70px !important;
            width: 70px !important;
            background-color: green !important;
            color: white !important;
            border-radius: 50%;
            border: none;
            cursor: pointer;
            display: flex;
            margin: 10px !important;
            &:hover {background-color: darkgreen !important;}
        }
        p {
            font-size: 48px !important;
            font-weight: bold !important;
        }
        &[class*="-disabled"] button {
            background-color: lightgray !important;
            color: white !important;
            border: 1px solid lightgray !important;
            &:hover {
                background-color: lightgray !important;
            }
        } 
        .stTooltipIcon {justify-content: flex-end !important;}
        &[class*="-sidebar"] .stTooltipIcon {
            justify-content: center !important;
        }
    }
    .stElementContainer[class*="file-preview"] {
        button {
            min-width: 500px !important;
            justify-content: flex-start !important;
        }
        &[class*="-selected"] {
            button {
                background-color: #1E88E5 !important;
                color: white !important;
            }
            p {
                font-weight: bold !important;
            }
        }
    }
    .stVerticalBlock[class*="file-download"] {
        .stElementContainer {justify-content: flex-end !important;}
    }  
    """)
    css.apply()
