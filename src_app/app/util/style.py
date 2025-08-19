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
            margin-bottom: 20px !important;
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
            /* color: #1E3A8A !important; */
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
            margin-bottom: 0px !important;
            padding-bottom: 0px !important;
        }
        .stExpander summary {
            padding-top: 4px !important;
            padding-bottom: 4px !important;
            span[data-testid="stExpanderIconCheck"] {
                color: green !important;
            }
            span[data-testid="stExpanderIconError"] {
                color: red !important;
            }
        }
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
        label[data-testid="stWidgetLabel"] > div:not([data-testid="stMarkdownContainer"]) {
            justify-content: flex-start !important;
            padding-left: 10px !important;
        }
        .stVerticalBlock[class*="task-stats"] .stMarkdown {
            margin-bottom: 0px !important;
            margin-top: -10px !important;
            padding-left: 15px !important;
        }
        '''

    def special_expander(self):
        return '''
        .stVerticalBlock[class*="special-expander"] > div > [data-testid="stExpander"] > details {
            background: transparent !important;
            border: none !important;
            > div {
                border: 1px solid #D1D5DB !important;
                border-radius: 12px !important;
                margin-bottom: 10px !important;
                padding-top: 20px !important;
                padding-bottom: 20px !important;
            }
            > summary {
                background: linear-gradient(75deg, lightblue -2%, white 1%, lightblue 200%) !important;
                border-radius: 12px !important;
                padding: 14px 18px !important;
                border: 2px dash #D1D5DB !important;
                text-transform: uppercase !important;
                span {
                    font-size: 24px !important;
                    font-weight: 900 !important;
                    letter-spacing: 3px !important;
                    white-space: nowrap !important;
                    &[class*="stMarkdownColoredText"] {
                        border-bottom: 2px solid lightgray !important;
                    }
                }
            }
            > summary:hover {
                background: linear-gradient(75deg, darkblue -2%, white 1%, darkblue 200%) !important;
                border-color: #1E3A8A !important;
                transform: translateY(-1px) !important;
                box-shadow: 0 4px 12px rgba(30, 58, 138, 0.15) !important;
            }
        }
        .expander-help-container {
            position: relative;
            margin: 10px 0;
        }
        .expander-help-container .help-tooltip {
            position: absolute;
            top: -30px;
            left: 0;
            padding: 10px 15px;
            border-radius: 6px;
            font-size: 16px;
            font-style: italic;
            white-space: nowrap;
            opacity: 0;
            visibility: hidden;
            transition: all 0.3s;
            pointer-events: none;
            text-align: left;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }

        .help-icon {
            position: absolute;
            top: 5px;
            left: 0px;
            color: gray !important;
            background-color: transparent !important;
            font-size: 24px !important;
            font-weight: 900 !important;
            letter-spacing: 3px !important;
            white-space: nowrap !important;
            padding: 14px 18px !important;
            width: 100% !important;
            cursor: help;
            z-index: 1000 !important;
        }

        .expander-help-container:hover .help-tooltip {
            opacity: 1;
            visibility: visible;
        }

        .expander-help-container .help-tooltip::after {
            content: '';
            position: absolute;
            top: 100%;
            left: 20px;
            border: 5px solid transparent;
            border-top-color: #333;
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
            min-width: 60px !important;
            height: 60px !important;
            width: 60px !important;
            span {
                font-size: 36px !important;
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
        [class*="-clean"] button {
            color: lightblue !important;
            &:hover {
                background-color: lightblue !important;
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
        [class*="click-content"] {
            button {
                justify-content: flex-start !important;
                text-align: left !important;
                padding-left: 6px !important;
                p {font-size: 16px !important;}
            }
            &[class*="-selected"] button {
                background-color: #1E88E5 !important;
                color: white !important;
            }
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
        [class*="-clear"] button {
            color: red !important;
            &:hover {
                background-color: red !important;
                color: white !important;
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
                color: black !important;
                background: linear-gradient(to right, #87CEEB -2%, white 1%, #87CEEB 200%) !important;
                &:hover {
                    background: linear-gradient(to right, #1E88E5 -2%, white 1%, #1E88E5 200%) !important;
                    color: black !important;
                }
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
            min-width: 60px !important;
            height: 60px !important;
            width: 60px !important;
            background-color: green !important;
            color: white !important;
            border-radius: 25%;
            border: none;
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
        &[class*="-sidebar"] {
            .stTooltipIcon {
                justify-content: center !important;
            }
            button {
                margin-top: 0px !important;
                margin-bottom: 0px !important;
                padding-top: 0px !important;
                padding-bottom: 0px !important;
            }
        }
    }
    .stElementContainer[class*="script-latest-task"] {
        button {
            min-width: 60px !important;
            height: 60px !important;
            width: 60px !important;
            background-color: lightblue !important;
            color: white !important;
            border-radius: 25%;
            border: none;
            display: flex;
            margin-top: 0px !important;
            margin-bottom: 0px !important;
            padding-top: 0px !important;
            padding-bottom: 0px !important;
            &:hover {background-color: blue !important;}
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
        .stTooltipIcon {
            justify-content: center !important;
        }
    }
    .stElementContainer[class*="file-preview"] {
        button {
            min-width: 500px !important;
            justify-content: flex-start !important;
        }
        &[class*="-select"] {
            button {
                background: #1E88E5 !important;
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
    div[data-testid*="stSidebarHeader"] {
        padding-bottom: 0px !important;
        margin-top: 0px !important;
        margin-bottom: 0px !important;
    }
    div[data-testid*="stSidebarUserContent"] {
        h3 {
            margin-top: 0px !important;
            margin-bottom: 0px !important;
            padding-bottom: 5px !important;
        }
        .stVerticalBlock[class*="sidebar-global-settings"] {
            .stColumn {
                margin-top: 0px !important;
                margin-bottom: 0px !important;
                padding-top: 0px !important;
                padding-bottom: 0px !important;
                padding-left: 10px !important;
                height: 16px !important;
                .stButtonGroup button {
                    margin-top: 0px !important;
                    margin-bottom: 0px !important;
                    padding-top: 0px !important;
                    padding-bottom: 0px !important;
                    height: 28px !important;
                    font-size: 12px !important;
                }
            }
        }
        .stVerticalBlock[class*="sidebar-intro-links"] {
            padding-top: 10px !important;
            .stTooltipIcon {
                justify-content: center !important;
            }
            button {
                height: 40px !important;
                width: 40px !important;
                border-radius: 25% !important;
                border: none !important;
                span {
                    font-size: 28px !important;
                    font-weight: bold !important;
                    justify-content: center !important;
                }
            }
            div[class*="-home"] button {
                background-color: lightgray !important; /* lightgray */
                color: white !important;
                &:hover {background-color: gray !important;}
            }
            div[class*="-developer"] button {
                background-color: #87CEEB !important; /* lightblue */
                color: white !important;
                &:hover {background-color: #1E88E5 !important;}
            }
            div[class*="-config"] button {
                background-color: #ff8080 !important; /* lightred */
                color: white !important;
                &:hover {background-color: #ff4040 !important;} /* darker red */
            }
            div[class*="-task"] button {
                background-color: #90EE90 !important; /* lightgreen */
                color: white !important;
                &:hover {background-color: #008000 !important;} /* darkergreen */
            }
            div[class*="-script"] button {
                background-color: #EE82EE !important; /* lightviolet */
                color: white !important;
                &:hover {background-color: #8B008B !important;} /* darkerviolet */
            }
        } 
        .stVerticalBlock[class*="sidebar-script-links"] {
            .stPageLink {
                margin-top: 0px !important;
                margin-bottom: -10px !important;
                padding-top: 0px !important;
                padding-bottom: 0px !important;
                padding-left: 10px !important;
            }
        } 
    }
    .stVerticalBlock[class*="home-intro-pages"] {
        button {
            height: 60px !important;
            width: 88% !important;
            color: white !important;
            border-radius: 12px !important;
            p {
                font-size: 20px !important;
                font-weight: bold !important;
            }            
        }
        div[class*="-developer"] button {
            /* background: radial-gradient(ellipse at center, #87CEEB 0%, white 90%) !important; */
            background: #87CEEB !important;
            &:hover {background: #1E88E5 !important;}
        }
        div[class*="-config"] button {
            background: #ff8080 !important; /* lightred */
            &:hover {background: #ff4040 !important;} /* darker red */
        }
        div[class*="-task"] button {
            background: #90EE90 !important; /* lightgreen */
            &:hover {background: #008000 !important;} /* darkergreen */
        }
        div[class*="-script"] button {
            background: #EE82EE !important; /* lightviolet */
            &:hover {background: #8B008B !important;} /* darkerviolet */
        }
    }
    .stElementContainer[class*="go-home-button"] {
        align-self: flex-end !important;
        button {
            height: 60px !important;
            width: 60px !important;
            border-radius: 25% !important;
            border: 2px solid lightgray !important;
            span {
                font-size: 40px !important;
                font-weight: bold !important;
                justify-content: center !important;
            }
        }
    }
    .stVerticalBlock[class*="detail-exit-info-container"] {
        p {
            margin-top: 0px !important;
            margin-bottom: 0px !important;
            padding-top: 0px !important;
            padding-bottom: 0px !important;
        }
    }
    """)
    css.apply()
