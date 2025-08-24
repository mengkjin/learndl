-----------------------------------------------
2023年12月27日更新
本项目添加了基础数据的生成包，分为basic_data、packages两个文件夹及update_example.py使用样例，其内容如下
1、basic_data中为基础数据的生成及读取函数，ashare_stkpool对应股票池，barra_model对应barra数据，daily_bar对应日行情及估值，
divnsplit对应分红，index_level对应指数日行情，index_weight对应指数权重，industry对应行业
2、packages中为常用工具的包，其中basic_src_data为数据库包，使用者需修改其中wind_conn函数；crosec_mem为文件数据库工具；financial_tools为财务函数的工具；events_system为日期工具
3、update_example.py为使用案例，使用者可修改日期，生成数据到root_path路径下
4、使用时将basic_data、packages、factors、portfolios添加到系统路径中(spyder菜单栏的PythonPath Manager)
5、各基准起始日期
000300.SH - 20101231
000905.SH - 20130329
000852.SH - 20150529
932000.CSI - 20230831
000906.SH - 20130329
931865.CSI - 20190329
-----------------------------------------------
2024年1月15日更新
本项目添加了因子测试、组合优化、组合测试的相关包，具体变动如下：
1、project中添加了factor_test_example文件夹和portfolio_test文件夹，前者为因子测试的示例，后者为组合优化及回测的示例
2、basic_data中添加了stk_basic_info包，用于基础信息数据的生成
3、添加了factors与portfolios两个文件夹，对应因子和组合的相关包
4、packages中添加了stk_index_utils、stk_ret_tools、factor_tools文件夹，分别用于股票收益计算、指数收益计算、因子处理
5、packages中basic_src_data>>wind_tools>>basic文件有修改
-----------------------------------------------
2024年1月31日更新
1、组合回测与组合优化环节支持非满仓的情况，组合优化参数文件中添加leverage参数，表示仓位
2、在收益分解环节增加交易成本一项
3、修复回测与优化环节的基准指数参数共用一个的bug
-----------------------------------------------
