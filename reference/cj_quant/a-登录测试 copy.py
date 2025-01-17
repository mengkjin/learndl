# 登录测试
try:
    import zmq
except:
    print('需要安装用于网络通讯的zmq包，可通过pip install pyzmq安装')
    exit()

import quant_db # 导入我们的云服务python接口

usrId = 'jinmeng9600' # 请填写您的账号

qdb = quant_db.DB_Api(usrId)
