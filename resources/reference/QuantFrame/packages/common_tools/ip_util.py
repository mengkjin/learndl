import socket


myname = socket.getfqdn(socket.gethostname())

isin_office = 'TFZQ' in myname