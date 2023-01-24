import qrcode
from datetime import datetime


def qr_gen(message, name):
    qr = qrcode.make(data=message)
    qr.save(stream=f'qrcodes/{name}.png')
    print(f'QR code was created! Open the {name}.png')


now = datetime.now()
time = datetime.time(now).strftime('%H-%M-%S')
title = 'zakaz-'+time

msg = 'Potato 30$\nPizza 35$\nBurger 15$'
print('name =', title)

qr_gen(message=msg, name=title)