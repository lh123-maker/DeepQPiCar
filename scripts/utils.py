import re
import math
import datetime

import subprocess


def get_signal_level_from_router(wifi_device):
    signal_cmd = 'iwconfig {} | grep -i --color signal'.format(wifi_device)
    proc = subprocess.Popen(signal_cmd, stdout=subprocess.PIPE, shell=True)
    output = proc.stdout.read()
    #Link Quality=67/70  Signal level=-43 dBm
    m = re.search('(level=)(?P<level>-\d{2})', output.strip())
    return int(m.group('level'))

def get_distance_from_router(singal_level):
    frequency = 2412
    return math.pow(10, ((27.55 - (20 * math.log10(frequency)) + abs(singal_level)) / 20.0))



if __name__ == '__main__':
    signal_level = get_signal_level_from_router('wlp4s0')
    distance_cm = get_distance_from_router(signal_level)
    message = "{'%s' : {'signal_level' : %s,'distance_cm': %s } }" % (datetime.datetime.now(), signal_level, distance_cm)
    import ast
    d = ast.literal_eval(message)
    print(d.keys())
