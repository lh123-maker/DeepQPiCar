import re
import math
import datetime

import subprocess

VELOCITIES = {
    '75-35' : .032,
    '35-75' : .030,
    '65-65' : .050
}

ACCELERATION = {
    '65-65-75-35' : .072,
    '65-65-35-75' : .080,
    '75-35-35-75' : .080,
    '75-35-65-65' : -.072,
    '35-75-75-35' : -.080,
    '35-75-65-65' : -.080,
}

def get_signal_level_from_router(wifi_device):
    signal_cmd = 'iwconfig {} | grep -i --color signal'.format(wifi_device)
    proc = subprocess.Popen(signal_cmd, stdout=subprocess.PIPE, shell=True)
    output = proc.stdout.read()
    #Link Quality=67/70  Signal level=-43 dBm
    m = re.search('(level=)(?P<level>-\d{2})', output.strip())
    return int(m.group('level'))

def calculate_distance_from_router(singal_level):
    frequency = 2412
    return math.pow(10, ((27.55 - (20 * math.log10(frequency)) + abs(singal_level)) / 20.0))

def get_distance_from_router():
    return calculate_distance_from_router(get_signal_level_from_router('wlan0'))

def get_velocity_from_motors(controls):
    return VELOCITIES['{}-{}'.format(abs(controls[0]), abs(controls[1]))]

def get_acceleration_from_motors(current, previous):
    return ACCELERATION['-'.join([str(abs(i)) for i in previous+current])]