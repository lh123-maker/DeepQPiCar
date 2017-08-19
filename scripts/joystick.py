#!/usr/bin/env python
# coding: Latin-1

# Load library functions we want
import time
import os
import sys
import pygame

from a_star import AStar

driver = AStar()

# Re-direct our output to standard error, we need to ignore standard out to hide some nasty print statements from pygame
sys.stdout = sys.stderr

# Settings for the joystick
axisUpDown = 1                          # Joystick axis to read for up / down position
axisUpDownInverted = False              # Set this to True if up and down appear to be swapped
axisLeftRight = 2                       # Joystick axis to read for left / right position
axisLeftRightInverted = False           # Set this to True if left and right appear to be swapped
buttonResetEpo = 3                      # Joystick button number to perform an EPO reset (Start)
buttonSlow = 8                          # Joystick button number for driving slowly whilst held (L2)
slowFactor = 0.5                        # Speed to slow to when the drive slowly button is held, e.g. 0.5 would be half speed
buttonFastTurn = 9                      # Joystick button number for turning fast (R2)
interval = 0.00                         # Time between updates in seconds, smaller responds faster but uses more processor time

# Power settings
voltageIn = 12.0                        # Total battery voltage to the PicoBorg Reverse
voltageOut = 12.0                        # Maximum motor voltage

maxPower = .5
# Setup the power limits
# if voltageOut > voltageIn:
#     maxPower = 1.0
# else:
#     maxPower = voltageOut / float(voltageIn)

os.environ["SDL_VIDEODRIVER"] = "dummy" # Removes the need to have a GUI window
pygame.init()
#pygame.display.set_mode((1,1))
print 'Waiting for joystick... (press CTRL+C to abort)'
while True:
    try:
        try:
            pygame.joystick.init()
            # Attempt to setup the joystick
            if pygame.joystick.get_count() < 1:
                # No joystick attached, toggle the LED
                pygame.joystick.quit()
                time.sleep(0.5)
            else:
                # We have a joystick, attempt to initialise it!
                joystick = pygame.joystick.Joystick(0)
                break
        except pygame.error as e:
            # Failed to connect to the joystick, toggle the LED
            # PBR.SetLed(not PBR.GetLed())
            pygame.joystick.quit()
            time.sleep(0.5)
            print('error ', e)
    except KeyboardInterrupt:
        # CTRL+C exit, give up
        print '\nUser aborted'
        # PBR.SetLed(True)
        sys.exit()
print 'Joystick found'
joystick.init()

try:
    print 'Press CTRL+C to quit'
    driveLeft = 0.0
    driveRight = 0.0
    running = True
    hadEvent = False
    upDown = 0.0
    leftRight = 0.0
    # Loop indefinitely
    while running:
        # Get the latest events from the system
        hadEvent = False
        events = pygame.event.get()
        # Handle each event individually
        for event in events:
            if event.type == pygame.QUIT:
                # User exit
                running = False
            elif event.type == pygame.JOYBUTTONDOWN:
                # A button on the joystick just got pushed down
                hadEvent = True                    
            elif event.type == pygame.JOYAXISMOTION:
                # A joystick has been moved
                hadEvent = True
            if hadEvent:
                # Read axis positions (-1 to +1)
                if axisUpDownInverted:
                    upDown = -joystick.get_axis(axisUpDown)
                else:
                    upDown = joystick.get_axis(axisUpDown)
                if axisLeftRightInverted:
                    leftRight = -joystick.get_axis(axisLeftRight)
                else:
                    leftRight = joystick.get_axis(axisLeftRight)
                # Apply steering speeds
                if not joystick.get_button(buttonFastTurn):
                    leftRight *= 0.5
                # Determine the drive power levels
                driveLeft = -upDown
                driveRight = -upDown
                if leftRight < -0.05:
                    # Turning left
                    driveLeft *= 1.0 + (2.0 * leftRight)
                elif leftRight > 0.05:
                    # Turning right
                    driveRight *= 1.0 - (2.0 * leftRight)
                # Check for button presses
                if joystick.get_button(buttonResetEpo):
                    pass
                if joystick.get_button(buttonSlow):
                    driveLeft *= slowFactor
                    driveRight *= slowFactor
                # Set the motors to the new speeds
                # PBR.SetMotor1(driveRight * maxPower)
                # PBR.SetMotor2(-driveLeft * maxPower)
                # print(driver.joystck(driveRight * maxPower), driver.joystck(-driveLeft * maxPower))
                driver.motors(driver.joystck(driveRight * maxPower), driver.joystck(driveLeft * maxPower))
        # Change the LED to reflect the status of the EPO latch
        # PBR.SetLed(PBR.GetEpo())
        # Wait for the interval period
        time.sleep(interval)
    # Disable all drives
    # PBR.MotorsOff()
except KeyboardInterrupt:
    # CTRL+C exit, disable all drives
    # PBR.MotorsOff()
    pass
