#!/usr/bin/env python
# This is a modified (reduced) version of mavextract.py from pymavlink source

from __future__ import print_function

import os
import struct
import numpy as np
import math

main_path = os.getcwd()

from pymavlink import mavutil
# For testing:
#land_target_lat = -35.3627622717389
#land_target_lon = 149.165164232254

def older_message(m, lastm):
    '''return true if m is older than lastm by timestamp'''
    atts = {'time_boot_ms' : 1.0e-3,
            'time_unix_usec' : 1.0e-6,
            'time_usec' : 1.0e-6}
    for a in list(atts.keys()):
        if hasattr(m, a):
            mul = atts[a]
            t1 = m.getattr(a) * mul
            t2 = lastm.getattr(a) * mul
            if t2 >= t1 and t2 - t1 < 60:
                return True
    return False

def process(filename):
    '''process one logfile'''
    #Create output files for each parameter of interest individually
    #Parameters - IMU 1/2, Mag, Baro, Arsp
    #Also need to grab state at time of GPS loss, including variances.
    #Also need EKF data

    imu1_log = main_path + "/IMU1.txt"
    imu2_log = main_path + "/IMU2.txt"
    mag_log = main_path + "/MAG.txt"
    baro_log = main_path + "/BARO.txt"
    arsp_log= main_path + "/ARSP.txt"
    gps_log = main_path + "/GPS.txt"
    ekf1_log = main_path + "/ekf1.txt"
    ekf2_log = main_path + "/ekf2.txt"
    ekf3_log = main_path + "/ekf3.txt"
    ekf4_log = main_path + "/ekf4.txt"

    logs = [imu1_log,imu2_log,mag_log,baro_log,arsp_log,gps_log,ekf1_log,ekf2_log,ekf3_log,ekf4_log]

    logs_w = []

    for log in logs:
        logs_w.append(open(log, 'w'))

    print("Processing %s" % filename)
    mlog = mavutil.mavlink_connection(filename, notimestamps=False,
                                      robust_parsing=False)


    dirname = os.path.dirname(filename)

    messages = []

    while True:
        m = mlog.recv_match()
        if m is None:
            break

        mtype = m.get_type()
        if mtype in messages:
            if older_message(m, messages[mtype]):
                continue
        #print(mtype)
        if mtype =="GPS":
            logs_w[5].write("%i,%i,%f,%f,%f\n" % (m.TimeUS,m.Status,m.Lat,m.Lng,m.Alt))
        elif mtype =="IMU":
            if m.I == 0:
                logs_w[0].write("%i,%f,%f,%f,%f,%f,%f\n" % (m.TimeUS,m.GyrX,m.GyrY,m.GyrZ,m.AccX,m.AccY,m.AccZ))
            elif m.I == 1:
                logs_w[1].write("%i,%f,%f,%f,%f,%f,%f\n" % (m.TimeUS,m.GyrX,m.GyrY,m.GyrZ,m.AccX,m.AccY,m.AccZ))

        elif mtype == "MAG":
            logs_w[2].write("%i,%f,%f,%f,%f,%f,%f,%f,%f,%f\n" % (m.TimeUS,m.MagX,m.MagY,m.MagZ,m.OfsX,m.OfsY,m.OfsZ,m.MOX,m.MOY,m.MOZ))

        elif mtype == "BARO":
            logs_w[3].write("%i,%f\n" % (m.TimeUS,m.Alt))

        elif mtype == "ARSP":
            logs_w[4].write("%i,%f\n" % (m.TimeUS,m.Airspeed))


        #XKF log messages relate to the EKF 3 instances, here I am just looking at the first (C=0)

        #XKF1 has some of the state estimates
        elif mtype == "XKF1":
            if m.C == 0:
                logs_w[6].write("%i,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n" % (m.TimeUS,m.Roll,m.Pitch,m.Yaw,m.VN,m.VE,m.VD,m.PN,m.PE,m.PD,m.GX,m.GY,m.GZ))
            else:
                pass
        
        #XKF2 Has the remaining state estimates

        elif mtype == "XKF2":
            if m.C == 0:
                logs_w[7].write("%i,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n" % (m.TimeUS,m.AX,m.AY,m.AZ,m.VWN,m.VWE,m.MN,m.ME,m.MD,m.MX,m.MY,m.MZ))
            else:
                pass

        # XKF3 has innovation values
        elif mtype == "XKF3":
            if m.C == 0:
                logs_w[8].write("%i,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n" % (m.TimeUS,m.IVN,m.IVE,m.IVD,m.IPN,m.IPE,m.IPD,m.IMX,m.IMY,m.IMZ,m.IYAW,m.IVT))
            else:
                pass

        # XKF4 has variance data (square root of the variances)
        elif mtype == "XKF4":
            if m.C == 0:
                logs_w[9].write("%i,%f,%f,%f,%f,%f\n" % (m.TimeUS,m.SV,m.SP,m.SH,m.SM,m.SVT))
            else:
                pass

        # Get stop time by waiting for it to transition to QLAND mode.
        elif mtype == "MODE":
            print(m)
            if m.Mode == 20:
                print("I made it to QLAND Mode")
                stop_time = m.TimeUS
            else:
                pass

        #Get time that we start constant heading run, standalone EKF will start at fixed time after this
        elif mtype == "MSG":
            if m.Message == "Mission: 3 WP":
                time_start_standalone = m.TimeUS
            else:
                pass

        elif mtype == "ORGN":
            orgn_lat =m.Lat
            orgn_lon = m.Lng
            orgn_alt =  m.Alt
            orgn = np.array([orgn_lat,orgn_lon,orgn_alt])

        else:
            pass

    for log in logs_w:
        log.close()

    #stop_time = 3374126476
    stats = main_path + "/stats.txt"
    stats_w = open(stats, 'w')
    stats_w.write("%i,%i\n" % (time_start_standalone, stop_time))
    stats_w.write("%f,%f,%f\n" % (orgn[0], orgn[1], orgn[2]))

    return time_start_standalone, stop_time, orgn
    #return total_time, total_xtrack_error, total_alt_error, land_miss_distance


# gps_distance function taken from mp_util.py file in MavProxy source
def gps_distance(lat1, lon1, lat2, lon2):
    '''return distance between two points in meters,
    coordinates are in degrees
    thanks to http://www.movable-type.co.uk/scripts/latlong.html'''

    radius_of_earth = 6378100.0 # in meters

    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    lon1 = math.radians(lon1)
    lon2 = math.radians(lon2)
    dLat = lat2 - lat1
    dLon = lon2 - lon1

    a = math.sin(0.5*dLat)**2 + math.sin(0.5*dLon)**2 * math.cos(lat1) * math.cos(lat2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0-a))
    return radius_of_earth * c