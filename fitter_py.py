import sys
import matplotlib.pyplot as plt
import time
import prediction_module
import serial_module


# start reading from port
def __main__():
    if len(sys.argv) >= 0:
        while True:
            start_t = time.time()
            data_obj = prediction_module.Data_class()
            prediction_obj = prediction_module.PredictionClass()

            # Read from serial port
            # ser = serial_module.serialRead('/dev/tty.usbmodem1411', 9600)
            # data_read = ser.readData(1500)

            # Read data from file
            input_f_name = "../testData/capture_0_2.txt"
            inputF = file(input_f_name, "r")
            data_read = [float(line) for line in inputF.readlines()]

            data_obj.fill(data_read, 100.0)
            prediction_obj.performFFT(data_obj)
            blah, blah2 = prediction_obj.performFIT_SSE(data_obj)
            # data_obj.showPlot()

            # use the learnt model to predict point
            time_to_lowest, time_to_move, vel, pred, t = prediction_obj.predict_sec(data_obj, 10, 0.1)
            e_time = time.time() - start_t
            time_to_move -= e_time
            if time_to_move > 0.0:
                print time_to_move
                plt.plot(t, pred)
                break

    else:
        print "ERROR: Input correct number of inputs"
    return


__main__()
