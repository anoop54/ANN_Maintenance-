# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 10:33:40 2018

@author: Anoop
"""
import math
import numpy as np 
import pandas as pd 
from numpy.random import *
import matplotlib.pyplot as plt
import peakutils
import paho.mqtt.client as paho
import datetime
import time
import ast
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


Nf = 3      #3 different waves
N = 1000   #t/x value for the frequencies 

t = np.arange(N, dtype = float)   #array of values from 0 to 10000
Ts = rand(3)*2000+10              #array of 3 different periods 
fs = 1/Ts                         #array of 3 different frequencies 
amp = rand(Nf)*200+10             #array of 3 different amplitudes 
phi = rand(Nf)*2*math.pi          #array of 3 different phases 
h = np.zeros(N)                   #Empty array of zeros of size 10000
count= 0 
vibrations = np.array([])
vibs = np.array([])

prevtime = 0

s1 = amp[0]*np.sin(2*math.pi*t*fs[0]+phi[0])  #first signal
s2 = amp[1]*np.sin(2*math.pi*t*fs[1]+phi[1])  #second signal
s3 = amp[2]*np.sin(2*math.pi*t*fs[2]+phi[2])  #third signal

sig = s1+s2+s3 #Total signal
sign= sig + randn(N)*3*sig +randn(N)*700  #adding random noise 

def sub_to_mqtt():
    client = paho.Client()
    client.on_subscribe = on_subscribe
    client.on_message = on_message
    client.connect("104.238.164.118", 8883)
    client.subscribe("LearningFactory/IIoT", qos=0)

def signal_to_freq(signal,sample_freq,N):      
    Hn = np.fft.fft(signal)    
    ind = np.arange(0,(N/2),dtype = int)    
    freq = np.fft.fftfreq(N,sample_freq)    
    freq = freq[ind]
    psd = abs(Hn[ind])**2 +  abs(Hn[ind])**2
    pssd = pd.DataFrame(psd)
    return freq,psd    
    #plt.plot(freq,psd)
    
    
def signal_peaks(psd,freq):
    indices = peakutils.indexes(psd, thres=0.04, min_dist=0) #Get peaks indexes in psd
    i=0  
    freq = np.array(freq)  #x array of fft data
    psd = np.array(psd)#y array of fft data    
    peak_freqs = np.array([]) 
    
    while(i<3): #3 repersents number of peaks
        peaks = psd[indices] #magnitude of peaks
        i = i+1
        maximum = np.argmax(peaks) #Get index max index value
        peak_freqs = np.append(peak_freqs,indices[maximum])
        indices = np.delete(indices,maximum)
    peak_freqs = peak_freqs.astype(int)
    
    return freq[peak_freqs]

def on_subscribe(client, userdata, mid, granted_qos):
    print("Subscribed: "+str(mid)+" "+str(granted_qos))

#singleMsgAtATime
'''
def on_message(client, userdata, msg):
    #print(str(int(msg.payload)))
    global count
    global vibrations
    global prevtime
    global vibs
    
    x = msg.payload
    vibs = np.append(vibs,int(x))
    if(count==1000):
        #print(vibrations)
        vibrations = vibs
        x = datetime.datetime.now().time()
        time = x.second +x.microsecond/1000000
       # freq,psd = signal_to_freq(vibrations,100,100)
        #print(vibrations)        
        #plt.plot(freq,psd)
        #plt.pause(5) # show it for 5 seconds
        vibs= np.array([])
        count = 0
        prevtime = time
    count=count+1'''
    

def on_message(client, userdata, msg):
    #print(str(int(msg.payload)))
    global count
    global vibrations
    global prevtime
    global vibs 
    x = msg.payload
    x = x.decode("utf-8")    
    x = x.replace("]", "")
    x = x.replace("[", "")   
    x =np.fromstring(x, dtype=int, sep=',')   
    vibrations = np.array(x)
    

    #print(x)

#main
client = paho.Client()
client.on_subscribe = on_subscribe
client.on_message = on_message
client.connect("104.238.164.118", 8883)
client.subscribe("LearningFactory/IIoT", qos=0)

client.loop_start()
#client.loop_forever()
time.sleep(5)
count = 0 
while(1):
   # print(vibrations)
    freq,psd = signal_to_freq(vibrations,1,128)
   
    print(predict(psd))
    #plt.clf()
    #plt.plot(psd[1:])
    #plt.pause(0.05)
    time.sleep(2)
    
    #predict using model that has been loaded 
    
    '''
    f = open('load.txt','a')
    
    save_array = str(psd).replace("\n","")
    save_array = save_array.replace("[","")
    save_array = save_array.replace("]","")
    save_array = save_array.replace(" ",",")
    print(save_array)
    f.write(save_array +',0'+'\n')
    f.close()
    '''
    
    
client.disconnect()


#TESTING
'''

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")







colums  = np.array(range(64))
colums = np.append(colums,'y')
df = pd.read_csv('load.txt',names = colums)

X = df.iloc[:,0:64]
Y = df.iloc[:,64:]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)




sc= StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

def predict(vib): 
    x= np.array([vib])
    return loaded_model.predict(sc.transform(x))
'''


'''
time.sleep(15)
while(1):
    time.sleep(1)
    print("GO")
    #print(vibrations)
    freq,psd = signal_to_freq(vibrations,1,1000)
    #peak_freqs = signal_peaks(psd,freq)
    plt.clf()
    plt.plot(psd[1:])
    plt.pause(0.05)
'''